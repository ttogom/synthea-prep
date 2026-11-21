"""Offline tests for Plan judging protocol 2 (judge_plans_v2.py).

No model, GPU, network, or Hugging Face download: a fake backend answers.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plan_judge_v2 import colab, protocol, runner, validate_models  # noqa: E402
from scripts.plan_judge_v2.backends import FakeBackend, Result, VllmBackend, redact  # noqa: E402
from scripts.plan_judge_v2.inputs import scan_results  # noqa: E402
from scripts.plan_judge_v2.store import (  # noqa: E402
    LEASE, Lease, LeaseHeld, LeaseLost, OutputConflict, atomic_write_json, open_manifest,
)

FAST_LEASE = {"settle": 0, "heartbeat": 3600}
PILOT_CASES = ROOT / "experiments/validation/plan_judge_v2_pilot_cases.jsonl"
NOTEBOOK = ROOT / "notebooks/plan_judge_v2_colab.ipynb"


def full_reply(messages, schema):
    """Mark every reference action full, citing generated item 1; no harmful actions."""
    count = schema["properties"]["reference_items"]["minItems"]
    return json.dumps({"reference_items": [{"ref": n, "match": "full", "generated": [1]} for n in range(1, count + 1)],
                       "harmful_actions": []})


def predictions(patient, tag):
    return [{"condition": f"dx {rank}", "plan": [f"{tag} {patient} rank{rank} action A", f"{tag} rank{rank} action B"]}
            for rank in (1, 2, 3)]


class Fixture:
    """A tiny base+agentic cohort written to a temporary directory."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.patients = self.dir / "so.jsonl"
        self.labels = self.dir / "labels.jsonl"
        self.results = self.dir / "run" / "results.jsonl"
        self.results.parent.mkdir()
        self.out = self.dir / "out"
        ids = ["p1", "p2", "p3", "p4"]
        self.patients.write_text("".join(json.dumps({"id": i, "subjective": f"S of {i}", "objective": f"O of {i}"}) + "\n"
                                         for i in ids))
        plans = {"p1": ["Penicillin V 500 MG", "Throat culture"], "p2": ["Lisinopril 10 MG"],
                 "p3": ["Care plan (record artifact)"], "p4": ["ECG"]}
        self.labels.write_text("".join(json.dumps({"id": i, "condition": "some_condition", "plan": plans[i],
                                                   "has_presenting_evidence": True}) + "\n" for i in ids))
        records = []
        for i in ids:
            slot = lambda tag: {"status": "completed", "predictions": predictions(i, tag), "evaluation": None}  # noqa: E731
            variants = {"llm_only": {"status": "completed", "assessment_plans": {"gen_a": slot("base-a"),
                                                                                  "gen_b": slot("base-b")}},
                        "agentic_both": {"status": "completed", "assessment_plans": {"gen_a": slot("agent-a"),
                                                                                      "gen_b": slot("agent-b")}}}
            records.append({"id": i, "status": "completed", "components": {}, "variants": variants})
        # Edge cases: empty rank-1 plan, failed generation.
        records[3]["variants"]["llm_only"]["assessment_plans"]["gen_a"]["predictions"][0]["plan"] = []
        records[3]["variants"]["llm_only"]["assessment_plans"]["gen_b"] = {"status": "failed_incomplete",
                                                                           "predictions": None}
        self.results.write_text("".join(json.dumps(r) + "\n" for r in records))

    def argv(self, *extra, variants=("llm_only",)):
        return ["--results", str(self.results), "--patients", str(self.patients), "--labels", str(self.labels),
                "--variants", *variants, "--out-dir", str(self.out), "--run-name", "run1", "--model", "fake/judge",
                "--backend", "vllm", *extra]

    def variant_dir(self, variant="llm_only"):
        return self.out / "run1" / variant

    def rows(self, variant="llm_only"):
        path = self.variant_dir(variant) / "judgments.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def manifest(self, variant="llm_only"):
        return json.loads((self.variant_dir(variant) / "manifest.json").read_text())


class Harness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fx = Fixture(self.temp.name)
        self.backends = []
        self.logs = []

    def run_judge(self, *extra, responder=full_reply, variants=("llm_only",), revision="rev-1"):
        def factory(name, model, **options):
            backend = FakeBackend(responder, model_id=model, revision=revision)
            self.backends.append(backend)
            return backend
        return runner.run(self.fx.argv(*extra, variants=variants), backend_factory=factory,
                          log=self.logs.append, lease_options=FAST_LEASE)


class ProtocolTests(unittest.TestCase):
    def test_version_two_and_distinct_from_protocol_one(self):
        self.assertEqual(protocol.PROTOCOL_VERSION, 2)
        self.assertEqual(protocol.JUDGED_RANK, 1)
        identity = protocol.protocol_identity()
        self.assertEqual(identity["version"], 2)
        self.assertEqual(identity["prompt_template_sha256"],
                         hashlib.sha256(protocol.PROMPT_TEMPLATE.encode()).hexdigest())
        v1 = (ROOT / "med_copilot/prompts/correctness_judges.py").read_text()
        self.assertNotIn(protocol.PROMPT_TEMPLATE[:200], v1)

    def test_prompt_numbers_both_plans_and_has_no_copyable_values(self):
        [message] = protocol.render_messages("strep", "S text", "O text", ["Pen V 500", "Culture"], ["Amox", "Rest"])
        content = message["content"]
        self.assertEqual(message["role"], "user")
        self.assertIn("1. Pen V 500\n2. Culture", content)
        self.assertIn("1. Amox\n2. Rest", content)
        self.assertIn("from 1 to 2 exactly once", content)
        # The form names types only; no filled-in example a small judge could copy.
        self.assertNotRegex(content, r'"ref": \d|"generated": \[?\d|"match": "(full|partial|none)"[,}]')

    def test_schema_bounds_counts(self):
        schema = protocol.response_schema(2, 3)
        items = schema["properties"]["reference_items"]
        self.assertEqual((items["minItems"], items["maxItems"]), (2, 2))
        self.assertEqual(items["items"]["properties"]["generated"]["items"]["maximum"], 3)
        self.assertEqual(items["items"]["properties"]["match"]["enum"], ["full", "partial", "none"])

    def reply(self, items, harmful=()):
        return json.dumps({"reference_items": items, "harmful_actions": list(harmful)})

    def test_parse_and_score(self):
        parsed = protocol.parse_response(self.reply(
            [{"ref": 1, "match": "full", "generated": [2]}, {"ref": 2, "match": "partial", "generated": [1, 3]}],
            [{"generated": 3, "why": " unsafe "}]), 2, 3)
        self.assertEqual(parsed["harmful_actions"], [{"generated": 3, "why": "unsafe"}])
        self.assertEqual(protocol.score(parsed), {"credits": [1.0, 0.5], "coverage": 0.75, "harmful_count": 1,
                                                  "penalized": 0.5})

    def test_harmful_count_never_changes_coverage_and_penalized_floors_at_zero(self):
        parsed = {"reference_items": [{"ref": 1, "match": "partial", "generated": [1]}],
                  "harmful_actions": [{"generated": 1, "why": "x"}, {"generated": 2, "why": "y"}]}
        self.assertEqual(protocol.score(parsed), {"credits": [0.5], "coverage": 0.5, "harmful_count": 2,
                                                  "penalized": 0.0})

    def test_fenced_reply_and_extra_top_level_keys_are_accepted(self):
        body = json.dumps({"reference_items": [{"ref": 1, "match": "none", "generated": []}],
                           "harmful_actions": [], "notes": "ignored"})
        parsed = protocol.parse_response(f"```json\n{body}\n```", 1, 1)
        self.assertEqual(parsed["reference_items"][0]["match"], "none")

    def test_invalid_replies_are_rejected(self):
        ok = {"ref": 1, "match": "full", "generated": [1]}
        cases = {
            "not json": "full",
            "list": "[]",
            "missing harmful": json.dumps({"reference_items": [ok]}),
            "wrong count": self.reply([ok, {**ok, "ref": 2}]),
            "out of order": self.reply([{**ok, "ref": 2}]),
            "bad match": self.reply([{**ok, "match": "mostly"}]),
            "bool ref": self.reply([{**ok, "ref": True}]),
            "citation out of range": self.reply([{**ok, "generated": [3]}]),
            "duplicate citation": self.reply([{**ok, "generated": [1, 1]}]),
            "none cites": self.reply([{"ref": 1, "match": "none", "generated": [1]}]),
            "full without citation": self.reply([{"ref": 1, "match": "full", "generated": []}]),
            "extra item key": self.reply([{**ok, "credit": 1}]),
            "harmful out of range": self.reply([ok], [{"generated": 9, "why": "x"}]),
            "harmful duplicate": self.reply([ok], [{"generated": 1, "why": "x"}, {"generated": 1, "why": "y"}]),
            "harmful blank why": self.reply([ok], [{"generated": 1, "why": " "}]),
            "duplicate key": '{"reference_items": [], "reference_items": [], "harmful_actions": []}',
            "nan": '{"reference_items": [{"ref": NaN}], "harmful_actions": []}',
            "empty": "",
        }
        for name, content in cases.items():
            with self.subTest(name), self.assertRaises(protocol.JudgeResponseError):
                protocol.parse_response(content, 1, 2)

    def test_unavailable_reference_rules_match_protocol_one(self):
        try:
            from scripts.med_copilot_judges import unavailable_plan_reason as v1
        except ImportError:  # the main pipeline's dependencies are not installed (e.g. on Colab)
            v1 = None
        for plan, reason in ((None, "missing_reference"), ([], "missing_reference"), ([" "], "missing_reference"),
                             (["none recorded"], "no_recorded_plan"), (["X ( Record  Artifact )"], "record_artifact_plan"),
                             (["Pen V"], None)):
            with self.subTest(plan=plan):
                self.assertEqual(protocol.unavailable_reference_reason(plan), reason)
                if v1 is not None:
                    self.assertEqual(v1(plan), reason)


class InputTests(Harness):
    def scan(self, variants, models=None):
        from scripts.plan_judge_v2.inputs import load_by_id
        return scan_results(self.fx.results, load_by_id(self.fx.labels, ("condition", "plan")),
                            load_by_id(self.fx.patients, ("subjective", "objective")), variants, models)

    def test_base_and_agentic_variants_and_rank_one_only(self):
        scan = self.scan(["llm_only", "agentic_both"])
        self.assertEqual(scan.variants_found, ["llm_only", "agentic_both"])
        self.assertEqual(len(scan.items["llm_only"]), 8)
        self.assertEqual(len(scan.items["agentic_both"]), 8)
        self.assertEqual(scan.generator_models, ["gen_a", "gen_b"])
        item = next(i for i in scan.items["agentic_both"] if i.patient_id == "p1")
        self.assertEqual(item.generated_actions, ["agent-a p1 rank1 action A", "agent-a rank1 action B"])
        self.assertEqual(scan.sha256, hashlib.sha256(self.fx.results.read_bytes()).hexdigest())

    def test_statuses_for_unusable_plans(self):
        items = {(i.patient_id, i.generator_model): i for i in self.scan(["llm_only"]).items["llm_only"]}
        self.assertEqual(items["p3", "gen_a"].reason, "record_artifact_plan")
        self.assertEqual((items["p4", "gen_a"].status, items["p4", "gen_a"].reason), ("empty", "empty_generated_plan"))
        self.assertEqual(items["p4", "gen_b"].reason, "generation_failed")
        self.assertEqual(items["p1", "gen_a"].status, "pending")

    def test_all_selects_every_variant_and_models_filter(self):
        scan = self.scan(["all"], ["gen_a"])
        self.assertEqual(set(scan.items), {"llm_only", "agentic_both"})
        self.assertTrue(all(i.generator_model == "gen_a" for items in scan.items.values() for i in items))

    def test_unknown_variant_or_model_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not in results.jsonl"):
            self.scan(["guidelines"])
        with self.assertRaisesRegex(ValueError, "gen_z"):
            self.scan(["llm_only"], ["gen_z"])

    def test_legacy_records_without_variants_are_rejected(self):
        self.fx.results.write_text(json.dumps({"id": "p1", "assessment_plans": {}}) + "\n")
        with self.assertRaisesRegex(ValueError, "schema-4"):
            self.scan(["llm_only"])


class DenominatorTests(Harness):
    """Every patient x selected variant x expected generator yields exactly one item or row."""

    scan = InputTests.scan

    def edit_records(self, change):
        records = [json.loads(line) for line in self.fx.results.read_text().splitlines()]
        change({r["id"]: r for r in records})
        self.fx.results.write_text("".join(json.dumps(r) + "\n" for r in records))

    def by_key(self, variant, models=None):
        items = self.scan([variant], models).items[variant]
        keys = [i.key for i in items]
        self.assertEqual(len(keys), len(set(keys)), "duplicate patient/generator items")
        return {i.key: i for i in items}

    def assert_complete(self, scan):
        for variant, items in scan.items.items():
            with self.subTest(variant=variant):
                self.assertEqual({i.key for i in items},
                                 {(p, m) for p in ("p1", "p2", "p3", "p4") for m in scan.generator_models})

    def test_runner_shaped_blocked_variant_keeps_every_generator(self):
        # run_experiments.py leaves every slot not_processed when evidence failure blocks a variant.
        def block(records):
            records["p2"]["variants"]["agentic_both"] = {
                "status": "failed_incomplete", "error": {"type": "EvidenceUnavailable", "message": "x"},
                "assessment_plans": {m: {"status": "not_processed", "error": None, "predictions": None}
                                     for m in ("gen_a", "gen_b")}}
        self.edit_records(block)
        items = self.by_key("agentic_both")
        self.assertEqual(len(items), 8)
        for model in ("gen_a", "gen_b"):
            item = items["p2", model]
            self.assertEqual((item.status, item.reason, item.reasons),
                             ("unavailable", "variant_not_completed", ("variant_not_completed",)))
            self.assertEqual(item.source, {"variant_status": "failed_incomplete", "generation_status": "not_processed",
                                           "generation_error_type": "EvidenceUnavailable"})

    def test_blocked_variant_without_slots_or_absent_variant_is_filled_in(self):
        def block(records):
            records["p1"]["variants"]["agentic_both"] = {"status": "failed_incomplete", "assessment_plans": {}}
            del records["p2"]["variants"]["agentic_both"]
        self.edit_records(block)
        for selection in (["agentic_both"], ["all"]):
            with self.subTest(selection=selection):
                scan = self.scan(selection)
                self.assert_complete(scan)
                items = {i.key: i for i in scan.items["agentic_both"]}
                for key in (("p1", "gen_a"), ("p1", "gen_b"), ("p2", "gen_a"), ("p2", "gen_b")):
                    self.assertEqual(items[key].reason, "variant_not_completed")
                    self.assertEqual(items[key].status, "unavailable")
                self.assertEqual(items["p2", "gen_a"].source["variant_status"], None)
                self.assertEqual(items["p1", "gen_a"].condition, "some condition")

    def test_distinct_generation_reasons(self):
        def damage(records):
            slots = records["p1"]["variants"]["agentic_both"]["assessment_plans"]
            slots["gen_a"]["predictions"] = None                     # completed slot, no Plan
            slots["gen_b"]["predictions"][0]["plan"] = "not a list"  # malformed Plan
            slots = records["p2"]["variants"]["agentic_both"]["assessment_plans"]
            slots["gen_a"] = {"status": "failed_incomplete", "error": {"type": "Timeout"}, "predictions": None}
            del slots["gen_b"]                                       # completed variant missing a slot
            slots = records["p4"]["variants"]["agentic_both"]["assessment_plans"]
            slots["gen_b"]["status"] = "not_processed"               # slot skipped in a completed variant
        self.edit_records(damage)
        items = self.by_key("agentic_both")
        self.assertEqual(len(items), 8)
        expected = {("p1", "gen_a"): "missing_plan", ("p1", "gen_b"): "invalid_generation",
                    ("p2", "gen_a"): "generation_failed", ("p2", "gen_b"): "missing_plan",
                    ("p4", "gen_b"): "generation_not_processed", ("p3", "gen_a"): "record_artifact_plan",
                    ("p4", "gen_a"): None}
        for key, reason in expected.items():
            with self.subTest(key=key):
                self.assertEqual(items[key].reason, reason)
        self.assertEqual(items["p2", "gen_a"].source["generation_error_type"], "Timeout")
        self.assertEqual(items["p2", "gen_b"].source["generation_status"], None)

    def test_generation_and_reference_reasons_are_both_kept(self):
        def damage(records):
            slots = records["p3"]["variants"]["llm_only"]["assessment_plans"]
            slots["gen_a"] = {"status": "failed_incomplete", "predictions": None}
            slots["gen_b"]["predictions"][0]["plan"] = ["  "]
        self.edit_records(damage)
        items = self.by_key("llm_only")
        failed, empty = items["p3", "gen_a"], items["p3", "gen_b"]
        self.assertEqual((failed.status, failed.reason, failed.reasons),
                         ("unavailable", "generation_failed", ("generation_failed", "record_artifact_plan")))
        # An empty Plan cannot be scored 0 against an unusable reference.
        self.assertEqual((empty.status, empty.reason, empty.reasons),
                         ("unavailable", "record_artifact_plan", ("record_artifact_plan", "empty_generated_plan")))
        self.assertIsNone(empty.reference_actions)

    def test_generator_filter_sets_the_denominator(self):
        scan = self.scan(["agentic_both"], ["gen_b"])
        self.assertEqual(scan.generator_models, ["gen_b"])
        self.assert_complete(scan)

    def test_judge_rows_manifest_and_csv_keep_the_denominator(self):
        def block(records):
            records["p1"]["variants"]["agentic_both"] = {
                "status": "failed_incomplete", "error": {"type": "EvidenceUnavailable"},
                "assessment_plans": {m: {"status": "not_processed", "predictions": None} for m in ("gen_a", "gen_b")}}
            del records["p2"]["variants"]["agentic_both"]["assessment_plans"]["gen_b"]
        self.edit_records(block)
        self.assertEqual(self.run_judge("--csv", variants=("agentic_both",)), runner.EXIT_OK)
        rows = self.fx.rows("agentic_both")
        self.assertEqual(len(rows), 8)
        by_key = {(r["patient_id"], r["generator_model"]): r for r in rows}
        self.assertEqual(by_key["p1", "gen_b"]["reason"], "variant_not_completed")
        self.assertEqual(by_key["p1", "gen_b"]["source"]["generation_error_type"], "EvidenceUnavailable")
        self.assertEqual(by_key["p2", "gen_b"]["reasons"], ["missing_plan"])
        manifest = self.fx.manifest("agentic_both")
        self.assertEqual(manifest["selection"]["generator_models_expected"], ["gen_a", "gen_b"])
        self.assertEqual(manifest["selection"]["patients_in_results"], 4)
        counts = manifest["counts"]
        self.assertEqual(counts["plans"], 8)
        self.assertEqual(counts["unavailable_by_reason"],
                         {"variant_not_completed": 2, "missing_plan": 1, "record_artifact_plan": 2})
        self.assertEqual(counts["scored"] + counts["unavailable"], counts["plans"])
        self.assertEqual(sum(counts["unavailable_by_reason"].values()), counts["unavailable"])
        with open(self.fx.variant_dir("agentic_both") / "judgments.csv", newline="") as handle:
            table = {(r["patient_id"], r["generator_model"]): r for r in csv.DictReader(handle)}
        self.assertEqual(len(table), 8)
        self.assertEqual((table["p1", "gen_a"]["reasons"], table["p1", "gen_a"]["variant_status"]),
                         ("variant_not_completed", "failed_incomplete"))

    def test_dry_run_reports_denominator_and_patients_missing_from_results(self):
        with open(self.fx.patients, "a") as handle:
            handle.write(json.dumps({"id": "p5", "subjective": "S", "objective": "O"}) + "\n")
        self.assertEqual(self.run_judge("--dry-run", variants=("llm_only",)), runner.EXIT_OK)
        log = "\n".join(self.logs)
        self.assertIn("4 patients x 2 generators (gen_a, gen_b)", log)
        self.assertIn("1 patient(s) in so.jsonl have no results record; only patients in the results file", log)
        self.assertIn('"generation_failed": 1', log)
        self.assertFalse(self.fx.out.exists())


class JudgeRunTests(Harness):
    def test_one_call_per_pending_rank_one_plan_and_inputs_untouched(self):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.fx.results, self.fx.patients, self.fx.labels)}
        prompts = []

        def responder(messages, schema):
            prompts.append(messages[0]["content"])
            return full_reply(messages, schema)
        self.assertEqual(self.run_judge("--csv", responder=responder), runner.EXIT_OK)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
        self.assertEqual(before, after)
        # p1, p2 x gen_a, gen_b judged; p3 record artifact; p4 empty and failed: 4 calls for 8 Plans.
        self.assertEqual(len(prompts), 4)
        self.assertTrue(all("rank1" in p and "rank2" not in p and "rank3" not in p for p in prompts))
        rows = self.fx.rows()
        self.assertEqual(len(rows), 8)
        by_key = {(r["patient_id"], r["generator_model"]): r for r in rows}
        self.assertEqual(by_key["p1", "gen_a"]["coverage"], 1.0)
        self.assertEqual(by_key["p4", "gen_a"]["reason"], "empty_generated_plan")
        self.assertEqual(by_key["p4", "gen_a"]["coverage"], 0.0)
        self.assertEqual(by_key["p3", "gen_a"]["status"], "unavailable")
        self.assertEqual(by_key["p1", "gen_a"]["raw_response"], full_reply(None, protocol.response_schema(2, 2)))
        self.assertIsNotNone(by_key["p1", "gen_a"]["parsed"])

    def test_manifest_holds_shared_metadata_and_rows_reference_it(self):
        self.run_judge()
        manifest = self.fx.manifest()
        self.assertEqual(manifest["protocol"]["version"], 2)
        self.assertEqual(manifest["variant"], "llm_only")
        self.assertEqual(manifest["inputs"]["results"]["sha256"],
                         hashlib.sha256(self.fx.results.read_bytes()).hexdigest())
        self.assertEqual(manifest["judge"]["model_id"], "fake/judge")
        self.assertEqual(manifest["judge"]["revision_resolved"], "rev-1")
        self.assertEqual(manifest["sessions"][0]["status"], "completed")
        self.assertEqual(manifest["counts"]["scored"], 5)
        for row in self.fx.rows():
            self.assertEqual(row["run_id"], manifest["run_id"])
            self.assertNotIn("inputs", row)
            self.assertNotIn("judge", row)
            self.assertNotIn("protocol", row)

    def test_separate_outputs_per_variant_with_csv(self):
        self.assertEqual(self.run_judge("--csv", variants=("llm_only", "agentic_both")), runner.EXIT_OK)
        for variant, plans in (("llm_only", 8), ("agentic_both", 8)):
            with self.subTest(variant=variant):
                self.assertEqual(self.fx.manifest(variant)["variant"], variant)
                with open(self.fx.variant_dir(variant) / "judgments.csv", newline="") as handle:
                    table = list(csv.DictReader(handle))
                self.assertEqual(len(table), plans)
                self.assertEqual({r["variant"] for r in table}, {variant})
        self.assertNotEqual(self.fx.manifest("llm_only")["run_id"], self.fx.manifest("agentic_both")["run_id"])
        self.assertEqual(len(self.backends), 1)  # loaded once for both variants
        self.assertTrue(self.backends[0].loaded)

    def test_batches_respect_batch_size(self):
        self.run_judge("--batch-size", "3")
        self.assertEqual(self.backends[0].batches, [3, 1])

    def test_resume_skips_finished_plans_and_does_not_reload_the_model(self):
        self.run_judge()
        self.assertEqual(self.run_judge(), runner.EXIT_OK)
        self.assertFalse(self.backends[1].loaded)
        self.assertEqual(len(self.fx.rows()), 8)
        self.assertEqual(len(self.fx.manifest()["sessions"]), 2)

    def test_limit_then_resume_completes(self):
        self.assertEqual(self.run_judge("--limit", "1"), runner.EXIT_INCOMPLETE)
        self.assertEqual(self.fx.manifest()["counts"]["not_attempted"], 3)
        self.assertEqual(self.run_judge(), runner.EXIT_OK)
        self.assertEqual(self.fx.manifest()["counts"]["scored"], 5)

    def test_invalid_replies_are_saved_and_retried_until_max_attempts(self):
        self.assertEqual(self.run_judge("--max-attempts", "2", responder=lambda m, s: "not json"),
                         runner.EXIT_INCOMPLETE)
        rows = self.fx.rows()
        invalid = [r for r in rows if r["status"] == "invalid_response"]
        self.assertEqual(len(invalid), 4)
        self.assertEqual(invalid[0]["raw_response"], "not json")
        self.assertEqual(invalid[0]["error"]["type"], "JudgeResponseError")
        self.run_judge("--max-attempts", "2", responder=lambda m, s: "still not json")
        self.assertEqual(self.fx.manifest()["counts"]["failed_exhausted"], 4)
        self.run_judge("--max-attempts", "2")  # exhausted: no further calls
        self.assertEqual(self.backends[-1].batches, [])
        self.assertEqual(self.run_judge("--max-attempts", "3"), runner.EXIT_OK)
        attempts = [r["attempt"] for r in self.fx.rows() if r["status"] == "scored" and r["prompt_sha256"]]
        self.assertEqual(attempts, [3, 3, 3, 3])

    def test_request_errors_are_recorded_and_redacted(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_secret_value_123"}):
            def responder(messages, schema):
                raise RuntimeError("401 for token hf_secret_value_123")
            self.assertEqual(self.run_judge(responder=responder), runner.EXIT_INCOMPLETE)
        errors = [r for r in self.fx.rows() if r["status"] == "error"]
        self.assertEqual(len(errors), 4)
        self.assertNotIn("hf_secret_value_123", json.dumps(self.fx.rows()))
        self.assertIn("[REDACTED]", errors[0]["error"]["message"])

    def test_truncated_tail_is_repaired_before_appending(self):
        self.run_judge("--limit", "1")
        path = self.fx.variant_dir() / "judgments.jsonl"
        with open(path, "ab") as handle:
            handle.write(b'{"run_id": "partial')
        self.assertEqual(self.run_judge(), runner.EXIT_OK)
        self.assertEqual(len(self.fx.rows()), 8)
        self.assertGreater(self.fx.manifest()["sessions"][-1]["repaired_tail_bytes"], 0)

    def test_corrupt_middle_line_is_refused(self):
        self.run_judge("--limit", "1")
        path = self.fx.variant_dir() / "judgments.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text(lines[0] + "garbage\n" + "".join(lines[1:]))
        self.assertEqual(self.run_judge(), runner.EXIT_FAILED)
        self.assertTrue(any("corrupt" in line for line in self.logs))

    def test_changed_judge_or_input_cannot_mix_into_existing_outputs(self):
        self.run_judge("--limit", "1")
        self.assertEqual(self.run_judge(revision="rev-2"), runner.EXIT_FAILED)
        self.assertTrue(any("revision_resolved" in line for line in self.logs))
        with open(self.fx.results, "a") as handle:
            handle.write("\n")  # same content otherwise; the hash and default run name change
        self.assertEqual(self.run_judge(), runner.EXIT_FAILED)
        self.assertTrue(any("inputs.results" in line for line in self.logs))

    def test_old_protocol_manifest_is_refused(self):
        self.run_judge("--limit", "1")
        path = self.fx.variant_dir() / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["protocol"]["version"] = 1
        atomic_write_json(path, manifest)
        self.assertEqual(self.run_judge(), runner.EXIT_FAILED)

    def test_dry_run_writes_nothing_and_builds_no_backend(self):
        self.assertEqual(self.run_judge("--dry-run"), runner.EXIT_OK)
        self.assertEqual(self.backends, [])
        self.assertFalse(self.fx.out.exists())
        self.assertTrue(any("Example prompt" in line for line in self.logs))

    def test_output_directory_may_not_contain_inputs(self):
        argv = self.fx.argv()
        argv[argv.index("--out-dir") + 1] = str(self.fx.dir)
        self.assertEqual(runner.run(argv, backend_factory=lambda *a, **k: None, log=self.logs.append),
                         runner.EXIT_FAILED)


class LeaseTests(Harness):
    def test_live_lease_blocks_a_second_writer(self):
        self.fx.variant_dir().mkdir(parents=True)
        holder = Lease(self.fx.variant_dir(), "someone-else", "Rasika", **FAST_LEASE)
        holder.acquire()
        self.addCleanup(holder.release)
        self.assertEqual(self.run_judge(), runner.EXIT_LEASE)
        self.assertTrue(any("Rasika" in line for line in self.logs))
        self.assertFalse((self.fx.variant_dir() / "judgments.jsonl").exists())

    def stale_lease(self):
        self.fx.variant_dir().mkdir(parents=True)
        atomic_write_json(self.fx.variant_dir() / LEASE, {"writer_id": "old", "owner": "Alan",
                                                          "heartbeat_unix": time.time() - 3600})

    def test_stale_lease_needs_explicit_takeover(self):
        self.stale_lease()
        self.assertEqual(self.run_judge(), runner.EXIT_LEASE)
        self.assertTrue(any("takeover_stale" in line for line in self.logs))
        self.assertEqual(self.run_judge("--takeover-stale"), runner.EXIT_OK)
        self.assertFalse((self.fx.variant_dir() / LEASE).exists())  # released afterwards

    def test_lost_lease_stops_before_the_next_append(self):
        calls = []

        def responder(messages, schema):
            calls.append(1)
            if len(calls) == 2:  # another writer takes over between batches
                atomic_write_json(self.fx.variant_dir() / LEASE, {"writer_id": "intruder", "heartbeat_unix": time.time()})
            return full_reply(messages, schema)
        self.assertEqual(self.run_judge("--batch-size", "2", responder=responder), runner.EXIT_FAILED)
        judged = [r for r in self.fx.rows() if r["prompt_sha256"]]
        self.assertEqual(judged, [])  # first batch held the intruding write; nothing appended after it
        self.assertEqual(json.loads((self.fx.variant_dir() / LEASE).read_text())["writer_id"], "intruder")

    def test_lease_unit_behaviour(self):
        directory = Path(self.temp.name) / "lease"
        directory.mkdir()
        first = Lease(directory, "a", **FAST_LEASE)
        first.acquire()
        with self.assertRaises(LeaseHeld):
            Lease(directory, "b", **FAST_LEASE).acquire()
        (directory / LEASE).unlink()
        with self.assertRaises(LeaseLost):
            first.verify()
        first.release()


class ManifestTests(unittest.TestCase):
    def test_compatible_manifest_is_reused_and_paths_may_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            proposed = {"manifest_version": 1, "protocol": protocol.protocol_identity(), "variant": "v",
                        "inputs": {"results": {"path": "/a/results.jsonl", "sha256": "x"}},
                        "selection": {"rank": 1}, "judge": {"model_id": "m"}, "outputs": {}}
            first = open_manifest(directory, proposed)
            moved = {**proposed, "inputs": {"results": {"path": "/drive/other/results.jsonl", "sha256": "x"}}}
            self.assertEqual(open_manifest(directory, moved)["run_id"], first["run_id"])
            with self.assertRaises(OutputConflict):
                open_manifest(directory, {**proposed, "variant": "w"})


class BackendTests(unittest.TestCase):
    def test_redact_removes_tokens(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_abcdefgh123"}):
            self.assertEqual(redact("bad hf_abcdefgh123 token"), "bad [REDACTED] token")

    def test_vllm_adapter_with_a_stub_engine(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return list(range(len(messages[0]["content"])))

        class Output:
            def __init__(self, text):
                self.outputs = [type("Completion", (), {"text": text, "finish_reason": "stop"})()]

        class Engine:
            def __init__(self):
                self.calls = []

            def get_tokenizer(self):
                return Tokenizer()

            def chat(self, conversations, sampling_params, use_tqdm, chat_template_kwargs):
                self.calls.append((len(conversations), len(sampling_params), chat_template_kwargs))
                return [Output("{}") for _ in conversations]

        backend = VllmBackend("org/model", revision="main", max_model_len=500, resolver=lambda m, r: "abc123",
                              chat_template_kwargs={"enable_thinking": False})
        backend.llm = Engine()
        with patch.object(VllmBackend, "sampling_params", staticmethod(lambda schema: schema)):
            from scripts.plan_judge_v2.backends import Request
            results = backend.generate([Request([{"role": "user", "content": "x" * 10}], {}),
                                        Request([{"role": "user", "content": "x" * 400}], {})])
        self.assertEqual(results[0].text, "{}")
        self.assertEqual(results[1].error["type"], "PromptTooLong")
        self.assertEqual(backend.llm.calls, [(1, 1, {"enable_thinking": False})])
        self.assertEqual(backend.identity()["revision_resolved"], "abc123")


class ColabTests(unittest.TestCase):
    CONFIG = {"RESULTS_PATH": "/d/results.jsonl", "PATIENTS_PATH": "/r/so.jsonl", "LABELS_PATH": "/r/labels.jsonl",
              "VARIANTS": ["llm_only", "agentic_both"], "GENERATOR_MODELS": ["gpt_4o_mini"], "OUTPUT_DIR": "/d/out",
              "RUN_NAME": None, "BACKEND": "vllm", "JUDGE_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
              "JUDGE_REVISION": "abc", "DTYPE": "float16", "MAX_MODEL_LEN": 4096, "GPU_MEMORY_UTILIZATION": 0.85,
              "CHAT_TEMPLATE_KWARGS": {"enable_thinking": False},
              "ENGINE_ARGS": {"limit_mm_per_prompt": {"image": 0}}, "BATCH_SIZE": 64, "MAX_ATTEMPTS": 3,
              "LIMIT": 20, "WRITE_CSV": True, "OWNER": "Alan", "TAKEOVER_STALE": False, "DRY_RUN": False}

    def test_judge_argv_round_trips_through_the_real_parser(self):
        argv = colab.judge_argv(self.CONFIG)
        self.assertEqual(argv[0], "judge")
        args = runner.build_parser().parse_args(argv[1:])
        self.assertEqual(args.variants, ["llm_only", "agentic_both"])
        self.assertEqual(args.generator_models, ["gpt_4o_mini"])
        self.assertEqual((args.model, args.revision, args.dtype, args.limit), ("Qwen/Qwen3-4B-Instruct-2507", "abc",
                                                                                "float16", 20))
        self.assertEqual(json.loads(args.chat_template_kwargs), {"enable_thinking": False})
        self.assertEqual(args.engine_arg, ['limit_mm_per_prompt={"image": 0}'])
        self.assertTrue(args.csv and not args.dry_run and not args.takeover_stale)

    def test_unknown_config_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            colab.judge_argv({**self.CONFIG, "HF_TOKEN": "x"})

    def test_validate_argv_parses(self):
        import argparse
        parser = argparse.ArgumentParser()
        validate_models.build_parsers(parser.add_subparsers(dest="command"))
        args = parser.parse_args(colab.validate_argv({"model": "google/medgemma-4b-it",
                                                      "engine_args": {"limit_mm_per_prompt": {"image": 0}}},
                                                     cases="c.jsonl", out_dir="r", allow_draft=True))
        self.assertEqual((args.command, args.model, args.allow_draft), ("validate", "google/medgemma-4b-it", True))

    def test_notebook_config_matches_the_argv_builder_and_holds_no_secrets(self):
        notebook = json.loads(NOTEBOOK.read_text())
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for source in sources:
            compile(source, "cell", "exec")
        self.assertIn("CONFIG: edit only this cell", sources[0])
        namespace = {}
        exec(sources[0], namespace)  # noqa: S102 - the config cell is plain assignments
        config_keys = {key for key in colab.JUDGE_KEYS if key != "DRY_RUN"}
        self.assertTrue(config_keys <= set(namespace), config_keys - set(namespace))
        self.assertIsNone(namespace["JUDGE_MODEL"])
        self.assertEqual(namespace["MODE"], "dry_run")
        self.assertIn('userdata.get("HF_TOKEN")', "".join(sources))
        self.assertNotRegex("".join(sources), r"hf_[A-Za-z0-9]{10,}")
        self.assertTrue(all(cell.get("outputs", []) == [] for cell in notebook["cells"]))


class PilotTests(unittest.TestCase):
    def test_pilot_case_file_is_balanced_and_valid(self):
        cases = validate_models.load_cases(PILOT_CASES)
        self.assertEqual(len(cases), 48)
        counts = {}
        for case in cases:
            counts[case["category"]] = counts.get(case["category"], 0) + 1
        self.assertEqual(set(counts), set(validate_models.CATEGORIES))
        self.assertTrue(all(5 <= n <= 7 for n in counts.values()), counts)
        self.assertEqual(len({c["condition"] for c in cases}), 6)

    def test_quadratic_kappa(self):
        self.assertEqual(validate_models.quadratic_kappa([(0, 0), (1, 1), (2, 2)]), 1.0)
        self.assertLess(validate_models.quadratic_kappa([(0, 2), (2, 0), (1, 1)]), 0)

    def gold_backend(self, cases):
        by_prompt = {}
        for case in cases:
            [message] = protocol.render_messages(case["condition"], case["subjective"], case["objective"],
                                                 case["reference_actions"], case["generated_actions"])
            items = [{"ref": n, "match": m, "generated": [] if m == "none" else [1]}
                     for n, m in enumerate(case["gold"]["matches"], 1)]
            harmful = [{"generated": h, "why": "gold"} for h in case["gold"]["harmful"]]
            by_prompt[message["content"]] = json.dumps({"reference_items": items, "harmful_actions": harmful})
        return FakeBackend(lambda messages, schema: by_prompt[messages[0]["content"]])

    def test_perfect_judge_scores_perfectly_and_draft_is_refused(self):
        cases = validate_models.load_cases(PILOT_CASES)
        outcomes, timing = validate_models.evaluate(cases, self.gold_backend(cases), batch_size=10)
        metrics = validate_models.metrics(cases, outcomes)
        self.assertEqual(metrics["valid_json_rate"], 1.0)
        self.assertEqual(metrics["item_quadratic_kappa"], 1.0)
        self.assertEqual(metrics["coverage_mae"], 0.0)
        self.assertEqual(metrics["harmful_precision"], 1.0)
        self.assertGreater(metrics["separation_correct_minus_worst_wrong"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            import argparse
            parser = argparse.ArgumentParser()
            validate_models.build_parsers(parser.add_subparsers(dest="command"))
            args = parser.parse_args(["validate", "--cases", str(PILOT_CASES), "--model", "fake/judge",
                                      "--out-dir", directory])
            factory = lambda name, model, **kw: self.gold_backend(cases)  # noqa: E731
            logs = []
            self.assertEqual(validate_models.validate(args, backend_factory=factory, log=logs.append), 1)
            args.allow_draft = True
            self.assertEqual(validate_models.validate(args, backend_factory=factory, log=logs.append), 0)
            report = json.loads((Path(directory) / "fake_judge.json").read_text())
            self.assertEqual(report["labels_status"], "draft")
            self.assertIn("pilot", report["pilot_note"].lower())
            compare_args = parser.parse_args(["compare", "--reports", directory])
            self.assertEqual(validate_models.compare(compare_args, log=logs.append), 0)
            self.assertTrue((Path(directory) / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
