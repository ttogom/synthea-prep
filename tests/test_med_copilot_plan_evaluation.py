"""Offline regression tests for the independently resumable Plan-only stage."""

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nltk
import yaml

# Direct execution starts in tests/; project and vendored imports need these roots.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
with patch.object(nltk, "download", return_value=True):
    from scripts import med_copilot_plan_evaluation as stage
    from scripts.med_copilot_judges import CorrectnessJudges, plan_entry
    from scripts.med_copilot_run_lock import RunLockError, exclusive_run_lock
    from scripts.med_copilot_scoring import PatientScorer
    from scripts.med_copilot_helpers import GUIDELINE_TABLES, guideline_resource_files, sha256_bundle
    with patch("dotenv.load_dotenv", return_value=False):
        import judge_treatment_plans as command


class FakeModels:
    model_ids = ("judge",)

    def __init__(self, reply=None):
        self.reply = reply
        self.requests = []
        self.lock = Lock()
        self.active = self.maximum = 0

    def judge(self, model, messages):
        with self.lock:
            self.requests.append((model, messages))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            content = self.reply(model, messages) if self.reply else None
            if isinstance(content, BaseException):
                raise content
            if content is None:
                content = ('{"credit": 1}' if messages[0]["content"].startswith(
                    "Grade whether the generated plan covers") else '{"unsupported_extra_count": 0}')
            return SimpleNamespace(id="reply", model="judge", usage=None,
                                   choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                                       content=content, refusal=None))])
        finally:
            with self.lock:
                self.active -= 1

    def metadata(self, root):
        return {"selected_models": ["judge"], "models": {"judge": {"model": "fake"}}}

    def redact(self, text):
        return text.replace("test-private-key", "[REDACTED]")


class PlanStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.patients = [{"id": f"patient-{number}", "subjective": f"S {number}", "objective": f"O {number}"}
                         for number in range(3)]
        self.labels = [{"id": patient["id"], "condition": "reference_condition",
                        "plan": [f"Reference action {number}"], "has_presenting_evidence": True}
                       for number, patient in enumerate(self.patients)]
        self.input_path = self.root / "input.jsonl"
        self.labels_path = self.root / "labels.jsonl"
        self.input_path.write_text("".join(json.dumps(row) + "\n" for row in self.patients))
        self.labels_path.write_text("".join(json.dumps(row) + "\n" for row in self.labels))
        self.scorer = PatientScorer(self.labels_path, self.patients, False, metric_evaluator=object())
        self.metadata = {
            "run_name": "run", "status": "completed", "errors": [],
            "finished_at": "2026-11-12T01:00:00-08:00",
            "input": {"file": "input.jsonl", "sha256": hashlib.sha256(self.input_path.read_bytes()).hexdigest()},
            "experiment": {"selected_variants": ["llm_only"], "plan_judge": "judge", "plan_judge_concurrency": 2},
            "assessment_plan_models": {"selected_models": ["generator"]},
            "evaluation": {"labels_path": "labels.jsonl",
                           "labels_sha256": hashlib.sha256(self.labels_path.read_bytes()).hexdigest(),
                           "selected_patient_ids": [patient["id"] for patient in self.patients],
                           "metric_implementation": {"original": "must remain"}},
        }
        self.records = []
        for number, patient in enumerate(self.patients):
            predictions = [{"condition": f"Generated diagnosis {rank}",
                            "plan": [f"Generated action {number} {rank}"]} for rank in range(3)]
            evaluation = [{"rank": rank, "condition": {"sentinel": ["text", rank]},
                           "plan": {"sentinel": ["text plan", rank]},
                           "diagnosis_correctness": {"sentinel": ["diagnosis", rank]},
                           "plan_correctness": {**plan_entry(rank, self.labels[number]["plan"],
                                                            predictions[rank - 1]["plan"]), "reason": "deferred"}}
                          for rank in (1, 2, 3)]
            self.records.append({"id": patient["id"], "status": "completed", "components": {"evidence": "kept"},
                                 "variants": {"llm_only": {"assessment_plans": {
                                     "generator": {"status": "completed", "predictions": predictions,
                                                   "assessment_plan_raw": "original reply", "evaluation": evaluation}}}}})
        self.write()

    def write(self):
        (self.run_dir / "run.json").write_text(json.dumps(self.metadata))
        (self.run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def saved(self):
        metadata = json.loads((self.run_dir / "run.json").read_text())
        records = [json.loads(line) for line in (self.run_dir / "results.jsonl").read_text().splitlines()]
        return metadata, records

    def run_stage(self, models=None, **kwargs):
        models = models or FakeModels()
        return stage.resume_plan_evaluation(self.run_dir, self.scorer, models,
                                             root=self.root, **kwargs)

    def entries(self, records):
        return [entry for row in records for variant in row["variants"].values()
                for output in variant["assessment_plans"].values() for entry in output["evaluation"]]

    def generation_correctness(self):
        return CorrectnessJudges(None, ("judge",), self.run_dir / "judge_responses.jsonl",
                                 plan_judge="judge", plan_execution="deferred").metadata(self.root)

    def assert_protocol_rejected_without_writes(self, message):
        paths = [self.run_dir / name for name in ("run.json", "results.jsonl", "judge_responses.jsonl")]
        before = {p.name: p.read_bytes() if p.exists() else None for p in paths}
        models = FakeModels()
        with patch("scripts.med_copilot_judges.JudgeCache", side_effect=AssertionError("cache opened")), \
                self.assertRaisesRegex(ValueError, message):
            self.run_stage(models)
        self.assertEqual(models.requests, [])
        self.assertEqual({p.name: p.read_bytes() if p.exists() else None for p in paths}, before)

    def test_first_plan_stage_uses_current_protocol_and_preserves_old_generation_fingerprints(self):
        original = self.generation_correctness()
        original["plan"]["extras_prompt_template_sha256"] = "old-generation-extras-prompt"
        original["diagnosis"]["model"] = "generation-only-diagnosis-model"
        self.metadata["evaluation"]["correctness"] = deepcopy(original)
        self.write()
        observed = []
        def reply(model, messages):
            observed.append(json.loads((self.run_dir / "run.json").read_text()))
        judges = self.run_stage(FakeModels(reply))
        metadata, _ = self.saved()
        self.assertEqual(judges.calls, 18)
        self.assertEqual(metadata["plan_evaluation"]["status"], "completed")
        self.assertEqual(metadata["evaluation"], self.metadata["evaluation"])
        self.assertEqual(metadata["experiment"], self.metadata["experiment"])
        protocol = metadata["plan_evaluation"]["protocol"]
        self.assertEqual(protocol["version"], 1)
        self.assertEqual(protocol["plan"]["extras_prompt_template_sha256"],
                         self.generation_correctness()["plan"]["extras_prompt_template_sha256"])
        self.assertTrue(observed)
        for snapshot in observed:
            self.assertEqual(snapshot["plan_evaluation"]["protocol"], protocol)
            self.assertEqual(snapshot["plan_evaluation"]["status"], "running")
            self.assertEqual(snapshot["evaluation"]["correctness"], original)

    def test_old_generation_hash_does_not_block_recovery_after_failed_export(self):
        self.metadata["evaluation"]["correctness"] = self.generation_correctness()
        self.metadata["evaluation"]["correctness"]["plan"]["rubric_sha256"] = "old-generation-rubric"
        self.write()
        original = (self.run_dir / "results.jsonl").read_bytes()
        with patch.object(stage, "_write_results", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run_stage()
        metadata, _ = self.saved()
        self.assertIn("protocol", metadata["plan_evaluation"])
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), original)
        judges = self.run_stage(FakeModels())
        self.assertEqual((judges.calls, judges.cache_hits), (0, 18))
        self.assertEqual(self.saved()[0]["evaluation"], self.metadata["evaluation"])
        self.assertEqual(self.saved()[0]["plan_evaluation"]["status"], "completed")

    def test_new_stage_protocol_rejects_drift_before_touching_cache_results_or_metadata(self):
        self.run_stage()
        cache = self.run_dir / "judge_responses.jsonl"
        with cache.open("ab") as handle:
            handle.write(b'{"torn"')
        before = {p.name: p.read_bytes() for p in self.run_dir.iterdir() if p.is_file()}
        models = FakeModels()
        with patch("scripts.med_copilot_judges.PLAN_EXTRAS_JUDGE_TEMPLATE", "changed prompt"), \
                patch("scripts.med_copilot_judges.JudgeCache", side_effect=AssertionError("cache opened")), \
                self.assertRaisesRegex(ValueError, "protocol.*extras_prompt"):
            self.run_stage(models)
        self.assertEqual(models.requests, [])
        self.assertEqual({p.name: p.read_bytes() for p in self.run_dir.iterdir() if p.is_file()}, before)

    def test_first_stage_allows_an_empty_cache_and_automatic_zero_without_treating_it_as_prior_judging(self):
        self.metadata["evaluation"]["correctness"] = self.generation_correctness()
        self.metadata["evaluation"]["correctness"]["plan"]["rubric_sha256"] = "old-rubric"
        output = self.records[0]["variants"]["llm_only"]["assessment_plans"]["generator"]
        output["predictions"][0]["plan"] = []
        output["evaluation"][0]["plan_correctness"] = plan_entry(1, self.labels[0]["plan"], [])
        output["evaluation"][1].pop("plan_correctness")
        (self.run_dir / "judge_responses.jsonl").touch()
        self.write()
        judges = self.run_stage()
        self.assertEqual(judges.calls, 16)
        self.assertEqual(self.saved()[0]["evaluation"], self.metadata["evaluation"])
        self.assertEqual(self.entries(self.saved()[1])[0]["plan_correctness"]["reason"], "empty_generated_plan")

    def test_legacy_matching_protocol_is_upgraded_and_reuses_completed_scores(self):
        self.metadata["evaluation"]["correctness"] = self.generation_correctness()
        self.write()
        self.run_stage()
        self.metadata, self.records = self.saved()
        self.metadata["plan_evaluation"].pop("protocol")
        self.write()
        before = (self.run_dir / "results.jsonl").read_bytes()
        cache = (self.run_dir / "judge_responses.jsonl").read_bytes()
        judges = self.run_stage()
        metadata, _ = self.saved()
        self.assertEqual((judges.calls, judges.cache_hits), (0, 0))
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), before)
        self.assertEqual((self.run_dir / "judge_responses.jsonl").read_bytes(), cache)
        self.assertEqual(metadata["evaluation"], self.metadata["evaluation"])
        self.assertEqual(metadata["plan_evaluation"]["protocol"]["version"], 1)
        self.assertNotIn("protocol", metadata["plan_evaluation_history"][-1])

    def test_legacy_evidence_prevents_initializing_over_changed_generation_fingerprints(self):
        original_records = deepcopy(self.records)
        for kind in ("cache", "judged entry", "aggregate", "scored marker", "failed entry", "interrupted entry", "stage", "history"):
            with self.subTest(evidence=kind):
                self.records = deepcopy(original_records)
                self.metadata.pop("plan_evaluation", None)
                self.metadata.pop("plan_evaluation_history", None)
                cache = self.run_dir / "judge_responses.jsonl"
                cache.unlink(missing_ok=True)
                self.metadata["evaluation"]["correctness"] = self.generation_correctness()
                self.metadata["evaluation"]["correctness"]["plan"]["extras_prompt_template_sha256"] = "old-prompt"
                entry = self.entries(self.records)[0]["plan_correctness"]
                if kind == "cache":
                    cache.write_text('{"torn"')
                elif kind == "judged entry":
                    entry["judges"] = [{"call": "action", "credit": 1, "status": "scored"}]
                elif kind == "aggregate":
                    entry.update(status="scored", score=1.0, weighted_score=1.0, credits=[1.0])
                elif kind == "scored marker":
                    entry.update(status="scored")
                elif kind == "failed entry":
                    entry.update(status="error", reason="judge_failed")
                elif kind == "interrupted entry":
                    entry.update(status="interrupted")
                elif kind == "stage":
                    self.metadata["plan_evaluation"] = {"status": "running"}
                else:
                    self.metadata["plan_evaluation_history"] = [{"status": "interrupted"}]
                self.write()
                self.assert_protocol_rejected_without_writes("correctness.*extras_prompt")

    def test_legacy_cache_without_correctness_provenance_is_not_silently_adopted(self):
        (self.run_dir / "judge_responses.jsonl").write_text('{"torn"')
        self.assert_protocol_rejected_without_writes("without correctness provenance")

    def test_protocol_drift_missing_fields_and_unsupported_versions_cannot_be_hidden_by_generation_metadata(self):
        self.run_stage()
        metadata, records = self.saved()
        changes = [
            (("version",), 2), (("version",), True),
            (("plan", "judge_model"), "other"), (("plan", "judge_temperature"), 1),
            (("plan", "rubric_file"), "other.md"), (("plan", "rubric_sha256"), "other"),
            (("plan", "action_prompt_template_sha256"), "other"),
            (("plan", "extras_prompt_template_sha256"), "other"),
            (("plan", "credit_values"), [0, 1]), (("plan", "deduction_per_unsupported_extra"), 1),
            (("rank_weighting", "formula"), "other"), (("rank_weighting", "weights"), {"1": 0.5}),
        ]
        for keys, value in changes:
            with self.subTest(field=keys):
                self.metadata, self.records = deepcopy(metadata), deepcopy(records)
                self.metadata["evaluation"]["correctness"] = self.generation_correctness()
                target = self.metadata["plan_evaluation"]["protocol"]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                self.write()
                self.assert_protocol_rejected_without_writes("protocol")
        for protocol in (None, [], {}, {"version": 1}, {"version": 1, "plan": None}):
            with self.subTest(protocol=protocol):
                self.metadata, self.records = deepcopy(metadata), deepcopy(records)
                self.metadata["plan_evaluation"]["protocol"] = protocol
                self.write()
                self.assert_protocol_rejected_without_writes("protocol")

    def test_versioned_history_cannot_be_ignored_when_current_protocol_is_removed(self):
        self.run_stage()
        self.metadata, self.records = self.saved()
        self.metadata["plan_evaluation_history"] = [deepcopy(self.metadata["plan_evaluation"])]
        self.metadata["plan_evaluation"].pop("protocol")
        self.write()
        self.assert_protocol_rejected_without_writes("protocol is missing")

    def test_stage_judge_settings_cannot_disagree_with_its_protocol(self):
        self.run_stage()
        self.metadata, self.records = self.saved()
        self.metadata["plan_evaluation"]["judge_settings"]["extras_prompt_template_sha256"] = "other-prompt"
        self.write()
        self.assert_protocol_rejected_without_writes("protocol.*extras_prompt")

    def test_protocol_and_judge_settings_describe_the_start_even_if_rubric_changes_during_work(self):
        rubric = self.root / "rubric.md"
        rubric.write_text("rubric at stage start")
        initial_hash = hashlib.sha256(rubric.read_bytes()).hexdigest()
        def change_rubric(model, messages):
            rubric.write_text("changed while work was running")
        with patch("scripts.med_copilot_judges.PLAN_RUBRIC", rubric):
            self.run_stage(FakeModels(change_rubric))
            saved = self.saved()[0]["plan_evaluation"]
            self.assertEqual(saved["protocol"]["plan"]["rubric_sha256"], initial_hash)
            self.assertEqual(saved["judge_settings"]["rubric_sha256"], initial_hash)
            self.assert_protocol_rejected_without_writes("protocol.*rubric_sha256")

    def test_same_judge_alias_can_resume_without_comparing_provider_settings(self):
        self.run_stage()
        models = FakeModels()
        provider = {"models": {"judge": {"model": "updated-underlying-model"}}}
        with patch.object(models, "metadata", return_value=provider):
            judges = self.run_stage(models)
        self.assertEqual((judges.calls, judges.cache_hits), (0, 0))
        self.assertEqual(self.saved()[0]["plan_evaluation"]["judge_settings"]["judge_model_config"], provider)

    def test_malformed_history_or_generation_provenance_is_rejected_without_writes(self):
        original = deepcopy(self.metadata)
        for key, value in (("plan_evaluation", []), ("plan_evaluation_history", None),
                           ("plan_evaluation_history", ["invalid"]), ("correctness", "invalid")):
            with self.subTest(key=key, value=value):
                self.metadata = deepcopy(original)
                target = self.metadata["evaluation"] if key == "correctness" else self.metadata
                target[key] = value
                self.write()
                self.assert_protocol_rejected_without_writes("must be")

    def test_only_plan_fields_change_and_generator_status_and_metric_provenance_survive(self):
        before = deepcopy(self.records)
        with patch.object(self.scorer, "score_result", side_effect=AssertionError("all metrics repeated")), \
                patch("scripts.med_copilot_judges.BioLordSimilarity.cosine", side_effect=AssertionError("diagnosis repeated")):
            judges = self.run_stage()
        metadata, records = self.saved()
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["finished_at"], self.metadata["finished_at"])
        self.assertEqual(metadata["evaluation"]["metric_implementation"], self.metadata["evaluation"]["metric_implementation"])
        self.assertEqual(metadata["experiment"], self.metadata["experiment"])
        self.assertEqual(metadata["plan_evaluation"]["status"], "completed")
        self.assertEqual(judges.calls, 18)
        self.assertTrue(all(e["plan_correctness"]["score"] == 1 for e in self.entries(records)))
        for entry in self.entries(records):
            entry.pop("plan_correctness")
        for entry in self.entries(before):
            entry.pop("plan_correctness")
        self.assertEqual(records, before)

    def test_requests_overlap_and_never_exceed_configured_concurrency(self):
        def reply(model, messages):
            time.sleep(0.01)
        models = FakeModels(reply)
        self.run_stage(models, concurrency=2)
        self.assertEqual(models.maximum, 2)

    def test_export_happens_once_and_a_completed_resume_does_not_rewrite_results(self):
        with patch.object(stage, "_write_results", wraps=stage._write_results) as write:
            self.run_stage()
            self.assertEqual(write.call_count, 1)
            write.reset_mock()
            models = FakeModels()
            self.run_stage(models)
            self.assertEqual(models.requests, [])
            write.assert_not_called()

    def test_bundled_extras_are_scored_and_resume_without_calls_or_result_writes(self):
        for row in self.records:
            output = row["variants"]["llm_only"]["assessment_plans"]["generator"]
            for prediction, entry in zip(output["predictions"], output["evaluation"]):
                prediction["plan"] = ["Antibiotics; imaging"]
                entry["plan_correctness"] = plan_entry(
                    entry["rank"], entry["plan_correctness"]["reference_actions"], prediction["plan"])
        self.write()
        models = FakeModels(lambda model, messages: '{"unsupported_extra_count": 2}'
                            if messages[0]["content"].startswith("Count the distinct generated actions") else None)
        self.run_stage(models)
        self.assertEqual(len(models.requests), 6)  # Identical plans across ranks share judge requests.
        metadata, records = self.saved()
        self.assertEqual(metadata["plan_evaluation"]["status"], "completed")
        for entry in self.entries(records):
            plan = entry["plan_correctness"]
            self.assertEqual((plan["status"], plan["unsupported_extra_count"], plan["deduction"], plan["score"]),
                             ("scored", 2, 1.0, 0.0))
        second = FakeModels()
        with patch.object(stage, "_write_results", side_effect=AssertionError("completed results rewritten")):
            self.run_stage(second)
        self.assertEqual(second.requests, [])

    def test_zero_coverage_is_complete_and_a_resume_preserves_the_export_without_calls(self):
        before = deepcopy(self.records)
        models = FakeModels(lambda model, messages: '{"credit": 0}')
        judges = self.run_stage(models, concurrency=3)
        metadata, records = self.saved()
        self.assertEqual(judges.calls, 9)
        self.assertEqual(metadata["plan_evaluation"]["status"], "completed")
        for entry in self.entries(records):
            plan = entry["plan_correctness"]
            self.assertEqual((plan["status"], plan["reason"], plan["score"], plan["weighted_score"]),
                             ("scored", "zero_reference_coverage", 0.0, 0.0))
            self.assertIsNone(plan["unsupported_extra_count"])
            self.assertIsNone(plan["deduction"])
            self.assertEqual([j["call"] for j in plan["judges"]], ["action"])
        for original, actual in zip(self.entries(before), self.entries(records)):
            self.assertEqual({k: v for k, v in original.items() if k != "plan_correctness"},
                             {k: v for k, v in actual.items() if k != "plan_correctness"})
        results = (self.run_dir / "results.jsonl").read_bytes()
        cache = (self.run_dir / "judge_responses.jsonl").read_bytes()
        with patch.object(stage, "_write_results", wraps=stage._write_results) as write:
            resumed = self.run_stage(FakeModels())
            self.assertEqual((resumed.calls, resumed.cache_hits), (0, 0))
            write.assert_not_called()
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), results)
        self.assertEqual((self.run_dir / "judge_responses.jsonl").read_bytes(), cache)

    def test_zero_coverage_survives_a_crash_before_export_using_only_action_cache(self):
        original = (self.run_dir / "results.jsonl").read_bytes()
        with patch.object(stage, "_write_results", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run_stage(FakeModels(lambda model, messages: '{"credit": 0}'))
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), original)
        self.assertEqual(len((self.run_dir / "judge_responses.jsonl").read_text().splitlines()), 9)
        models = FakeModels()
        judges = self.run_stage(models)
        self.assertEqual((models.requests, judges.calls, judges.cache_hits), ([], 0, 9))
        self.assertEqual(self.saved()[0]["plan_evaluation"]["status"], "completed")
        self.assertTrue(all(e["plan_correctness"]["reason"] == "zero_reference_coverage"
                            for e in self.entries(self.saved()[1])))

    def test_tampered_zero_coverage_requires_complete_matching_zero_action_evidence(self):
        self.run_stage(FakeModels(lambda model, messages: '{"credit": 0}'))
        metadata, records = self.saved()
        cases = {
            "missing action": lambda p: p["judges"].clear(),
            "invalid action": lambda p: p["judges"][0].update(status="invalid_response"),
            "wrong judge": lambda p: p["judges"][0].update(judge_model="other"),
            "wrong key": lambda p: p["judges"][0].update(request_key="wrong"),
            "wrong index": lambda p: p["judges"][0].update(reference_item=2),
            "positive evidence": lambda p: p["judges"][0].update(credit=0.5, raw_response='{"credit": 0.5}'),
            "invalid raw response": lambda p: p["judges"][0].update(raw_response="invalid"),
            "wrong credits": lambda p: p.update(credits=[0.5]),
            "wrong score": lambda p: p.update(score=0.5),
            "wrong weighted score": lambda p: p.update(weighted_score=0.5),
            "fabricated extras": lambda p: p.update(unsupported_extra_count=0),
            "fabricated deduction": lambda p: p.update(deduction=0),
            "missing reason": lambda p: p.update(reason=None),
        }
        for name, tamper in cases.items():
            with self.subTest(case=name):
                self.metadata, self.records = deepcopy(metadata), deepcopy(records)
                tamper(self.entries(self.records)[0]["plan_correctness"])
                self.write()
                with patch.object(stage, "_write_results", wraps=stage._write_results) as write:
                    judges = self.run_stage(FakeModels())
                    self.assertEqual((judges.calls, judges.cache_hits), (0, 1))
                    write.assert_called_once()
                self.assertEqual(self.saved()[1], records)

    def test_legacy_zero_score_with_measured_extras_remains_reusable(self):
        self.run_stage(FakeModels(lambda model, messages: '{"credit": 0}'))
        self.metadata, self.records = self.saved()
        plan = self.entries(self.records)[0]["plan_correctness"]
        patient = self.patients[0]
        messages = stage.plan_extras_messages("reference condition", patient["subjective"], patient["objective"],
                                              plan["reference_actions"], plan["generated_actions"])
        plan["judges"].append({"call": "extras", "reference_item": None, "judge_model": "judge",
                               "status": "scored", "raw_response": '{"unsupported_extra_count": 1}',
                               "unsupported_extra_count": 1, "request_key": stage.request_key("judge", messages)})
        plan.update(reason=None, unsupported_extra_count=1, deduction=0.5)
        self.write()
        before = (self.run_dir / "results.jsonl").read_bytes()
        with patch.object(stage, "_write_results", wraps=stage._write_results) as write:
            judges = self.run_stage(FakeModels())
            self.assertEqual((judges.calls, judges.cache_hits), (0, 0))
            write.assert_not_called()
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), before)

    def test_existing_exact_pending_entries_are_exported_after_judging(self):
        for entry in self.entries(self.records):
            entry["plan_correctness"]["reason"] = None
        self.write()
        self.run_stage()
        self.assertTrue(all(e["plan_correctness"]["status"] == "scored" for e in self.entries(self.saved()[1])))

    def test_hard_crash_before_export_reconstructs_from_durable_responses_without_calls(self):
        original = (self.run_dir / "results.jsonl").read_bytes()
        first = FakeModels()
        with patch.object(stage, "_write_results", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.run_stage(first)
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), original)
        self.assertEqual(self.saved()[0]["plan_evaluation"]["status"], "failed_incomplete")
        second = FakeModels()
        judges = self.run_stage(second)
        self.assertEqual(second.requests, [])
        self.assertEqual(judges.cache_hits, len(first.requests))
        self.assertEqual(self.saved()[0]["plan_evaluation"]["status"], "completed")

    def test_interruption_saves_responses_and_restarts_only_missing_requests(self):
        calls = []
        def reply(model, messages):
            calls.append(messages)
            if len(calls) == 2:
                return KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage(FakeModels(reply), concurrency=1)
        self.assertEqual(self.saved()[0]["plan_evaluation"]["status"], "interrupted")
        cached = len((self.run_dir / "judge_responses.jsonl").read_text().splitlines())
        second = FakeModels()
        self.run_stage(second, concurrency=1)
        self.assertEqual(cached + len(second.requests), 18)

    def test_failed_responses_are_retried_and_valid_responses_reused(self):
        self.run_stage(FakeModels(lambda model, messages: "invalid" if messages[0]["content"].startswith(
            "Grade whether the generated plan covers") else None))
        self.assertEqual(self.saved()[0]["plan_evaluation"]["status"], "failed_incomplete")
        second = FakeModels()
        self.run_stage(second)
        self.assertEqual(len(second.requests), 9)
        self.assertTrue(all(messages[0]["content"].startswith("Grade whether") for _, messages in second.requests))

    def test_guideline_chat_rate_change_allows_plan_only_resume_and_reuses_valid_scalar_cache(self):
        directory = self.root / "guidelines"
        (directory / "output").mkdir(parents=True)
        path = directory / "settings.yaml"
        settings = {"models": {"default_chat_model": {
            "requests_per_minute": "auto", "tokens_per_minute": "auto"}}}
        path.write_text(yaml.safe_dump(settings))
        for name in GUIDELINE_TABLES:
            (directory / "output" / (name + ".parquet")).write_text("offline " + name)
        original_hash = sha256_bundle(guideline_resource_files(directory, path, directory / "output"))
        self.metadata["experiment"]["guidelines"] = "guidelines"
        self.metadata["reference_data"] = {"guidelines": {"directory": "guidelines", "sha256": original_hash}}
        self.write()
        original_results = (self.run_dir / "results.jsonl").read_bytes()
        first = FakeModels()
        with patch.object(stage, "_write_results", side_effect=OSError("export interrupted")), self.assertRaises(OSError):
            self.run_stage(first)
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), original_results)
        cache = (self.run_dir / "judge_responses.jsonl").read_bytes()
        settings["models"]["default_chat_model"].update(requests_per_minute=12, tokens_per_minute=200000)
        path.write_text(yaml.safe_dump(settings))
        self.assertNotEqual(sha256_bundle(guideline_resource_files(directory, path, directory / "output")), original_hash)
        second = FakeModels()
        judges = self.run_stage(second)
        metadata, records = self.saved()
        self.assertEqual(second.requests, [])
        self.assertEqual(judges.cache_hits, len(first.requests))
        self.assertEqual((self.run_dir / "judge_responses.jsonl").read_bytes(), cache)
        self.assertEqual(metadata["experiment"], self.metadata["experiment"])
        self.assertEqual(metadata["reference_data"], self.metadata["reference_data"])
        self.assertEqual(metadata["plan_evaluation"]["status"], "completed")
        for before, after in zip(self.entries(self.records), self.entries(records)):
            self.assertEqual({key: value for key, value in after.items() if key != "plan_correctness"},
                             {key: value for key, value in before.items() if key != "plan_correctness"})

    def test_changed_prediction_invalidates_only_affected_plan(self):
        self.run_stage()
        self.metadata, self.records = self.saved()
        self.records[0]["variants"]["llm_only"]["assessment_plans"]["generator"]["predictions"][0]["plan"] = ["Different action"]
        self.write()
        models = FakeModels()
        self.run_stage(models)
        self.assertEqual(len(models.requests), 2)

    def test_tampered_scored_aggregate_is_rebuilt_from_cache(self):
        self.run_stage()
        self.metadata, self.records = self.saved()
        self.entries(self.records)[0]["plan_correctness"]["credits"] = [0]
        self.write()
        models = FakeModels()
        self.run_stage(models)
        self.assertEqual(models.requests, [])
        self.assertEqual(self.entries(self.saved()[1])[0]["plan_correctness"]["credits"], [1.0])

    def test_empty_and_unavailable_plans_do_not_call_the_model(self):
        self.records = self.records[:1]
        self.patients = self.patients[:1]
        self.metadata["evaluation"]["selected_patient_ids"] = [self.patients[0]["id"]]
        self.scorer = PatientScorer(self.labels_path, self.patients, False, metric_evaluator=object())
        output = self.records[0]["variants"]["llm_only"]["assessment_plans"]["generator"]
        output["predictions"] = [{"condition": "A", "plan": []}, {"condition": "A", "plan": None},
                                 {"condition": "A", "plan": []}]
        self.write()
        models = FakeModels()
        self.run_stage(models)
        self.assertEqual(models.requests, [])
        self.assertEqual([e["plan_correctness"]["status"] for e in self.entries(self.saved()[1])],
                         ["scored", "unavailable", "scored"])

    def test_invalid_results_are_rejected_before_cache_tail_repair_or_requests(self):
        cache = self.run_dir / "judge_responses.jsonl"
        cache.write_text("unfinished")
        for mutation in (lambda: self.records.append(deepcopy(self.records[0])),
                         lambda: self.records[0].update(id="unknown"),
                         lambda: self.records[0]["variants"]["llm_only"]["assessment_plans"]["generator"].update(predictions=[])):
            before = deepcopy(self.records)
            mutation()
            self.write()
            models = FakeModels()
            with self.assertRaises(ValueError):
                self.run_stage(models)
            self.assertEqual(models.requests, [])
            self.assertEqual(cache.read_text(), "unfinished")
            self.records = before

    def test_numeric_overflow_is_rejected_before_any_paid_requests(self):
        for name in ("run.json", "results.jsonl"):
            self.write()
            path = self.run_dir / name
            text = path.read_text()
            path.write_text(text.replace('"status": "completed"', '"unrelated": 1e999, "status": "completed"', 1))
            models = FakeModels()
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.run_stage(models)
            self.assertEqual(models.requests, [])
            self.assertFalse((self.run_dir / "judge_responses.jsonl").exists())

    def test_unfinished_generation_is_rejected(self):
        for status in ("running", "failed_incomplete"):
            self.metadata["status"] = status
            self.write()
            with self.assertRaisesRegex(ValueError, "Generation must finish"):
                self.run_stage()

    def test_invalid_concurrency_is_rejected(self):
        for value in (True, 0, -1, 1.5, "2", 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_stage(concurrency=value)

    def test_runtime_context_and_concurrency_preserve_experiment_and_archive_previous_stages(self):
        self.metadata["evaluation"]["correctness"] = self.generation_correctness()
        self.metadata["experiment"]["execution"] = {"ollama_judge_context_length": 8192}
        old_history = [{"status": "completed", "concurrency": 2, "context_length": 8192, "calls": 7}]
        previous = {"status": "interrupted", "concurrency": 4, "calls": 5, "cache_hits": 3}
        self.metadata.update(plan_evaluation_history=deepcopy(old_history), plan_evaluation=deepcopy(previous))
        self.write()
        self.run_stage(concurrency=8, context_length=4096)
        metadata, _ = self.saved()
        self.assertEqual(metadata["experiment"], self.metadata["experiment"])
        self.assertEqual(metadata["plan_evaluation_history"],
                         [*old_history, {**previous, "context_length": 8192}])
        self.assertEqual(metadata["plan_evaluation"]["concurrency"], 8)
        self.assertEqual(metadata["plan_evaluation"]["context_length"], 4096)
        completed = deepcopy(metadata["plan_evaluation"])
        contents = (self.run_dir / "results.jsonl").read_bytes()
        second = FakeModels()
        self.run_stage(second, concurrency=8, context_length=4096)
        self.assertEqual(second.requests, [])
        self.assertEqual((self.run_dir / "results.jsonl").read_bytes(), contents)
        self.assertEqual(self.saved()[0]["plan_evaluation_history"][-1], completed)
        self.assertEqual(len(self.saved()[0]["plan_evaluation_history"]), 3)

    def test_invalid_runtime_context_is_rejected_before_artifact_changes(self):
        original = (self.run_dir / "run.json").read_bytes()
        for value in (True, 0, -1, 1, 511, 262145, 1.5, "4096"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "context_length"):
                self.run_stage(context_length=value)
            self.assertEqual((self.run_dir / "run.json").read_bytes(), original)
            self.assertFalse((self.run_dir / "judge_responses.jsonl").exists())
        for value in (512, 262144):
            with self.subTest(context_boundary=value):
                self.run_stage(context_length=value)
                self.assertEqual(self.saved()[0]["plan_evaluation"]["context_length"], value)

    def test_cli_runtime_overrides_reach_server_and_stage_and_restart_without_flags_reuses_them(self):
        self.metadata["experiment"]["execution"] = {"ollama_judge_context_length": 8192}
        self.write()
        models = FakeModels()
        with patch.object(command, "_prepare", return_value=(self.scorer, models, models)), \
                patch.object(command, "ollama_for_saved_run", return_value=nullcontext()) as server, \
                patch.object(command, "generate_report", return_value=self.run_dir / "report.html"):
            self.assertEqual(command.main([str(self.run_dir), "--concurrency", "8", "--context-length", "4096"],
                                          root=self.root), 0)
            server.assert_called_once_with(self.run_dir.resolve(), concurrency=8, context_length=4096)
            before = len(models.requests)
            server.reset_mock()
            self.assertEqual(command.main([str(self.run_dir)], root=self.root), 0)
            server.assert_called_once_with(self.run_dir.resolve(), concurrency=8, context_length=4096)
            self.assertEqual(len(models.requests), before)
        metadata, _ = self.saved()
        self.assertEqual(metadata["experiment"], self.metadata["experiment"])
        self.assertEqual(metadata["plan_evaluation"]["context_length"], 4096)
        self.assertEqual(metadata["plan_evaluation"]["concurrency"], 8)

    def test_cli_invalid_runtime_override_does_not_start_server_or_rewrite_metadata(self):
        original = (self.run_dir / "run.json").read_bytes()
        for args in (("--concurrency", "0"), ("--context-length", "0"),
                     ("--context-length", "511"), ("--context-length", "262145")):
            with self.subTest(args=args), patch.object(command, "ollama_for_saved_run") as server, \
                    patch.object(command, "_prepare") as prepare:
                self.assertEqual(command.main([str(self.run_dir), *args], root=self.root), 1)
                server.assert_not_called()
                prepare.assert_not_called()
                self.assertEqual((self.run_dir / "run.json").read_bytes(), original)

    def test_changed_input_labels_and_scorer_context_are_refused(self):
        self.scorer._patients[self.patients[0]["id"]]["subjective"] = "Changed S/O"
        with self.assertRaisesRegex(ValueError, "S/O"):
            self.run_stage()

    def test_unavailable_judge_does_not_create_false_success(self):
        models = FakeModels()
        models.model_ids = ("other",)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.run_stage(models)
        self.assertEqual(models.requests, [])
        self.assertNotIn("plan_evaluation", self.saved()[0])

    def test_terminal_model_errors_stop_the_bounded_queue(self):
        class NotFoundError(Exception):
            pass
        models = FakeModels(lambda model, messages: NotFoundError("missing model"))
        with self.assertRaises(stage.PlanJudgeUnavailableError):
            self.run_stage(models, concurrency=1)
        self.assertLess(len(models.requests), 18)
        self.assertEqual(self.saved()[0]["plan_evaluation"]["status"], "failed_incomplete")

    def test_model_error_and_invalid_reply_secrets_are_redacted_in_every_artifact(self):
        for value in (RuntimeError("test-private-key"), '{"credit": "test-private-key"}'):
            self.write()
            cache = self.run_dir / "judge_responses.jsonl"
            cache.unlink(missing_ok=True)
            models = FakeModels(lambda model, messages: value if messages[0]["content"].startswith(
                "Grade whether the generated plan covers") else None)
            self.run_stage(models)
            for path in (self.run_dir / "run.json", self.run_dir / "results.jsonl", cache):
                if path.exists():
                    self.assertNotIn("test-private-key", path.read_text())
            self.assertIn("[REDACTED]", (self.run_dir / "results.jsonl").read_text())

    def test_lock_excludes_a_second_owner_and_remains_reusable_after_release(self):
        with exclusive_run_lock(self.run_dir):
            with self.assertRaises(RunLockError):
                self.run_stage()
        self.assertTrue((self.run_dir / ".run.lock").exists())
        self.run_stage()


if __name__ == "__main__":
    unittest.main()
