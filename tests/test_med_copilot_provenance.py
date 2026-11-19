"""Offline tests for end-of-run usage and recovery provenance."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

# Direct execution starts in tests/; resolve project modules before importing them.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime import provenance
from scripts.med_copilot_run_file import RunFile


STAMP = datetime.fromisoformat("2026-11-12T13:30:00-08:00")


def response(identifier, model="gpt-4o-mini-2024-07-18", input_tokens=100, output_tokens=20):
    return {"response_id": identifier, "reported_model": model,
            "token_usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}}


def patient(identifier="A"):
    return {"id": identifier, "components": {"evidence": {"status": "completed", "capture": {
        "model_responses": {"keyword_extraction": [response("prep-" + identifier)],
            "guideline_query_embedding": [response(None, "text-embedding-3-small", 50, None)]}}}},
        "variants": {"llm_only": {"assessment_plan_prompts": {"system": "s", "user": "u"},
            "assessment_plans": {
                "hosted": {"status": "completed", "model_responses": [response("final-" + identifier)]},
                "local": {"status": "completed", "model_responses": [response("local-" + identifier, "gemma3:4b")]}}}}}


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.metadata = {"status": "completed", "resumptions": [],
            "experiment": {"patient_keyword_model": "gpt-4o-mini"},
            "assessment_plan_models": {"models": {
                "hosted": {"config": {"model": "gpt-4o-mini", "api_base": None}},
                "local": {"config": {"model": "gemma3:4b", "api_base": "http://127.0.0.1:11435/v1"}}}}}
        self.rows = [patient()]
        self.save_sources()
        self.clock = patch.object(provenance, "pacific_now", return_value=STAMP)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def save_sources(self):
        (self.run / "run.json").write_text(json.dumps(self.metadata))
        (self.run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def write(self):
        paths = provenance.write_provenance(self.run)
        return {key: json.loads(path.read_text()) for key, path in paths.items()}

    def test_final_report_schema_math_and_usage_ledger_without_mutating_sources(self):
        before = {name: (self.run / name).read_bytes() for name in ("run.json", "results.jsonl")}
        reports = self.write()
        cost = reports["usage"]
        self.assertEqual(set(cost), {"observed_at_pst", "prepared_patient_count", "completed_hosted_output_count",
            "usage", "missing_or_malformed_usage", "captured_usage_estimated_usd", "usd_per_million_tokens",
            "price_sources", "limitations", "deduplication", "measured_usage", "api_usage_file"})
        self.assertEqual(cost["observed_at_pst"], STAMP.isoformat())
        self.assertEqual(cost["prepared_patient_count"], 1)
        self.assertEqual(cost["completed_hosted_output_count"], 1)
        self.assertEqual(cost["usage"]["preparation_gpt"],
            {"responses": 1, "input_tokens": 100, "output_tokens": 20, "estimated_usd": 0.000027})
        self.assertEqual(cost["usage"]["final_gpt"], cost["usage"]["preparation_gpt"])
        self.assertEqual(cost["usage"]["embedding"],
            {"responses": 1, "input_tokens": 50, "estimated_usd": 0.000001})
        self.assertAlmostEqual(cost["captured_usage_estimated_usd"], 0.000055)
        self.assertEqual(cost["missing_or_malformed_usage"], 0)
        self.assertEqual(reports["recovery"], {"run_restarts": 0, "events": []})
        self.assertEqual({p.name for p in (self.run / "provenance").iterdir()},
                         {"reported-usage-cost.json", "recovery-incidents.json", "api-usage.jsonl"})
        self.assertFalse(list(self.run.rglob("command.txt")))
        self.assertFalse(list(self.run.rglob("launch.json")))
        self.assertEqual({name: (self.run / name).read_bytes() for name in before}, before)

    def test_unique_response_ids_deduplicate_cache_hits_but_idless_calls_remain_distinct(self):
        second = deepcopy(self.rows[0]); second["id"] = "B"
        # Same response IDs are cached metadata; ID-less embedding captures cannot be deduplicated.
        self.rows.append(second); self.save_sources()
        cost = self.write()["usage"]
        self.assertEqual(cost["usage"]["preparation_gpt"]["responses"], 1)
        self.assertEqual(cost["usage"]["final_gpt"]["responses"], 1)
        self.assertEqual(cost["usage"]["embedding"]["responses"], 2)
        self.assertEqual(cost["prepared_patient_count"], 2)
        self.assertEqual(cost["completed_hosted_output_count"], 2)

    def test_partial_duplicate_usage_can_be_completed_by_a_later_capture(self):
        entries = self.rows[0]["components"]["evidence"]["capture"]["model_responses"]["keyword_extraction"]
        entries.insert(0, {"response_id": "prep-A", "reported_model": "gpt-4o-mini", "token_usage": None})
        self.save_sources()
        cost = self.write()["usage"]
        self.assertEqual(cost["usage"]["preparation_gpt"]["responses"], 1)
        self.assertEqual(cost["missing_or_malformed_usage"], 0)

    def test_duplicate_model_alias_and_version_metadata_do_not_double_count_or_conflict(self):
        entries = self.rows[0]["components"]["evidence"]["capture"]["model_responses"]["keyword_extraction"]
        entries.insert(0, response("prep-A", "gpt-4o-mini"))
        self.save_sources()
        self.assertEqual(self.write()["usage"]["usage"]["preparation_gpt"]["responses"], 1)

    def test_duplicate_with_later_model_identity_can_supply_a_supported_price(self):
        capture = self.rows[0]["components"]["evidence"]["capture"]["model_responses"]
        capture["guideline_evidence"] = [response("guideline", None), response("guideline")]
        self.save_sources()
        cost = self.write()["usage"]
        self.assertEqual(cost["usage"]["preparation_gpt"]["responses"], 2)
        self.assertFalse(any("unknown (" in note for note in cost["limitations"]))

    def test_bad_usage_is_explicit_and_never_priced_as_zero_token_success(self):
        variants = [None, {}, {"prompt_tokens": True, "completion_tokens": 1},
                    {"prompt_tokens": -1, "completion_tokens": 1},
                    {"prompt_tokens": 1.5, "completion_tokens": 1},
                    {"prompt_tokens": 1, "completion_tokens": "2"}]
        for usage in variants:
            with self.subTest(usage=usage):
                row = patient(); row["variants"]["llm_only"]["assessment_plans"]["hosted"]["model_responses"][0]["token_usage"] = usage
                cost = provenance.write_provenance(self.run, metadata=self.metadata, records=[row])
                report = json.loads(cost["usage"].read_text())
                self.assertEqual(report["missing_or_malformed_usage"], 1)
                self.assertEqual(report["usage"]["final_gpt"]["responses"], 0)

    def test_zero_reported_usage_is_valid(self):
        row = patient(); row["variants"]["llm_only"]["assessment_plans"]["hosted"]["model_responses"] = [response("free", input_tokens=0, output_tokens=0)]
        paths = provenance.write_provenance(self.run, metadata=self.metadata, records=[row])
        cost = json.loads(paths["usage"].read_text())
        self.assertEqual(cost["usage"]["final_gpt"]["responses"], 1)
        self.assertEqual(cost["usage"]["final_gpt"]["estimated_usd"], 0)

    def test_unknown_hosted_model_is_counted_but_not_given_an_incorrect_gpt_price(self):
        self.rows[0]["variants"]["llm_only"]["assessment_plans"]["hosted"]["model_responses"] = [response("other", "other-provider-model")]
        self.save_sources()
        report = self.write()["usage"]
        self.assertEqual(report["completed_hosted_output_count"], 1)
        self.assertEqual(report["usage"]["final_gpt"]["responses"], 0)
        self.assertTrue(any("other-provider-model (1)" in note for note in report["limitations"]))

    def test_loopback_local_models_never_add_api_credit_usage(self):
        for endpoint in ("http://127.0.0.1:11435/v1", "http://localhost:11435/v1", "http://[::1]:11435/v1"):
            with self.subTest(endpoint=endpoint):
                self.metadata["assessment_plan_models"]["models"]["local"]["config"]["api_base"] = endpoint
                self.save_sources()
                report = self.write()["usage"]
                self.assertEqual(report["completed_hosted_output_count"], 1)
                self.assertFalse(any("gemma" in note for note in report["limitations"]))

    def test_failed_attempts_are_priced_even_when_only_one_output_completes(self):
        output = self.rows[0]["variants"]["llm_only"]["assessment_plans"]["hosted"]
        output["model_responses"].insert(0, response("capped-attempt", input_tokens=200, output_tokens=100))
        self.save_sources()
        cost = self.write()["usage"]
        self.assertEqual(cost["completed_hosted_output_count"], 1)
        self.assertEqual(cost["usage"]["final_gpt"]["responses"], 2)
        self.assertEqual(cost["usage"]["final_gpt"]["input_tokens"], 300)

    def test_preparation_and_hosted_counts_exclude_incomplete_work(self):
        self.rows[0]["components"]["evidence"]["status"] = "failed_incomplete"
        self.rows[0]["variants"]["llm_only"]["assessment_plans"]["hosted"]["status"] = "failed_incomplete"
        self.save_sources()
        cost = self.write()["usage"]
        self.assertEqual(cost["prepared_patient_count"], 0)
        self.assertEqual(cost["completed_hosted_output_count"], 0)
        self.assertEqual(cost["usage"]["final_gpt"]["responses"], 1)

    def test_agent_calls_and_failed_attempts_have_actual_counts_and_variant_attribution(self):
        component = self.rows[0]["components"]["evidence"]
        component.update(status="failed_incomplete", settings={"agent_model": "gpt-4o-mini"})
        component["capture"]["model_responses"]["agent_control"] = [
            response("agent-failed", input_tokens=123, output_tokens=7),
            response("agent-finish", input_tokens=456, output_tokens=9)]
        calls = component["capture"]["model_responses"]["agent_control"]
        for call in calls:
            call["token_usage"]["total_tokens"] = (call["token_usage"]["prompt_tokens"]
                                                    + call["token_usage"]["completion_tokens"])
            call["token_usage"]["cached_tokens"] = 64
        # Legacy convenience copies must not be counted again.
        component["capture"]["agent"] = {"usage": [call["token_usage"] for call in calls]}
        self.rows[0]["variants"]["llm_only"]["components"] = {"agent_patients": "evidence"}
        self.save_sources()
        report = self.write()["usage"]
        measured = report["measured_usage"]["by_stage"]["agent_control"]
        self.assertEqual(measured["responses"], 2)
        self.assertEqual(measured["reported_tokens"], {"prompt_tokens": 579, "completion_tokens": 16,
                                                     "total_tokens": 595, "cached_tokens": 128})
        self.assertEqual(measured["missing_usage_fields"], dict.fromkeys(measured["reported_tokens"], 0))
        self.assertEqual(report["usage"]["preparation_gpt"]["input_tokens"], 679)
        ledger = [json.loads(line) for line in (self.run / report["api_usage_file"]).read_text().splitlines()]
        agents = [row for row in ledger if row["contexts"][0]["stage"] == "agent_control"]
        self.assertEqual([row["response_id"] for row in agents], ["agent-failed", "agent-finish"])
        for index, row in enumerate(agents):
            self.assertEqual(row["contexts"], [{"patient_id": "A", "component": "evidence",
                "stage": "agent_control", "variants": ["llm_only"], "cache_hit": False, "capture_index": index}])

    def test_old_agent_count_captures_are_included_when_per_response_metadata_is_absent(self):
        component = self.rows[0]["components"]["evidence"]
        component["settings"] = {"agent_model": "gpt-4o-mini"}
        component["capture"]["agent"] = {"usage": [{"prompt_tokens": 17, "completion_tokens": 3}]}
        self.save_sources()
        report = self.write()["usage"]
        self.assertEqual(report["usage"]["preparation_gpt"]["input_tokens"], 117)
        actual = report["measured_usage"]["by_stage"]["agent_control"]
        self.assertEqual(actual["reported_tokens"]["prompt_tokens"], 17)
        self.assertEqual(actual["missing_usage_fields"]["total_tokens"], 1)
        self.assertEqual(actual["reported_tokens"]["total_tokens"], 0)
        ledger = [json.loads(line) for line in (self.run / report["api_usage_file"]).read_text().splitlines()]
        legacy = next(row for row in ledger if row["contexts"][0]["stage"] == "agent_control")
        self.assertIsNone(legacy["reported_model"])
        self.assertEqual(legacy["model_source"], "configured")

    def test_request_ids_deduplicate_embeddings_and_local_cache_hits_add_no_new_tokens(self):
        first = self.rows[0]["components"]["evidence"]["capture"]["model_responses"]
        embedding = first["guideline_query_embedding"][0]
        embedding["request_id"] = "embedding-request"
        first["guideline_query_embedding"].append(deepcopy(embedding))
        first["guideline_evidence"] = [{**response("cached-answer", input_tokens=9999), "cache_hit": True}]
        self.save_sources()
        report = self.write()["usage"]
        self.assertEqual(report["usage"]["embedding"]["responses"], 1)
        self.assertEqual(report["measured_usage"]["local_cache_hits"], 1)
        self.assertEqual(report["measured_usage"]["responses"], 3)
        self.assertEqual(report["usage"]["preparation_gpt"]["input_tokens"], 100)

    def test_cached_copy_of_a_response_already_used_live_does_not_add_tokens_for_another_variant(self):
        output = self.rows[0]["variants"]["llm_only"]
        component = self.rows[0]["components"]["evidence"]
        component["capture"]["model_responses"]["keyword_extraction"] = []
        component["capture"]["model_responses"]["agent_control"] = [response("shared")]
        output["components"] = {"agent_patients": "evidence"}
        other = deepcopy(component)
        other["capture"]["model_responses"]["agent_control"][0]["cache_hit"] = True
        other["capture"]["model_responses"]["guideline_query_embedding"] = []
        self.rows[0]["components"]["cached_evidence"] = other
        self.rows[0]["variants"]["cached_variant"] = {"components": {"agent_patients": "cached_evidence"}}
        self.save_sources()
        report = self.write()["usage"]
        self.assertEqual(report["measured_usage"]["local_cache_hits"], 1)
        self.assertEqual(report["measured_usage"]["by_stage"]["agent_control"]["responses"], 1)
        self.assertNotIn("cached_variant", report["measured_usage"]["by_variant"])

    def test_partial_duplicate_metadata_merges_cached_counts_but_conflicts_are_rejected(self):
        entries = self.rows[0]["components"]["evidence"]["capture"]["model_responses"]["keyword_extraction"]
        extra = deepcopy(entries[0])
        extra["token_usage"].update(total_tokens=120, cached_tokens=32)
        entries.append(extra)
        self.save_sources()
        report = self.write()["usage"]
        actual = report["measured_usage"]["by_stage"]["keyword_extraction"]
        self.assertEqual(actual["responses"], 1)
        self.assertEqual(actual["reported_tokens"]["cached_tokens"], 32)
        conflict = deepcopy(extra)
        conflict["token_usage"]["cached_tokens"] = 64
        entries.append(conflict)
        self.save_sources()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.write()

    def test_unpriced_models_and_missing_fields_still_have_measured_token_records(self):
        output = self.rows[0]["variants"]["llm_only"]["assessment_plans"]["hosted"]
        output["model_responses"] = [response("unpriced", "new-model", 321, 12)]
        self.save_sources()
        report = self.write()["usage"]
        actual = report["measured_usage"]["by_model"]["new-model"]
        self.assertEqual(actual["reported_tokens"]["prompt_tokens"], 321)
        self.assertEqual(actual["reported_tokens"]["completion_tokens"], 12)
        self.assertEqual(actual["missing_usage_fields"]["total_tokens"], 1)
        self.assertEqual(actual["missing_usage_fields"]["cached_tokens"], 1)

    def test_resume_preserves_prior_failed_agent_usage_and_rebuilds_a_stable_ledger(self):
        capture = self.rows[0]["components"]["evidence"]["capture"]["model_responses"]
        capture["agent_control"] = [response("old-agent")]
        self.save_sources()
        self.write()
        self.metadata["resumptions"] = [{"started_at": "later"}]
        capture["agent_control"].append(response("new-agent", input_tokens=250))
        self.save_sources()
        report = self.write()["usage"]
        ledger_path = self.run / report["api_usage_file"]
        contents = ledger_path.read_bytes()
        measured = report["measured_usage"]["by_stage"]["agent_control"]
        self.assertEqual(measured["responses"], 2)
        self.assertEqual(measured["reported_tokens"]["prompt_tokens"], 350)
        self.write()
        self.assertEqual(ledger_path.read_bytes(), contents)

    def test_llm_only_preparation_requires_a_saved_prompt(self):
        self.rows[0]["components"] = {}
        self.rows[0]["variants"]["llm_only"]["assessment_plan_prompts"] = None
        self.save_sources()
        self.assertEqual(self.write()["usage"]["prepared_patient_count"], 0)

    def test_resume_recomputes_cumulative_usage_preserves_prices_and_detailed_recovery(self):
        self.write()
        cost_path = self.run / "provenance/reported-usage-cost.json"
        old = json.loads(cost_path.read_text()); old["usd_per_million_tokens"]["gpt_4o_mini_input"] = 1
        old.update(budget_usd=100, prior_work_allowance_usd=5, planning_total_usd=10,
                   remaining_budget_usd_with_allowance=90)
        old["limitations"].append("Old budget allowance")
        cost_path.write_text(json.dumps(old))
        detail = {"event": "experiment_resume_completed", "verified": True}
        recovery = self.run / "provenance/recovery-incidents.json"
        recovery.write_text(json.dumps({"run_restarts": 1, "events": [detail]}))
        self.metadata["resumptions"] = [{"started_at": "original-resume"}, {"started_at": "new-resume"}]
        self.metadata["preparation_recovery"] = {"events": [{"worker": 1, "outcome": "recovered"}]}
        self.rows.append(patient("B")); self.save_sources()
        reports = self.write()
        self.assertEqual(reports["usage"]["usage"]["final_gpt"]["responses"], 2)
        self.assertEqual(reports["usage"]["usd_per_million_tokens"]["gpt_4o_mini_input"], 1)
        self.assertFalse(any("budget" in key or "allowance" in key for key in reports["usage"]))
        self.assertFalse(any("allowance" in note for note in reports["usage"]["limitations"]))
        self.assertEqual(reports["recovery"]["run_restarts"], 2)
        self.assertEqual(reports["recovery"]["events"], [detail,
            {"event": "experiment_resumed", "started_at": "new-resume"},
            {"event": "preparation_worker_recovery", "worker": 1, "outcome": "recovered"}])
        self.assertEqual(self.write()["recovery"], reports["recovery"])

    def test_invalid_source_or_conflicting_response_never_replaces_previous_reports(self):
        self.write()
        original = {path: path.read_bytes() for path in (self.run / "provenance").iterdir()}
        for mode in ("duplicate-patient", "conflicting-usage", "invalid-json"):
            with self.subTest(mode=mode):
                self.rows = [patient()]
                if mode == "duplicate-patient": self.rows.append(patient())
                if mode == "conflicting-usage":
                    self.rows[0]["components"]["evidence"]["capture"]["model_responses"]["keyword_extraction"].append(response("prep-A", input_tokens=200))
                self.save_sources()
                if mode == "invalid-json": (self.run / "results.jsonl").write_text('{"invalid"\n')
                with self.assertRaises(ValueError): self.write()
                self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_running_report_is_refused_without_creating_provenance(self):
        self.metadata["status"] = "running"; self.save_sources()
        with self.assertRaisesRegex(ValueError, "runner has stopped"): self.write()
        self.assertFalse((self.run / "provenance").exists())

    def test_invalid_prices_or_recovery_fail_before_any_report_replacement(self):
        self.write()
        cost = self.run / "provenance/reported-usage-cost.json"
        recovery = self.run / "provenance/recovery-incidents.json"
        for value in (True, -1, float("nan"), "0.15"):
            report = json.loads(cost.read_text()); report["usd_per_million_tokens"]["gpt_4o_mini_input"] = value
            cost.write_text(json.dumps(report))
            before = cost.read_bytes()
            with self.assertRaisesRegex(ValueError, "prices"): self.write()
            self.assertEqual(cost.read_bytes(), before)
        cost.unlink(); self.write()
        recovery.write_text(json.dumps({"run_restarts": 1, "events": []}))
        before = cost.read_bytes()
        with self.assertRaisesRegex(ValueError, "recovery history"): self.write()
        self.assertEqual(cost.read_bytes(), before)

    def test_failed_startup_can_finalize_empty_costs_without_results(self):
        self.metadata["status"] = "failed_incomplete"; self.save_sources()
        (self.run / "results.jsonl").unlink()
        cost = self.write()["usage"]
        self.assertEqual(cost["captured_usage_estimated_usd"], 0)
        self.assertEqual(cost["prepared_patient_count"], 0)

    def test_atomic_write_failure_preserves_old_file_and_removes_temporary(self):
        self.write()
        original = {path: path.read_bytes() for path in (self.run / "provenance").iterdir()}
        with patch.object(provenance.os, "replace", side_effect=OSError("disk unavailable")), \
                self.assertRaisesRegex(OSError, "disk unavailable"):
            self.write()
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertFalse(list((self.run / "provenance").glob("*.tmp")))

    def test_context_waits_for_resource_cleanup_and_redacts_failure_without_masking_primary(self):
        run_file = RunFile(self.run, self.metadata)
        run_file.redact = lambda text: text.replace("secret", "[REDACTED]")
        events = []
        with patch.object(provenance, "write_provenance", side_effect=lambda *args, **kw: events.append("report")):
            with ExitStack() as stack:
                stack.enter_context(provenance.finalize_provenance(run_file))
                stack.callback(events.append, "resources stopped")
                self.assertEqual(events, [])
            self.assertEqual(events, ["resources stopped", "report"])
        error = RuntimeError("primary failure")
        stderr = StringIO()
        with patch.object(provenance, "write_provenance", side_effect=OSError("secret disk failure")), \
                patch.object(sys, "stderr", stderr), self.assertRaises(RuntimeError) as raised:
            with provenance.finalize_provenance(run_file): raise error
        self.assertIs(raised.exception, error)
        self.assertIn("[REDACTED]", stderr.getvalue())
        self.assertNotIn("secret", stderr.getvalue())

    def test_provenance_failure_on_otherwise_successful_exit_is_not_silenced(self):
        run_file = RunFile(self.run, self.metadata)
        with patch.object(provenance, "write_provenance", side_effect=OSError("disk failure")), \
                self.assertRaisesRegex(OSError, "disk failure"):
            with provenance.finalize_provenance(run_file): pass

    def test_outer_handled_exception_does_not_hide_new_provenance_failure(self):
        run_file = RunFile(self.run, self.metadata)
        try:
            raise ValueError("already handled outside the runner")
        except ValueError:
            with patch.object(provenance, "write_provenance", side_effect=OSError("new failure")), \
                    self.assertRaisesRegex(OSError, "new failure"):
                with provenance.finalize_provenance(run_file): pass

    def test_invalid_recovery_event_values_do_not_partially_refresh_cost_report(self):
        self.write()
        cost = self.run / "provenance/reported-usage-cost.json"
        before = cost.read_bytes()
        self.metadata["preparation_recovery"] = {"events": [{"value": float("nan")} ]}
        self.save_sources()
        with self.assertRaises(ValueError): self.write()
        self.assertEqual(cost.read_bytes(), before)

    def test_git_status_exposes_only_approved_results_and_provenance_artifacts(self):
        repository = self.run / "ignore-test"
        repository.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
        (repository / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
        subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
        prefix = "experiments/output/future-run/"
        approved = {prefix + name for name in ("results.jsonl", "run.json", "results_summary.html",
            "provenance/reported-usage-cost.json", "provenance/recovery-incidents.json",
            "provenance/api-usage.jsonl")}
        operational = {prefix + name for name in ("jobs.sqlite3", "jobs.sqlite3-wal", "jobs.sqlite3-shm",
            ".run.lock", "experiment.log", "supervisor.out", "supervisor.pid", "launcher.sh", "watch_run.py",
            "progress.jsonl", "progress.latest.json", "command.txt", "launch.json", "unexpected.tmp",
            "pendingresume-01/pending.json", "provenance/command.txt", "provenance/launch.json",
            "provenance/unexpected.json")}
        operational.update({"experiments/output/unexpected.json", "experiments/output/launch_future/watch_run.py"})
        for name in approved | operational:
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        status = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repository, text=True)
        self.assertEqual(set(status.splitlines()), approved)


if __name__ == "__main__":
    unittest.main()
