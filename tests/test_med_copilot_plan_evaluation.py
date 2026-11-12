"""Offline regression tests for the independently resumable Plan-only stage."""

from copy import deepcopy
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

# Direct execution starts in tests/; project and vendored imports need these roots.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
with patch.object(nltk, "download", return_value=True):
    from scripts import med_copilot_plan_evaluation as stage
    from scripts.med_copilot_judges import CorrectnessJudges, plan_entry
    from scripts.med_copilot_run_lock import RunLockError, exclusive_run_lock
    from scripts.med_copilot_scoring import PatientScorer


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
