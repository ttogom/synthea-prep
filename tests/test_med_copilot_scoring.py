"""Offline tests for label validation, field routing, and upstream metric reuse."""

from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
from importlib import metadata as package_metadata
from io import StringIO
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from metric_doubles import SingletonMetricDouble
import nltk
from nltk.translate.meteor_score import single_meteor_score
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu
import torch

# Direct execution puts tests/ on sys.path; project imports require its parent.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The upstream module requests NLTK downloads at import. Offline tests suppress
# only that side effect; production imports retain the unchanged upstream code.
with patch.object(nltk, "download", return_value=True):
    from scripts.med_copilot_scoring import (
        METRIC_NAMES,
        PatientScorer,
        UPSTREAM_EVALUATOR,
        parse_scoring_args,
    )
    from med_copilot.upstream.evaluate_metrics import Evaluator
    from scripts.med_copilot_judges import JUDGE_FILE, CorrectnessJudges
    from scripts.med_copilot_scoring import resume_evaluation
    from scripts.med_copilot_run_file import RunFile, _ERROR_SCOPE, error_scope


class FakeSimilarity:
    """Offline stand-in for BioLORD-2023: identical text 1.0, anything else 0.75."""

    def __init__(self, *args, **kwargs):
        pass

    def cosine(self, reference, generated):
        return 1.0 if reference.strip().lower() == generated.strip().lower() else 0.75


def setUpModule():
    # Never load the real BioLORD model, and judge Plans with a fake selected model.
    for target, value in (("scripts.med_copilot_judges.BioLordSimilarity", FakeSimilarity),):
        patcher = patch(target, value)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


class PatientScorerTests(unittest.TestCase):
    def test_score_results_batches_patients_and_routes_distinct_scores_by_full_identity(self):
        self.labels[1]["plan"] = ["Other reference plan"]
        self.write_labels(self.labels)
        records = [self.result("patient-b", condition="B"), self.result("patient-a", condition="A")]
        before = deepcopy(records)
        batches = []

        def score_pairs(pairs):
            batches.append(list(pairs))
            return iter(SimpleNamespace(key=pair.key, error=None,
                                        metrics={name: (index + 1) / 100 for name in METRIC_NAMES})
                        for index, pair in enumerate(pairs))

        scorer = self.scorer(presenting_evidence_only=False,
                             metric_evaluator=SimpleNamespace(score_pairs=score_pairs))
        returned = scorer.score_results(iter(records))
        self.assertEqual(returned, records)
        self.assertIs(returned[0], records[0])
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 12)
        self.assertEqual([pair.key for pair in batches[0]], [
            (patient_id, None, None, rank, field)
            for patient_id in ("patient-b", "patient-a") for rank in (1, 2, 3)
            for field in ("condition", "plan")])
        self.assertEqual([pair.reference for pair in batches[0]][:2], ["normal", "Other reference plan"])
        for offset, (record, original) in enumerate(zip(records, before)):
            self.assertEqual(record["predictions"], original["predictions"])
            for index, entry in enumerate(record["evaluation"]):
                for field_index, field in enumerate(("condition", "plan")):
                    expected = (offset * 6 + index * 2 + field_index + 1) / 100
                    self.assertEqual(set(entry[field]["metrics"].values()), {expected})

    def test_score_results_resume_uses_only_pending_pairs_and_rescore_is_explicit(self):
        self.labels[1]["plan"] = ["Other reference plan"]
        self.write_labels(self.labels)
        records = [self.result(), self.result("patient-b")]
        scorer = self.scorer(presenting_evidence_only=False)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            scorer.score_results(records)
        self.assertEqual(evaluate.call_count, 12)
        kept = deepcopy(records[0]["evaluation"])
        records[1]["evaluation"][1]["plan"] = {"status": "error", "metrics": {}, "error": {}}
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            scorer.score_results(records)
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args, (["Other reference plan"], ["P"]))
        self.assertEqual(records[0]["evaluation"], kept)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            scorer.score_results(records, rescore=True)
        self.assertEqual(evaluate.call_count, 12)

    def test_score_results_validates_entire_batch_before_mutating_first_record(self):
        record = self.result()
        for invalid in (self.result("unknown"), record):
            with self.subTest(invalid=invalid["id"]):
                before = deepcopy(record)
                with patch.object(Evaluator, "evaluate_metrics") as evaluate, self.assertRaises(ValueError):
                    self.scorer().score_results([record, invalid])
                evaluate.assert_not_called()
                self.assertEqual(record, before)
        self.assertEqual(self.scorer().score_results([]), [])

    def test_score_results_supports_disjoint_slots_for_same_patient(self):
        records = [{"id": "patient-a", "variants": {variant: {"assessment_plans": {
            model: self.result(condition=model)}}}} for variant, model in (
                ("llm_only", "mini"), ("guidelines", "mini"), ("guidelines", "llama"))]
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            self.scorer().score_results(records)
        self.assertEqual(evaluate.call_count, 18)
        self.assertEqual([evaluate.call_args_list[index].args[1] for index in (0, 6, 12)],
                         [["mini"], ["mini"], ["llama"]])

    def test_score_results_interruption_preserves_completed_patient_and_resumes_remaining(self):
        self.labels[1]["plan"] = ["Other reference plan"]
        self.write_labels(self.labels)
        records = [self.result(condition="FIRST"), self.result("patient-b", condition="SECOND")]
        scorer = self.scorer(presenting_evidence_only=False)

        def metric(references, candidates, lang):
            if candidates == ["SECOND"]:
                raise KeyboardInterrupt()
            return self.metrics

        with patch.object(Evaluator, "evaluate_metrics", side_effect=metric), self.assertRaises(KeyboardInterrupt):
            scorer.score_results(records)
        saved_first = deepcopy(records[0]["evaluation"])
        self.assertTrue(all(entry[field]["status"] == "scored" for entry in saved_first
                            for field in ("condition", "plan")))
        self.assertEqual(records[1]["evaluation"][0]["condition"]["status"], "interrupted")
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            scorer.score_results(records)
        self.assertEqual(evaluate.call_count, 6)
        self.assertEqual(records[0]["evaluation"], saved_first)

    def test_score_results_attributes_checkpoints_and_judge_errors_to_each_patient(self):
        self.labels[1]["plan"] = ["Other reference plan"]
        self.write_labels(self.labels)
        records = [self.result(), self.result("patient-b")]
        scorer = self.scorer(presenting_evidence_only=False)
        owner = RunFile(self.directory, {"errors": []})
        checkpoints = []
        owner.checkpoint = lambda: checkpoints.append((_ERROR_SCOPE.get() or {}).get("patient_id"))

        def cosine(reference, generated):
            if reference == "normal":
                raise RuntimeError("patient b diagnosis failed")
            return 1.0

        scorer.attach_judges(CorrectnessJudges(None, (), self.directory / JUDGE_FILE,
            plan_judge="judge", plan_execution="deferred", similarity=SimpleNamespace(cosine=cosine)))
        with error_scope(owner, patient_id="outer"), \
                patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_results(records)
            self.assertEqual(_ERROR_SCOPE.get()["patient_id"], "outer")
        self.assertEqual(set(checkpoints), {"patient-a", "patient-b"})
        self.assertEqual({(error["patient_id"], error["stage"]) for error in owner.metadata["errors"]},
                         {("patient-b", "scoring")})
        self.assertTrue(all(entry["diagnosis_correctness"]["status"] == "scored"
                            for entry in records[0]["evaluation"]))
        self.assertTrue(all(entry["diagnosis_correctness"]["status"] == "error"
                            for entry in records[1]["evaluation"]))

    def test_each_model_scores_against_same_patient_without_mutating_predictions(self):
        first, second = self.result(condition="mini"), self.result(condition="llama")
        first.pop("id")
        second.pop("id")
        record = {"id": "patient-a", "assessment_plans": {"mini": first, "llama": second}}
        before = deepcopy(record)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as metric:
            self.assertIs(self.scorer().score_result(record), record)
        self.assertEqual(metric.call_count, 12)
        self.assertEqual(metric.call_args_list[0].args, (["acute bronchitis"], ["mini"]))
        self.assertEqual(metric.call_args_list[6].args, (["acute bronchitis"], ["llama"]))
        for key in record["assessment_plans"]:
            self.assertEqual(record["assessment_plans"][key]["predictions"],
                             before["assessment_plans"][key]["predictions"])
        self.assertNotIn("evaluation", record)
        first["evaluation"][0]["condition"]["metrics"]["BLEU"] = 999
        self.assertEqual(second["evaluation"][0]["condition"]["metrics"], self.metrics)

    def test_failed_and_unattempted_models_are_not_scored(self):
        failed = {"status": "failed_incomplete", "predictions": None, "evaluation": None}
        waiting = {"status": "not_processed", "predictions": None, "evaluation": None}
        good = self.result()
        record = {"id": "patient-a", "assessment_plans": {"failed": failed, "waiting": waiting, "good": good}}
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as metric:
            self.scorer().score_result(record)
        self.assertEqual(metric.call_count, 6)
        self.assertIsNone(failed["evaluation"])
        self.assertIsNone(waiting["evaluation"])

    def test_interruption_keeps_prior_model_scores_and_later_models_unscored(self):
        record = {"id": "patient-a", "assessment_plans": {
            name: self.result(condition=name) for name in ("mini", "llama", "later")}}
        def metric(references, candidates, lang):
            if candidates == ["llama"]:
                raise KeyboardInterrupt()
            return self.metrics
        with patch.object(Evaluator, "evaluate_metrics", side_effect=metric), self.assertRaises(KeyboardInterrupt):
            self.scorer().score_result(record)
        first, second, last = record["assessment_plans"].values()
        self.assertEqual(first["evaluation"][2]["plan"]["status"], "scored")
        self.assertEqual(second["evaluation"][0]["condition"]["status"], "interrupted")
        self.assertTrue(all(entry[field]["status"] == "pending" for entry in last["evaluation"]
                            for field in ("condition", "plan")))

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.labels_path = self.directory / "labels.jsonl"
        self.labels = [
            {
                "id": "patient-a", "condition": "acute_bronchitis",
                "plan": ["Rest", " Drink fluids "], "has_presenting_evidence": True,
                "bin": "ignored metadata",
            },
            {
                "id": "patient-b", "condition": "normal",
                "plan": ["none recorded"], "has_presenting_evidence": False,
            },
        ]
        self.patients = [
            {"id": "patient-b", "subjective": "b", "objective": "b"},
            {"id": "patient-a", "subjective": "a", "objective": "a"},
        ]
        self.write_labels(self.labels)
        self.metrics = {name: index / 10 for index, name in enumerate(METRIC_NAMES, 1)}

    def write_labels(self, labels):
        self.labels_path.write_text(
            "".join(json.dumps(label, ensure_ascii=False) + "\n" for label in labels), encoding="utf-8",
        )

    def scorer(self, **kwargs):
        kwargs.setdefault("metric_evaluator", SingletonMetricDouble(
            lambda *args, **kw: Evaluator.evaluate_metrics(*args, **kw)))
        return PatientScorer(self.labels_path, self.patients, **kwargs)

    def result(self, patient_id="patient-a", **kwargs):
        condition = kwargs.pop("condition", "A")
        plan = kwargs.pop("plan", ["P"])
        predictions = kwargs.pop("predictions", [
            {"condition": condition, "plan": plan} for _ in range(3)
        ])
        return {"id": patient_id, "status": "completed", "predictions": predictions, **kwargs}

    def assert_unscored(self, entry, status):
        self.assertEqual(entry["status"], status)
        self.assertEqual(entry["metrics"], dict.fromkeys(METRIC_NAMES))
        self.assertEqual(set(entry), {"status", "metrics", "error"})

    def test_filter_defaults_to_evidence_and_preserves_original_patient_objects(self):
        with patch.object(Evaluator, "evaluate_metrics") as evaluate:
            scorer = self.scorer()
        self.assertEqual([patient["id"] for patient in scorer.patients], ["patient-a"])
        self.assertIs(scorer.patients[0], self.patients[1])
        self.assertEqual(len(self.patients), 2)
        evaluate.assert_not_called()

    def test_explicit_false_retains_original_order(self):
        self.assertEqual(self.scorer(presenting_evidence_only=False).patients, self.patients)

    def test_patient_limit_runs_after_evidence_filter_and_preserves_input(self):
        before = deepcopy(self.patients)
        scorer = self.scorer(max_patients=1)
        self.assertEqual(scorer.patients, [self.patients[1]])
        self.assertIs(scorer.patients[0], self.patients[1])
        self.assertEqual(self.patients, before)
        unfiltered = self.scorer(presenting_evidence_only=False, max_patients=1)
        self.assertEqual(unfiltered.patients, [self.patients[0]])
        metadata = unfiltered.metadata(self.directory)
        self.assertEqual(metadata["max_patients"], 1)
        self.assertEqual(metadata["input_patient_count"], 2)
        self.assertEqual(metadata["selected_patient_count"], 1)
        self.assertEqual(metadata["selected_patient_ids"], ["patient-b"])
        self.assertEqual(metadata["excluded_patient_ids"], ["patient-a"])
        with self.assertRaisesRegex(ValueError, "selected input patient"):
            unfiltered.score_result(self.result("patient-a"))

    def test_null_or_large_limit_keeps_the_entire_filtered_cohort(self):
        for value in (None, 2, 10):
            with self.subTest(value=value):
                scorer = self.scorer(presenting_evidence_only=False, max_patients=value)
                self.assertEqual(scorer.patients, self.patients)
                self.assertEqual(scorer.metadata(self.directory)["excluded_patient_ids"], [])
                self.assertEqual(scorer.metadata(self.directory)["max_patients"], value)

    def test_invalid_patient_limit_fails_before_reading_labels(self):
        self.labels_path.unlink()
        for value in (0, -1, True, False, 1.0, "1", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_patients"):
                self.scorer(max_patients=value)

    def test_limit_does_not_hide_missing_labels_or_bad_rows_beyond_selected_patients(self):
        self.write_labels(self.labels[1:])
        with self.assertRaisesRegex(ValueError, "Missing labels.*patient-a"):
            self.scorer(presenting_evidence_only=False, max_patients=1)
        self.write_labels([self.labels[1], {**self.labels[0], "plan": [None]}])
        with self.assertRaisesRegex(ValueError, "plan"):
            self.scorer(presenting_evidence_only=False, max_patients=1)
        self.write_labels(self.labels)
        self.patients.append(self.patients[0])
        with self.assertRaisesRegex(ValueError, "Duplicate input patient"):
            self.scorer(presenting_evidence_only=False, max_patients=1)

    def test_limit_does_not_allow_empty_eligible_cohort(self):
        self.write_labels([{**label, "has_presenting_evidence": False} for label in self.labels])
        with self.assertRaisesRegex(ValueError, "No patients selected"):
            self.scorer(max_patients=1)

    def test_boolean_filter_rejects_truthy_nonbooleans(self):
        for value in (0, 1, "true", None, [], {}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.scorer(presenting_evidence_only=value)

    def test_label_superset_is_allowed_but_validated(self):
        self.write_labels(self.labels + [{
            "id": "extra", "condition": "extra", "plan": [], "has_presenting_evidence": True,
        }])
        self.assertEqual(self.scorer().metadata(self.directory)["label_count"], 3)
        extra = {"id": "extra", "condition": "extra", "plan": [], "has_presenting_evidence": 1}
        self.write_labels(self.labels + [extra])
        with self.assertRaisesRegex(ValueError, "has_presenting_evidence"):
            self.scorer()

    def test_missing_original_input_labels_fail_even_if_filter_would_exclude(self):
        self.write_labels(self.labels[:1])
        with self.assertRaisesRegex(ValueError, "Missing labels.*patient-b"):
            self.scorer()

    def test_duplicate_gold_rows_fail_before_filter(self):
        self.write_labels(self.labels + [self.labels[1]])
        with self.assertRaisesRegex(ValueError, "duplicate label id.*patient-b"):
            self.scorer()

    def test_invalid_label_schema_is_rejected_for_each_field(self):
        changes = [
            ("id", None), ("id", " "), ("id", 3),
            ("condition", None), ("condition", 4), ("condition", " "), ("condition", "___"),
            ("plan", "rest"), ("plan", None), ("plan", [3]), ("plan", [None]),
            ("has_presenting_evidence", None), ("has_presenting_evidence", 1),
            ("has_presenting_evidence", "false"),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                invalid = {**self.labels[1], field: value}
                self.write_labels([self.labels[0], invalid])
                with self.assertRaisesRegex(ValueError, field):
                    self.scorer()

    def test_missing_label_fields_are_rejected(self):
        for field in ("id", "condition", "plan", "has_presenting_evidence"):
            with self.subTest(field=field):
                invalid = {key: value for key, value in self.labels[0].items() if key != field}
                self.write_labels([invalid, self.labels[1]])
                with self.assertRaisesRegex(ValueError, field):
                    self.scorer()

    def test_malformed_nonobject_and_empty_label_files(self):
        for content, error in (("{bad}\n", "invalid label JSON"), ("[]\n", "JSON object"),
                               ("null\n", "JSON object"), ("\n", "Missing labels")):
            with self.subTest(content=content):
                self.labels_path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self.scorer()

    def test_bad_input_ids_and_objects_fail_validation(self):
        for patients in (({"id": ""},), ({"id": 2},), (None,),
                         (self.patients[0], self.patients[0])):
            with self.subTest(patients=patients), self.assertRaises(ValueError):
                PatientScorer(self.labels_path, patients)

    def test_empty_selected_cohort_fails(self):
        for label in self.labels:
            label["has_presenting_evidence"] = False
        self.write_labels(self.labels)
        with self.assertRaisesRegex(ValueError, "No patients selected"):
            self.scorer()
        with self.assertRaisesRegex(ValueError, "No patients selected"):
            PatientScorer(self.labels_path, [])

    def test_blank_lines_and_unicode_ids_are_supported_without_stripping(self):
        self.labels[0]["id"] = "patient café"
        self.patients[1]["id"] = "patient café"
        self.write_labels(self.labels)
        self.labels_path.write_text("\n" + self.labels_path.read_text() + "\n", encoding="utf-8")
        scorer = self.scorer()
        self.assertEqual(scorer.patients[0]["id"], "patient café")

    def test_exact_singleton_inputs_preserve_reference_order_and_whitespace(self):
        self.labels[0]["condition"] = " acute__bronchitis "
        self.write_labels(self.labels)
        result = self.result(condition="  Diagnosis  ", plan=["  Proposed plan\n"])
        original = deepcopy(result)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            returned = self.scorer().score_result(result)
        self.assertIs(returned, result)
        self.assertEqual([call.args for call in evaluate.call_args_list], [
            ([" acute  bronchitis "], ["  Diagnosis  "]),
            (["Rest,  Drink fluids "], ["  Proposed plan\n"]),
        ] * 3)
        self.assertEqual([call.kwargs for call in evaluate.call_args_list], [{"lang": "en"}] * 6)
        self.assertEqual({key: result[key] for key in original}, original)
        self.assertEqual([entry["rank"] for entry in result["evaluation"]], [1, 2, 3])
        for entry in result["evaluation"]:
            self.assertEqual(set(entry), {"rank", "condition", "plan"})
            for field in ("condition", "plan"):
                self.assertEqual(entry[field], {"status": "scored", "metrics": self.metrics, "error": None})
        self.assertIsNot(result["evaluation"][0]["condition"], result["evaluation"][1]["condition"])

    def test_distinct_ranked_pairs_are_scored_separately_without_weighting_or_reordering(self):
        predictions = [
            {"condition": "Likely diagnosis", "plan": ["First action", "Second action", "Third action"]},
            {"condition": "Alternative diagnosis", "plan": ["Different action"]},
            {"condition": "Third diagnosis", "plan": ["Another action", "Follow-up, if worse"]},
        ]
        result = self.result(predictions=predictions)
        original = deepcopy(result)
        metrics = [{name: (index + 1) / 10 for name in METRIC_NAMES} for index in range(6)]
        with patch.object(Evaluator, "evaluate_metrics", side_effect=metrics) as evaluate:
            self.scorer().score_result(result)
        self.assertEqual([call.args for call in evaluate.call_args_list], [
            (["acute bronchitis"], ["Likely diagnosis"]),
            (["Rest,  Drink fluids "], ["First action, Second action, Third action"]),
            (["acute bronchitis"], ["Alternative diagnosis"]),
            (["Rest,  Drink fluids "], ["Different action"]),
            (["acute bronchitis"], ["Third diagnosis"]),
            (["Rest,  Drink fluids "], ["Another action, Follow-up, if worse"]),
        ])
        self.assertIs(result["predictions"], predictions)
        self.assertEqual({key: result[key] for key in original}, original)
        for index, entry in enumerate(result["evaluation"]):
            self.assertEqual(entry["rank"], index + 1)
            self.assertEqual(entry["condition"]["metrics"], metrics[index * 2])
            self.assertEqual(entry["plan"]["metrics"], metrics[index * 2 + 1])
        self.assertEqual(set(result) - set(original), {"evaluation"})

    def test_unicode_line_separators_inside_label_text_are_not_jsonl_delimiters(self):
        self.labels[0]["condition"] = "acute\u2028bronchitis"
        self.labels[0]["plan"] = ["Rest\u2029then drink fluids"]
        self.write_labels(self.labels)
        self.assertIn("\u2028", self.labels_path.read_text(encoding="utf-8"))
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            self.scorer().score_result(self.result())
        for index in range(0, 6, 2):
            self.assertEqual(evaluate.call_args_list[index].args[0], ["acute\u2028bronchitis"])
            self.assertEqual(evaluate.call_args_list[index + 1].args[0], ["Rest\u2029then drink fluids"])

    def test_generated_plan_items_keep_commas_duplicates_order_and_raw_output(self):
        plan = ["Follow-up, if worse", "  First treatment\n", "Follow-up, if worse"]
        result = self.result(condition="Diagnosis café", plan=plan,
                             assessment_plan_raw='{"predictions":[{"condition":"Diagnosis café","plan":["raw"]}]}')
        original = deepcopy(result)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            self.scorer().score_result(result)
        for index in (1, 3, 5):
            self.assertEqual(evaluate.call_args_list[index].args,
                             (["Rest,  Drink fluids "],
                              ["Follow-up, if worse,   First treatment\n, Follow-up, if worse"]))
        self.assertEqual({key: result[key] for key in original}, original)
        self.assertIs(result["predictions"][0]["plan"], plan)
        self.assertEqual(result["evaluation"][0]["plan"]["status"], "scored")

    def test_old_top_level_fields_are_never_a_prediction_fallback(self):
        result = self.result(assessment="LEGACY ASSESSMENT", condition="IGNORED", plan=["IGNORED"])
        del result["predictions"]
        result.update(condition="LEGACY CONDITION", plan=["LEGACY PLAN"])
        with patch.object(Evaluator, "evaluate_metrics") as evaluate:
            self.scorer().score_result(result)
        evaluate.assert_not_called()
        for entry in result["evaluation"]:
            for field in ("condition", "plan"):
                self.assert_unscored(entry[field], "missing_generation")
        self.assertEqual(result["assessment"], "LEGACY ASSESSMENT")
        self.assertEqual(result["condition"], "LEGACY CONDITION")
        self.assertEqual(result["plan"], ["LEGACY PLAN"])

    def test_exact_plan_sentinel_skips_all_ranked_plans_even_when_generation_missing(self):
        scorer = self.scorer(presenting_evidence_only=False)
        for predictions, call_count in ((None, 0), ([{"condition": "D", "plan": None}] * 3, 3)):
            with self.subTest(predictions=predictions):
                result = self.result("patient-b", predictions=predictions)
                with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
                    scorer.score_result(result)
                self.assertEqual(evaluate.call_count, call_count)
                for entry in result["evaluation"]:
                    self.assert_unscored(entry["plan"], "skipped_no_recorded_plan")

    def test_similar_sentinel_strings_are_regular_plan_references(self):
        for plan in (["None recorded"], ["none recorded "], ["none recorded", "Rest"]):
            with self.subTest(plan=plan):
                self.labels[0]["plan"] = plan
                self.write_labels(self.labels)
                with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
                    result = self.scorer().score_result(self.result())
                self.assertEqual(evaluate.call_count, 6)
                for index, entry in enumerate(result["evaluation"]):
                    self.assertEqual(entry["plan"]["status"], "scored")
                    self.assertEqual(evaluate.call_args_list[index * 2 + 1].args[0], [", ".join(plan)])

    def test_empty_and_whitespace_only_reference_take_priority_over_missing_generation(self):
        for plan in ([], [""], [" \n"]):
            with self.subTest(plan=plan):
                self.labels[0]["plan"] = plan
                self.write_labels(self.labels)
                with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
                    result = self.scorer().score_result(self.result(plan=None))
                self.assertEqual(evaluate.call_count, 3)
                for entry in result["evaluation"]:
                    self.assert_unscored(entry["plan"], "missing_reference")

    def test_generated_plan_limit_does_not_truncate_long_reference_plans(self):
        self.labels[0]["plan"] = ["Gold one", "Gold two", "Gold three", "Gold four"]
        self.write_labels(self.labels)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            result = self.scorer().score_result(self.result(plan=["Generated one", "Generated two", "Generated three"]))
        self.assertEqual(evaluate.call_count, 6)
        for index, entry in enumerate(result["evaluation"]):
            self.assertEqual(evaluate.call_args_list[index * 2 + 1].args,
                             (["Gold one, Gold two, Gold three, Gold four"],
                              ["Generated one, Generated two, Generated three"]))
            self.assertEqual(entry["plan"]["status"], "scored")

    def test_missing_generation_is_reported_per_field_and_rank_without_metric_calls(self):
        scorer = self.scorer()
        for condition, plan in ((None, None), ("", []), (" \n", [" \n"]), (None, [""])):
            with self.subTest(condition=condition, plan=plan), patch.object(Evaluator, "evaluate_metrics") as evaluate:
                result = scorer.score_result(self.result(condition=condition, plan=plan))
                evaluate.assert_not_called()
                for entry in result["evaluation"]:
                    for field in ("condition", "plan"):
                        self.assert_unscored(entry[field], "missing_generation")
        result = {"id": "patient-a", "status": "failed_incomplete"}
        scorer.score_result(result)
        self.assertEqual(result["status"], "failed_incomplete")
        for entry in result["evaluation"]:
            for field in ("condition", "plan"):
                self.assert_unscored(entry[field], "missing_generation")

    def test_wrong_prediction_containers_and_counts_are_errors_not_truncated_or_coerced(self):
        pair = {"condition": "D", "plan": ["P"]}
        for predictions in ("text", {}, True, 1, [], [pair], [pair] * 2, [pair] * 4, (pair,) * 3):
            with self.subTest(predictions=predictions), patch.object(Evaluator, "evaluate_metrics") as evaluate:
                result = self.scorer().score_result(self.result(predictions=predictions))
                evaluate.assert_not_called()
                for entry in result["evaluation"]:
                    for field in ("condition", "plan"):
                        self.assert_unscored(entry[field], "error")
                        self.assertEqual(entry[field]["error"]["type"], "TypeError")

    def test_wrong_pair_structure_only_blocks_its_own_rank(self):
        for pair in (None, [], "D", 1, {"condition": "D", "plan": ["P"], "extra": "bad"}):
            with self.subTest(pair=pair), patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
                result = self.scorer().score_result(self.result(predictions=[
                    {"condition": "D1", "plan": ["P1"]}, pair, {"condition": "D3", "plan": ["P3"]},
                ]))
                self.assertEqual(evaluate.call_count, 4)
                for field in ("condition", "plan"):
                    self.assertEqual(result["evaluation"][0][field]["status"], "scored")
                    self.assert_unscored(result["evaluation"][1][field], "error")
                    self.assertEqual(result["evaluation"][2][field]["status"], "scored")

    def test_missing_pair_fields_are_independent_and_do_not_shift_ranks(self):
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            result = self.scorer().score_result(self.result(predictions=[
                {"condition": "D1"}, {"plan": ["P2"]}, {},
            ]))
        self.assertEqual([call.args[1] for call in evaluate.call_args_list], [["D1"], ["P2"]])
        self.assertEqual(result["evaluation"][0]["condition"]["status"], "scored")
        self.assert_unscored(result["evaluation"][0]["plan"], "missing_generation")
        self.assert_unscored(result["evaluation"][1]["condition"], "missing_generation")
        self.assertEqual(result["evaluation"][1]["plan"]["status"], "scored")
        for field in ("condition", "plan"):
            self.assert_unscored(result["evaluation"][2][field], "missing_generation")

    def test_nonstring_condition_error_does_not_block_other_fields_or_ranks(self):
        for condition in ({"diagnosis": "A"}, ["A"], 1, True):
            with self.subTest(condition=condition):
                predictions = [{"condition": "D", "plan": ["P"]} for _ in range(3)]
                predictions[1]["condition"] = condition
                with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
                    result = self.scorer().score_result(self.result(predictions=predictions))
                self.assertEqual(evaluate.call_count, 5)
                self.assert_unscored(result["evaluation"][1]["condition"], "error")
                self.assertEqual(result["evaluation"][1]["condition"]["error"]["type"], "TypeError")
                self.assertEqual(result["evaluation"][1]["plan"]["status"], "scored")

    def test_invalid_plan_types_and_overlong_lists_are_rank_local_errors(self):
        for plan in ("P", "", " \n", 1, True, {}, ("P",), [None], [1], [True], ["P", {}], ["P"] * 4):
            with self.subTest(plan=plan):
                predictions = [{"condition": "D", "plan": ["P"]} for _ in range(3)]
                predictions[1]["plan"] = plan
                with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
                    result = self.scorer().score_result(self.result(predictions=predictions))
                self.assertEqual(evaluate.call_count, 5)
                self.assertEqual(result["evaluation"][1]["condition"]["status"], "scored")
                self.assert_unscored(result["evaluation"][1]["plan"], "error")
                self.assertEqual(result["evaluation"][1]["plan"]["error"]["type"], "TypeError")

    def test_each_metric_failure_leaves_nulls_and_attempts_other_fields_and_ranks(self):
        for position in range(6):
            with self.subTest(position=position):
                effects = [self.metrics] * 6
                effects[position] = RuntimeError("model failed")
                with patch.object(Evaluator, "evaluate_metrics", side_effect=effects) as evaluate:
                    result = self.scorer().score_result(self.result())
                self.assertEqual(evaluate.call_count, 6)
                for index, entry in enumerate(result["evaluation"]):
                    for offset, field in enumerate(("condition", "plan")):
                        if index * 2 + offset == position:
                            self.assert_unscored(entry[field], "error")
                            self.assertEqual(entry[field]["error"], {"type": "RuntimeError", "message": "model failed"})
                        else:
                            self.assertEqual(entry[field]["metrics"], self.metrics)
                self.assertEqual(result["status"], "completed")

    def test_invalid_metric_shapes_and_values_are_saved_as_errors(self):
        bad = [None, [], {}, {**self.metrics, "extra": 0},
               {key: value for key, value in self.metrics.items() if key != "BLEU"}]
        bad += [{**self.metrics, "BLEU": value} for value in
                (True, False, None, "0.4", float("nan"), float("inf"), -float("inf"), 1j)]
        for metrics in bad:
            with self.subTest(metrics=metrics):
                with patch.object(Evaluator, "evaluate_metrics", side_effect=[metrics] + [self.metrics] * 5) as evaluate:
                    result = self.scorer().score_result(self.result())
                self.assertEqual(evaluate.call_count, 6)
                self.assert_unscored(result["evaluation"][0]["condition"], "error")
                for entry in result["evaluation"][1:]:
                    for field in ("condition", "plan"):
                        self.assertEqual(entry[field]["status"], "scored")
                self.assertEqual(result["evaluation"][0]["plan"]["status"], "scored")
                json.dumps(result, allow_nan=False)

    def test_numpy_numbers_and_out_of_unit_interval_values_are_preserved_as_floats(self):
        metrics = {name: np.float64(-0.25) for name in METRIC_NAMES}
        metrics["BLEU"] = np.int64(2)
        with patch.object(Evaluator, "evaluate_metrics", return_value=metrics):
            result = self.scorer().score_result(self.result())
        for entry in result["evaluation"]:
            self.assertEqual(entry["condition"]["metrics"]["BLEU"], 2.0)
            self.assertEqual(entry["plan"]["metrics"]["BERTScore_F1"], -0.25)
            for field in ("condition", "plan"):
                self.assertTrue(all(type(value) is float for value in entry[field]["metrics"].values()))
        json.dumps(result, allow_nan=False)

    def test_interrupt_marks_only_current_field_and_preserves_prior_and_pending_ranks(self):
        for exception in (KeyboardInterrupt("cancelled"), SystemExit(4)):
            for position in range(6):
                with self.subTest(exception=exception, position=position):
                    result = self.result()
                    effects = [self.metrics] * position + [exception]
                    with patch.object(Evaluator, "evaluate_metrics", side_effect=effects) as evaluate:
                        with self.assertRaises(type(exception)):
                            self.scorer().score_result(result)
                    self.assertEqual(evaluate.call_count, position + 1)
                    for index, entry in enumerate(result["evaluation"]):
                        for offset, field in enumerate(("condition", "plan")):
                            metric_position = index * 2 + offset
                            if metric_position < position:
                                self.assertEqual(entry[field]["status"], "scored")
                            else:
                                self.assert_unscored(entry[field], "interrupted" if metric_position == position else "pending")
                    self.assertEqual(result["status"], "completed")
                    json.dumps(result, allow_nan=False)

    def test_unselected_and_unknown_results_are_rejected(self):
        scorer = self.scorer()
        for result in (None, {}, self.result([]), self.result(1), self.result("patient-b"), self.result("unknown")):
            with self.subTest(result=result), self.assertRaises(ValueError):
                scorer.score_result(result)

    def test_metadata_fingerprints_source_snapshot_and_cohort(self):
        original_bytes = self.labels_path.read_bytes()
        scorer = self.scorer()
        self.labels_path.write_text("replaced after construction", encoding="utf-8")
        metadata = scorer.metadata(self.directory)
        self.assertEqual(metadata["labels_path"], "labels.jsonl")
        self.assertEqual(metadata["labels_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(metadata["evaluator_sha256"], hashlib.sha256(UPSTREAM_EVALUATOR.read_bytes()).hexdigest())
        self.assertEqual(metadata["selected_patient_ids"], ["patient-a"])
        self.assertEqual(metadata["excluded_patient_ids"], ["patient-b"])
        self.assertEqual(metadata["input_patient_count"], 2)
        self.assertEqual(metadata["selected_patient_count"], 1)
        self.assertTrue(metadata["presenting_evidence_only"])
        self.assertEqual(metadata["settings"]["aggregation"], "per_patient_per_rank_per_field_singleton")
        self.assertEqual(metadata["settings"]["lang"], "en")
        self.assertEqual(metadata["settings"]["prediction_count"], 3)
        self.assertEqual(metadata["settings"]["plan_item_limit"], 3)
        self.assertIsNone(metadata["settings"]["rank_weighting"])
        self.assertEqual(metadata["settings"]["generated_plan"],
                         "join original generated items with comma and space")
        self.assertNotIn("model_revision", metadata)
        self.assertNotIn("subjective", json.dumps(metadata))
        json.dumps(metadata, allow_nan=False)

    def test_metadata_package_not_found_is_explicit(self):
        with patch("scripts.med_copilot_scoring.package_metadata.version",
                   side_effect=package_metadata.PackageNotFoundError):
            metadata = self.scorer().metadata(ROOT)
        self.assertTrue(all(value is None for value in metadata["metric_packages"].values()))

    def test_real_lexical_metric_parity_with_only_bert_mocked(self):
        # This suite never downloads resources. Provision English Punkt data
        # separately (and use NLTK_DATA) to exercise actual upstream tokenization.
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            self.skipTest("English Punkt_tab is not provisioned; set NLTK_DATA to its cache")
        reference_a = "acute bronchitis"
        candidate_a = "acute Bronchitis"
        reference_p = "Rest,  Drink fluids "
        candidate_p = "Rest and drink fluids"
        bert_f1 = torch.tensor([0.765], dtype=torch.float32)
        def bert(candidates, references, lang):
            self.assertEqual(lang, "en")
            self.assertEqual(len(candidates), 1)
            self.assertEqual(len(references), 1)
            return bert_f1, bert_f1, bert_f1
        with patch("med_copilot.upstream.evaluate_metrics.bert_score", side_effect=bert) as bert_call:
            result = self.scorer().score_result(self.result(condition=candidate_a, plan=[candidate_p]))
        self.assertEqual(bert_call.call_count, 6)
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for field, reference, candidate in (("condition", reference_a, candidate_a),
                                             ("plan", reference_p, candidate_p)):
            expected_rouge = scorer.score(reference, candidate)
            expected = {
                "BLEU": sacrebleu.corpus_bleu([candidate], [[reference]]).score / 100,
                "METEOR": single_meteor_score(nltk.word_tokenize(reference), nltk.word_tokenize(candidate)),
                "ROUGE-1": expected_rouge["rouge1"].fmeasure,
                "ROUGE-2": expected_rouge["rouge2"].fmeasure,
                "ROUGE-L": expected_rouge["rougeL"].fmeasure,
                "BERTScore_F1": float(bert_f1.mean()),
            }
            for entry in result["evaluation"]:
                self.assertEqual(entry[field]["status"], "scored")
                self.assertEqual(entry[field]["metrics"], expected)

    def test_known_lexical_scores_preserve_short_bleu_and_meteor_penalty(self):
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            self.skipTest("English Punkt_tab is not provisioned; set NLTK_DATA to its cache")
        self.labels[0]["plan"] = ["one two three four"]
        self.write_labels(self.labels)
        bert_values = torch.tensor([0.5])
        with patch("med_copilot.upstream.evaluate_metrics.bert_score",
                   return_value=(bert_values, bert_values, bert_values)) as bert:
            result = self.scorer().score_result(
                self.result(condition="acute bronchitis", plan=["one two three four"]),
            )
        condition_metrics = result["evaluation"][0]["condition"]["metrics"]
        plan_metrics = result["evaluation"][0]["plan"]["metrics"]
        self.assertEqual(condition_metrics["BLEU"], 0.0)
        self.assertEqual(condition_metrics["METEOR"], 0.9375)
        self.assertAlmostEqual(plan_metrics["BLEU"], 1.0)
        self.assertEqual(plan_metrics["METEOR"], 0.9921875)
        for metrics in (condition_metrics, plan_metrics):
            self.assertEqual([metrics[key] for key in ("ROUGE-1", "ROUGE-2", "ROUGE-L")], [1.0] * 3)
            self.assertEqual(metrics["BERTScore_F1"], 0.5)
        self.assertEqual(bert.call_count, 6)
        self.assertEqual([call.kwargs for call in bert.call_args_list], [{"lang": "en"}] * 6)
        for entry in result["evaluation"]:
            self.assertEqual(entry["condition"]["metrics"], condition_metrics)
            self.assertEqual(entry["plan"]["metrics"], plan_metrics)


class ScoringArgumentTests(unittest.TestCase):
    def test_config_and_model_cli_overrides_are_rejected(self):
        self.assertEqual(vars(parse_scoring_args([])), {"presenting_evidence_only": None, "resume": None})
        for option in ("--config", "--experiment-config", "--model-config", "--models"):
            with self.subTest(option=option), redirect_stderr(StringIO()) as stderr:
                with self.assertRaises(SystemExit) as failure:
                    parse_scoring_args([option, "override"])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_default_bare_and_explicit_boolean_forms(self):
        for arguments, expected in (([], None), (["--presenting-evidence-only"], True),
                                    (["--presenting-evidence-only", "true"], True),
                                    (["--presenting-evidence-only=true"], True),
                                    (["--presenting-evidence-only", "FALSE"], False),
                                    (["--presenting-evidence-only=false"], False)):
            with self.subTest(arguments=arguments):
                self.assertIs(parse_scoring_args(arguments).presenting_evidence_only, expected)

    def test_invalid_values_and_unknown_flags_fail(self):
        for arguments in (["--presenting-evidence-only", "1"],
                          ["--presenting-evidence-only", "yes"],
                          ["--presenting-evidence-only", ""], ["--unknown"]):
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    parse_scoring_args(arguments)
                self.assertEqual(failure.exception.code, 2)

    def test_none_uses_process_arguments(self):
        with patch.object(sys, "argv", ["runner", "--presenting-evidence-only", "false"]):
            self.assertFalse(parse_scoring_args().presenting_evidence_only)

    def test_resume_takes_an_existing_run_path_without_changing_cohort_default(self):
        args = parse_scoring_args(["--resume", "saved runs/patient run"])
        self.assertEqual(args.resume, Path("saved runs/patient run"))
        self.assertIsNone(args.presenting_evidence_only)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as failure:
            parse_scoring_args(["--resume"])
        self.assertEqual(failure.exception.code, 2)


class VariantScoringTests(unittest.TestCase):
    setUp = PatientScorerTests.setUp
    write_labels = PatientScorerTests.write_labels
    scorer = PatientScorerTests.scorer
    result = PatientScorerTests.result

    def test_seeded_sample_runs_after_both_filters_and_records_exclusions(self):
        self.patients = [dict(id=name, subjective="S", objective="O")
                         for name in ("no-evidence", "no-plan", "first", "limited", "last")]
        self.write_labels([dict(id=row["id"], condition="gold", plan=["none recorded"] if row["id"] == "no-plan" else ["plan"],
                                has_presenting_evidence=row["id"] != "no-evidence") for row in self.patients])
        options = dict(max_patients=2, exclude_none_recorded_plans=True)
        scorer = self.scorer(**options)
        self.assertEqual([row["id"] for row in scorer.patients], ["first", "last"])
        self.assertEqual(self.scorer(**options).patients, scorer.patients)
        metadata = scorer.metadata(self.directory)
        self.assertEqual(metadata["patient_sampling_seed"], 42)
        self.assertEqual([metadata[key] for key in ("input_patient_count", "after_presenting_evidence_count", "eligible_patient_count", "selected_patient_count")], [5, 4, 3, 2])
        self.assertEqual(metadata["exclusion_reasons"], {"no-evidence": ["missing_presenting_evidence"], "no-plan": ["no_recorded_plan"], "limited": ["max_patients"]})
        self.assertEqual(metadata["excluded_plan_value"], ["none recorded"])
        self.assertTrue(metadata["exclude_none_recorded_plans"])

    def test_filter_does_not_normalize_case_whitespace_or_other_plan_shapes(self):
        plans = [["none recorded"], ["None recorded"], [" none recorded "], [], ["none recorded", "other"]]
        self.patients = [dict(id=str(i), subjective="S", objective="O") for i in range(len(plans))]
        self.write_labels([dict(id=str(i), condition="gold", plan=plan, has_presenting_evidence=True) for i, plan in enumerate(plans)])
        self.assertEqual([p["id"] for p in self.scorer(exclude_none_recorded_plans=True).patients], ["1", "2", "3", "4"])
        self.assertEqual(len(self.scorer(exclude_none_recorded_plans=False).patients), 5)
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.scorer(exclude_none_recorded_plans=value)

    def test_all_filtered_cohort_and_bad_later_labels_fail_before_scoring(self):
        self.write_labels([dict(id="patient-a", condition="gold", plan=["none recorded"], has_presenting_evidence=True), self.labels[1]])
        with self.assertRaisesRegex(ValueError, "No patients selected"):
            self.scorer(exclude_none_recorded_plans=True)
        self.labels[-1]["plan"] = "invalid"
        self.write_labels(self.labels)
        with self.assertRaisesRegex(ValueError, "plan"):
            self.scorer(max_patients=1, exclude_none_recorded_plans=True)

    def test_nested_variants_score_each_model_without_mixing_predictions(self):
        scorer = self.scorer()
        record = {"id": "patient-a", "variants": {}}
        for variant in ("llm_only", "guidelines"):
            record["variants"][variant] = {"assessment_plans": {
                model: {**self.result(condition=f"{variant}:{model}"), "status": "completed"}
                for model in ("one", "two")}}
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            self.assertIs(scorer.score_result(record), record)
        self.assertEqual(evaluate.call_count, 24)
        for variant, data in record["variants"].items():
            for model, output in data["assessment_plans"].items():
                self.assertEqual(output["predictions"][0]["condition"], f"{variant}:{model}")
                self.assertEqual(len(output["evaluation"]), 3)
                self.assertEqual(output["evaluation"][0]["condition"]["metrics"], self.metrics)
        self.assertNotIn("evaluation", record)

    def test_schema_four_components_and_arbitrary_variant_ids_preserve_scoring_contract(self):
        components = {
            "shared_patient": {"status": "completed", "outputs": {"example": "Reference SOAP"},
                               "trace": [{"query": "Prepared S/O", "ranked_ids": ["reference-1"]}]},
            "new_guidelines": {"status": "failed_incomplete", "outputs": {},
                               "error": {"type": "RuntimeError", "message": "Unavailable"}},
        }
        good = {**self.result(condition="custom:one"), "evaluation": None}
        failed = {"status": "failed_incomplete", "predictions": None, "evaluation": None}
        waiting = {"status": "not_processed", "predictions": None, "evaluation": None}
        record = {
            "id": "patient-a", "status": "failed_incomplete", "components": deepcopy(components),
            "variants": {
                "custom_variant": {"status": "failed_incomplete", "components": {"patient_retrieval": "shared_patient"},
                                   "assessment_plans": {"one": good, "failed": failed}},
                "another_new_variant": {"status": "completed", "components": {},
                                        "assessment_plans": {
                                            model: {**self.result(condition=f"another:{model}"), "evaluation": None}
                                            for model in ("one", "two")}},
                "blocked_variant": {"status": "failed_incomplete", "components": {"guidelines": "new_guidelines"},
                                    "assessment_plans": {"waiting": waiting}},
            },
        }
        before = deepcopy(record)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            self.assertIs(self.scorer().score_result(record), record)
        self.assertEqual(evaluate.call_count, 18)
        self.assertEqual(record["components"], components)
        self.assertEqual(record["status"], before["status"])
        for variant_id, variant in record["variants"].items():
            self.assertEqual(variant["components"], before["variants"][variant_id]["components"])
            for model_id, output in variant["assessment_plans"].items():
                previous = before["variants"][variant_id]["assessment_plans"][model_id]
                self.assertEqual(output["predictions"], previous["predictions"])
                if output["status"] == "completed":
                    self.assertEqual(len(output["evaluation"]), 3)
                    self.assertEqual(output["evaluation"][0]["condition"]["metrics"], self.metrics)
                else:
                    self.assertIsNone(output["evaluation"])
        self.assertNotIn("evaluation", record)

    def test_new_variant_interruption_preserves_component_captures_and_prior_scores(self):
        record = {
            "id": "patient-a", "components": {"custom": {"status": "completed", "outputs": {"text": "Evidence"}}},
            "variants": {name: {"assessment_plans": {
                "one": {**self.result(condition=name), "evaluation": None}}}
                for name in ("new_first", "new_interrupted", "new_later")},
        }
        components = deepcopy(record["components"])
        with patch.object(Evaluator, "evaluate_metrics", side_effect=[self.metrics] * 6 + [KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                self.scorer().score_result(record)
        self.assertEqual(record["components"], components)
        self.assertEqual(record["variants"]["new_first"]["assessment_plans"]["one"]["evaluation"][2]["plan"]["status"], "scored")
        self.assertEqual(record["variants"]["new_interrupted"]["assessment_plans"]["one"]["evaluation"][0]["condition"]["status"], "interrupted")
        self.assertTrue(all(entry[field]["status"] == "pending" for entry in
                            record["variants"]["new_later"]["assessment_plans"]["one"]["evaluation"]
                            for field in ("condition", "plan")))


JUDGE_MODELS = ("mini", "llama", "gemma", "healthcare")


def judge_response(content):
    return SimpleNamespace(id="judge", model="judge-model", usage=None, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content, refusal=None))])


ACTION_PROMPT = "Use the patient's Subjective and Objective notes to grade whether the generated plan's actions cover"
EXTRAS_PROMPT = "Count the distinct actions in the generated plan"


class FakeJudgeModels:
    """Full credit and no extras unless a reply override matches; records judge, prompt pairs."""

    model_ids = JUDGE_MODELS

    def __init__(self, reply=None):
        self.reply = reply
        self.requests = []

    def judge(self, model_id, messages):
        prompt = messages[-1]["content"]
        self.requests.append((model_id, prompt))
        content = self.reply(model_id, prompt) if self.reply else None
        if isinstance(content, BaseException):
            raise content
        if content is None:
            content = ('{"credit": 1}' if prompt.startswith(ACTION_PROMPT)
                       else '{"unsupported_extra_count": 0}')
        return judge_response(content)

    def redact(self, text):
        return text


class CorrectnessScoringTests(unittest.TestCase):
    setUp = PatientScorerTests.setUp
    write_labels = PatientScorerTests.write_labels
    scorer = PatientScorerTests.scorer
    result = PatientScorerTests.result

    def judged(self, record, models=None):
        models = models or FakeJudgeModels()
        scorer = self.scorer()
        scorer.attach_judges(CorrectnessJudges(models.judge, models.model_ids, self.directory / JUDGE_FILE, plan_judge="llama"))
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_result(record)
        return scorer, models

    def multi_model(self, **kwargs):
        return {"id": "patient-a", "variants": {"llm_only": {"assessment_plans": {
            model: self.result(**kwargs) for model in JUDGE_MODELS}}}}

    def test_every_rank_keeps_text_metrics_and_adds_separate_diagnosis_and_plan_scores(self):
        record = self.multi_model(condition="Bronchitis", plan=["Rest", "Fluids"])
        _, models = self.judged(record)
        for output in record["variants"]["llm_only"]["assessment_plans"].values():
            for rank, entry in enumerate(output["evaluation"], 1):
                self.assertEqual(set(entry), {"rank", "condition", "plan", "diagnosis_correctness", "plan_correctness"})
                self.assertEqual(entry["condition"]["metrics"], self.metrics)
                self.assertEqual(entry["plan"]["metrics"], self.metrics)
                for kind, score in (("diagnosis_correctness", 0.75), ("plan_correctness", 1.0)):
                    self.assertEqual(entry[kind]["status"], "scored")
                    self.assertEqual(entry[kind]["score"], score)
                    self.assertAlmostEqual(entry[kind]["weighted_score"], score / rank)
                self.assertEqual(entry["diagnosis_correctness"]["method"], "biolord_2023_cosine")
                self.assertEqual(entry["diagnosis_correctness"]["reference"], "acute bronchitis")
                self.assertEqual(entry["plan_correctness"]["reference_actions"], ["Rest", " Drink fluids "])
                self.assertEqual(entry["plan_correctness"]["generated_actions"], ["Rest", "Fluids"])
        self.assertNotIn("combined", json.dumps(record))

    def test_only_the_single_plan_judge_is_asked_and_diagnoses_use_no_judge(self):
        record = {"id": "patient-a", "variants": {"llm_only": {"assessment_plans": {
            model: self.result(condition=f"DX by {model}") for model in JUDGE_MODELS}}}}
        _, models = self.judged(record)
        self.assertEqual({model for model, _ in models.requests}, {"llama"})
        self.assertFalse(any("DX by" in prompt for _, prompt in models.requests))
        self.assertEqual({prompt.split("\n", 1)[0] for _, prompt in models.requests},
                         {"Use the patient's Subjective and Objective notes to grade whether the generated plan's actions cover one reference action for the labeled condition.",
                          "Count the distinct actions in the generated plan that are \"unsupported extras\", meaning that they neither cover an action in the reference plan nor represent appropriate additional care for the patient. Absence from the reference plan alone does not make an extra unsupported."})
        gemma = record["variants"]["llm_only"]["assessment_plans"]["gemma"]["evaluation"][0]
        self.assertEqual(gemma["plan_correctness"]["judges"][0]["judge_model"], "llama")

    def test_unavailable_plan_labels_are_never_judged_but_diagnoses_are(self):
        for plan, reason in ((["none recorded"], "no_recorded_plan"),
                             (["COPD clinical management plan (record artifact)"], "record_artifact_plan"),
                             ([], "missing_reference")):
            with self.subTest(plan=plan):
                (self.directory / JUDGE_FILE).unlink(missing_ok=True)
                self.labels[0]["plan"] = plan
                self.write_labels(self.labels)
                record = self.multi_model()
                _, models = self.judged(record)
                entry = record["variants"]["llm_only"]["assessment_plans"]["mini"]["evaluation"][0]
                self.assertEqual((entry["plan_correctness"]["status"], entry["plan_correctness"]["reason"]),
                                 ("unavailable", reason))
                self.assertEqual(entry["diagnosis_correctness"]["status"], "scored")
                self.assertEqual(models.requests, [])

    def test_invalid_or_missing_generations_are_unavailable_per_rank(self):
        predictions = [{"condition": None, "plan": None}, {"condition": 3, "plan": "Rest"}, ["bad"]]
        record = self.multi_model(predictions=predictions)
        self.judged(record)
        entries = record["variants"]["llm_only"]["assessment_plans"]["mini"]["evaluation"]
        self.assertEqual([(e["diagnosis_correctness"]["reason"], e["plan_correctness"]["reason"]) for e in entries],
                         [("missing_generation", "missing_generation"), ("invalid_generation", "invalid_generation"),
                          ("invalid_generation", "invalid_generation")])
        record = self.multi_model(predictions=None)
        self.judged(record)
        entry = record["variants"]["llm_only"]["assessment_plans"]["mini"]["evaluation"][2]
        self.assertEqual(entry["plan_correctness"]["reason"], "missing_generation")

    def test_without_judges_evaluation_keeps_its_previous_shape(self):
        record = self.multi_model()
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            self.scorer().score_result(record)
        entry = record["variants"]["llm_only"]["assessment_plans"]["mini"]["evaluation"][0]
        self.assertEqual(set(entry), {"rank", "condition", "plan"})
        self.assertNotIn("correctness", self.scorer().metadata(self.directory))

    def test_metadata_records_the_judge_method_when_attached(self):
        scorer, _ = self.judged(self.multi_model())
        correctness = scorer.metadata(ROOT)["correctness"]
        self.assertEqual(correctness["selected_models"], list(JUDGE_MODELS))
        self.assertEqual(correctness["status"], "preliminary single-model midterm metric")
        self.assertEqual(correctness["plan"]["judge_model"], "llama")
        self.assertEqual(correctness["rank_weighting"]["formula"], "1 / rank")

    def test_judge_interruption_marks_the_panel_and_leaves_later_ranks_pending(self):
        def reply(model, prompt):
            return KeyboardInterrupt()
        record = self.multi_model()
        with self.assertRaises(KeyboardInterrupt):
            self.judged(record, FakeJudgeModels(reply))
        entries = record["variants"]["llm_only"]["assessment_plans"]["mini"]["evaluation"]
        self.assertEqual(entries[0]["diagnosis_correctness"]["status"], "scored")
        self.assertEqual(entries[0]["plan_correctness"]["status"], "interrupted")
        self.assertEqual(entries[1]["diagnosis_correctness"]["status"], "pending")
        self.assertEqual(entries[2]["condition"]["status"], "scored")


class IncrementalScoringTests(unittest.TestCase):
    """A runner restart completes only unfinished scoring, including partial ranks."""

    setUp = PatientScorerTests.setUp
    write_labels = PatientScorerTests.write_labels
    scorer = PatientScorerTests.scorer
    result = PatientScorerTests.result

    def test_completed_nested_model_scores_are_unchanged_and_make_no_calls(self):
        output = self.result()
        record = {"id": "patient-a", "variants": {"baseline": {"assessment_plans": {"mini": output}}}}
        scorer = self.scorer()
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_result(record)
        before = deepcopy(record)
        with patch.object(Evaluator, "evaluate_metrics", side_effect=AssertionError("completed metric repeated")) as evaluate:
            self.assertIs(scorer.score_result(record), record)
        evaluate.assert_not_called()
        self.assertEqual(record, before)

    def test_interrupted_scoring_reuses_earlier_fields_and_finishes_remaining_fields(self):
        output = self.result()
        scorer = self.scorer()
        with patch.object(Evaluator, "evaluate_metrics", side_effect=[self.metrics, KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                scorer.score_result(output)
        first = deepcopy(output["evaluation"][0]["condition"])
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            scorer.score_result(output)
        self.assertEqual(evaluate.call_count, 5)
        self.assertEqual(evaluate.call_args_list[0].args, (["Rest,  Drink fluids "], ["P"]))
        self.assertEqual(output["evaluation"][0]["condition"], first)
        self.assertTrue(all(e[field]["status"] == "scored" for e in output["evaluation"]
                            for field in ("condition", "plan")))

    def test_failed_field_is_retried_without_repeating_other_five_fields(self):
        output = self.result()
        scorer = self.scorer()
        with patch.object(Evaluator, "evaluate_metrics", side_effect=[self.metrics, RuntimeError("temporary failure")] + [self.metrics] * 4):
            scorer.score_result(output)
        before = deepcopy(output["evaluation"])
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            scorer.score_result(output)
        self.assertEqual(evaluate.call_count, 1)
        before[0]["plan"] = output["evaluation"][0]["plan"]
        self.assertEqual(output["evaluation"], before)

    def test_scored_but_invalid_metrics_are_recomputed_instead_of_reused(self):
        output = self.result()
        scorer = self.scorer()
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_result(output)
        output["evaluation"][1]["condition"]["metrics"].pop("BLEU")
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            scorer.score_result(output)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(output["evaluation"][1]["condition"]["metrics"], self.metrics)

    def test_correctness_retry_preserves_successful_diagnoses_and_other_plans(self):
        output = self.result()
        scorer = self.scorer()
        models = FakeJudgeModels()
        judges = CorrectnessJudges(models.judge, models.model_ids, self.directory / JUDGE_FILE, plan_judge="llama")
        scorer.attach_judges(judges)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_result(output)
        before = deepcopy(output["evaluation"])
        output["evaluation"][1]["plan_correctness"].update(status="error", reason="judge_failed", score=None)
        with patch.object(Evaluator, "evaluate_metrics", side_effect=AssertionError("text scoring repeated")) as evaluate, \
                patch.object(judges.similarity, "cosine", side_effect=AssertionError("diagnosis scoring repeated")) as similarity:
            scorer.score_result(output)
        evaluate.assert_not_called()
        similarity.assert_not_called()
        self.assertEqual(output["evaluation"], before)
        self.assertEqual(len(models.requests), 3)  # All retry requests reuse valid judge responses.

    def test_unavailable_correctness_is_retried_when_judge_becomes_available(self):
        output = self.result()
        scorer = self.scorer()
        unavailable = CorrectnessJudges(None, (), self.directory / JUDGE_FILE, plan_judge="llama")
        scorer.attach_judges(unavailable)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_result(output)
        self.assertEqual(output["evaluation"][0]["plan_correctness"]["reason"], "judge_unavailable")
        models = FakeJudgeModels()
        scorer.attach_judges(CorrectnessJudges(models.judge, models.model_ids, self.directory / JUDGE_FILE, plan_judge="llama"))
        with patch.object(Evaluator, "evaluate_metrics") as evaluate:
            scorer.score_result(output)
        evaluate.assert_not_called()
        self.assertTrue(all(e["plan_correctness"]["status"] == "scored" for e in output["evaluation"]))

    def test_checkpoint_after_each_text_field_retains_progress_before_next_metric(self):
        output = self.result()
        snapshots = []
        with patch("scripts.med_copilot_scoring.checkpoint", create=True,
                   side_effect=lambda: snapshots.append(deepcopy(output))) as checkpoint, \
                patch.object(Evaluator, "evaluate_metrics", side_effect=[self.metrics, KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                self.scorer().score_result(output)
        self.assertGreaterEqual(checkpoint.call_count, 1)
        self.assertEqual(snapshots[0]["evaluation"][0]["condition"]["status"], "scored")
        self.assertEqual(snapshots[0]["evaluation"][0]["plan"]["status"], "pending")

    def test_explicit_rescore_recalculates_completed_text_and_diagnosis_scores(self):
        output = self.result()
        scorer = self.scorer()
        models = FakeJudgeModels()
        judges = CorrectnessJudges(models.judge, models.model_ids, self.directory / JUDGE_FILE, plan_judge="llama")
        scorer.attach_judges(judges)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_result(output)
        replacement = {name: 0.125 for name in METRIC_NAMES}
        with patch.object(Evaluator, "evaluate_metrics", return_value=replacement) as evaluate, \
                patch.object(judges.similarity, "cosine", return_value=0.25) as similarity:
            scorer.score_result(output, rescore=True)
        self.assertEqual(evaluate.call_count, 6)
        self.assertEqual(similarity.call_count, 3)
        self.assertEqual(output["evaluation"][0]["condition"]["metrics"], replacement)
        self.assertEqual(output["evaluation"][0]["diagnosis_correctness"]["score"], 0.25)
        self.assertEqual(len(models.requests), 3)

    def test_checkpoint_write_failure_propagates_instead_of_becoming_a_metric_error(self):
        output = self.result()
        with patch("scripts.med_copilot_scoring.checkpoint", side_effect=RuntimeError("checkpoint failed")), \
                patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
                self.scorer().score_result(output)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(output["evaluation"][0]["condition"]["status"], "scored")

    def test_invalid_scored_correctness_is_recomputed(self):
        output = self.result()
        scorer = self.scorer()
        models = FakeJudgeModels()
        judges = CorrectnessJudges(models.judge, models.model_ids, self.directory / JUDGE_FILE, plan_judge="llama")
        scorer.attach_judges(judges)
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            scorer.score_result(output)
        output["evaluation"][0]["diagnosis_correctness"].update(score=2.0, weighted_score=2.0)
        with patch.object(Evaluator, "evaluate_metrics") as evaluate, \
                patch.object(judges.similarity, "cosine", return_value=0.25) as similarity:
            scorer.score_result(output)
        evaluate.assert_not_called()
        similarity.assert_called_once()
        self.assertEqual(output["evaluation"][0]["diagnosis_correctness"]["score"], 0.25)


class ResumeEvaluationTests(unittest.TestCase):
    setUp = PatientScorerTests.setUp
    write_labels = PatientScorerTests.write_labels
    scorer = PatientScorerTests.scorer
    result = PatientScorerTests.result

    def write_run(self):
        self.run_dir = self.directory / "run"
        self.run_dir.mkdir()
        scorer = self.scorer()
        metadata = {"run_name": "run", "status": "completed", "errors": [],
                    "experiment": {"plan_judge": "llama"},
                    "evaluation": {"selected_patient_ids": [patient["id"] for patient in scorer.patients]}}
        (self.run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
        record = {"id": "patient-a", "status": "completed", "variants": {"llm_only": {"assessment_plans": {
            model: self.result(condition=f"DX {model}") for model in JUDGE_MODELS}}}}
        (self.run_dir / "results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    def resume(self, models):
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics):
            return resume_evaluation(self.run_dir, self.scorer(), models, root=self.directory)

    def saved(self):
        record = json.loads((self.run_dir / "results.jsonl").read_text(encoding="utf-8"))
        return record, json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))

    def test_resume_scores_saved_predictions_and_records_the_method(self):
        self.write_run()
        judges = self.resume(FakeJudgeModels())
        record, run = self.saved()
        entry = record["variants"]["llm_only"]["assessment_plans"]["gemma"]["evaluation"][1]
        self.assertEqual((entry["diagnosis_correctness"]["score"], entry["diagnosis_correctness"]["weighted_score"]),
                         (0.75, 0.375))
        self.assertEqual(run["evaluation"]["correctness"]["selected_models"], list(JUDGE_MODELS))
        self.assertEqual(run["evaluation"]["correctness"]["status"], "preliminary single-model midterm metric")
        self.assertEqual(run["status"], "completed")
        # Every generator and rank shares one plan text: 2 action calls + 1 extras call are
        # distinct; the other 4 x 3 x 3 - 3 requests reuse them.
        self.assertEqual(judges.calls, 3)
        self.assertEqual(judges.calls + judges.cache_hits, 4 * 3 * 3)

    def test_interrupted_evaluation_resumes_without_repeating_completed_calls(self):
        self.write_run()
        calls = []
        def reply(model, prompt):
            calls.append(model)
            return KeyboardInterrupt() if len(calls) == 2 else None
        with self.assertRaises(KeyboardInterrupt):
            self.resume(FakeJudgeModels(reply))
        before, _ = self.saved()
        self.assertNotIn("evaluation", before["variants"]["llm_only"]["assessment_plans"]["mini"])
        resumed = FakeJudgeModels()
        judges = self.resume(resumed)
        # 3 distinct requests: 1 completed before the interruption, 2 remain.
        self.assertEqual(judges.calls, 3 - 1)
        self.assertEqual(judges.calls + judges.cache_hits, 4 * 3 * 3)
        self.assertEqual(len(resumed.requests), 2)
        third = self.resume(FakeJudgeModels())
        self.assertEqual(third.calls, 0)

    def test_scoring_failures_are_saved_under_the_scoring_stage(self):
        self.write_run()
        def reply(model, prompt):
            return RuntimeError("judge offline") if model == "llama" else None
        self.resume(FakeJudgeModels(reply))
        record, run = self.saved()
        self.assertTrue(run["errors"])
        self.assertEqual({(error["patient_id"], error["stage"], error["exception_type"]) for error in run["errors"]},
                         {("patient-a", "scoring", "JudgeCallError")})
        self.assertTrue(all("RuntimeError: judge offline" in error["message"] for error in run["errors"]))
        # Resuming retries the failed calls but adds no duplicate error entries.
        self.resume(FakeJudgeModels(reply))
        self.assertEqual(self.saved()[1]["errors"], run["errors"])
        entry = record["variants"]["llm_only"]["assessment_plans"]["mini"]["evaluation"][0]
        self.assertEqual(entry["plan_correctness"]["reason"], "judge_failed")
        # BioLORD diagnosis scores need no judge, so they are unaffected.
        self.assertEqual(entry["diagnosis_correctness"]["status"], "scored")

    def test_resume_refuses_a_different_cohort(self):
        self.write_run()
        metadata = json.loads((self.run_dir / "run.json").read_text())
        metadata["evaluation"]["selected_patient_ids"] = ["someone-else"]
        (self.run_dir / "run.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "cohort"):
            self.resume(FakeJudgeModels())

    def test_legacy_full_rescore_function_recalculates_previously_saved_text_scores(self):
        self.write_run()
        self.resume(FakeJudgeModels())
        replacement = {name: 0.125 for name in METRIC_NAMES}
        with patch.object(Evaluator, "evaluate_metrics", return_value=replacement) as evaluate:
            resume_evaluation(self.run_dir, self.scorer(), FakeJudgeModels(), root=self.directory)
        self.assertEqual(evaluate.call_count, len(JUDGE_MODELS) * 6)
        record, _ = self.saved()
        self.assertTrue(all(e[field]["metrics"] == replacement
                            for output in record["variants"]["llm_only"]["assessment_plans"].values()
                            for e in output["evaluation"] for field in ("condition", "plan")))


RESUME_SCRIPT = ROOT / "judge_treatment_plans.py"


class ResumeCommandTests(unittest.TestCase):
    """Command-level checks of judge_treatment_plans.py with fake judge models."""
    setUp = PatientScorerTests.setUp
    write_labels = PatientScorerTests.write_labels
    result = PatientScorerTests.result

    def build_run(self):
        profile = self.directory / "experiments/configs/plan_judging.yaml"
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_bytes((ROOT / "experiments/configs/plan_judging.yaml").read_bytes())
        self.input_path = self.directory / "input.jsonl"
        self.input_path.write_text("".join(json.dumps(p) + "\n" for p in self.patients), encoding="utf-8")
        self.run_dir = self.directory / "experiments/output/run"
        self.run_dir.mkdir(parents=True)
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.metadata = {
            "run_name": "run", "status": "completed", "errors": [], "schema_version": 4,
            "input": {"file": "input.jsonl", "sha256": sha(self.input_path)},
            "experiment": {"selected_variants": ["llm_only"], "temperature": 1, "max_retries": 3, "plan_judge": "llama"},
            "assessment_plan_models": {"config_file": "models.yaml", "selected_models": list(JUDGE_MODELS),
                                       "models": {}},
            "evaluation": {"labels_path": "labels.jsonl", "labels_sha256": sha(self.labels_path),
                           "presenting_evidence_only": False, "max_patients": None,
                           "exclude_none_recorded_plans": False,
                           "selected_patient_ids": ["patient-b", "patient-a"]},
        }
        (self.run_dir / "run.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        rows = [{"id": patient_id, "status": "completed", "variants": {"llm_only": {"assessment_plans": {
            model: self.result(patient_id, condition=f"DX {model}") for model in JUDGE_MODELS}}}}
            for patient_id in ("patient-b", "patient-a")]
        (self.run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def command(self, *argv, reply=None, models=None, models_factory=None):
        """Run the command's main(); return (exit status, stdout, stderr, fake models)."""
        models = models or FakeJudgeModels(reply)
        namespace = runpy.run_path(str(RESUME_SCRIPT), run_name="resume_command")
        stdout, stderr = StringIO(), StringIO()
        with patch.object(Evaluator, "evaluate_metrics", return_value=self.metrics), \
                patch("scripts.med_copilot_scoring.TextMetricEvaluator", side_effect=lambda: SingletonMetricDouble(
                    lambda *args, **kw: Evaluator.evaluate_metrics(*args, **kw))), \
                patch.dict(namespace["main"].__globals__, {"save_verified_runtime": lambda *a: {"file": "provenance/fake.json"}}), \
                redirect_stderr(stderr), patch("sys.stdout", stdout):
            status = namespace["main"](list(argv) or [str(self.run_dir)], root=self.directory,
                                       models_factory=models_factory or (lambda *args, **kwargs: models))
        return status, stdout.getvalue(), stderr.getvalue(), models

    def saved(self):
        rows = [json.loads(line) for line in (self.run_dir / "results.jsonl").read_text().splitlines()]
        return rows, json.loads((self.run_dir / "run.json").read_text())

    def test_success_judges_only_plans_writes_the_report_and_exits_zero(self):
        self.build_run()
        status, stdout, stderr, models = self.command()
        self.assertEqual((status, stderr), (0, ""))
        self.assertIn("Judge calls made: ", stdout)
        self.assertTrue((self.run_dir / "results_summary.html").is_file())
        rows, run = self.saved()
        entry = rows[1]["variants"]["llm_only"]["assessment_plans"]["gemma"]["evaluation"][0]
        self.assertEqual(entry["plan_correctness"]["status"], "scored")
        self.assertNotIn("diagnosis_correctness", entry)
        self.assertNotIn("condition", entry)
        self.assertNotIn("plan", entry)
        # patient-b's label is ["none recorded"], so its Plans are never judged.
        none_recorded = rows[0]["variants"]["llm_only"]["assessment_plans"]["mini"]["evaluation"][0]
        self.assertEqual(none_recorded["plan_correctness"]["reason"], "no_recorded_plan")
        self.assertNotIn("correctness", run["evaluation"])
        self.assertEqual(run["plan_evaluation"]["protocol"]["version"], 1)
        self.assertEqual(self.command()[0], 0)
        self.assertEqual(self.command()[3].requests, [])  # Fully resumed: nothing is requested again.

    def test_evaluation_resume_uses_the_judge_alias_recorded_in_experiment_metadata(self):
        self.build_run()
        self.metadata["experiment"]["plan_judge"] = "custom_judge"
        (self.run_dir / "run.json").write_text(json.dumps(self.metadata))
        generators, judge = FakeJudgeModels(), FakeJudgeModels()
        judge.model_ids = ("custom_judge",)
        requested = []
        def factory(config, selected_models, **kwargs):
            requested.append(selected_models)
            return judge if selected_models == ["custom_judge"] else generators
        status, _, stderr, _ = self.command(models=generators, models_factory=factory)
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(requested, [["custom_judge"]])
        self.assertEqual({model for model, _ in judge.requests}, {"custom_judge"})
        self.assertEqual(self.saved()[1]["plan_evaluation"]["protocol"]["plan"]["judge_model"], "custom_judge")

    def test_invalid_run_directories_exit_one_without_scoring(self):
        self.build_run()
        status, _, stderr, models = self.command(str(self.directory / "missing-run"))
        self.assertEqual(status, 1)
        self.assertIn("Cannot resume evaluation", stderr)
        self.assertEqual(models.requests, [])
        del self.metadata["evaluation"]["labels_sha256"]
        (self.run_dir / "run.json").write_text(json.dumps(self.metadata))
        status, _, stderr, _ = self.command()
        self.assertEqual(status, 1)
        self.assertIn("missing run.json field 'labels_sha256'", stderr)

    def test_changed_labels_or_input_are_refused(self):
        self.build_run()
        self.labels[0]["condition"] = "changed"
        self.write_labels(self.labels)
        status, _, stderr, models = self.command()
        self.assertEqual(status, 1)
        self.assertIn("Labels changed since the run", stderr)
        self.assertEqual(models.requests, [])

    def test_interruption_exits_130_and_the_next_run_resumes_without_repeats(self):
        self.build_run()
        calls = []
        def reply(model, prompt):
            calls.append(model)
            return KeyboardInterrupt() if len(calls) == 2 else None
        status, _, stderr, _ = self.command(reply=reply)
        self.assertEqual(status, 130)
        self.assertIn("rerun this command to resume", stderr)
        cached = len((self.run_dir / "judge_responses.jsonl").read_text().splitlines())
        # Other already-running requests may finish while interruption drains
        # the bounded executor; every such valid response must be retained.
        self.assertGreaterEqual(cached, 1)
        status, _, _, models = self.command()
        self.assertEqual(status, 0)
        requested = {(model, prompt) for model, prompt in models.requests}
        self.assertEqual(len(requested), len(models.requests))
        self.assertEqual(cached + len(models.requests),
                         len((self.run_dir / "judge_responses.jsonl").read_text().splitlines()))

    def test_failed_judges_exit_one_record_errors_once_and_succeed_after_retry(self):
        self.build_run()
        def reply(model, prompt):
            return "not json" if prompt.startswith(ACTION_PROMPT) else None
        status, _, stderr, _ = self.command(reply=reply)
        self.assertEqual(status, 1)
        self.assertIn("correctness entries failed", stderr)
        _, run = self.saved()
        errors = run["errors"]
        self.assertTrue(errors)
        self.assertEqual({(e["stage"], e["exception_type"]) for e in errors}, {("scoring", "JudgeResponseError")})
        # patient-b's Plan label is ["none recorded"], so only patient-a's Plans are judged.
        self.assertEqual({e["patient_id"] for e in errors}, {"patient-a"})
        self.assertEqual(self.command(reply=reply)[0], 1)
        self.assertEqual(self.saved()[1]["errors"], errors)  # retried, but not recorded twice
        self.assertEqual(self.command()[0], 0)
        self.assertEqual(self.saved()[1]["errors"], errors)

    def test_script_entry_point_exits_with_the_status(self):
        with patch.object(sys, "argv", [str(RESUME_SCRIPT), str(self.directory / "missing-run")]), \
                redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            runpy.run_path(str(RESUME_SCRIPT), run_name="__main__")
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
