"""Offline tests for label validation, field routing, and upstream metric reuse."""

from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
from importlib import metadata as package_metadata
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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


class PatientScorerTests(unittest.TestCase):
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
    def test_default_bare_and_explicit_boolean_forms(self):
        for arguments, expected in (([], True), (["--presenting-evidence-only"], True),
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


if __name__ == "__main__":
    unittest.main()
