"""Offline tests for the standalone, read-only MED-COPILOT results report."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Direct execution puts tests/ on sys.path; project imports need the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collate_med_copilot_results import (
    METRICS, format_top_rank_summary, generate_report, load_report_data, main, render_report,
)


SCRIPT = ROOT / "scripts/collate_med_copilot_results.py"


def score(value=0.5, status="scored"):
    return {"status": status,
            "metrics": dict.fromkeys(METRICS, value if status == "scored" else None),
            "error": None}


def output(value=0.5):
    return {
        "status": "completed", "error": None,
        "predictions": [{"condition": f"Condition {rank}",
                         "plan": [f"Action {rank}.1", f"Action {rank}.2", f"Action {rank}.3"]}
                        for rank in range(1, 4)],
        "evaluation": [{"rank": rank, "condition": score(value), "plan": score(value / 2)}
                       for rank in range(1, 4)],
        "model_responses": [{"reported_model": "reported/model-version", "response_id": "PRIVATE_RESPONSE_ID",
                             "token_usage": {"total_tokens": 123}}],
    }


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "experiments/output/example"
        self.run.mkdir(parents=True)
        self.labels_path = self.root / "labels.jsonl"
        self.labels = [{"id": "patient-1", "condition": "reference_condition", "plan": ["Reference action one", "Reference action two"]}]
        self.metadata = {
            "run_name": "PRIVATE_RUN_NAME", "git_commit": "PRIVATE_COMMIT", "status": "completed",
            "started_at": "PRIVATE_TIMESTAMP", "api_key": "PRIVATE_API_KEY",
            "evaluation": {"labels_path": "labels.jsonl", "selected_patient_ids": ["patient-1"]},
        }
        self.rows = [{"id": "patient-1", **output()}]
        self.rows[0]["model_responses"] = {
            "keyword_extraction": [{"reported_model": "wrong-stage-model"}],
            "assessment_plan": [{"reported_model": "legacy-final-version", "response_id": "PRIVATE_RESPONSE_ID"}],
        }
        self.write()

    def write(self, *, fingerprint=True):
        self.labels_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in self.labels), encoding="utf-8")
        if fingerprint:
            self.metadata["evaluation"]["labels_sha256"] = hashlib.sha256(self.labels_path.read_bytes()).hexdigest()
        else:
            self.metadata["evaluation"].pop("labels_sha256", None)
        (self.run / "run.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        (self.run / "results.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in self.rows), encoding="utf-8")

    def load(self):
        return load_report_data(self.run, root=self.root)

    def native_v3(self):
        self.metadata["schema_version"] = 3
        self.metadata["experiment"] = {"selected_variants": ["llm_only", "guidelines"]}
        self.metadata["assessment_plan_models"] = {"selected_models": ["mini", "llama"], "models": {
            "mini": {"config": {"model": "mini"}}, "llama": {"config": {"model": "llama"}},
        }}
        self.rows = [{"id": "patient-1", "status": "completed", "variants": {
            variant: {"status": "completed", "error": None,
                      "assessment_plan_prompts": {"user": "PRIVATE_PROMPT"},
                      "assessment_plans": {model: output(value) for model in ("mini", "llama")}}
            for variant, value in (("llm_only", 0.4), ("guidelines", 0.8))
        }}]

    def test_v3_normalizes_one_patient_variant_model_matrix_without_duplicate_shared_content(self):
        self.native_v3()
        self.write()
        before = [(self.run / name).read_bytes() for name in ("run.json", "results.jsonl")]
        data = self.load()
        self.assertEqual(len(data["patients"]), 1)
        outputs = data["patients"][0]["outputs"]
        self.assertEqual([(v["variant"], v["model_id"]) for v in outputs],
                         [(variant, model) for variant in ("llm_only", "guidelines") for model in ("mini", "llama")])
        self.assertEqual(len(data["summary"]), 24)  # 2 variants * 2 models * 3 ranks * 2 fields
        self.assertEqual(len(data["comparisons"]), 12)
        html = generate_report(self.run, root=self.root).read_text()
        self.assertIn("Best evidence variant by model", html)
        self.assertNotIn("PRIVATE_PROMPT", html)
        self.assertEqual(before, [(self.run / name).read_bytes() for name in ("run.json", "results.jsonl")])

    def test_v3_missing_patients_variants_and_models_remain_explicitly_unprocessed(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["action"]})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        del self.rows[0]["variants"]["guidelines"]
        del self.rows[0]["variants"]["llm_only"]["assessment_plans"]["llama"]
        self.write()
        data = self.load()
        self.assertEqual(len(data["patients"]), 2)
        self.assertEqual(len(data["patients"][1]["outputs"]), 4)
        outputs = [v for patient in data["patients"] for v in patient["outputs"]]
        self.assertEqual(sum(v["status"] == "not_processed" for v in outputs), 7)
        self.assertTrue(all(pair["scores"]["condition"]["metrics"]["BERTScore_F1"] is None
                            for v in outputs if v["status"] == "not_processed" for pair in v["predictions"]))
        summary = format_top_rank_summary(data)
        self.assertIn("Best variant: llm_only", summary)
        self.assertIn("guidelines: condition: N/A (n=0); plan: N/A (n=0)", summary)

    def test_v3_rejects_invalid_variant_metadata_rows_and_duplicate_patients_cleanly(self):
        cases = ["bad_selection", "duplicate_selection", "unknown_variant", "invalid_variants", "invalid_slot",
                 "missing_variants", "duplicate_patient", "invalid_patient", "invalid_selected_patient", "empty_selection"]
        for case in cases:
            with self.subTest(case=case):
                self.native_v3()
                self.metadata["evaluation"]["selected_patient_ids"] = ["patient-1"]
                if case == "bad_selection": self.metadata["experiment"]["selected_variants"] = [{}]
                if case == "duplicate_selection": self.metadata["experiment"]["selected_variants"] = ["llm_only"] * 2
                if case == "unknown_variant": self.rows[0]["variants"]["bad"] = {}
                if case == "invalid_variants": self.rows[0]["variants"] = []
                if case == "invalid_slot": self.rows[0]["variants"]["llm_only"] = {"assessment_plans": []}
                if case == "missing_variants": del self.rows[0]["variants"]
                if case == "duplicate_patient": self.rows *= 2
                if case == "invalid_patient": self.rows[0]["id"] = []
                if case == "invalid_selected_patient": self.metadata["evaluation"]["selected_patient_ids"] = [{}]
                if case == "empty_selection": self.metadata["experiment"]["selected_variants"] = []
                self.write()
                with self.assertRaises(ValueError): self.load()

    def test_best_variant_uses_only_top_condition_bert_mean_and_plan_statistics_are_separate(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["action"]})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        later = deepcopy(self.rows[0])
        later["id"] = "patient-2"
        self.rows.append(later)
        for row, value in zip(self.rows, (0.0, 1.0)):
            baseline = row["variants"]["llm_only"]["assessment_plans"]["mini"]
            baseline["evaluation"][0]["condition"] = score(value)
            baseline["evaluation"][0]["plan"] = score(0.9)
            baseline["evaluation"][1]["condition"] = score(100)  # rank 2 cannot choose the winner
        self.write()
        summary = format_top_rank_summary(self.load())
        mini, llama = summary.split("Model: mini\n", 1)[1].split("Model: llama", 1)
        self.assertIn("llm_only: condition: 0.5000 (n=2); plan: 0.9000 (n=2)", mini)
        self.assertIn("Best variant: guidelines", mini)
        self.assertIn("Top-ranked condition: mean=0.8000, population_sd=0.0000, min=0.8000, max=0.8000, n=2", mini)
        self.assertIn("Top-ranked plan: mean=0.4000", mini)
        self.assertIn("Best variant: guidelines", llama)

    def test_summary_zero_missing_scores_ties_and_counts_are_not_conflated(self):
        self.native_v3()
        for slot in self.rows[0]["variants"].values():
            slot["assessment_plans"]["mini"] = output(0)
            slot["assessment_plans"]["mini"]["evaluation"][0]["plan"] = score(status="skipped_no_recorded_plan")
            slot["assessment_plans"]["llama"] = {"status": "failed_incomplete", "error": {"type": "Error"}}
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("llm_only: condition: 0.0000 (n=1); plan: N/A (n=0)", summary)
        self.assertIn("Best variant: llm_only", summary)
        self.assertIn("Top-ranked plan: N/A (n=0)", summary)
        self.assertIn("Best variant: unavailable (no scored top-ranked conditions)", summary)
        self.metadata["experiment"]["selected_variants"].reverse()
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("Best variant: guidelines", summary)

    def test_summary_population_spread_and_html_escaping_and_empty_cohort(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        second = deepcopy(self.rows[0])
        second["id"] = "patient-2"
        self.rows.append(second)
        self.metadata["assessment_plan_models"]["selected_models"] = ["<script>mini</script>"]
        for row, value in zip(self.rows, (0.0, 1.0)):
            for variant in row["variants"].values():
                variant["assessment_plans"] = {"<script>mini</script>": output(value)}
        self.write()
        data = self.load()
        summary = format_top_rank_summary(data)
        self.assertIn("mean=0.5000, population_sd=0.5000, min=0.0000, max=1.0000, n=2", summary)
        html = render_report(data)
        self.assertNotIn("Model: <script>mini</script>", html)
        self.assertIn("Model: &lt;script&gt;mini&lt;/script&gt;", html)
        self.rows = []
        self.metadata["evaluation"]["selected_patient_ids"] = []
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("Model: <script>mini</script>", summary)
        self.assertIn("Best variant: unavailable", summary)
        self.assertIn("condition: N/A (n=0)", summary)

    def test_summary_of_large_finite_saved_scores_does_not_overflow(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        second = deepcopy(self.rows[0])
        second["id"] = "patient-2"
        self.rows.append(second)
        for row in self.rows:
            for variant in row["variants"].values():
                for model in variant["assessment_plans"]:
                    variant["assessment_plans"][model] = output(1e308)
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("Best variant: llm_only", summary)
        self.assertNotIn("inf", summary)
        self.assertNotIn("nan", summary)
        self.assertIn("population_sd=0.0000", summary)

    def test_legacy_results_use_only_the_final_response_model_and_preserve_all_pairs(self):
        data = self.load()
        patient = data["patients"][0]
        self.assertEqual(patient["id"], "patient-1")
        self.assertEqual(patient["reference_condition"], "reference_condition")
        self.assertEqual(patient["reference_plan"], self.labels[0]["plan"])
        model = patient["outputs"][0]
        self.assertEqual(model["model_id"], "legacy-final-version")
        self.assertEqual(model["model_name"], "legacy-final-version")
        self.assertEqual(model["variant"], "not_recorded")
        self.assertEqual([pair["rank"] for pair in model["predictions"]], [1, 2, 3])
        for pair, original in zip(model["predictions"], self.rows[0]["predictions"]):
            self.assertEqual(pair["condition"], original["condition"])
            self.assertEqual(pair["plan"], original["plan"])
            for field in ("condition", "plan"):
                self.assertEqual(set(pair["scores"][field]["metrics"]), set(METRICS))
        self.assertEqual(data["comparisons"], [])
        self.assertIn("not recorded", data["notices"][0])

    def test_no_evidence_variant_is_inferred_from_saved_retrieval_or_guidelines(self):
        self.rows[0].update(selected_reference_patient={"id": "example"}, guideline_evidence="Some evidence")
        self.metadata["reference_data"] = {"patient_retrieval": {}, "guidelines": {}}
        self.write()
        self.assertEqual(self.load()["patients"][0]["outputs"][0]["variant"], "not_recorded")

    def test_multi_model_metadata_supplies_aliases_and_missing_model_names(self):
        self.metadata["assessment_plan_models"] = {
            "selected_models": ["model_a", "model_b"],
            "models": {"model_a": {"config": {"model": "configured/a"}},
                       "model_b": {"config": {"model": "configured/b"}}},
        }
        a, b = output(), output(0.8)
        b["model_responses"] = []
        self.rows = [{"id": "patient-1", "status": "completed", "assessment_plans": {"model_a": a, "model_b": b}}]
        self.write()
        outputs = self.load()["patients"][0]["outputs"]
        self.assertEqual([item["model_id"] for item in outputs], ["model_a", "model_b"])
        self.assertEqual([item["model_name"] for item in outputs], ["reported/model-version", "configured/b"])
        self.assertEqual(outputs[1]["predictions"][0]["scores"]["condition"]["metrics"]["BLEU"], 0.8)

    def test_last_reported_final_model_is_used_without_reading_current_configs(self):
        self.rows[0]["model_responses"]["assessment_plan"].extend([{}, {"reported_model": "newest-final-model"}])
        self.write()
        with patch("builtins.open", side_effect=AssertionError("No configuration reads")):
            data = self.load()
        self.assertEqual(data["patients"][0]["outputs"][0]["model_name"], "newest-final-model")

    def test_absolute_label_paths_do_not_depend_on_root_or_working_directory(self):
        self.metadata["evaluation"]["labels_path"] = str(self.labels_path)
        self.write()
        data = load_report_data(self.run, root=self.root / "unrelated")
        self.assertEqual(data["patients"][0]["id"], "patient-1")

    def test_label_fingerprint_mismatch_prevents_silently_joining_changed_labels(self):
        self.labels_path.write_text(self.labels_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, "no longer match.*labels_sha256"):
            self.load()

    def test_missing_label_fingerprint_is_explicitly_explained(self):
        self.write(fingerprint=False)
        self.assertTrue(any("could not be verified" in note for note in self.load()["notices"]))

    def test_missing_label_path_missing_file_and_unmatched_result_id_fail(self):
        original = deepcopy(self.metadata)
        for evaluation in ({}, {"labels_path": "not-found.jsonl"}):
            with self.subTest(evaluation=evaluation):
                self.metadata["evaluation"] = evaluation
                self.write()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load()
        self.metadata = original
        self.rows[0]["id"] = "no-label"
        self.write()
        with self.assertRaisesRegex(ValueError, "No reference label.*no-label"):
            self.load()

    def test_duplicate_or_invalid_labels_are_rejected(self):
        original = deepcopy(self.labels)
        for labels in ([*original, original[0]], [{"id": "", "condition": "x", "plan": []}],
                       [{"id": "patient-1", "condition": None, "plan": []}],
                       [{"id": "patient-1", "condition": "x", "plan": "not an array"}],
                       [{"id": "patient-1", "condition": "x", "plan": [5]}]):
            with self.subTest(labels=labels):
                self.labels = labels
                self.write()
                with self.assertRaises(ValueError):
                    self.load()

    def test_blank_lines_and_unicode_line_separators_in_text_are_preserved(self):
        self.rows[0]["predictions"][0]["condition"] = "Condition\u2028with\u2029separators"
        self.write()
        path = self.run / "results.jsonl"
        path.write_text('\n \n' + path.read_text() + '\n')
        condition = self.load()["patients"][0]["outputs"][0]["predictions"][0]["condition"]
        self.assertEqual(condition, "Condition\u2028with\u2029separators")

    def test_invalid_json_objects_duplicate_keys_and_nonfinite_literals_name_the_source(self):
        path = self.run / "results.jsonl"
        for value in ('[]', '{', '{"id":"patient-1","id":"patient-1"}', '{"value":NaN}'):
            with self.subTest(value=value):
                path.write_text('\n' + value + '\n')
                with self.assertRaisesRegex(ValueError, r"results.jsonl:2"):
                    self.load()

    def test_duplicate_patient_variant_is_rejected_but_different_variants_are_allowed(self):
        self.rows.append(deepcopy(self.rows[0]))
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate result"):
            self.load()
        self.rows[0]["evidence_variant"] = "llm_only"
        self.rows[1]["evidence_variant"] = "guidelines"
        self.write()
        self.assertEqual(len(self.load()["patients"][0]["outputs"]), 2)

    def test_explicit_record_variant_takes_precedence_over_experiment_variant(self):
        self.metadata["experiment"] = {"evidence_variant": "llm_only"}
        self.write()
        self.assertEqual(self.load()["patients"][0]["outputs"][0]["variant"], "llm_only")
        self.rows[0]["evidence_variant"] = "patient_retrieval_guidelines"
        self.write()
        data = self.load()
        self.assertEqual(data["patients"][0]["outputs"][0]["variant"], "patient_retrieval_guidelines")
        self.assertEqual(data["notices"], [])

    def test_unknown_variants_are_rejected_instead_of_being_guessed(self):
        for variant in ("bad_variant", 12, []):
            with self.subTest(variant=variant):
                self.rows[0]["evidence_variant"] = variant
                self.write()
                with self.assertRaisesRegex(ValueError, "Unsupported evidence_variant"):
                    self.load()

    def test_summaries_keep_ranks_fields_and_models_separate_and_include_zero(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["reference"]})
        second = deepcopy(self.rows[0])
        second["id"] = "patient-2"
        second["evaluation"][0]["condition"] = score(0)
        second["evaluation"][1]["condition"] = score(0.9)
        second["evaluation"][2]["plan"] = score(None, "skipped_no_recorded_plan")
        self.rows.append(second)
        self.write()
        summary = {(row["rank"], row["field"]): row for row in self.load()["summary"]}
        self.assertEqual(len(summary), 6)
        self.assertEqual(summary[(1, "condition")]["means"]["BLEU"], 0.25)
        self.assertEqual(summary[(2, "condition")]["means"]["BLEU"], 0.7)
        self.assertEqual(summary[(1, "plan")]["means"]["BLEU"], 0.25)
        self.assertEqual(summary[(3, "plan")]["counts"], {"scored": 1, "skipped_no_recorded_plan": 1})

    def test_paired_differences_use_matching_patient_model_rank_and_field(self):
        self.labels = [{"id": f"p{number}", "condition": "reference", "plan": ["reference"]} for number in (1, 2, 3)]
        self.metadata["evaluation"]["selected_patient_ids"] = ["p1", "p2", "p3"]
        self.rows = []
        for patient_id, variant, value in (("p1", "llm_only", 0.2), ("p1", "guidelines", 0.8),
                                          ("p2", "llm_only", 0.4), ("p2", "guidelines", 0.5),
                                          ("p3", "guidelines", 1.0)):
            self.rows.append({"id": patient_id, "evidence_variant": variant,
                              "assessment_plans": {"model-a": output(value), "model-b": output(value / 2)}})
        self.write()
        data = self.load()
        comparisons = {(r["model_id"], r["rank"], r["field"]): r for r in data["comparisons"]}
        self.assertEqual(len(comparisons), 12)
        condition = comparisons[("model-a", 1, "condition")]
        self.assertEqual(condition["paired_patients"], 2)
        self.assertAlmostEqual(condition["differences"]["BLEU"], 0.35)
        self.assertAlmostEqual(comparisons[("model-a", 1, "plan")]["differences"]["BLEU"], 0.175)
        self.assertAlmostEqual(comparisons[("model-b", 1, "condition")]["differences"]["BLEU"], 0.175)
        self.assertEqual(len(data["patients"]), 3)

    def test_skipped_failed_and_missing_fields_never_enter_paired_comparisons(self):
        baseline, evidence = deepcopy(self.rows[0]), deepcopy(self.rows[0])
        baseline["evidence_variant"], evidence["evidence_variant"] = "llm_only", "patient_retrieval"
        baseline["evaluation"][0]["plan"] = score(None, "error")
        evidence["evaluation"][1]["condition"] = score(None, "missing_generation")
        baseline["evaluation"][2]["plan"] = score(None, "skipped_no_recorded_plan")
        self.rows = [baseline, evidence]
        self.write()
        comparisons = {(r["rank"], r["field"]): r for r in self.load()["comparisons"]}
        for key in ((1, "plan"), (2, "condition"), (3, "plan")):
            self.assertEqual(comparisons[key]["paired_patients"], 0)
            self.assertTrue(all(value is None for value in comparisons[key]["differences"].values()))
        self.assertEqual(comparisons[(1, "condition")]["paired_patients"], 1)

    def test_every_evidence_variant_is_supported_without_creating_fake_baselines(self):
        for variant in ("llm_only", "patient_retrieval", "guidelines", "patient_retrieval_guidelines"):
            with self.subTest(variant=variant):
                self.rows[0]["evidence_variant"] = variant
                self.write()
                data = self.load()
                self.assertTrue(all(row["paired_patients"] == 0 for row in data["comparisons"]))

    def test_unscored_statuses_preserve_explanations_and_null_all_metric_values(self):
        statuses = ("skipped_no_recorded_plan", "missing_reference", "missing_generation", "error",
                    "interrupted", "pending", "other_future_status")
        for status in statuses:
            with self.subTest(status=status):
                self.rows[0]["evaluation"][0]["plan"] = score(0.5)
                self.rows[0]["evaluation"][0]["plan"]["status"] = status
                self.write()
                data = self.load()
                entry = data["patients"][0]["outputs"][0]["predictions"][0]["scores"]["plan"]
                self.assertEqual(entry["status"], status)
                self.assertTrue(all(value is None for value in entry["metrics"].values()))
                self.assertTrue(entry["reason"])
                summary = next(row for row in data["summary"] if row["rank"] == 1 and row["field"] == "plan")
                self.assertTrue(all(value is None for value in summary["means"].values()))

    def test_incomplete_boolean_string_and_overflowing_metrics_are_unavailable(self):
        for invalid in ({"BLEU": 0.2}, dict.fromkeys(METRICS, True), dict.fromkeys(METRICS, "0.5"),
                        dict.fromkeys(METRICS, None), dict.fromkeys(METRICS, 10 ** 400)):
            with self.subTest(invalid=invalid):
                self.rows[0]["evaluation"][0]["condition"]["metrics"] = invalid
                self.write()
                data = self.load()
                entry = data["patients"][0]["outputs"][0]["predictions"][0]["scores"]["condition"]
                self.assertEqual(entry["status"], "invalid_metrics")
                self.assertTrue(all(value is None for value in entry["metrics"].values()))

    def test_evaluation_rank_is_used_even_when_saved_entries_are_out_of_order(self):
        self.rows[0]["evaluation"][0]["condition"] = score(0.1)
        self.rows[0]["evaluation"].reverse()
        self.write()
        pair = self.load()["patients"][0]["outputs"][0]["predictions"][0]
        self.assertEqual(pair["scores"]["condition"]["metrics"]["BLEU"], 0.1)

    def test_invalid_and_duplicate_evaluation_ranks_are_rejected(self):
        for rank in (True, 0, 4, "1", 2):
            with self.subTest(rank=rank):
                self.rows[0]["evaluation"][0]["rank"] = rank
                self.write()
                with self.assertRaisesRegex(ValueError, "invalid or duplicate score ranks"):
                    self.load()

    def test_invalid_generation_status_and_model_slots_are_rejected(self):
        original = deepcopy(self.rows)
        for slot in (None, [], 1, "invalid"):
            with self.subTest(slot=slot):
                self.rows = [{"id": "patient-1", "assessment_plans": slot}]
                self.write()
                with self.assertRaisesRegex(ValueError, "assessment_plans must be an object"):
                    self.load()
        for slot in ({"status": "unknown"}, {"status": None}, "invalid"):
            with self.subTest(model_output=slot):
                self.rows = [{"id": "patient-1", "assessment_plans": {"model": slot}}]
                self.write()
                with self.assertRaises(ValueError):
                    self.load()
        self.rows = original

    def test_invalid_recorded_selection_and_config_structures_fail_cleanly(self):
        original = deepcopy(self.metadata)
        cases = [
            {"experiment": "not an object"},
            {"assessment_plan_models": ["model"]},
            {"assessment_plan_models": {"models": ["model"]}},
            {"assessment_plan_models": {"selected_models": "model"}},
            {"assessment_plan_models": {"selected_models": ["model", "model"]}},
            {"assessment_plan_models": {"selected_models": [1]}},
        ]
        for update in cases:
            with self.subTest(update=update):
                self.metadata = {**deepcopy(original), **update}
                self.write()
                with self.assertRaises(ValueError):
                    self.load()
        self.metadata = deepcopy(original)
        for selected in ("patient-1", ["patient-1", "patient-1"], [None]):
            with self.subTest(selected=selected):
                self.metadata["evaluation"]["selected_patient_ids"] = selected
                self.write()
                with self.assertRaisesRegex(ValueError, "selected_patient_ids"):
                    self.load()

    def test_empty_missing_and_invalid_plan_shapes_are_shown_without_inventing_actions(self):
        for plan in ([], None, "not a list", [""], ["one", "two", "three", "four"]):
            with self.subTest(plan=plan):
                self.rows[0]["predictions"][0]["plan"] = plan
                self.write()
                data = self.load()
                generated = data["patients"][0]["outputs"][0]["predictions"][0]["plan"]
                self.assertEqual(generated, [] if plan == [] else None)
                html = render_report(data)
                self.assertIn("Results need attention", html)
                self.assertIn("No plan steps recorded" if plan == [] else "Plan unavailable", html)

    def test_missing_predictions_evaluations_and_partial_failures_remain_visible(self):
        self.rows[0].update(status="failed_incomplete", predictions=None, evaluation=None,
                            error={"type": "AuthenticationError", "message": "PRIVATE_API_KEY"})
        self.write()
        data = self.load()
        model = data["patients"][0]["outputs"][0]
        self.assertEqual(len(model["predictions"]), 3)
        self.assertEqual(model["generation_note"], "Generation failed (AuthenticationError)")
        self.assertTrue(all(pair["condition"] is None for pair in model["predictions"]))
        self.assertTrue(all(pair["scores"]["plan"]["status"] == "not_processed" for pair in model["predictions"]))
        self.assertIn("Results need attention", render_report(data))

    def test_completed_generation_without_evaluation_is_flagged_for_attention(self):
        self.rows[0]["evaluation"] = None
        self.write()
        data = self.load()
        self.assertIn("Results need attention", render_report(data))
        entry = data["patients"][0]["outputs"][0]["predictions"][0]["scores"]["plan"]
        self.assertEqual(entry["status"], "not_scored")

    def test_missing_selected_patient_and_model_slots_are_counted_as_unprocessed(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        self.metadata["assessment_plan_models"] = {"selected_models": ["a", "b"], "models": {}}
        self.rows = [{"id": "patient-1", "assessment_plans": {"a": output()}}]
        self.write()
        data = self.load()
        self.assertEqual(len(data["patients"]), 2)
        self.assertEqual(data["patients"][0]["outputs"][1]["status"], "not_processed")
        self.assertTrue(all(item["status"] == "not_processed" for item in data["patients"][1]["outputs"]))
        a = next(row for row in data["summary"] if row["model_id"] == "a" and row["rank"] == 1 and row["field"] == "condition")
        self.assertEqual(a["counts"], {"scored": 1, "not_processed": 1})
        self.assertEqual(a["total"], 2)

    def test_unprocessed_legacy_patient_uses_known_model_when_possible(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        self.write()
        data = self.load()
        self.assertEqual(data["patients"][1]["outputs"][0]["model_id"], "legacy-final-version")

    def test_empty_run_generates_an_explicit_empty_report(self):
        self.metadata["evaluation"]["selected_patient_ids"] = []
        self.rows = []
        self.write()
        data = self.load()
        self.assertEqual(data["patients"], [])
        self.assertEqual(data["summary"], [])
        self.assertIn("No results available", render_report(data))
        self.assertIn("No patient results", ' '.join(data["notices"]))

    def test_html_contains_labels_three_pairs_and_all_twelve_scores_per_pair(self):
        self.assertEqual(METRICS, ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1"))
        data = self.load()
        html = render_report(data)
        document = Document(html)
        text = ' '.join(document.text)
        for phrase in ("reference_condition", "Reference action one", "Reference action two", "patient-1",
                       "legacy-final-version", "Condition 1", "Condition 2", "Condition 3", "Action 3.3"):
            self.assertIn(phrase, text)
        self.assertEqual(sum(tag == "li" and attrs.get("class") == "prediction" for tag, attrs in document.tags), 3)
        self.assertEqual(sum(tag == "table" and attrs.get("class") == "score-table" for tag, attrs in document.tags), 3)
        self.assertIn('title="0.5">0.5000', html)
        self.assertIn('class="rank-label">Rank 1', html)
        for metric in METRICS:
            self.assertGreaterEqual(text.count(metric), 5)
        payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)[1]
        self.assertEqual(json.loads(payload), data)

    def test_reference_plan_sentinel_and_real_zero_scores_are_distinct_from_na(self):
        self.labels[0]["plan"] = ["none recorded"]
        for entry in self.rows[0]["evaluation"]:
            entry["condition"] = score(0)
            entry["plan"] = score(None, "skipped_no_recorded_plan")
        self.write()
        html = render_report(self.load())
        self.assertIn("No plan recorded in label", html)
        self.assertIn("No reference plan recorded; scoring skipped", html)
        self.assertIn('title="0">0.0000', html)
        self.assertIn('class="na">N/A', html)
        self.assertIn("All patients evaluated", html)
        self.assertIn("3 condition/plan fields scored; 3 skipped; 0 unavailable", html)

    def test_header_keeps_only_patient_and_model_cards(self):
        html = render_report(self.load())
        stats = re.search(r'<div class="stats">(.*?)</header>', html, re.S)[1]
        document = Document(stats)
        self.assertEqual([text for text in document.text if text.strip()], ["Patients", "1", "Models", "1"])
        self.assertNotIn("Model outputs completed", html)
        self.assertNotIn("Condition / plan fields scored", html)
        self.assertNotIn("model outputs", html)
        self.assertIn("1/1 patients completed generation for all selected models and evidence variants", html)

    def test_patient_completion_counts_once_across_multiple_models_and_variants(self):
        self.rows = [{"id": "patient-1", "evidence_variant": variant,
                      "assessment_plans": {"a": output(), "b": output()}}
                     for variant in ("llm_only", "guidelines")]
        self.write()
        html = render_report(self.load())
        self.assertIn("1/1 patients completed generation", html)
        self.assertNotIn("4/4 patients", html)
        self.rows[1]["assessment_plans"]["b"].update(status="failed_incomplete", evaluation=None, predictions=None)
        self.write()
        html = render_report(self.load())
        self.assertIn("0/1 patients completed generation", html)
        self.assertIn("Results need attention", html)

    def test_grouped_overview_keeps_each_model_together_with_ordered_ranks_and_two_subrows(self):
        self.rows = [{"id": "patient-1", "evidence_variant": variant,
                      "assessment_plans": {"a": output(), "b": output()}}
                     for variant in ("llm_only", "guidelines")]
        self.write()
        data = self.load()
        # Rendering must order rank numbers even if an upstream consumer
        # rearranges the aggregate entries before passing them to the renderer.
        data["summary"].reverse()
        html = render_report(data)
        for table_id, expected_groups in (("summary-table", 4), ("comparison-table", 2)):
            table = re.search(rf'<table id="{table_id}">(.*?)</table>', html, re.S)[1]
            groups = re.findall(r'<tbody class="model-group"([^>]*)>(.*?)</tbody>', table, re.S)
            self.assertEqual(len(groups), expected_groups)
            model_sequence = []
            for attributes, body in groups:
                attrs = dict(re.findall(r'(data-[\w-]+)="([^"]*)"', attributes))
                model = attrs["data-model"]
                model_sequence.append(model)
                rows = re.findall(r'<tr data-rank="(\d)" data-field="(.*?)">', body)
                self.assertEqual(rows, [(str(rank), field) for rank in (1, 2, 3) for field in ("condition", "plan")])
                self.assertEqual(body.count('class="group-label" scope="rowgroup" rowspan="6"'), 1)
                self.assertEqual(body.count(f'<strong>{model}</strong>'), 1)
                self.assertEqual(body.count('class="rank-cell" scope="row" rowspan="2"'), 3)
                self.assertEqual(re.findall(r'class="rank-label">Rank (\d)', body), ["1", "2", "3"])
                self.assertIn('class="field-label" scope="row">Condition', body)
                self.assertIn('class="field-label" scope="row">Plan', body)
            self.assertEqual(len(list(dict.fromkeys(model_sequence))), 2)
            self.assertEqual(sum(left != right for left, right in zip(model_sequence, model_sequence[1:])), 1)
        self.assertNotIn('id="field-filter"', html)
        self.assertNotIn('id="rank-filter"', html)

    def test_model_name_appears_once_with_rank_between_model_and_condition_plan(self):
        html = render_report(self.load())
        table = re.search(r'<table id="summary-table">(.*?)</table>', html, re.S)[1]
        self.assertEqual(table.count('<strong>legacy-final-version</strong>'), 1)
        header = re.search(r'<thead>(.*?)</thead>', table, re.S)[1]
        labels = [re.sub('<[^>]*>', '', label) for label in re.findall(r'<th[^>]*>(.*?)</th>', header)]
        self.assertEqual(labels[:3], ["Model / evidence", "Rank", "Field / coverage"])
        rows = re.findall(r'<tr data-rank="(\d)" data-field="(.*?)">(.*?)</tr>', table, re.S)
        for rank, field, body in rows:
            headings = [attrs.get("class") for tag, attrs in Document(body).tags if tag == "th"]
            expected = (["group-label"] if rank == "1" and field == "condition" else [])
            expected += (["rank-cell"] if field == "condition" else []) + ["field-label"]
            self.assertEqual(headings, expected)

    def test_all_score_tables_put_bert_first_then_rouge_l_without_misaligning_values(self):
        values = {"BLEU": 0.11, "METEOR": 0.22, "ROUGE-1": 0.33,
                  "ROUGE-2": 0.44, "ROUGE-L": 0.55, "BERTScore_F1": 0.66}
        baseline, evidence = deepcopy(self.rows[0]), deepcopy(self.rows[0])
        baseline["evidence_variant"], evidence["evidence_variant"] = "llm_only", "guidelines"
        for entry in baseline["evaluation"]:
            entry["condition"] = score(0)
        for entry in evidence["evaluation"]:
            entry["condition"]["metrics"] = values.copy()
        self.rows = [baseline, evidence]
        self.write()
        html = render_report(self.load())
        expected = ["BERTScore_F1", "ROUGE-L", "BLEU", "METEOR", "ROUGE-1", "ROUGE-2"]
        for table in re.findall(r'<table[^>]*>(.*?)</table>', html, re.S):
            header = re.search(r'<thead>(.*?)</thead>', table, re.S)[1]
            labels = [re.sub('<[^>]*>', '', label) for label in re.findall(r'<th[^>]*>(.*?)</th>', header)]
            self.assertEqual(labels[-6:], expected)
        comparison = re.search(r'<table id="comparison-table">(.*?)</table>', html, re.S)[1]
        first_row = re.search(r'<tr data-rank="1" data-field="condition">(.*?)</tr>', comparison, re.S)[1]
        cells = [re.sub('<[^>]*>', '', cell) for cell in re.findall(r'<td>(.*?)</td>', first_row)]
        self.assertEqual(cells, ["+0.6600", "+0.5500", "+0.1100", "+0.2200", "+0.3300", "+0.4400"])
        prediction_table = re.findall(r'<table class="score-table">(.*?)</table>', html, re.S)[3]
        first_prediction_row = re.search(r'<tbody><tr>(.*?)</tr>', prediction_table, re.S)[1]
        cells = [re.sub('<[^>]*>', '', cell) for cell in re.findall(r'<td>(.*?)</td>', first_prediction_row)]
        self.assertEqual(cells, ["0.6600", "0.5500", "0.1100", "0.2200", "0.3300", "0.4400"])

    def test_patient_rows_preserve_full_ids_in_compact_headings_with_more_controls(self):
        html = render_report(self.load())
        self.assertIn('class="patient-heading"', html)
        self.assertIn('class="patient-id" title="patient-1">patient-1', html)
        self.assertIn('id="show-more"', html)
        self.assertIn('id="show-less"', html)
        self.assertIn("The first 10 matching patients are shown initially", html)
        self.assertIn("Search checks all patients", html)
        self.assertIn("Expand shown patients", html)

    def test_html_and_embedded_json_escape_untrusted_patient_model_and_label_text(self):
        attack = '</script><img src=x onerror="alert(1)"> & \u2028'
        self.labels[0]["condition"] = attack
        self.labels[0]["plan"] = [attack]
        self.rows[0]["predictions"][0] = {"condition": attack, "plan": [attack]}
        self.rows[0]["model_responses"]["assessment_plan"][0]["reported_model"] = attack
        self.write()
        html = render_report(self.load())
        self.assertNotIn(attack, html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;img src=x', html)
        self.assertEqual(html.count('</script>'), 2)
        payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)[1]
        self.assertEqual(json.loads(payload)["patients"][0]["reference_condition"], attack)

    def test_run_metadata_raw_responses_prompts_tokens_and_secret_messages_are_excluded(self):
        self.rows[0].update(assessment_plan_raw="PRIVATE_RAW_REPLY", assessment_plan_prompts="PRIVATE_PROMPT")
        self.rows[0]["evaluation"][0]["plan"].update(status="error", error={"type": "RuntimeError", "message": "PRIVATE_ERROR_MESSAGE PRIVATE_API_KEY"})
        self.write()
        html = render_report(self.load())
        for text in ("PRIVATE_RUN_NAME", "PRIVATE_COMMIT", "PRIVATE_TIMESTAMP", "PRIVATE_API_KEY",
                     "PRIVATE_RESPONSE_ID", "PRIVATE_RAW_REPLY", "PRIVATE_PROMPT", "PRIVATE_ERROR_MESSAGE",
                     "token_usage", "labels_sha256", "started_at", "api_key"):
            self.assertNotIn(text, html)
        self.assertIn("Scoring failed (RuntimeError)", html)

    def test_page_is_self_contained_searchable_and_accessible_without_javascript(self):
        html = render_report(self.load())
        document = Document(html)
        self.assertFalse(any(tag == "script" and "src" in attrs for tag, attrs in document.tags))
        self.assertFalse(any(tag == "link" or tag == "iframe" for tag, _ in document.tags))
        self.assertIn('<noscript>', html)
        for control in ("model-filter", "variant-filter", "patient-search", "expand-all", "reset", "show-more", "show-less"):
            self.assertTrue(any(attrs.get("id") == control for _, attrs in document.tags))
        self.assertFalse(any(attrs.get("id") in ("rank-filter", "field-filter") for _, attrs in document.tags))
        self.assertEqual(sum(tag == "details" for tag, _ in document.tags), 1)
        self.assertIn('lang="en"', html)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the offline JavaScript interaction check")
    def test_javascript_filters_all_ranks_paginates_searches_all_patients_and_expands_shown(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["reference"]})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        self.rows = [
            {"id": "patient-1", "evidence_variant": "llm_only", "assessment_plans": {"a": output()}},
            {"id": "patient-2", "evidence_variant": "guidelines", "assessment_plans": {"b": output()}},
        ]
        for number in range(3, 24):
            patient_id = f"patient-{number}"
            self.labels.append({"id": patient_id, "condition": "reference", "plan": ["reference"]})
            self.metadata["evaluation"]["selected_patient_ids"].append(patient_id)
            self.rows.append({"id": patient_id, "evidence_variant": "llm_only", "assessment_plans": {"a": output()}})
        self.write()
        data = self.load()
        html = render_report(data)
        javascript = re.findall(r'<script>(.*?)</script>', html, re.S)[0]
        # Exercise the actual report script against a minimal offline DOM. The
        # separate HTML test verifies these control IDs and data attributes exist.
        harness = r'''
const assert = require("node:assert/strict");
const vm = require("node:vm");
class Element {
  constructor(dataset = {}, children = []) {
    this.dataset = dataset; this.children = children; this.hidden = false;
    this.open = false; this.value = ""; this.textContent = ""; this.listeners = {};
  }
  querySelectorAll() { return this.children; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  fire(name) { this.listeners[name](); }
}
const elements = {};
for (const id of ["model-filter", "variant-filter", "patient-search", "show-more", "show-less",
                  "summary-empty", "comparison-empty", "patients-empty", "patient-count", "expand-all", "reset"])
  elements[id] = new Element();
for (const [name, rows] of [["summary", DATA.summary], ["comparison", DATA.comparisons]]) {
  const groups = rows.filter(row => row.field === "condition" && row.rank === 1);
  elements[name + "-table"] = new Element({}, groups.map(row => new Element({
    model: row.model_id, variant: row.variant, rank: String(row.rank), field: row.field
  })));
}
const cards = DATA.patients.map(patient => new Element({search: patient.id + " " + patient.reference_condition},
  patient.outputs.map(output => new Element({model: output.model_id, variant: output.variant}))));
const document = {getElementById: id => elements[id], querySelectorAll: () => cards};
vm.runInNewContext(SCRIPT, {document});
const visibleGroups = name => elements[name + "-table"].children.filter(group => !group.hidden);
const visibleCards = () => cards.filter(card => !card.hidden);
assert.equal(visibleGroups("summary").length, 2); // One group contains all ranks per model/evidence.
assert.equal(elements["comparison-empty"].hidden, true); // A zero-pair row explicitly displays N/A.
assert.equal(visibleCards().length, 10);
assert.equal(elements["patient-count"].textContent, "Showing 10 of 23 matching patients");
assert.equal(elements["show-more"].textContent, "Show 10 more patients (13 remaining)");
assert.equal(elements["show-less"].hidden, true);
elements["expand-all"].fire("click");
assert.ok(cards.slice(0, 10).every(card => card.open)); assert.equal(cards[10].open, false);
assert.equal(visibleCards().length, 10); // Expanding details never bypasses the patient limit.
elements["expand-all"].fire("click"); assert.ok(cards.every(card => !card.open));
elements["show-more"].fire("click"); assert.equal(visibleCards().length, 20);
assert.equal(elements["show-more"].textContent, "Show 3 more patients (3 remaining)");
assert.equal(elements["show-less"].hidden, false);
elements["show-more"].fire("click"); assert.equal(visibleCards().length, 23);
assert.equal(elements["show-more"].hidden, true);
elements["show-less"].fire("click"); assert.equal(visibleCards().length, 10);
elements["model-filter"].value = "a"; elements["model-filter"].fire("change");
assert.equal(cards[0].hidden, false); assert.equal(cards[1].hidden, true);
assert.equal(visibleGroups("summary").length, 1);
assert.equal(elements["patient-count"].textContent, "Showing 10 of 22 matching patients");
elements["variant-filter"].value = "guidelines"; elements["variant-filter"].fire("change");
assert.equal(visibleGroups("summary").length, 0); assert.equal(elements["patients-empty"].hidden, false);
assert.equal(elements["show-more"].hidden, true);
elements["reset"].fire("click");
assert.equal(visibleGroups("summary").length, 2); assert.equal(visibleCards().length, 10);
elements["patient-search"].value = "PATIENT-23"; elements["patient-search"].fire("input");
assert.equal(cards[0].hidden, true); assert.equal(cards[22].hidden, false);
assert.equal(elements["patient-count"].textContent, "Showing 1 of 1 matching patients");
assert.equal(visibleGroups("summary").length, 2); // Search changes patient details only.
elements["expand-all"].fire("click");
assert.equal(cards[0].open, false); assert.equal(cards[22].open, true);
assert.equal(elements["expand-all"].textContent, "Collapse shown patients");
elements["expand-all"].fire("click"); assert.equal(cards[22].open, false);
elements["patient-search"].value = "does not exist"; elements["patient-search"].fire("input");
assert.equal(visibleCards().length, 0); assert.equal(elements["patients-empty"].hidden, false);
elements["reset"].fire("click");
assert.equal(elements["patient-count"].textContent, "Showing 10 of 23 matching patients");
elements["show-more"].fire("click");
elements["patient-search"].value = "patient"; elements["patient-search"].fire("input");
assert.equal(visibleCards().length, 10); // Changing filters starts a fresh ten-patient page.
'''
        program = 'const DATA = ' + json.dumps(data) + ';\nconst SCRIPT = ' + json.dumps(javascript) + ';\n' + harness
        result = subprocess.run([shutil.which("node"), "-e", program], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generate_writes_only_html_and_preserves_inputs_byte_for_byte(self):
        snapshots = {path: path.read_bytes() for path in (self.run / "run.json", self.run / "results.jsonl", self.labels_path)}
        path = generate_report(self.run, root=self.root)
        self.assertEqual(path, self.run / "results_summary.html")
        self.assertTrue(path.read_text().startswith('<!doctype html>'))
        self.assertEqual({p: p.read_bytes() for p in snapshots}, snapshots)
        self.assertEqual({p.name for p in self.run.iterdir()}, {"run.json", "results.jsonl", "results_summary.html"})
        first = path.read_bytes()
        self.assertEqual(generate_report(self.run, root=self.root).read_bytes(), first)

    def test_validation_failure_preserves_old_report_and_leaves_no_temporary_file(self):
        path = generate_report(self.run, root=self.root)
        original = path.read_bytes()
        self.labels_path.write_text('changed labels')
        with self.assertRaises(ValueError):
            generate_report(self.run, root=self.root)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.run.glob('.results_summary-*.tmp')), [])

    def test_replace_failure_preserves_old_report_and_cleans_up_temporary_file(self):
        path = generate_report(self.run, root=self.root)
        original = path.read_bytes()
        with patch("scripts.collate_med_copilot_results.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                generate_report(self.run, root=self.root)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.run.glob('.results_summary-*.tmp')), [])

    def test_main_reports_its_destination(self):
        self.metadata["evaluation"]["labels_path"] = str(self.labels_path)
        self.write()
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main([str(self.run)]), 0)
        self.assertIn(str(self.run / "results_summary.html"), stdout.getvalue())
        self.assertTrue((self.run / "results_summary.html").exists())

    def test_main_reports_actionable_failure_and_nonzero_exit_without_creating_report(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main([str(self.root / "missing")])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Cannot generate results summary", stderr.getvalue())
        self.assertFalse((self.run / "results_summary.html").exists())

    def test_script_runs_from_another_working_directory_without_model_packages_or_network(self):
        self.metadata["evaluation"]["labels_path"] = str(self.labels_path)
        self.write()
        program = (
            "import builtins, runpy, socket, sys\n"
            "def blocked(*args, **kwargs): raise AssertionError('Network access attempted')\n"
            "socket.socket.connect = blocked\n"
            "socket.getaddrinfo = blocked\n"
            "original_import = builtins.__import__\n"
            "def guarded_import(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in ('torch', 'pandas', 'openai', 'litellm', 'transformers', 'graphrag'):\n"
            "        raise AssertionError('Model package imported: ' + name)\n"
            "    return original_import(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded_import\n"
            f"sys.argv = [{str(SCRIPT)!r}, {str(self.run)!r}]\n"
            f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
        )
        result = subprocess.run([sys.executable, "-I", "-c", program], cwd=self.root,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Results summary written", result.stdout)
        self.assertTrue((self.run / "results_summary.html").exists())


if __name__ == "__main__":
    unittest.main()
