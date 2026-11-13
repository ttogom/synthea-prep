"""Offline verification of temporary Plan-validation workloads and recovery."""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nltk

# Direct execution needs project/vendor roots before importing scoring adapters.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
with patch.object(nltk, "download", return_value=True), patch("dotenv.load_dotenv", return_value=False):
    from scripts import validate_plan_judging as validation
    from scripts.med_copilot_scoring import PatientScorer


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / "source", self.root / "validation"
        self.source.mkdir()
        (self.root / "med_copilot/prompts").mkdir(parents=True)
        (self.root / "med_copilot/prompts/correctness_judges.py").write_text("prompt fingerprint\n")
        (self.root / "experiments/configs").mkdir(parents=True)
        (self.root / "experiments/configs/plan_judging.yaml").write_text(
            "concurrency: 4\ncontext_length: 4096\nflash_attention: true\nkv_cache_type: f16\nhost_cache_mib: 2048\n")
        self.patients, self.labels, self.records = [], [], []
        for index in range(80):
            patient = {"id": f"p{index:03}", "subjective": "S" * (index + 1), "objective": "O"}
            reference = [f"Reference {number}" for number in range(3 if index >= 76 else 1 + index % 2)]
            self.patients.append(patient)
            self.labels.append({"id": patient["id"], "condition": "condition", "plan": reference,
                                "has_presenting_evidence": True})
            output = {"status": "completed", "raw_response": "untouched", "predictions": [
                {"condition": "diagnosis", "plan": [f"Generated {index} {rank}"]} for rank in range(3)],
                "evaluation": [{"rank": rank, "diagnosis_correctness": {"score": .4},
                                "plan": {"metrics": {"sentinel": rank}}} for rank in (1, 2, 3)]}
            self.records.append({"id": patient["id"], "status": "completed", "evidence": "intact",
                                 "variants": {variant: {"assessment_plans": {model: deepcopy(output)
                                                                            for model in ("g1", "g2")}}
                                              for variant in ("llm_only", "guidelines")}})
        for name, rows in (("input.jsonl", self.patients), ("labels.jsonl", self.labels)):
            (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.metadata = {"status": "completed", "run_name": "source", "errors": [],
                         "input": {"file": "input.jsonl", "sha256": validation._sha(self.root / "input.jsonl")},
                         "experiment": {"selected_variants": ["llm_only", "guidelines"], "plan_judge": "judge"},
                         "assessment_plan_models": {"selected_models": ["g1", "g2"]},
                         "evaluation": {"selected_patient_ids": [patient["id"] for patient in self.patients],
                                        "labels_path": "labels.jsonl", "labels_sha256": validation._sha(self.root / "labels.jsonl")}}
        (self.source / "run.json").write_text(json.dumps(self.metadata))
        (self.source / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def test_preparation_preserves_complete_selected_workloads_and_source_artifacts(self):
        before = {path: path.read_bytes() for path in self.source.iterdir()}
        resident, subset, manifest = validation.prepare_validation(self.source, self.output, root=self.root)
        self.assertEqual(resident, self.records)
        self.assertEqual(len(subset), 8)
        self.assertEqual(len(set(manifest["selected_cells"].values())), 8)
        self.assertEqual(manifest["resident_patient_count"], 80)
        self.assertEqual(subset, [record for record in self.records if record["id"] in manifest["selected_cells"]])
        self.assertEqual(validation._records(self.output / "results.jsonl"), subset)
        expected_metadata = deepcopy(self.metadata)
        expected_metadata["evaluation"]["selected_patient_ids"] = [record["id"] for record in subset]
        self.assertEqual(json.loads((self.output / "run.json").read_text()), expected_metadata)
        self.assertEqual({path: path.read_bytes() for path in self.source.iterdir()}, before)
        _, again, manifest_again = validation.prepare_validation(self.source, self.root / "again", root=self.root)
        self.assertEqual((again, manifest_again["selected_cells"]), (subset, manifest["selected_cells"]))

    def test_existing_or_nested_output_is_rejected_without_writes(self):
        for destination in (self.source, self.source / "child"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "outside"):
                validation.prepare_validation(self.source, destination, root=self.root)
        self.output.mkdir()
        (self.output / "sentinel").write_text("keep")
        with self.assertRaisesRegex(ValueError, "new directory"):
            validation.prepare_validation(self.source, self.output, root=self.root)
        self.assertEqual(list(self.output.iterdir()), [self.output / "sentinel"])

    def test_changed_input_or_incomplete_result_set_is_rejected_before_output_creation(self):
        original = (self.root / "input.jsonl").read_bytes()
        (self.root / "input.jsonl").write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "changed since"):
            validation.prepare_validation(self.source, self.output, root=self.root)
        self.assertFalse(self.output.exists())
        (self.root / "input.jsonl").write_bytes(original)
        (self.source / "results.jsonl").write_text("\n".join(json.dumps(row) for row in self.records[:-1]))
        with self.assertRaisesRegex(ValueError, "recorded cohort"):
            validation.prepare_validation(self.source, self.output, root=self.root)
        self.assertFalse(self.output.exists())

    def test_nonplan_comparison_retains_all_other_nested_fields(self):
        original = {"predictions": [{"plan": ["keep"]}], "evaluation": [
            {"plan": {"metrics": 1}, "diagnosis_correctness": {"score": 1}, "plan_correctness": {"score": 0}}]}
        self.assertEqual(validation.without_plan_scores(original), {"predictions": [{"plan": ["keep"]}],
                         "evaluation": [{"plan": {"metrics": 1}, "diagnosis_correctness": {"score": 1}}]})
        self.assertIn("plan_correctness", original["evaluation"][0])

    def test_offline_production_stage_interrupts_resumes_and_keeps_source_and_nonplan_data(self):
        scorer = PatientScorer(self.root / "labels.jsonl", self.patients, False, metric_evaluator=object())
        class Models:
            model_ids = ("judge",)
            redact = staticmethod(lambda text: text)
            metadata = staticmethod(lambda root: {"models": {"judge": {"config": {"model": "fake"}}}})
            def judge(self, _, messages):
                text = messages[0]["content"]
                if "<reference_action>" in text:
                    action = re.search(r"<reference_action>\n(.*?)\n</reference_action>", text, re.S).group(1)
                    generated = re.search(r"<generated_plan>\n(.*?)\n</generated_plan>", text, re.S).group(1)
                    content = json.dumps({"credit": int(action in generated)})
                else:
                    content = '{"unsupported_extra_count": 0}'
                return SimpleNamespace(id="fake", model="fake", usage=None, choices=[SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=content, refusal=None))])
        models = Models()
        before = {path: path.read_bytes() for path in self.source.iterdir()}
        with patch.object(validation.command, "_prepare", return_value=(scorer, models, models)), \
                patch.object(validation, "ollama_for_saved_run", return_value=nullcontext(object())), \
                patch.object(validation, "save_verified_runtime", return_value={"file": "verified.json", "sha256": "a" * 64}):
            summary = validation.validate(self.source, self.output, root=self.root)
        self.assertEqual(summary["interrupted_stage"]["status"], "interrupted")
        self.assertTrue(summary["source_unchanged"] and summary["prompt_unchanged"] and summary["non_plan_preserved"])
        self.assertEqual(summary["controls"]["plans"], 9)
        self.assertEqual(summary["stages"][-1]["calls"], 0)
        self.assertEqual({path: path.read_bytes() for path in self.source.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
