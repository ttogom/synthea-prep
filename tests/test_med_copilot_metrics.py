"""Offline adapter tests; optional cached-weight parity via MEDCOPILOT_BERT_PARITY=1."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nltk
import torch

# Direct execution adds tests/; project imports need the checkout root first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.med_copilot_metrics import (
    BATCH_SIZE, BERT_SETTINGS, METRIC_NAMES, MetricPair, TextMetricEvaluator,
    lexical_metrics, resolved_device, validate_metrics,
)

# Only the semantic reference imports upstream; suppress its download side effects.
with patch.object(nltk, "download", return_value=True):
    from scripts.med_copilot_scoring import METRIC_ADAPTER, PatientScorer, resume_evaluation
    from med_copilot.upstream.evaluate_metrics import Evaluator


LEXICAL = {name: 0.25 for name in METRIC_NAMES if name != "BERTScore_F1"}


class BertDouble:
    def __init__(self):
        self.calls = []
        self.reply = None

    def score(self, candidates, references, **kwargs):
        self.calls.append((list(candidates), list(references), kwargs))
        if self.reply is not None:
            return self.reply(candidates, references)
        return None, None, torch.tensor([len(text) / 1000 for text in candidates])


class MetricAdapterTests(unittest.TestCase):
    def setUp(self):
        self.bert = BertDouble()
        self.factory = Mock(return_value=self.bert)
        self.lexical = Mock(return_value=LEXICAL)
        self.adapter = TextMetricEvaluator(bert_scorer_factory=self.factory, lexical=self.lexical)

    def pairs(self, count):
        return [MetricPair(("patient", index), f"reference {index}", "a" * (index % 9 + 1))
                for index in range(count)]

    def test_true_batches_tail_identity_and_lazy_reuse_across_calls(self):
        self.factory.assert_not_called()
        pairs = list(reversed(self.pairs(139)))  # Duplicate text, distinct shuffled identities.
        scores = list(self.adapter.score_pairs(pairs))
        self.assertEqual([len(call[0]) for call in self.bert.calls], [64, 64, 11])
        self.assertEqual([score.key for score in scores], [pair.key for pair in pairs])
        self.assertEqual([text for call in self.bert.calls for text in call[0]],
                         [pair.candidate for pair in pairs])
        self.assertEqual([text for call in self.bert.calls for text in call[1]],
                         [pair.reference for pair in pairs])
        for pair, score in zip(pairs, scores):
            self.assertAlmostEqual(score.metrics["BERTScore_F1"], len(pair.candidate) / 1000)
            self.assertEqual({key: score.metrics[key] for key in LEXICAL}, LEXICAL)
            self.assertIsNone(score.error)
        list(self.adapter.score_pairs(self.pairs(1)))
        self.factory.assert_called_once_with(**BERT_SETTINGS)
        self.assertTrue(all(call[2] == {"batch_size": BATCH_SIZE, "verbose": False}
                            for call in self.bert.calls))
        self.assertEqual(self.lexical.call_count, 140)

    def test_empty_batch_never_initializes_model(self):
        self.assertEqual(list(self.adapter.score_pairs([])), [])
        self.factory.assert_not_called()
        self.lexical.assert_not_called()

    def test_invalid_inputs_and_duplicate_keys_fail_before_inference(self):
        for pairs in ([object()], [MetricPair((1,), None, "text")],
                      [MetricPair("bad key", "ref", "cand")], self.pairs(1) * 2):
            with self.subTest(pairs=pairs), self.assertRaises((TypeError, ValueError)):
                list(self.adapter.score_pairs(pairs))
        self.factory.assert_not_called()

    def test_initialization_failure_once_per_call_and_retry_next_call(self):
        self.factory.side_effect = [RuntimeError("load failed"), self.bert]
        scores = list(self.adapter.score_pairs(self.pairs(70)))
        self.assertEqual(self.factory.call_count, 1)
        self.assertTrue(all(isinstance(score.error, RuntimeError) and score.metrics is None for score in scores))
        self.lexical.assert_not_called()
        self.assertIsNone(list(self.adapter.score_pairs(self.pairs(1)))[0].error)
        self.assertEqual(self.factory.call_count, 2)

    def test_batch_failure_falls_back_on_same_model_and_isolates_bad_pair(self):
        def reply(candidates, references):
            if len(candidates) > 1 or references == ["reference 2"]:
                raise RuntimeError("bad batch or pair")
            return None, None, [0.7]
        self.bert.reply = reply
        scores = list(self.adapter.score_pairs(self.pairs(4)))
        self.assertEqual([score.error is None for score in scores], [True, True, False, True])
        self.assertEqual([len(call[0]) for call in self.bert.calls], [4, 1, 1, 1, 1])
        self.factory.assert_called_once()
        self.assertEqual(self.lexical.call_count, 3)

    def test_malformed_vector_length_is_not_silently_zipped(self):
        self.bert.reply = lambda cands, refs: (None, None, [0.8] if len(cands) == 1 else [])
        scores = list(self.adapter.score_pairs(self.pairs(3)))
        self.assertEqual([score.metrics["BERTScore_F1"] for score in scores], [0.8] * 3)
        self.bert.reply = lambda cands, refs: (None, None, [])
        scores = list(self.adapter.score_pairs(self.pairs(3)))
        self.assertTrue(all(isinstance(score.error, ValueError) for score in scores))

    def test_invalid_f1_values_are_field_local_and_no_partial_metrics_escape(self):
        for value in (float("nan"), float("inf"), True, [0.4], "0.4", None):
            with self.subTest(value=value):
                self.bert.reply = lambda cands, refs: (None, None, [0.6, value, 0.8])
                scores = list(self.adapter.score_pairs(self.pairs(3)))
                self.assertEqual([score.error is None for score in scores], [True, False, True])
                self.assertIsNone(scores[1].metrics)

    def test_lexical_failure_does_not_discard_other_fields(self):
        self.lexical.side_effect = [LEXICAL, LookupError("missing resource"), LEXICAL]
        scores = list(self.adapter.score_pairs(self.pairs(3)))
        self.assertEqual([score.error is None for score in scores], [True, False, True])
        self.assertIsNone(scores[1].metrics)

    def test_bert_interrupt_propagates_without_fallback_or_model_reload(self):
        self.bert.reply = Mock(side_effect=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            list(self.adapter.score_pairs(self.pairs(3)))
        self.assertEqual(len(self.bert.calls), 1)
        self.bert.reply = None
        list(self.adapter.score_pairs(self.pairs(1)))
        self.factory.assert_called_once()

    def test_lexical_interrupt_releases_lock_and_preserves_yielded_results(self):
        self.lexical.side_effect = [LEXICAL, KeyboardInterrupt()]
        scores = self.adapter.score_pairs(self.pairs(3))
        self.assertIsNone(next(scores).error)
        with self.assertRaises(KeyboardInterrupt):
            next(scores)
        self.lexical.side_effect = None
        self.assertIsNone(list(self.adapter.score_pairs(self.pairs(1)))[0].error)

    def test_explicit_close_releases_lock_for_next_scoring_call(self):
        scores = self.adapter.score_pairs(self.pairs(3))
        next(scores)
        scores.close()
        self.assertFalse(self.adapter._lock.locked())
        self.assertEqual(len(list(self.adapter.score_pairs(self.pairs(1)))), 1)

    def test_cross_process_use_is_rejected_before_touching_model_or_lock(self):
        with patch("scripts.med_copilot_metrics.os.getpid", return_value=-1):
            with self.assertRaisesRegex(RuntimeError, "owning scoring process"):
                list(self.adapter.score_pairs(self.pairs(1)))
        self.factory.assert_not_called()

    def test_finite_metrics_validation_and_copy(self):
        metrics = {**LEXICAL, "BERTScore_F1": 0.5}
        values = validate_metrics(metrics)
        self.assertEqual(values, metrics)
        self.assertIsNot(values, metrics)
        for invalid in (None, {}, {**metrics, "other": 1}, {**metrics, "BLEU": True},
                        {**metrics, "BLEU": float("nan")}, {**metrics, "BLEU": "1"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_metrics(invalid)

    def test_device_policy_preserves_cuda_cpu_and_never_selects_mps(self):
        for available, expected in ((True, "cuda"), (False, "cpu")):
            with patch("scripts.med_copilot_metrics.torch.cuda.is_available", return_value=available):
                self.assertEqual(resolved_device(), expected)

    def test_real_lexical_metrics_equal_upstream_singletons(self):
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            self.skipTest("Cached NLTK resources unavailable")
        for reference, candidate in (("acute bronchitis", "acute bronchitis"),
                                     ("one two three four", "one two three four"),
                                     ("Rest, drink fluids. Follow-up.", "drink fluids, then rest")):
            with patch("med_copilot.upstream.evaluate_metrics.bert_score",
                       return_value=(None, None, torch.tensor([0.5]))):
                expected = Evaluator.evaluate_metrics([reference], [candidate], lang="en")
            self.assertEqual(lexical_metrics(reference, candidate),
                             {key: expected[key] for key in LEXICAL})


class BatchedPatientScorerTests(unittest.TestCase):
    def setUp(self):
        MetricAdapterTests.setUp(self)
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.labels = self.directory / "labels.jsonl"
        self.patients = [{"id": name, "subjective": "S", "objective": "O"} for name in ("A", "B")]
        self.labels.write_text("".join(json.dumps({"id": name, "condition": "gold_condition",
                                                  "plan": ["gold plan"], "has_presenting_evidence": True}) + "\n"
                                       for name in ("A", "B")))
        self.scorer = PatientScorer(self.labels, self.patients, metric_evaluator=self.adapter)

    def record(self, patient="A"):
        return {"id": patient, "variants": {variant: {"assessment_plans": {
            model: {"status": "completed", "predictions": [
                {"condition": f"{variant}/{model}/{rank}", "plan": ["duplicate action"]}
                for rank in range(3)]} for model in ("m3", "m1", "m2", "m0")}}
            for variant in ("v3", "v1", "v2", "v0")}}

    def fields(self, record):
        return [entry[field] for body in record["variants"].values()
                for output in body["assessment_plans"].values()
                for entry in output["evaluation"] for field in ("condition", "plan")]

    def test_pipeline_batches_across_all_outputs_and_reuses_model_across_patients(self):
        records = [self.record(name) for name in ("A", "B")]
        originals = deepcopy(records)
        for record in records:
            self.scorer.score_result(record)
            self.assertTrue(all(field["status"] == "scored" for field in self.fields(record)))
        self.factory.assert_called_once()
        self.assertEqual([len(call[0]) for call in self.bert.calls], [64, 32, 64, 32])
        for record, original in zip(records, originals):
            expected = [text for variant in original["variants"].values()
                        for output in variant["assessment_plans"].values()
                        for prediction in output["predictions"]
                        for text in (prediction["condition"], ", ".join(prediction["plan"]))]
            self.assertEqual([field["metrics"]["BERTScore_F1"] for field in self.fields(record)],
                             [float(torch.tensor(len(text) / 1000)) for text in expected])
        before = deepcopy(records)
        for record in records:
            self.scorer.score_result(record)
        self.assertEqual(records, before)
        self.assertEqual(len(self.bert.calls), 4)

    def test_pending_filtering_preserves_completed_and_excludes_invalid_missing_failed(self):
        record = self.record()
        self.scorer.score_result(record)
        fields = self.fields(record)
        completed = deepcopy(fields[0])
        output = record["variants"]["v3"]["assessment_plans"]["m3"]
        output["evaluation"][0]["plan"] = {"status": "pending"}
        output["evaluation"][1]["condition"] = {"status": "error"}
        output["predictions"][1]["condition"] = None
        output["evaluation"][2]["plan"] = {"status": "pending"}
        output["predictions"][2]["plan"] = "invalid"
        failed = record["variants"]["v1"]["assessment_plans"]["m1"]
        failed.update(status="failed_incomplete", evaluation=None)
        self.bert.calls.clear()
        self.scorer.score_result(record)
        self.assertEqual([len(call[0]) for call in self.bert.calls], [1])
        self.assertEqual(output["evaluation"][0]["condition"], completed)
        self.assertEqual(output["evaluation"][1]["condition"]["status"], "missing_generation")
        self.assertEqual(output["evaluation"][2]["plan"]["status"], "error")
        self.assertIsNone(failed["evaluation"])

    def test_no_recorded_plan_and_missing_generations_never_enter_bert_batches(self):
        self.scorer._labels["A"]["plan"] = ["none recorded"]
        record = self.record()
        output = record["variants"]["v3"]["assessment_plans"]["m3"]
        output["predictions"] = None
        self.scorer.score_result(record)
        self.assertEqual([len(call[0]) for call in self.bert.calls], [45])
        self.assertTrue(all(field["status"] in ("scored", "missing_generation", "skipped_no_recorded_plan")
                            for field in self.fields(record)))
        self.assertTrue(all(entry["condition"]["status"] == "missing_generation" for entry in output["evaluation"]))
        self.assertTrue(all(entry["plan"]["status"] == "skipped_no_recorded_plan" for entry in output["evaluation"]))

    def test_all_missing_outputs_do_not_initialize_model(self):
        record = {"id": "A", "predictions": None}
        self.scorer.score_result(record)
        self.factory.assert_not_called()
        self.assertTrue(all(entry[field]["status"] == "missing_generation" for entry in record["evaluation"]
                            for field in ("condition", "plan")))

    def test_initialization_errors_are_retryable_fields_without_false_completions(self):
        record = self.record()
        self.factory.side_effect = [RuntimeError("load failed"), self.bert]
        self.scorer.score_result(record)
        self.assertTrue(all(field["status"] == "error" and all(value is None for value in field["metrics"].values())
                            for field in self.fields(record)))
        self.assertEqual(self.factory.call_count, 1)
        self.scorer.score_result(record)
        self.assertTrue(all(field["status"] == "scored" for field in self.fields(record)))
        self.assertEqual(self.factory.call_count, 2)

    def test_mismatched_result_identity_cannot_score_another_field(self):
        self.scorer.metric_evaluator = SimpleNamespace(score_pairs=lambda pairs: iter([
            SimpleNamespace(key=("wrong",), metrics={**LEXICAL, "BERTScore_F1": 0.99}, error=None)]))
        record = self.record()
        self.scorer.score_result(record)
        self.assertTrue(all(field["status"] == "error" for field in self.fields(record)))
        self.assertIn("identity", self.fields(record)[0]["error"]["message"])

    def test_batch_interruption_checkpoints_completed_fields_and_resumes_pending_only(self):
        record = self.record()
        snapshots = []
        self.lexical.side_effect = [LEXICAL] * 65 + [KeyboardInterrupt()]
        with patch("scripts.med_copilot_scoring.checkpoint", side_effect=lambda: snapshots.append(deepcopy(record))):
            with self.assertRaises(KeyboardInterrupt):
                self.scorer.score_result(record)
        self.assertEqual([field["status"] for field in self.fields(snapshots[-1])],
                         ["scored"] * 65 + ["interrupted"] + ["pending"] * 30)
        saved = deepcopy(self.fields(record)[:65])
        self.lexical.side_effect = None
        self.scorer.score_result(record)
        self.assertEqual([len(call[0]) for call in self.bert.calls], [64, 32, 31])
        self.assertEqual(self.fields(record)[:65], saved)
        self.assertTrue(all(field["status"] == "scored" for field in self.fields(record)))
        self.factory.assert_called_once()

    def test_checkpoint_failure_closes_model_iterator_and_propagates(self):
        record = self.record()
        with patch("scripts.med_copilot_scoring.checkpoint", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.scorer.score_result(record)
        self.assertEqual(self.fields(record)[0]["status"], "scored")
        self.assertFalse(self.adapter._lock.locked())
        self.scorer.score_result(record)
        self.assertTrue(all(field["status"] == "scored" for field in self.fields(record)))

    def test_metadata_records_adapter_and_pipeline_fingerprints_without_loading(self):
        metadata = self.scorer.metadata(ROOT)
        implementation = metadata["metric_implementation"]
        self.assertEqual(implementation["path"], "scripts/med_copilot_metrics.py")
        self.assertEqual(implementation["sha256"], hashlib.sha256(METRIC_ADAPTER.read_bytes()).hexdigest())
        self.assertEqual(implementation["scoring_sha256"], hashlib.sha256(
            (ROOT / implementation["scoring_path"]).read_bytes()).hexdigest())
        self.assertEqual(metadata["settings"]["bert_score"]["batch_size"], 64)
        self.assertEqual(metadata["settings"]["bert_score"]["resolved_device"], resolved_device())
        self.factory.assert_not_called()

    def test_standalone_evaluation_uses_real_batches_and_one_model(self):
        run_dir = self.directory / "run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"errors": [], "experiment": {"plan_judge": "unused"},
            "evaluation": {"selected_patient_ids": ["A", "B"]}}))
        (run_dir / "results.jsonl").write_text("".join(json.dumps(self.record(name)) + "\n" for name in ("A", "B")))
        models = SimpleNamespace(judge=None, model_ids=(), redact=lambda text: text)
        similarity = SimpleNamespace(cosine=lambda *args: 0.5)
        with patch("scripts.med_copilot_judges.BioLordSimilarity", return_value=similarity):
            resume_evaluation(run_dir, self.scorer, models, root=self.directory)
        self.assertEqual([len(call[0]) for call in self.bert.calls], [64, 32, 64, 32])
        self.factory.assert_called_once()
        saved = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
        self.assertTrue(all(field["status"] == "scored" for record in saved for field in self.fields(record)))


@unittest.skipUnless(os.environ.get("MEDCOPILOT_BERT_PARITY") == "1", "Opt-in cached-weight numerical parity")
class CachedWeightParityTests(unittest.TestCase):
    def test_original_singletons_cached_singletons_and_batches_agree(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1", "Parity must never download weights")
        pairs = [MetricPair((index,), reference, candidate) for index, (reference, candidate) in enumerate([
            ("acute bronchitis", "bronchitis"),
            ("Rest and drink fluids; follow up if worse.", "Drink fluids and rest."),
            ("The patient has fever and cough. " * 100, "Cough and fever. " * 100),
            ("Café: follow-up", "Café: follow-up"),
            ("acute bronchitis", "bronchitis"),
        ])]
        expected = [Evaluator.evaluate_metrics([pair.reference], [pair.candidate], lang="en") for pair in pairs]
        adapter = TextMetricEvaluator()
        singleton = [next(adapter.score_pairs([pair])) for pair in pairs]
        batch = list(adapter.score_pairs(pairs))
        for original, single, batched in zip(expected, singleton, batch):
            self.assertIsNone(single.error)
            self.assertIsNone(batched.error)
            for name in METRIC_NAMES:
                tolerance = 1e-6 if name == "BERTScore_F1" else 0.0
                self.assertAlmostEqual(single.metrics[name], original[name], delta=tolerance)
                self.assertAlmostEqual(batched.metrics[name], original[name], delta=tolerance)


if __name__ == "__main__":
    unittest.main()
