"""Offline tests for the diagnosis and treatment-plan judge panels.

Run directly: python tests/test_med_copilot_judges.py
Or discover: python -m unittest discover -s tests -p 'test_*.py'
Judge models are fakes; no API or local model is called.
"""

import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from med_copilot.prompts.correctness_judges import PLAN_JUDGE_TEMPLATE
from scripts.med_copilot_judges import (
    JUDGE_FILE, CorrectnessJudges, JudgeCache, diagnosis_entry, parse_plan_judgement,
    plan_entry, plan_messages, rank_weight, unavailable_plan_reason,
)
from scripts.med_copilot_run_file import RunFile, error_scope, pipeline_stage

MODELS = ("gpt_4o_mini", "ollama_healthcare", "ollama_llama3", "ollama_gemma2_2b")
LABEL_CONDITION = "acute viral pharyngitis"
PATIENT = {"subjective": "Sore throat for 2 days.", "objective": "Temperature 38.1 °C."}


def response(content, refusal=None):
    return SimpleNamespace(id="resp", model="judge-model", usage=None, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content, refusal=refusal))])


class FakeModels:
    """Answer judge requests from per-judge replies; record every request."""

    def __init__(self, replies):
        self.replies = replies
        self.requests = []

    def __call__(self, model_id, messages):
        self.requests.append((model_id, messages))
        reply = self.replies[model_id]
        reply = reply(messages) if callable(reply) else reply
        if isinstance(reply, BaseException):
            raise reply
        return response(reply)


def scores(*values):
    return {model: json.dumps({"score": value}) for model, value in zip(MODELS[1:], values)}


def plans(*judgements):
    return {model: json.dumps({"credits": credits, "deduction": deduction, "strength_mismatch_items": []})
            for model, (credits, deduction) in zip(MODELS[1:], judgements)}


class JudgeTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cache = self.directory / JUDGE_FILE

    def judges(self, replies, models=MODELS):
        fake = FakeModels(replies)
        return CorrectnessJudges(fake, models, self.cache), fake

    def diagnosis(self, replies, generated="viral pharyngitis", rank=1):
        judges, fake = self.judges(replies)
        entry = diagnosis_entry(rank, LABEL_CONDITION, generated)
        judges.judge_diagnosis(entry, MODELS[0])
        return entry, fake

    def plan(self, replies, reference=("Throat culture (procedure)",), generated=("Throat culture",), rank=1):
        judges, fake = self.judges(replies)
        entry = plan_entry(rank, list(reference), None if generated is None else list(generated))
        judges.judge_plan(entry, MODELS[0], LABEL_CONDITION, PATIENT)
        return entry, fake


class PromptTests(JudgeTestCase):
    def test_plan_template_is_the_rubric_judge_prompt_verbatim(self):
        doc = (ROOT / "med_copilot/docs/treatment_plan_evaluation.md").read_text(encoding="utf-8")
        block = re.search(r"## Judge prompt\n\n```text\n(.*?)\n```", doc, re.S).group(1)
        fields = {name: "{" + name + "}" for name in
                  ("condition", "subjective", "objective", "reference_plan", "generated_plan")}
        self.assertEqual(PLAN_JUDGE_TEMPLATE.format(**fields), block)

    def test_plan_actions_stay_separate_and_numbered(self):
        [message] = plan_messages(LABEL_CONDITION, "S", "O", ["Amlodipine 5 mg", "Lifestyle education"],
                                  ["Diet, exercise", "ECG"])
        self.assertEqual(message["role"], "user")
        self.assertIn("Reference plan: 1. Amlodipine 5 mg\n2. Lifestyle education\n", message["content"])
        self.assertIn("Generated plan: 1. Diet, exercise\n2. ECG\n", message["content"])

    def test_judges_never_see_generator_variant_or_rank(self):
        replies = {**scores(1, 1, 1)}
        _, fake = self.diagnosis(replies, rank=3)
        _, plan_fake = self.plan(plans(([1], 0), ([1], 0), ([1], 0)), rank=2)
        for _, messages in fake.requests + plan_fake.requests:
            text = json.dumps(messages)
            for hidden in (*MODELS, "patient_retrieval", "guidelines", "llm_only", "rank 2", "rank 3",
                           "Rank:", "variant"):
                self.assertNotIn(hidden, text)

    def test_diagnosis_prompt_contains_only_the_reference_and_generated_diagnosis(self):
        _, fake = self.diagnosis(scores(1, 1, 1), generated="Pharyngitis, viral")
        prompt = fake.requests[0][1][0]["content"]
        self.assertIn(f"Reference condition: {LABEL_CONDITION}\n", prompt)
        self.assertIn("Generated diagnosis: Pharyngitis, viral\n", prompt)
        self.assertNotIn(PATIENT["subjective"], prompt)


class PanelTests(JudgeTestCase):
    def test_each_panel_is_the_three_other_selected_models(self):
        judges, _ = self.judges({})
        for generator in MODELS:
            self.assertEqual(judges.panel(generator), [model for model in MODELS if model != generator])

    def test_panel_requires_exactly_three_other_selected_models(self):
        self.assertIsNone(self.judges({}, MODELS[:3])[0].panel(MODELS[0]))
        self.assertIsNone(self.judges({})[0].panel("unselected_model"))
        entry, fake = self.diagnosis(scores(1, 1, 1))
        judges, fake = self.judges({}, MODELS[:2])
        entry = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
        judges.judge_diagnosis(entry, MODELS[0])
        self.assertEqual((entry["status"], entry["reason"], fake.requests), ("unavailable", "panel_unavailable", []))

    def test_generator_is_excluded_from_every_request(self):
        _, fake = self.diagnosis(scores(1, 1, 1))
        self.assertEqual([model for model, _ in fake.requests], list(MODELS[1:]))


class DiagnosisTests(JudgeTestCase):
    def test_synonym_judged_equivalent_scores_one(self):
        entry, _ = self.diagnosis(scores(1, 1, 1), generated="Viral sore throat")
        self.assertEqual((entry["status"], entry["score"], entry["weighted_score"]), ("scored", 1.0, 1.0))

    def test_partial_match_takes_the_median_of_three(self):
        entry, _ = self.diagnosis(scores(0.5, 1, 0.5), generated="acute pharyngitis")
        self.assertEqual(entry["score"], 0.5)
        self.assertEqual([judge["score"] for judge in entry["judges"]], [0.5, 1.0, 0.5])

    def test_incorrect_diagnosis_scores_zero_even_if_one_judge_disagrees(self):
        entry, _ = self.diagnosis(scores(0, 1, 0), generated="Anemia")
        self.assertEqual((entry["score"], entry["weighted_score"]), (0.0, 0.0))

    def test_rank_weights_divide_the_unweighted_score(self):
        for rank, weight in ((1, 1.0), (2, 0.5), (3, 1 / 3)):
            entry, _ = self.diagnosis(scores(1, 1, 0.5), rank=rank)
            self.assertEqual(entry["rank_weight"], weight)
            self.assertEqual(entry["score"], 1.0)
            self.assertAlmostEqual(entry["weighted_score"], 1.0 / rank)
        self.assertAlmostEqual(rank_weight(3), 0.333333, places=6)

    def test_raw_replies_and_parsed_scores_are_kept_per_judge(self):
        replies = {**scores(1, 1, 1), MODELS[2]: '{"score": 1, "reason": "same disease"}'}
        entry, _ = self.diagnosis(replies)
        self.assertEqual([judge["judge_model"] for judge in entry["judges"]], list(MODELS[1:]))
        self.assertEqual(entry["judges"][1]["raw_response"], '{"score": 1, "reason": "same disease"}')
        self.assertEqual(entry["judges"][1]["model_response"]["response_id"], "resp")
        self.assertTrue(all(len(judge["request_key"]) == 64 for judge in entry["judges"]))

    def test_missing_or_blank_generated_diagnosis_is_unavailable_without_calls(self):
        for generated in (None, "  "):
            entry, fake = self.diagnosis(scores(1, 1, 1), generated=generated)
            self.assertEqual((entry["status"], entry["reason"], fake.requests),
                             ("unavailable", "missing_generation", []))


class PlanTests(JudgeTestCase):
    def test_rubric_worked_examples(self):
        cases = [
            (["Throat culture"], [[1]] * 3, [0] * 3, 1.0),
            (["Amlodipine", "Lifestyle education"], [[0, 1]] * 3, [0] * 3, 0.5),
            (["ECG", "Stress testing", "NYHA classification"], [[1, 0, 0]] * 3, [0] * 3, 1 / 3),
            (["Throat culture"], [[1]] * 3, [0.5] * 3, 0.5),
        ]
        for reference, credits, deductions, expected in cases:
            with self.subTest(reference=reference, deduction=deductions[0]):
                self.cache.unlink(missing_ok=True)  # Identical requests would reuse cached replies.
                entry, _ = self.plan(plans(*zip(credits, deductions)), reference=reference)
                self.assertEqual(entry["status"], "scored")
                self.assertAlmostEqual(entry["score"], expected)

    def test_item_and_deduction_medians_are_taken_separately(self):
        entry, _ = self.plan(plans(([1, 0], 0), ([0.5, 0], 0.5), ([1, 1], 1)),
                             reference=["Culture", "Rest"], generated=["Culture", "Antibiotics"])
        self.assertEqual(entry["median_credits"], [1.0, 0.0])
        self.assertEqual(entry["median_deduction"], 0.5)
        self.assertEqual(entry["score"], 0.25)

    def test_unsupported_extras_reduce_the_score_to_a_zero_floor(self):
        entry, _ = self.plan(plans(([0.5], 1), ([0.5], 1.5), ([0], 1)))
        self.assertEqual((entry["score"], entry["weighted_score"]), (0.0, 0.0))

    def test_weighted_plan_score_uses_the_rank(self):
        entry, _ = self.plan(plans(([1], 0), ([1], 0), ([1], 0)), rank=2)
        self.assertEqual((entry["score"], entry["weighted_score"], entry["rank_weight"]), (1.0, 0.5, 0.5))

    def test_explicitly_empty_generated_plan_scores_zero_without_calls(self):
        entry, fake = self.plan(plans(), reference=["Culture", "Rest"], generated=[])
        self.assertEqual((entry["status"], entry["reason"], entry["score"], fake.requests),
                         ("scored", "empty_generated_plan", 0.0, []))
        self.assertEqual(entry["median_credits"], [0.0, 0.0])

    def test_unavailable_references_are_never_judged(self):
        for reference, reason in ((["none recorded"], "no_recorded_plan"), ([], "missing_reference"),
                                  (["  "], "missing_reference"),
                                  (["COPD clinical management plan (record artifact)"], "record_artifact_plan"),
                                  (["Inhaler", "COPD plan (record artifact)"], "record_artifact_plan"),
                                  (["COPD plan (Record Artifact)"], "record_artifact_plan"),
                                  (["COPD plan ( RECORD   artifact )"], "record_artifact_plan"),
                                  (["COPD plan (record\tartifact)"], "record_artifact_plan")):
            with self.subTest(reference=reference):
                entry, fake = self.plan(plans(), reference=reference, generated=[])
                self.assertEqual((entry["status"], entry["reason"], entry["score"], fake.requests),
                                 ("unavailable", reason, None, []))
                self.assertEqual(unavailable_plan_reason(reference), reason)
        self.assertIsNone(unavailable_plan_reason(["None recorded today"]))
        self.assertIsNone(unavailable_plan_reason(["Review the record (artifacts excluded)"]))

    def test_missing_generated_plan_is_unavailable(self):
        entry, fake = self.plan(plans(), generated=None)
        self.assertEqual((entry["status"], entry["reason"], fake.requests), ("unavailable", "missing_generation", []))


class MalformedJudgementTests(JudgeTestCase):
    def test_invalid_replies_fail_the_panel_and_keep_the_raw_text(self):
        valid = json.dumps({"credits": [1], "deduction": 0, "strength_mismatch_items": []})
        cases = {
            "not_json": "Score: 1", "fenced": f"```json\n{valid}\n```", "array": "[1]", "empty": "",
            "missing_key": '{"credits": [1], "deduction": 0}', "wrong_credit": '{"credits": [0.7], "deduction": 0, "strength_mismatch_items": []}',
            "too_many_credits": '{"credits": [1, 1], "deduction": 0, "strength_mismatch_items": []}',
            "bool_credit": '{"credits": [true], "deduction": 0, "strength_mismatch_items": []}',
            "odd_deduction": '{"credits": [1], "deduction": 0.3, "strength_mismatch_items": []}',
            "negative_deduction": '{"credits": [1], "deduction": -0.5, "strength_mismatch_items": []}',
            "mismatch_out_of_range": '{"credits": [1], "deduction": 0, "strength_mismatch_items": [2]}',
            "duplicate_key": '{"credits": [1], "credits": [0], "deduction": 0, "strength_mismatch_items": []}',
            "nonfinite": '{"credits": [1], "deduction": NaN, "strength_mismatch_items": []}',
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                self.cache.unlink(missing_ok=True)
                entry, _ = self.plan({MODELS[1]: valid, MODELS[2]: content, MODELS[3]: valid})
                self.assertEqual((entry["status"], entry["reason"], entry["score"]), ("error", "judge_failed", None))
                self.assertEqual(entry["judges"][1]["status"], "invalid_response")
                self.assertEqual(entry["judges"][1]["raw_response"], content)
                self.assertEqual([judge["status"] for judge in entry["judges"]],
                                 ["scored", "invalid_response", "scored"])

    def test_invalid_diagnosis_scores_and_refusals_are_invalid_responses(self):
        for content in ('{"score": 0.75}', '{"score": "1"}', '{"verdict": 1}', '{"score": true}'):
            with self.subTest(content=content):
                self.cache.unlink(missing_ok=True)
                entry, _ = self.diagnosis({**scores(1, 1, 1), MODELS[3]: content})
                self.assertEqual(entry["status"], "error")
                self.assertEqual(entry["judges"][2]["status"], "invalid_response")
        self.cache.unlink(missing_ok=True)
        judges = CorrectnessJudges(lambda model, messages: response('{"score": 1}', refusal="no"), MODELS, self.cache)
        entry = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
        judges.judge_diagnosis(entry, MODELS[0])
        self.assertEqual({judge["status"] for judge in entry["judges"]}, {"invalid_response"})

    def test_strength_mismatches_are_parsed_but_do_not_change_credit(self):
        judgement = parse_plan_judgement('{"credits": [0.5, 1], "deduction": 0, "strength_mismatch_items": [1]}', 2)
        self.assertEqual(judgement, {"credits": [0.5, 1.0], "deduction": 0.0, "strength_mismatch_items": [1]})


class JudgeFailureTests(JudgeTestCase):
    def test_failed_call_is_saved_under_the_scoring_stage_and_fails_only_that_panel(self):
        run = RunFile.begin(self.directory / "output")
        replies = {**scores(1, 1, 1), MODELS[2]: RuntimeError("judge server down")}
        judges, _ = self.judges(replies)
        entry = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
        with error_scope(run, patient_id="patient-a"), pipeline_stage("scoring"):
            judges.judge_diagnosis(entry, MODELS[0])
        self.assertEqual((entry["status"], entry["reason"]), ("error", "judge_failed"))
        self.assertEqual(entry["judges"][1]["error"], {"type": "RuntimeError", "message": "judge server down"})
        [error] = json.loads((run.run_dir / "run.json").read_text())["errors"]
        self.assertEqual((error["patient_id"], error["stage"], error["exception_type"]),
                         ("patient-a", "scoring", "JudgeCallError"))
        self.assertIn(f"diagnosis judge {MODELS[2]}", error["message"])
        self.assertIn("RuntimeError: judge server down", error["message"])
        self.assertIn("RuntimeError: judge server down", error["traceback"])  # chained cause
        # Only completed calls are cached; the failed one will be retried.
        self.assertEqual(len(JudgeCache(self.cache).entries), 2)

    def test_malformed_and_refused_replies_are_saved_once_per_patient_across_resumes(self):
        refusal = lambda messages: response('{"score": 1}', refusal="I cannot grade this")
        replies = {**scores(1, 1, 1), MODELS[1]: "not json", MODELS[3]: None}
        fake_models = FakeModels(replies)
        def complete(model_id, messages):
            if model_id == MODELS[3]:
                fake_models.requests.append((model_id, messages))
                return refusal(messages)
            return fake_models(model_id, messages)
        run = RunFile.begin(self.directory / "output")
        for _ in range(2):
            judges = CorrectnessJudges(complete, MODELS, self.cache)
            entry = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
            with error_scope(run, patient_id="patient-a"), pipeline_stage("scoring"):
                judges.judge_diagnosis(entry, MODELS[0])
            self.assertEqual([judge["status"] for judge in entry["judges"]],
                             ["invalid_response", "scored", "invalid_response"])
        errors = json.loads((run.run_dir / "run.json").read_text())["errors"]
        self.assertEqual([(e["patient_id"], e["stage"], e["exception_type"]) for e in errors],
                         [("patient-a", "scoring", "JudgeResponseError")] * 2)
        self.assertIn(f"diagnosis judge {MODELS[1]}", errors[0]["message"])
        self.assertIn("invalid JSON", errors[0]["message"])
        self.assertIn("refused", errors[1]["message"])
        # Both bad replies were requested again on the second pass; the valid one was reused.
        self.assertEqual([model for model, _ in fake_models.requests],
                         [MODELS[1], MODELS[2], MODELS[3], MODELS[1], MODELS[3]])

    def test_a_different_patient_with_the_same_failure_gets_its_own_error(self):
        run = RunFile.begin(self.directory / "output")
        for patient in ("patient-a", "patient-b"):
            judges, _ = self.judges({**scores(1, 1, 1), MODELS[2]: RuntimeError("down")})
            with error_scope(run, patient_id=patient), pipeline_stage("scoring"):
                judges.judge_diagnosis(diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis"), MODELS[0])
        errors = json.loads((run.run_dir / "run.json").read_text())["errors"]
        self.assertEqual([e["patient_id"] for e in errors], ["patient-a", "patient-b"])


class ResumptionTests(JudgeTestCase):
    def test_interrupted_panel_resumes_without_repeating_completed_calls(self):
        replies = {**plans(([1], 0), ([1], 0.5), ([0.5], 0)), MODELS[3]: KeyboardInterrupt()}
        judges, first = self.judges(replies)
        entry = plan_entry(1, ["Throat culture"], ["Throat culture", "Antibiotics"])
        with self.assertRaises(KeyboardInterrupt):
            judges.judge_plan(entry, MODELS[0], LABEL_CONDITION, PATIENT)
        self.assertEqual(entry["status"], "interrupted")
        self.assertEqual([judge["judge_model"] for judge in entry["judges"]], list(MODELS[1:3]))
        self.assertEqual(len(first.requests), 3)
        resumed, second = self.judges(plans(([1], 0), ([1], 0.5), ([0.5], 0)))
        entry = plan_entry(1, ["Throat culture"], ["Throat culture", "Antibiotics"])
        resumed.judge_plan(entry, MODELS[0], LABEL_CONDITION, PATIENT)
        self.assertEqual([model for model, _ in second.requests], [MODELS[3]])
        self.assertEqual((resumed.calls, resumed.cache_hits), (1, 2))
        self.assertEqual((entry["status"], entry["score"]), ("scored", 1.0))

    def test_malformed_reply_is_kept_for_audit_but_requested_again_on_resume(self):
        judges, _ = self.judges({**scores(1, 1, 1), MODELS[1]: "not json"})
        first = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
        judges.judge_diagnosis(first, MODELS[0])
        self.assertEqual(first["status"], "error")
        again, fake = self.judges(scores(1, 1, 1))
        entry = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
        again.judge_diagnosis(entry, MODELS[0])
        self.assertEqual([model for model, _ in fake.requests], [MODELS[1]])
        self.assertEqual((entry["status"], entry["score"]), ("scored", 1.0))
        saved = [json.loads(line) for line in self.cache.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(row["judge_model"], row["valid"]) for row in saved],
                         [(MODELS[1], False), (MODELS[2], True), (MODELS[3], True), (MODELS[1], True)])
        self.assertEqual(saved[0]["content"], "not json")
        self.assertIn("invalid JSON", saved[0]["error"]["message"])

    def test_partial_trailing_cache_line_is_dropped_and_appends_stay_valid(self):
        judges, _ = self.judges(scores(1, 1, 1))
        judges.judge_diagnosis(diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis"), MODELS[0])
        with self.cache.open("a", encoding="utf-8") as handle:
            handle.write('{"key": "torn')
        cache = JudgeCache(self.cache)
        self.assertEqual(len(cache.entries), 3)
        self.assertTrue(self.cache.read_text(encoding="utf-8").endswith("\n"))

    def test_identical_requests_to_the_same_judge_are_asked_once(self):
        judges, fake = self.judges(scores(1, 1, 1))
        for rank in (1, 2):
            judges.judge_diagnosis(diagnosis_entry(rank, LABEL_CONDITION, "viral pharyngitis"), MODELS[0])
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(judges.cache_hits, 3)


class MetadataTests(JudgeTestCase):
    def test_metadata_records_method_models_rubric_and_rank_weights(self):
        metadata = self.judges({})[0].metadata(ROOT)
        self.assertEqual(metadata["selected_models"], list(MODELS))
        self.assertEqual(metadata["judge_temperature"], 0)
        self.assertEqual(metadata["hidden_from_judges"], ["generating_model", "evidence_variant", "rank"])
        self.assertEqual(metadata["plan"]["rubric_file"], "med_copilot/docs/treatment_plan_evaluation.md")
        self.assertEqual(len(metadata["plan"]["rubric_sha256"]), 64)
        self.assertEqual(metadata["diagnosis"]["scale"]["0.5"],
                         "clinically related but meaningfully broader, narrower, or missing an important qualifier")
        self.assertEqual(metadata["rank_weighting"]["weights"], {"1": 1.0, "2": 0.5, "3": 1 / 3})
        self.assertIsNone(metadata["rank_weighting"]["combined_score"])
        json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
