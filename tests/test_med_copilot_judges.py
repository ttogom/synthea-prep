"""Offline tests for the preliminary single-model midterm correctness metric.

Run directly: python tests/test_med_copilot_judges.py
Or discover: python -m unittest discover -s tests -p 'test_*.py'
BioLORD embeddings and the Plan judge are fakes; no API or local model is called, except that
BioLordSanityOrderingTests uses the pinned BioLORD-2023 when it is already cached (never downloads).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.med_copilot_config import PIPELINE_DEFAULTS
from med_copilot.prompts.correctness_judges import (
    PLAN_ACTION_JUDGE_TEMPLATE, PLAN_EXTRAS_JUDGE_TEMPLATE, PLAN_JUDGE_TEMPLATE,
)
from scripts.med_copilot_judges import (
    BIOLORD_MODEL, BIOLORD_REVISION, DIAGNOSIS_ABBREVIATIONS, EXPERIMENTAL_UNVALIDATED, JUDGE_FILE,
    METRIC_STATUS, PLAN_VALIDATION_STATUS, BioLordSimilarity,
    CorrectnessJudges, JudgeCache, diagnosis_entry, diagnosis_score, normalize_diagnosis, parse_action_judgement,
    parse_extras_judgement,
    plan_action_messages, plan_entry, plan_extras_messages, plan_score, rank_weight, request_key, unavailable_plan_reason,
)
from scripts.med_copilot_run_file import RunFile, RunMetadataError, error_scope, pipeline_stage

DEFAULT_PLAN_JUDGE = PIPELINE_DEFAULTS["plan_judge"]
MODELS = ("gpt_4o_mini", "ollama_healthcare", "ollama_llama3", "ollama_gemma2_2b")
JUDGE = MODELS[2]
LABEL_CONDITION = "acute viral pharyngitis"
PATIENT = {"subjective": "Sore throat for 2 days.", "objective": "Temperature 38.1 °C."}
# Fake 3-dimensional embeddings with known cosines to the reference.
VECTORS = {
    LABEL_CONDITION: [1.0, 0.0, 0.0],
    "viral pharyngitis": [0.8, 0.6, 0.0],          # cosine 0.8
    "Viral sore throat": [2.0, 0.0, 0.0],          # cosine 1.0 (norm differs)
    "Anemia": [0.0, 0.0, 1.0],                     # cosine 0
    "Limited social contact": [-0.6, 0.8, 0.0],    # cosine -0.6, clipped to 0
}


def response(content, refusal=None):
    return SimpleNamespace(id="resp", model="judge-model", usage=None, choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content, refusal=refusal))])


def is_extras(messages):
    return messages[0]["content"].startswith("Count the distinct generated actions")


def reference_action(messages):
    return re.search(r"Reference action: (.*)\n", messages[0]["content"]).group(1)


def replies(credits=None, extras=0, action_content=None, extras_content=None):
    """Judge replies: per-action credits (default 1) and an unsupported-extra count."""
    def reply(messages):
        if is_extras(messages):
            return extras_content if extras_content is not None else json.dumps(
                {"unsupported_extra_count": extras})
        action = reference_action(messages)
        if isinstance(action_content, dict) and action in action_content:
            return action_content[action]
        return json.dumps({"credit": (credits or {}).get(action, 1)})
    return reply


class FakeJudge:
    """Answer judge requests; record every (model, messages) request."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def __call__(self, model_id, messages):
        self.requests.append((model_id, messages))
        value = self.reply(messages)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, SimpleNamespace) else response(value)


class FakeEncoder:
    def __init__(self, vectors=VECTORS, error=None):
        self.vectors, self.error, self.calls = vectors, error, []

    def __call__(self, texts):
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return [self.vectors[text] for text in texts]


class JudgeTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cache = self.directory / JUDGE_FILE

    def judges(self, reply=None, models=MODELS, encoder=None):
        fake = FakeJudge(reply or replies())
        encoder = encoder or FakeEncoder()
        judges = CorrectnessJudges(fake, models, self.cache, similarity=BioLordSimilarity(encoder),
                                   plan_judge=JUDGE)
        return judges, fake, encoder

    def diagnosis(self, generated="viral pharyngitis", rank=1, encoder=None):
        judges, fake, encoder = self.judges(encoder=encoder)
        entry = diagnosis_entry(rank, LABEL_CONDITION, generated)
        judges.score_diagnosis(entry)
        return entry, encoder, fake

    def plan(self, reply=None, reference=("Throat culture (procedure)",), generated=("Throat culture",),
             rank=1, models=MODELS):
        judges, fake, _ = self.judges(reply, models)
        entry = plan_entry(rank, list(reference), None if generated is None else list(generated))
        judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
        return entry, fake


class PromptTests(JudgeTestCase):
    def test_rubric_prompt_is_kept_verbatim_for_reference(self):
        doc = (ROOT / "med_copilot/docs/treatment_plan_evaluation.md").read_text(encoding="utf-8")
        block = re.search(r"## Judge prompt\n\n```text\n(.*?)\n```", doc, re.S).group(1)
        fields = {name: "{" + name + "}" for name in
                  ("condition", "subjective", "objective", "reference_plan", "generated_plan")}
        self.assertEqual(PLAN_JUDGE_TEMPLATE.format(**fields), block)

    def test_action_prompt_uses_the_rubric_scale_verbatim(self):
        scale = re.search(r"(Score each reference action:\n.*?counseling\.\n)", PLAN_JUDGE_TEMPLATE, re.S).group(1)
        self.assertIn(scale.replace("Score each reference action:", "Score the reference action:"),
                      PLAN_ACTION_JUDGE_TEMPLATE)

    def test_action_prompt_shows_one_reference_action_and_numbered_generated_actions(self):
        [message] = plan_action_messages(LABEL_CONDITION, "S", "O", "Amlodipine 5 mg", ["Diet, exercise", "ECG"])
        self.assertEqual(message["role"], "user")
        self.assertIn("Reference action: Amlodipine 5 mg\n", message["content"])
        self.assertIn("Generated plan: 1. Diet, exercise\n2. ECG\n", message["content"])
        self.assertIn('Valid outputs are {"credit": 0}, {"credit": 0.5}, or {"credit": 1}.', message["content"])

    def test_extras_prompt_requires_a_nonnegative_integer_and_numbers_both_plans(self):
        [message] = plan_extras_messages(LABEL_CONDITION, "S", "O", ["Amlodipine", "Lifestyle education"],
                                         ["Diet", "ECG", "Antibiotics"])
        self.assertIn("Reference plan: 1. Amlodipine\n2. Lifestyle education\n", message["content"])
        self.assertIn("Generated plan: 1. Diet\n2. ECG\n3. Antibiotics\n", message["content"])
        self.assertIn('{"unsupported_extra_count": N}, where N is a nonnegative integer.', message["content"])
        self.assertIn("bundled actions, count each distinct unsupported action once", message["content"])
        self.assertNotIn("from 0 to", message["content"])

    def test_prompts_never_contain_example_credit_arrays(self):
        for template in (PLAN_ACTION_JUDGE_TEMPLATE, PLAN_EXTRAS_JUDGE_TEMPLATE):
            self.assertNotIn('"credits"', template)

    def test_judge_never_sees_generator_variant_or_rank(self):
        _, fake = self.plan(rank=2)
        for _, messages in fake.requests:
            text = json.dumps(messages)
            for hidden in (*MODELS, "patient_retrieval", "guidelines", "llm_only", "rank 2",
                           "Rank:", "variant"):
                self.assertNotIn(hidden, text)


class BioLordDiagnosisTests(JudgeTestCase):
    def test_score_is_the_cosine_similarity_and_weight_divides_by_rank(self):
        for rank in (1, 2, 3):
            entry, _, _ = self.diagnosis(rank=rank)
            self.assertEqual((entry["status"], entry["method"]), ("scored", "biolord_2023_cosine"))
            self.assertAlmostEqual(entry["cosine_similarity"], 0.8)
            self.assertAlmostEqual(entry["score"], 0.8)
            self.assertAlmostEqual(entry["weighted_score"], 0.8 / rank)
        self.assertAlmostEqual(rank_weight(3), 0.333333, places=6)

    def test_cosine_ignores_vector_length(self):
        entry, _, _ = self.diagnosis(generated="Viral sore throat")
        self.assertAlmostEqual(entry["score"], 1.0)

    def test_negative_similarity_is_clipped_to_zero_and_raw_cosine_kept(self):
        entry, _, _ = self.diagnosis(generated="Limited social contact", rank=2)
        self.assertAlmostEqual(entry["cosine_similarity"], -0.6)
        self.assertEqual((entry["score"], entry["weighted_score"]), (0.0, 0.0))
        entry, _, _ = self.diagnosis(generated="Anemia")
        self.assertEqual(entry["score"], 0.0)

    def test_embeddings_are_cached_and_no_judge_is_asked(self):
        judges, fake, encoder = self.judges()
        for rank in (1, 2, 3):
            judges.score_diagnosis(diagnosis_entry(rank, LABEL_CONDITION, "viral pharyngitis"))
        judges.score_diagnosis(diagnosis_entry(1, LABEL_CONDITION, "Anemia"))
        self.assertEqual(encoder.calls, [[LABEL_CONDITION, "viral pharyngitis"], ["Anemia"]])
        self.assertEqual(fake.requests, [])

    def test_missing_or_blank_generated_diagnosis_is_unavailable_without_encoding(self):
        for generated in (None, "  "):
            entry, encoder, _ = self.diagnosis(generated=generated)
            self.assertEqual((entry["status"], entry["reason"], entry["score"], encoder.calls),
                             ("unavailable", "missing_generation", None, []))

    def test_similarity_failure_is_an_error_saved_under_the_scoring_stage(self):
        run = RunFile.begin(self.directory / "output")
        for encoder in (FakeEncoder(error=OSError("model download failed")),
                        FakeEncoder(vectors={LABEL_CONDITION: [1.0, 0.0], "viral pharyngitis": [0.0, 0.0]})):
            entry = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
            judges, _, _ = self.judges(encoder=encoder)
            with error_scope(run, patient_id="patient-a"), pipeline_stage("scoring"):
                judges.score_diagnosis(entry)
            self.assertEqual((entry["status"], entry["reason"], entry["score"]),
                             ("error", "similarity_failed", None))
        errors = json.loads((run.run_dir / "run.json").read_text())["errors"]
        self.assertEqual([(e["patient_id"], e["stage"], e["exception_type"]) for e in errors],
                         [("patient-a", "scoring", "SimilarityError")] * 2)
        self.assertIn("OSError: model download failed", errors[0]["message"])
        self.assertIn("OSError: model download failed", errors[0]["traceback"])  # chained cause

    def test_default_similarity_loads_the_pinned_biolord_revision_on_first_use(self):
        loaded = []

        class FakeSentenceTransformer:
            def __init__(self, name, revision=None):
                loaded.append((name, revision))

            def encode(self, texts, convert_to_numpy):
                return [VECTORS[text] for text in texts]

        module = ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeSentenceTransformer
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            similarity = BioLordSimilarity()
            self.assertEqual(loaded, [])
            self.assertAlmostEqual(similarity.cosine(LABEL_CONDITION, "viral pharyngitis"), 0.8)
            similarity.cosine(LABEL_CONDITION, "Anemia")
        self.assertEqual(loaded, [(BIOLORD_MODEL, BIOLORD_REVISION)])


class DiagnosisNormalizationTests(JudgeTestCase):
    # The four sanity pairs: the generated text BioLORD encodes after normalization.
    SANITY_PAIRS = [("essential hypertension", "HTN", "hypertension"),
                    ("chronic congestive heart failure", "CHF", "congestive heart failure"),
                    ("chronic obstructive pulmonary disease", "COPD", "chronic obstructive pulmonary disease"),
                    ("essential hypertension", "Hypertension", "Hypertension")]

    def test_abbreviations_expand_as_whole_words_in_any_case(self):
        self.assertEqual(DIAGNOSIS_ABBREVIATIONS, {"HTN": "hypertension", "CHF": "congestive heart failure",
                                                   "COPD": "chronic obstructive pulmonary disease"})
        for reference, generated, normalized in self.SANITY_PAIRS:
            self.assertEqual(normalize_diagnosis(generated), normalized)
            self.assertEqual(normalize_diagnosis(reference), reference)
        self.assertEqual(normalize_diagnosis("  uncontrolled htn,  with CHF "),
                         "uncontrolled hypertension, with congestive heart failure")
        self.assertEqual(normalize_diagnosis("copd exacerbation"), "chronic obstructive pulmonary disease exacerbation")
        for unchanged in ("HTNX", "CHFs", "pre-COPDish", "hypertension"):
            self.assertEqual(normalize_diagnosis(unchanged), unchanged)

    def test_entry_keeps_original_and_normalized_text_and_encodes_the_normalized_text(self):
        encoder = FakeEncoder({"essential hypertension": [1.0, 0.0], "hypertension": [0.6, 0.8]})
        judges, _, _ = self.judges(encoder=encoder)
        entry = diagnosis_entry(2, "essential hypertension", " HTN ")
        judges.score_diagnosis(entry)
        self.assertEqual((entry["reference"], entry["generated"]), ("essential hypertension", " HTN "))
        self.assertEqual((entry["normalized_reference"], entry["normalized_generated"]),
                         ("essential hypertension", "hypertension"))
        self.assertEqual(encoder.calls, [["essential hypertension", "hypertension"]])
        self.assertAlmostEqual(entry["score"], 0.6)
        self.assertAlmostEqual(entry["weighted_score"], 0.3)

    def test_metadata_documents_the_normalization(self):
        normalization = self.judges()[0].metadata(ROOT)["diagnosis"]["normalization"]
        self.assertEqual(normalization["abbreviations"], DIAGNOSIS_ABBREVIATIONS)
        self.assertIn("whole word in any case", normalization["rule"])

    def test_original_sanity_pairs_have_no_abbreviations_to_expand(self):
        for first, second, _ in ORIGINAL_SANITY_PAIRS:
            self.assertEqual((normalize_diagnosis(first), normalize_diagnosis(second)), (first, second))


# The four original BioLORD sanity pairs: (text, text, related?).
ORIGINAL_SANITY_PAIRS = [("heart failure", "congestive heart failure", True),
                         ("heart failure", "anemia", False),
                         ("acute viral pharyngitis", "acute pharyngitis", True),
                         ("essential hypertension", "hypertension", True)]
# (labeled condition, abbreviated generated diagnosis); normalization must raise the similarity.
ABBREVIATION_PAIRS = [("essential hypertension", "HTN"), ("chronic congestive heart failure", "CHF"),
                      ("chronic obstructive pulmonary disease", "COPD")]


def _cached_biolord():
    """The pinned BioLORD-2023 revision from the local Hugging Face cache, or None (never downloads)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(BIOLORD_MODEL, revision=BIOLORD_REVISION, local_files_only=True)
    except Exception:
        return None
    return BioLordSimilarity(lambda texts: model.encode(texts, convert_to_numpy=True))


class BioLordSanityOrderingTests(unittest.TestCase):
    """Real-model regression on orderings, not exact floats; skipped unless BioLORD is cached locally."""

    @classmethod
    def setUpClass(cls):
        cls.similarity = _cached_biolord()
        if cls.similarity is None:
            raise unittest.SkipTest(f"{BIOLORD_MODEL}@{BIOLORD_REVISION} is not in the local cache")

    def score(self, first, second):
        return diagnosis_score(self.similarity.cosine(normalize_diagnosis(first), normalize_diagnosis(second)))

    def test_every_related_pair_scores_above_every_incorrect_pair(self):
        related = {pair[:2]: self.score(*pair[:2]) for pair in ORIGINAL_SANITY_PAIRS if pair[2]}
        incorrect = {pair[:2]: self.score(*pair[:2]) for pair in ORIGINAL_SANITY_PAIRS if not pair[2]}
        for pair, value in related.items():
            for other, lower in incorrect.items():
                self.assertGreater(value, lower, f"{pair} should score above {other}")

    def test_normalization_raises_each_abbreviation_to_at_least_the_incorrect_pair(self):
        incorrect = self.score("heart failure", "anemia")
        for reference, abbreviation in ABBREVIATION_PAIRS:
            raw = diagnosis_score(self.similarity.cosine(reference, abbreviation))
            normalized = self.score(reference, abbreviation)
            self.assertGreaterEqual(normalized, raw, abbreviation)
            self.assertGreater(normalized, incorrect, abbreviation)
        self.assertGreater(self.score("essential hypertension", "HTN"),
                           self.score("essential hypertension", "chronic low back pain"))


class PlanTests(JudgeTestCase):
    def test_rubric_worked_examples(self):
        cases = [
            (["Throat culture"], {}, 0, 1.0),
            (["Amlodipine", "Lifestyle education"], {"Amlodipine": 0}, 0, 0.5),
            (["ECG", "Stress testing", "NYHA classification"], {"Stress testing": 0, "NYHA classification": 0}, 0, 1 / 3),
            (["Throat culture"], {}, 1, 0.5),
        ]
        for reference, credits, extras, expected in cases:
            with self.subTest(reference=reference, extras=extras):
                self.cache.unlink(missing_ok=True)  # Identical requests would reuse cached replies.
                entry, _ = self.plan(replies(credits, extras), reference=reference,
                                     generated=["Throat culture", "Antibiotics"])
                self.assertEqual(entry["status"], "scored")
                self.assertAlmostEqual(entry["score"], expected)
                self.assertEqual(entry["deduction"], 0.5 * extras)

    def test_one_call_per_reference_action_then_one_extras_call_to_the_single_judge(self):
        entry, fake = self.plan(replies({"Rest": 0.5}, 1), reference=["Culture", "Rest"],
                                generated=["Culture", "Antibiotics", "Imaging"])
        self.assertEqual({model for model, _ in fake.requests}, {JUDGE})
        self.assertEqual([("extras" if is_extras(m) else reference_action(m)) for _, m in fake.requests],
                         ["Culture", "Rest", "extras"])
        self.assertEqual([(judge["call"], judge["reference_item"]) for judge in entry["judges"]],
                         [("action", 1), ("action", 2), ("extras", None)])
        self.assertEqual((entry["credits"], entry["unsupported_extra_count"], entry["deduction"]),
                         ([1.0, 0.5], 1, 0.5))
        self.assertAlmostEqual(entry["score"], 0.5)
        self.assertTrue(all(len(judge["request_key"]) == 64 for judge in entry["judges"]))
        self.assertEqual(entry["judges"][1]["raw_response"], '{"credit": 0.5}')

    def test_unsupported_extras_reduce_the_score_to_a_zero_floor(self):
        entry, _ = self.plan(replies({"Throat culture (procedure)": 0.5}, 2), generated=["A", "B", "C"])
        self.assertEqual((entry["score"], entry["weighted_score"], entry["deduction"]), (0.0, 0.0, 1.0))

    def test_bundled_extras_can_exceed_the_number_of_plan_items(self):
        entry, _ = self.plan(replies(extras=2), reference=["Culture", "Rest"],
                             generated=["Culture; rest; antibiotics; imaging"], rank=2)
        self.assertEqual(entry["status"], "scored")
        self.assertEqual((entry["unsupported_extra_count"], entry["deduction"]), (2, 1.0))
        self.assertEqual((entry["score"], entry["weighted_score"]), (0.5, 0.25))

    def test_weighted_plan_score_uses_the_rank(self):
        entry, _ = self.plan(rank=2)
        self.assertEqual((entry["score"], entry["weighted_score"], entry["rank_weight"]), (1.0, 0.5, 0.5))

    def test_plan_judge_outside_the_selected_models_is_unavailable_without_calls(self):
        entry, fake = self.plan(models=("gpt_4o_mini", "ollama_healthcare"))
        self.assertEqual((entry["status"], entry["reason"], fake.requests), ("unavailable", "judge_unavailable", []))

    def test_explicitly_empty_generated_plan_scores_zero_without_calls(self):
        entry, fake = self.plan(reference=["Culture", "Rest"], generated=[])
        self.assertEqual((entry["status"], entry["reason"], entry["score"], fake.requests),
                         ("scored", "empty_generated_plan", 0.0, []))
        self.assertEqual((entry["credits"], entry["deduction"]), ([0.0, 0.0], 0.0))

    def test_unavailable_references_are_never_judged(self):
        for reference, reason in ((["none recorded"], "no_recorded_plan"), ([], "missing_reference"),
                                  (["  "], "missing_reference"),
                                  (["COPD clinical management plan (record artifact)"], "record_artifact_plan"),
                                  (["Inhaler", "COPD plan (record artifact)"], "record_artifact_plan"),
                                  (["COPD plan (Record Artifact)"], "record_artifact_plan"),
                                  (["COPD plan ( RECORD   artifact )"], "record_artifact_plan"),
                                  (["COPD plan (record\tartifact)"], "record_artifact_plan")):
            with self.subTest(reference=reference):
                entry, fake = self.plan(reference=reference, generated=[])
                self.assertEqual((entry["status"], entry["reason"], entry["score"], fake.requests),
                                 ("unavailable", reason, None, []))
                self.assertEqual(unavailable_plan_reason(reference), reason)
        self.assertIsNone(unavailable_plan_reason(["None recorded today"]))
        self.assertIsNone(unavailable_plan_reason(["Review the record (artifacts excluded)"]))

    def test_missing_generated_plan_is_unavailable(self):
        entry, fake = self.plan(generated=None)
        self.assertEqual((entry["status"], entry["reason"], fake.requests), ("unavailable", "missing_generation", []))


class ZeroCoverageTests(JudgeTestCase):
    def test_all_valid_zero_credits_skip_extras_and_keep_them_unmeasured(self):
        for count in (1, 2, 3):
            for rank in (1, 2, 3):
                with self.subTest(reference_count=count, rank=rank):
                    self.cache.unlink(missing_ok=True)
                    reference = [f"Action {number}" for number in range(count)]
                    zero = replies(dict.fromkeys(reference, 0))
                    def reply(messages):
                        if is_extras(messages):
                            raise AssertionError("Extras cannot change an all-zero score")
                        return zero(messages)
                    entry, fake = self.plan(reply, reference=reference, generated=["Other action"], rank=rank)
                    self.assertEqual((entry["status"], entry["reason"], entry["score"], entry["weighted_score"]),
                                     ("scored", "zero_reference_coverage", 0.0, 0.0))
                    self.assertEqual(entry["credits"], [0.0] * count)
                    self.assertIsNone(entry["unsupported_extra_count"])
                    self.assertIsNone(entry["deduction"])
                    self.assertIsNone(entry["error"])
                    self.assertEqual(len(fake.requests), count)
                    self.assertEqual([(j["call"], j["reference_item"]) for j in entry["judges"]],
                                     [("action", number) for number in range(1, count + 1)])
                    cached = [json.loads(line) for line in self.cache.read_text().splitlines()]
                    self.assertEqual([row["task"] for row in cached], ["plan_action"] * count)
                    # Holds even when a bundled generated item has many unsupported actions.
                    for extras in (0, 1, 2, 3, 100):
                        self.assertEqual(entry["score"], plan_score([0.0] * count, 0.5 * extras))

    def test_any_positive_credit_still_requests_extras_including_the_last_reference(self):
        for credits in ((0, 0.5), (1, 0), (0, 0, 0.5)):
            with self.subTest(credits=credits):
                self.cache.unlink(missing_ok=True)
                reference = [f"Action {number}" for number in range(len(credits))]
                entry, fake = self.plan(replies(dict(zip(reference, credits)), extras=1),
                                        reference=reference, generated=["Other action"])
                self.assertEqual(len(fake.requests), len(credits) + 1)
                self.assertTrue(is_extras(fake.requests[-1][1]))
                self.assertIsNone(entry["reason"])
                self.assertEqual((entry["unsupported_extra_count"], entry["deduction"]), (1, 0.5))
                self.assertEqual(entry["score"], plan_score(credits, 0.5))

    def test_a_failed_or_invalid_action_is_not_treated_as_zero_coverage(self):
        for failure in ('{"credit": true}', "not JSON", RuntimeError("judge down")):
            with self.subTest(failure=str(failure)):
                self.cache.unlink(missing_ok=True)
                base = replies({"Culture": 0, "Rest": 0})
                def reply(messages):
                    return failure if not is_extras(messages) and reference_action(messages) == "Culture" else base(messages)
                entry, fake = self.plan(reply, reference=["Culture", "Rest"])
                self.assertEqual((entry["status"], entry["reason"], entry["score"]),
                                 ("error", "judge_failed", None))
                self.assertEqual(len(fake.requests), 3)
                self.assertTrue(is_extras(fake.requests[-1][1]))

    def test_invalid_coverage_recovers_to_zero_without_reusing_an_old_extras_count(self):
        base = replies({"Culture": 0, "Rest": 0}, extras=2,
                       action_content={"Culture": "not JSON"})
        judges, _, _ = self.judges(base)
        first = plan_entry(1, ["Culture", "Rest"], ["Other action", "Another action"])
        judges.judge_plan(first, LABEL_CONDITION, PATIENT)
        self.assertEqual(first["status"], "error")
        self.assertEqual(first["judges"][-1]["unsupported_extra_count"], 2)
        resumed, fake, _ = self.judges(replies({"Culture": 0, "Rest": 0}))
        entry = plan_entry(1, ["Culture", "Rest"], ["Other action", "Another action"])
        resumed.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual((resumed.calls, resumed.cache_hits), (1, 1))
        self.assertEqual([reference_action(messages) for _, messages in fake.requests], ["Culture"])
        self.assertEqual((entry["status"], entry["reason"], entry["score"]),
                         ("scored", "zero_reference_coverage", 0.0))
        self.assertIsNone(entry["unsupported_extra_count"])
        self.assertIsNone(entry["deduction"])

    def test_concurrent_identical_zero_coverage_plans_share_only_their_action_requests(self):
        judges, fake, _ = self.judges(replies({"Culture": 0, "Rest": 0, "Fluids": 0}))
        entries = [plan_entry(rank, ["Culture", "Rest", "Fluids"], ["Other action"])
                   for rank in (1, 2, 3, 1, 2, 3)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda entry: judges.judge_plan(entry, LABEL_CONDITION, PATIENT), entries))
        self.assertEqual((judges.calls, judges.cache_hits, len(fake.requests)), (3, 15, 3))
        self.assertTrue(all(entry["reason"] == "zero_reference_coverage" and entry["weighted_score"] == 0
                            for entry in entries))
        self.assertTrue(all(not is_extras(messages) for _, messages in fake.requests))

    def test_interrupted_zero_coverage_resumes_from_durable_action_prefix(self):
        base = replies({"Culture": 0, "Rest": 0, "Fluids": 0})
        def interrupted(messages):
            return KeyboardInterrupt() if reference_action(messages) == "Rest" else base(messages)
        judges, _, _ = self.judges(interrupted)
        entry = plan_entry(2, ["Culture", "Rest", "Fluids"], ["Other action"])
        with self.assertRaises(KeyboardInterrupt):
            judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual(entry["status"], "interrupted")
        self.assertEqual(len(entry["judges"]), 1)
        resumed, fake, _ = self.judges(base)
        entry = plan_entry(2, ["Culture", "Rest", "Fluids"], ["Other action"])
        resumed.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual([reference_action(messages) for _, messages in fake.requests], ["Rest", "Fluids"])
        self.assertEqual((resumed.calls, resumed.cache_hits), (2, 1))
        self.assertEqual((entry["status"], entry["reason"], entry["weighted_score"]),
                         ("scored", "zero_reference_coverage", 0.0))


class MalformedJudgementTests(JudgeTestCase):
    def test_invalid_action_replies_fail_the_plan_keep_the_raw_text_and_still_ask_the_rest(self):
        cases = {
            "not_json": "Credit: 1", "fenced": '```json\n{"credit": 1}\n```', "array": "[1]", "empty": "",
            "missing_key": '{"score": 1}', "wrong_credit": '{"credit": 0.7}', "string_credit": '{"credit": "1"}',
            "bool_credit": '{"credit": true}', "duplicate_key": '{"credit": 1, "credit": 0}',
            "nonfinite": '{"credit": NaN}', "credits_array": '{"credits": [1, 0.5]}',
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                self.cache.unlink(missing_ok=True)
                entry, fake = self.plan(replies(action_content={"Culture": content}),
                                        reference=["Culture", "Rest"], generated=["Culture"])
                self.assertEqual((entry["status"], entry["reason"], entry["score"]), ("error", "judge_failed", None))
                self.assertEqual([judge["status"] for judge in entry["judges"]],
                                 ["invalid_response", "scored", "scored"])
                self.assertEqual(entry["judges"][0]["raw_response"], content)
                self.assertEqual(len(fake.requests), 3)

    def test_invalid_extras_counts_fail_the_plan(self):
        for content in ('{"unsupported_extra_count": -1}', '{"unsupported_extra_count": null}',
                        '{"unsupported_extra_count": 1.5}', '{"unsupported_extra_count": 1.0}',
                        '{"unsupported_extra_count": true}', '{"unsupported_extra_count": false}',
                        '{"unsupported_extra_count": "1"}', '{"extras": 1}'):
            with self.subTest(content=content):
                self.cache.unlink(missing_ok=True)
                entry, _ = self.plan(replies(extras_content=content), generated=["Culture", "Antibiotics"])
                self.assertEqual((entry["status"], entry["judges"][-1]["status"]), ("error", "invalid_response"))

    def test_refusals_are_invalid_responses(self):
        refusal = lambda messages: response('{"credit": 1}', refusal="I cannot grade this")
        entry, _ = self.plan(refusal)
        self.assertEqual({judge["status"] for judge in entry["judges"]}, {"invalid_response"})
        self.assertEqual(entry["status"], "error")

    def test_parsers_accept_only_the_rubric_values(self):
        self.assertEqual(parse_action_judgement('{"credit": 0.5, "reason": "less specific"}'), {"credit": 0.5})
        for count in (0, 1, 2, 1000):
            with self.subTest(count=count):
                self.assertEqual(parse_extras_judgement(json.dumps({"unsupported_extra_count": count})),
                                 {"unsupported_extra_count": count})


class JudgeFailureTests(JudgeTestCase):
    def test_judge_cache_write_failure_is_fatal_for_valid_and_malformed_responses(self):
        for content in ('{"credit": 1}', 'not json'):
            with self.subTest(content=content):
                judges, fake, _ = self.judges(lambda messages: content)
                run = RunFile.begin(self.directory / "output")
                original = (run.run_dir / "run.json").read_bytes()
                entry = plan_entry(1, ["Culture", "Rest"], ["Culture"])
                failure = OSError("judge response disk full")
                with patch.object(judges.cache, "add", side_effect=failure), \
                        error_scope(run, patient_id="patient-a"), pipeline_stage("scoring"):
                    with self.assertRaisesRegex(RunMetadataError, "Could not save judge response") as caught:
                        judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
                self.assertIs(caught.exception.__cause__, failure)
                self.assertEqual(len(fake.requests), 1)  # No later action or extras requests.
                self.assertEqual((run.run_dir / "run.json").read_bytes(), original)
                self.assertEqual(judges.cache.entries, {})
                self.assertFalse(self.cache.exists())

    def test_failed_call_is_saved_under_the_scoring_stage_and_other_calls_are_kept(self):
        run = RunFile.begin(self.directory / "output")
        base = replies()
        failing = lambda messages: RuntimeError("judge server down") if is_extras(messages) else base(messages)
        judges, _, _ = self.judges(failing)
        entry = plan_entry(1, ["Culture"], ["Culture", "Antibiotics"])
        with error_scope(run, patient_id="patient-a"), pipeline_stage("scoring"):
            judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual((entry["status"], entry["reason"]), ("error", "judge_failed"))
        self.assertEqual(entry["judges"][1]["error"], {"type": "RuntimeError", "message": "judge server down"})
        [error] = json.loads((run.run_dir / "run.json").read_text())["errors"]
        self.assertEqual((error["patient_id"], error["stage"], error["exception_type"]),
                         ("patient-a", "scoring", "JudgeCallError"))
        self.assertIn(f"plan_extras judge {JUDGE}", error["message"])
        self.assertIn("RuntimeError: judge server down", error["traceback"])  # chained cause
        # Only completed calls are cached; the failed one will be retried.
        self.assertEqual(len(JudgeCache(self.cache).entries), 1)

    def test_malformed_replies_are_saved_once_per_patient_across_resumes(self):
        reply = replies(action_content={"Culture": "not json"})
        fake = FakeJudge(reply)
        run = RunFile.begin(self.directory / "output")
        for _ in range(2):
            judges = CorrectnessJudges(fake, MODELS, self.cache, similarity=BioLordSimilarity(FakeEncoder()),
                                       plan_judge=JUDGE)
            entry = plan_entry(1, ["Culture", "Rest"], ["Culture"])
            with error_scope(run, patient_id="patient-a"), pipeline_stage("scoring"):
                judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
            self.assertEqual([judge["status"] for judge in entry["judges"]],
                             ["invalid_response", "scored", "scored"])
        errors = json.loads((run.run_dir / "run.json").read_text())["errors"]
        self.assertEqual([(e["patient_id"], e["stage"], e["exception_type"]) for e in errors],
                         [("patient-a", "scoring", "JudgeResponseError")])
        self.assertIn(f"plan_action judge {JUDGE}", errors[0]["message"])
        self.assertIn("invalid JSON", errors[0]["message"])
        # The bad reply was requested again on the second pass; valid ones were reused.
        self.assertEqual(["extras" if is_extras(m) else reference_action(m) for _, m in fake.requests],
                         ["Culture", "Rest", "extras", "Culture"])

    def test_a_different_patient_with_the_same_failure_gets_its_own_error(self):
        run = RunFile.begin(self.directory / "output")
        for patient in ("patient-a", "patient-b"):
            judges, _, _ = self.judges(lambda messages: RuntimeError("down"))
            with error_scope(run, patient_id=patient), pipeline_stage("scoring"):
                judges.judge_plan(plan_entry(1, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
        errors = json.loads((run.run_dir / "run.json").read_text())["errors"]
        self.assertEqual(sorted({e["patient_id"] for e in errors}), ["patient-a", "patient-b"])
        self.assertEqual(len(errors), 4)  # action and extras failures for each patient


class ResumptionTests(JudgeTestCase):
    def test_interrupted_plan_resumes_without_repeating_completed_calls(self):
        base = replies()
        interrupting = lambda messages: KeyboardInterrupt() if is_extras(messages) else base(messages)
        judges, first, _ = self.judges(interrupting)
        entry = plan_entry(1, ["Culture", "Rest"], ["Culture", "Antibiotics"])
        with self.assertRaises(KeyboardInterrupt):
            judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual(entry["status"], "interrupted")
        self.assertEqual([judge["call"] for judge in entry["judges"]], ["action", "action"])
        self.assertEqual(len(first.requests), 3)
        resumed, second, _ = self.judges(replies())
        entry = plan_entry(1, ["Culture", "Rest"], ["Culture", "Antibiotics"])
        resumed.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual([is_extras(m) for _, m in second.requests], [True])
        self.assertEqual((resumed.calls, resumed.cache_hits), (1, 2))
        self.assertEqual((entry["status"], entry["score"]), ("scored", 1.0))

    def test_malformed_reply_is_kept_for_audit_but_requested_again_on_resume(self):
        judges, _, _ = self.judges(replies(extras_content="not json"))
        first = plan_entry(1, ["Culture"], ["Culture"])
        judges.judge_plan(first, LABEL_CONDITION, PATIENT)
        self.assertEqual(first["status"], "error")
        again, fake, _ = self.judges(replies())
        entry = plan_entry(1, ["Culture"], ["Culture"])
        again.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual([is_extras(m) for _, m in fake.requests], [True])
        self.assertEqual((entry["status"], entry["score"]), ("scored", 1.0))
        saved = [json.loads(line) for line in self.cache.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(row["task"], row["valid"]) for row in saved],
                         [("plan_action", True), ("plan_extras", False), ("plan_extras", True)])
        self.assertEqual(saved[1]["content"], "not json")
        self.assertIn("invalid JSON", saved[1]["error"]["message"])

    def test_partial_trailing_cache_line_is_dropped_and_appends_stay_valid(self):
        judges, _, _ = self.judges()
        judges.judge_plan(plan_entry(1, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
        with self.cache.open("a", encoding="utf-8") as handle:
            handle.write('{"key": "torn')
        cache = JudgeCache(self.cache)
        self.assertEqual(len(cache.entries), 2)
        self.assertTrue(self.cache.read_text(encoding="utf-8").endswith("\n"))

    def test_corrupt_complete_line_is_rejected_before_torn_tail_repair(self):
        original = b'not JSON\n{"key":"torn'
        self.cache.write_bytes(original)
        with self.assertRaises(ValueError):
            JudgeCache(self.cache)
        self.assertEqual(self.cache.read_bytes(), original)

    def test_identical_requests_are_asked_once(self):
        judges, fake, _ = self.judges()
        for rank in (1, 2):
            judges.judge_plan(plan_entry(rank, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(judges.cache_hits, 2)

    def test_judge_cache_directory_flush_failure_is_fatal_before_reply_can_be_reused(self):
        judges, fake, _ = self.judges()
        with patch("scripts.med_copilot_judges._fsync_directory", side_effect=OSError("directory flush failed")), \
                self.assertRaises(RunMetadataError):
            judges.judge_plan(plan_entry(1, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(judges.cache.entries, {})
        self.assertEqual(len(self.cache.read_text().splitlines()), 1)  # Kept for recovery/auditing.


class ResumeValidationTests(JudgeTestCase):
    def saved(self):
        judges = CorrectnessJudges(None, [DEFAULT_PLAN_JUDGE], self.cache, plan_judge=DEFAULT_PLAN_JUDGE)
        return {"evaluation": {"correctness": judges.metadata(ROOT)}}

    def test_identical_semantics_validate_without_models_cache_calls_or_writes(self):
        metadata = self.saved()
        before = deepcopy(metadata)
        with patch("scripts.med_copilot_judges.JudgeCache", side_effect=AssertionError("cache touched")), \
                patch("scripts.med_copilot_judges.BioLordSimilarity", side_effect=AssertionError("model touched")):
            CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)
        self.assertEqual(metadata, before)
        self.assertFalse(self.cache.exists())

    def test_changed_scoring_semantics_are_rejected(self):
        original = self.saved()
        cases = [
            (("diagnosis", "model"), "different-encoder"),
            (("diagnosis", "model_revision"), "different-revision"),
            (("diagnosis", "normalization", "abbreviations"), {}),
            (("diagnosis", "normalization", "rule"), "different rule"),
            (("plan", "judge_model"), "different-judge"),
            (("plan", "judge_temperature"), 1),
            (("plan", "rubric_sha256"), "different-rubric"),
            (("plan", "action_prompt_template_sha256"), "different-action-prompt"),
            (("plan", "extras_prompt_template_sha256"), "different-extras-prompt"),
            (("plan", "credit_values"), [0, 1]),
            (("plan", "deduction_per_unsupported_extra"), 1),
            (("rank_weighting", "formula"), "1 / rank squared"),
            (("rank_weighting", "weights"), {"1": 1, "2": 0.25, "3": 1 / 9}),
        ]
        for path, value in cases:
            with self.subTest(setting=".".join(path)):
                metadata = deepcopy(original)
                target = metadata["evaluation"]["correctness"]
                for part in path[:-1]:
                    target = target[part]
                target[path[-1]] = value
                with self.assertRaisesRegex(ValueError, "correctness"):
                    CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)
        self.assertFalse(self.cache.exists())

    def test_availability_and_runtime_status_can_change_without_changing_scores(self):
        metadata = self.saved()
        correctness = metadata["evaluation"]["correctness"]
        correctness.update(status="interrupted", selected_models=[])
        correctness["plan"].update(judge_available=False, judge_model_config=None,
                                   validation_status="unavailable", reporting="unavailable")
        CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)

    def test_legacy_without_correctness_and_malformed_correctness_metadata(self):
        for metadata in ({}, {"evaluation": {}}, {"evaluation": {"correctness": None}}):
            with self.subTest(metadata=metadata):
                CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)
        for correctness in ([], "invalid", {}, {"diagnosis": None}):
            with self.subTest(correctness=correctness), self.assertRaisesRegex(ValueError, "correctness"):
                CorrectnessJudges.validate_resume({"evaluation": {"correctness": correctness}}, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)

    def test_changed_runtime_prompt_is_rejected_against_saved_prompt_hash(self):
        metadata = self.saved()
        for template in ("PLAN_ACTION_JUDGE_TEMPLATE", "PLAN_EXTRAS_JUDGE_TEMPLATE"):
            with self.subTest(template=template), \
                    patch("scripts.med_copilot_judges." + template, "changed prompt"), \
                    self.assertRaisesRegex(ValueError, "correctness"):
                CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)

    def test_resume_checks_the_selected_judge_alias(self):
        metadata = self.saved()
        metadata["evaluation"]["correctness"]["plan"]["judge_model"] = "custom_judge"
        CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge="custom_judge")
        with self.assertRaisesRegex(ValueError, "judge_model"):
            CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)

    def test_same_alias_does_not_require_matching_effective_judge_configuration(self):
        for config in (None, {"models": {DEFAULT_PLAN_JUDGE: {"model": "old model", "api_base": "old endpoint"}}}):
            with self.subTest(config=config):
                metadata = self.saved()
                metadata["evaluation"]["correctness"]["plan"]["judge_model_config"] = config
                CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=DEFAULT_PLAN_JUDGE)
        self.assertFalse(self.cache.exists())


class MetadataTests(JudgeTestCase):
    def test_judge_alias_is_explicit_and_invalid_values_fail_before_opening_cache(self):
        with self.assertRaises(TypeError):
            CorrectnessJudges(None, MODELS, self.cache)
        for value in (None, "", " \n", False, 1, [], {}):
            with self.subTest(alias=value), self.assertRaisesRegex(ValueError, "plan_judge"):
                CorrectnessJudges(None, MODELS, self.cache, plan_judge=value)
        self.assertFalse(self.cache.exists())

    def test_metadata_labels_the_preliminary_metric_and_records_models_prompts_and_weights(self):
        metadata = self.judges()[0].metadata(ROOT)
        self.assertEqual(metadata["status"], METRIC_STATUS)
        self.assertEqual(METRIC_STATUS, "preliminary single-model midterm metric")
        self.assertIn("preliminary single-model midterm metric", metadata["method"])
        self.assertEqual(metadata["selected_models"], list(MODELS))
        self.assertEqual(metadata["hidden_from_judges"], ["generating_model", "evidence_variant", "rank"])
        self.assertEqual((metadata["diagnosis"]["model"], metadata["diagnosis"]["model_revision"]),
                         (BIOLORD_MODEL, BIOLORD_REVISION))
        plan = metadata["plan"]
        self.assertEqual((plan["judge_model"], plan["judge_count"], plan["judge_available"],
                          plan["judge_temperature"]), (JUDGE, 1, True, 0))
        self.assertIsNone(plan["self_judging"])  # Generators unknown to this panel-less fixture.
        judges = CorrectnessJudges(FakeJudge(replies()), [JUDGE], self.cache, plan_judge=JUDGE,
                                   similarity=BioLordSimilarity(FakeEncoder()), generator_models=MODELS,
                                   judge_model_config={"selected_models": [JUDGE]})
        plan_with_generators = judges.metadata(ROOT)["plan"]
        self.assertEqual(plan_with_generators["self_judging"], "the judge also generated Plans and judges its own outputs")
        self.assertEqual(plan_with_generators["generator_models"], list(MODELS))
        self.assertEqual(plan_with_generators["judge_model_config"], {"selected_models": [JUDGE]})
        separate = CorrectnessJudges(FakeJudge(replies()), ["qwen"], self.cache, plan_judge="qwen",
                                     similarity=BioLordSimilarity(FakeEncoder()), generator_models=MODELS)
        self.assertEqual(separate.metadata(ROOT)["plan"]["self_judging"], "the judge is not a generating model")
        self.assertEqual(plan["rubric_file"], "med_copilot/docs/treatment_plan_evaluation.md")
        for key in ("rubric_sha256", "action_prompt_template_sha256", "extras_prompt_template_sha256"):
            self.assertEqual(len(plan[key]), 64)
        self.assertEqual(metadata["rank_weighting"]["weights"], {"1": 1.0, "2": 0.5, "3": 1 / 3})
        self.assertIsNone(metadata["rank_weighting"]["combined_score"])
        json.dumps(metadata, allow_nan=False)

    def test_plan_entries_and_metadata_carry_the_plan_validation_status(self):
        entry, _ = self.plan()
        self.assertEqual(entry["validation"], PLAN_VALIDATION_STATUS)
        plan = self.judges()[0].metadata(ROOT)["plan"]
        self.assertEqual(plan["validation_status"], PLAN_VALIDATION_STATUS)
        self.assertTrue(plan["validation_evidence"])
        if PLAN_VALIDATION_STATUS == EXPERIMENTAL_UNVALIDATED:
            self.assertIn("excluded from reported aggregate results", plan["reporting"])
            self.assertIn("not an accuracy measure", plan["reporting"])

    def test_shipped_plan_judge_is_a_local_registry_model_that_passed_validation(self):
        registry = yaml.safe_load((ROOT / "experiments/configs/condition_plan_models.yaml").read_text())["models"]
        self.assertEqual(DEFAULT_PLAN_JUDGE, "ollama_qwen2_5_14b")
        self.assertEqual(registry[DEFAULT_PLAN_JUDGE]["model"], "qwen2.5:14b")
        self.assertEqual(registry[DEFAULT_PLAN_JUDGE]["api_base"], "http://localhost:11434/v1")
        self.assertNotIn("api_key_env", registry[DEFAULT_PLAN_JUDGE])
        self.assertEqual(PLAN_VALIDATION_STATUS, "passed_balanced_sample")
        judges = CorrectnessJudges(FakeJudge(replies()), MODELS, self.cache,
                                   similarity=BioLordSimilarity(FakeEncoder()), plan_judge=DEFAULT_PLAN_JUDGE)
        self.assertEqual(judges.plan_judge, DEFAULT_PLAN_JUDGE)


class ConcurrentPlanTests(JudgeTestCase):
    def test_independent_plans_overlap_and_keep_their_own_scores(self):
        barrier = Barrier(2)
        base = replies(credits={"Culture": 1, "Rest": 0.5})

        def reply(messages):
            barrier.wait(timeout=3)
            return base(messages)

        judges, fake, _ = self.judges(reply)
        entries = [plan_entry(1, ["Culture"], ["Culture"]),
                   plan_entry(2, ["Rest"], ["Rest"])]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(judges.judge_plan, entry, LABEL_CONDITION, PATIENT)
                       for entry in entries]
            for future in futures:
                future.result(timeout=5)
        self.assertEqual([entry["score"] for entry in entries], [1.0, 0.5])
        self.assertEqual([entry["weighted_score"] for entry in entries], [1.0, 0.25])
        self.assertEqual((judges.calls, judges.cache_hits, len(fake.requests)), (4, 0, 4))
        self.assertEqual(len(JudgeCache(self.cache).entries), 4)

    def test_simultaneous_equal_plans_share_durable_calls_and_resume_without_calls(self):
        entered, release = Event(), Event()
        start = Barrier(8)
        base = replies()

        def reply(messages):
            if not is_extras(messages):
                entered.set()
                if not release.wait(timeout=3):
                    raise RuntimeError("test did not release the judge")
            return base(messages)

        judges, fake, _ = self.judges(reply)

        def score(rank):
            start.wait(timeout=3)
            entry = plan_entry(rank, ["Culture"], ["Culture"])
            return judges.judge_plan(entry, LABEL_CONDITION, PATIENT)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(score, index % 3 + 1) for index in range(8)]
            try:
                self.assertTrue(entered.wait(timeout=3))
            finally:
                release.set()
            entries = [future.result(timeout=5) for future in futures]
        self.assertTrue(all(entry["status"] == "scored" for entry in entries))
        self.assertEqual((judges.calls, judges.cache_hits, len(fake.requests)), (2, 14, 2))
        self.assertEqual(len(self.cache.read_text().splitlines()), 2)
        resumed, again, _ = self.judges(lambda messages: AssertionError("completed request repeated"))
        resumed.judge_plan(plan_entry(1, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
        self.assertEqual((resumed.calls, resumed.cache_hits, again.requests), (0, 2, []))

    def test_concurrent_cache_appends_remain_line_aligned_and_recoverable(self):
        cache = JudgeCache(self.cache)
        entries = [{"key": str(index), "valid": True, "content": "x" * 4096}
                   for index in range(32)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(cache.add, entries))
        recovered = JudgeCache(self.cache)
        self.assertEqual(recovered.entries, {entry["key"]: entry for entry in entries})
        self.assertEqual(len(self.cache.read_text().splitlines()), len(entries))

    def test_failed_cache_append_stops_other_calls_until_restart(self):
        judges, fake, _ = self.judges()
        with patch("scripts.med_copilot_judges._fsync_directory", side_effect=OSError("disk failed")):
            with self.assertRaises(RunMetadataError):
                judges.judge_plan(plan_entry(1, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
        # No unrelated request may append behind a possibly torn cache write.
        with self.assertRaises(RunMetadataError):
            judges.judge_plan(plan_entry(1, ["Rest"], ["Rest"]), LABEL_CONDITION, PATIENT)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(judges.cache.entries, {})
        # A fresh owner validates the complete persisted line and can use it.
        resumed, _, _ = self.judges()
        resumed.judge_plan(plan_entry(1, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
        self.assertEqual((resumed.calls, resumed.cache_hits), (1, 1))

    def test_unserializable_judge_reply_is_a_fatal_persistence_failure(self):
        for failure in (TypeError("unsupported usage value"), ValueError("nonfinite usage value")):
            with self.subTest(error=type(failure).__name__):
                judges, fake, _ = self.judges()
                with patch.object(judges.cache, "add", side_effect=failure), \
                        self.assertRaises(RunMetadataError) as caught:
                    judges.judge_plan(plan_entry(1, ["Culture"], ["Culture"]), LABEL_CONDITION, PATIENT)
                self.assertIs(caught.exception.__cause__, failure)
                self.assertEqual(len(fake.requests), 1)
                self.assertEqual(judges.cache.entries, {})

    def test_invalid_reply_does_not_block_later_retry_or_break_counters(self):
        judges, fake, _ = self.judges(replies(extras_content="invalid JSON"))
        first = plan_entry(1, ["Culture"], ["Culture"])
        judges.judge_plan(first, LABEL_CONDITION, PATIENT)
        self.assertEqual(first["status"], "error")
        fake.reply = replies()
        second = plan_entry(1, ["Culture"], ["Culture"])
        judges.judge_plan(second, LABEL_CONDITION, PATIENT)
        self.assertEqual(second["status"], "scored")
        self.assertEqual((judges.calls, judges.cache_hits), (3, 1))


class DeferredPlanTests(JudgeTestCase):
    def test_deferred_plans_leave_judge_artifacts_untouched_and_diagnoses_still_score(self):
        # The separate stage owns tail repair. Generation must not truncate it.
        original = b'{"key":"torn'
        self.cache.write_bytes(original)
        fake = FakeJudge(lambda messages: AssertionError("deferred judge called"))
        encoder = FakeEncoder()
        with patch("scripts.med_copilot_judges.JudgeCache", side_effect=AssertionError("cache opened")):
            judges = CorrectnessJudges(fake, (), self.cache, plan_judge=JUDGE,
                                       similarity=BioLordSimilarity(encoder), plan_execution="deferred")
            entry = plan_entry(1, ["Culture"], ["Culture"])
            judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
            diagnosis = diagnosis_entry(1, LABEL_CONDITION, "viral pharyngitis")
            judges.score_diagnosis(diagnosis)
            metadata = judges.metadata(ROOT)
        self.assertEqual((entry["status"], entry["reason"]), ("pending", "deferred"))
        self.assertEqual(diagnosis["status"], "scored")
        self.assertEqual(self.cache.read_bytes(), original)
        self.assertEqual(fake.requests, [])
        self.assertEqual(metadata["plan"]["execution"], "deferred")
        self.assertEqual(metadata["plan"]["judge_loading"], "deferred")
        self.assertFalse(metadata["plan"]["judge_available"])

    def test_deferred_mode_does_not_create_cache_and_preserves_final_zero_and_unavailable_entries(self):
        judges = CorrectnessJudges(None, (), self.cache, plan_judge=JUDGE, plan_execution="deferred")
        for generated, reference, expected in (([], ["Culture"], "scored"),
                                                (["Culture"], ["none recorded"], "unavailable")):
            with self.subTest(generated=generated, reference=reference):
                entry = plan_entry(1, reference, generated)
                original = deepcopy(entry)
                judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
                self.assertEqual(entry["status"], expected)
                self.assertEqual(entry, original)
        self.assertFalse(self.cache.exists())

    def test_invalid_execution_mode_fails_before_opening_cache(self):
        for mode in (None, "", "parallel", False, 1, []):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "plan_execution"):
                CorrectnessJudges(None, (), self.cache, plan_judge=JUDGE, plan_execution=mode)
        self.assertFalse(self.cache.exists())


class JudgeRedactionTests(JudgeTestCase):
    def test_transport_error_is_redacted_in_returned_entries_and_run_errors(self):
        secret = "private-provider-api-key"
        redact = lambda text: text.replace(secret, "[REDACTED]")
        run = RunFile.begin(self.directory / "output")
        run.redact = redact
        fake = FakeJudge(lambda messages: RuntimeError(f"Authorization failed: {secret}"))
        judges = CorrectnessJudges(fake, MODELS, self.cache, plan_judge=JUDGE, redact=redact)
        entry = plan_entry(1, ["Culture"], ["Culture"])
        with error_scope(run, patient_id="patient-a"), pipeline_stage("scoring"):
            judges.judge_plan(entry, LABEL_CONDITION, PATIENT)
        self.assertEqual(entry["status"], "error")
        self.assertNotIn(secret, json.dumps(entry))
        self.assertIn("[REDACTED]", json.dumps(entry))
        self.assertNotIn(secret, (run.run_dir / "run.json").read_text())
        self.assertFalse(self.cache.exists())

    def test_refusal_provider_metadata_and_stored_prompt_are_redacted(self):
        secret = "private-provider-api-key"
        redact = lambda text: text.replace(secret, "[REDACTED]")
        patient = {**PATIENT, "subjective": f"S/O text containing {secret}"}

        def reply(messages):
            result = response(f"refused {secret}", refusal=f"policy {secret}")
            result.model = f"reported-model-{secret}"
            return result

        judges = CorrectnessJudges(FakeJudge(reply), MODELS, self.cache, plan_judge=JUDGE, redact=redact)
        entry = plan_entry(1, ["Culture"], ["Culture"])
        judges.judge_plan(entry, LABEL_CONDITION, patient)
        self.assertEqual(entry["status"], "error")
        self.assertNotIn(secret, json.dumps(entry))
        self.assertNotIn(secret, self.cache.read_text())
        saved = [json.loads(line) for line in self.cache.read_text().splitlines()]
        actual_messages = plan_action_messages(LABEL_CONDITION, patient["subjective"],
                                               patient["objective"], "Culture", ["Culture"])
        self.assertEqual(saved[0]["key"], request_key(JUDGE, actual_messages))
        self.assertEqual(entry["judges"][0]["request_key"], saved[0]["key"])

    def test_redacted_valid_replies_keep_request_identity_and_are_reused(self):
        secret = "private-provider-api-key"
        redact = lambda text: text.replace(secret, "[REDACTED]")
        patient = {**PATIENT, "subjective": f"S/O text containing {secret}"}

        def reply(messages):
            value = ({"unsupported_extra_count": 0} if is_extras(messages) else {"credit": 1})
            return json.dumps({**value, "note": secret})

        judges = CorrectnessJudges(FakeJudge(reply), MODELS, self.cache, plan_judge=JUDGE, redact=redact)
        entry = plan_entry(1, ["Culture"], ["Culture"])
        judges.judge_plan(entry, LABEL_CONDITION, patient)
        self.assertEqual(entry["score"], 1)
        self.assertNotIn(secret, json.dumps(entry))
        self.assertNotIn(secret, self.cache.read_text())
        fake = FakeJudge(lambda messages: AssertionError("completed request repeated"))
        resumed = CorrectnessJudges(fake, MODELS, self.cache, plan_judge=JUDGE, redact=redact)
        resumed.judge_plan(plan_entry(2, ["Culture"], ["Culture"]), LABEL_CONDITION, patient)
        self.assertEqual((resumed.calls, resumed.cache_hits, fake.requests), (0, 2, []))

    def test_invalid_redactor_fails_before_opening_cache(self):
        for redact in ("redact", False, 1, []):
            with self.subTest(redact=redact), self.assertRaisesRegex(ValueError, "redact"):
                CorrectnessJudges(None, (), self.cache, plan_judge=JUDGE, redact=redact)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
