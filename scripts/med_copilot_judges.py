"""Three-model judge panels for diagnosis and treatment-plan correctness.

Each generated prediction is judged by the three selected models other than its
generator. Judges see only the reference, the generated text, and (for Plans)
the patient's S/O; the generator, evidence variant, and rank stay hidden. Every
completed judge call is appended to judge_responses.jsonl in the run directory
before it is used. Valid judgements are reused, so a resumed evaluation never
repeats them; malformed or refused replies stay in the file for auditing but are
requested again.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median

from med_copilot.prompts.correctness_judges import DIAGNOSIS_JUDGE_TEMPLATE, PLAN_JUDGE_TEMPLATE
from scripts.med_copilot_helpers import model_response_metadata, repo_path
from scripts.med_copilot_run_file import pacific_now, record_error


JUDGE_FILE = "judge_responses.jsonl"
PANEL_SIZE = 3
JUDGE_TEMPERATURE = 0
CREDIT_VALUES = (0, 0.5, 1)
DEDUCTION_PER_UNSUPPORTED_EXTRA = 0.5
RECORD_ARTIFACT_MARKER = "(record artifact)"
# Any case and spacing, e.g. "( Record  Artifact )".
RECORD_ARTIFACT = re.compile(r"\(\s*record\s+artifact\s*\)", re.IGNORECASE)
PLAN_RUBRIC = Path(__file__).resolve().parents[1] / "med_copilot/docs/treatment_plan_evaluation.md"
DIAGNOSIS_SCALE = {
    "1": "clinically equivalent diagnosis, including valid synonyms",
    "0.5": "clinically related but meaningfully broader, narrower, or missing an important qualifier",
    "0": "incorrect or a different condition",
}


def rank_weight(rank):
    """Reciprocal-rank weight: 1, 0.5, and 1/3 for ranks 1-3."""
    return 1 / rank


def unavailable_plan_reason(plan):
    """Why a reference Plan cannot be judged, or None when it can."""
    if not isinstance(plan, list) or not plan or not any(item.strip() for item in plan):
        return "missing_reference"
    if plan == ["none recorded"]:
        return "no_recorded_plan"
    if any(RECORD_ARTIFACT.search(item) for item in plan):
        return "record_artifact_plan"
    return None


def _numbered(actions):
    return "\n".join(f"{number}. {action}" for number, action in enumerate(actions, 1))


def diagnosis_messages(reference_condition, generated_diagnosis):
    prompt = DIAGNOSIS_JUDGE_TEMPLATE.format(reference_condition=reference_condition,
                                             generated_diagnosis=generated_diagnosis)
    return [{"role": "user", "content": prompt}]


def plan_messages(condition, subjective, objective, reference_actions, generated_actions):
    """Number each action so judges score items, not one joined string."""
    prompt = PLAN_JUDGE_TEMPLATE.format(
        condition=condition, subjective=subjective, objective=objective,
        reference_plan=_numbered(reference_actions), generated_plan=_numbered(generated_actions))
    return [{"role": "user", "content": prompt}]


def request_key(judge_model, messages):
    """Identify one judge request; identical requests to the same judge are asked once."""
    payload = json.dumps({"judge_model": judge_model, "temperature": JUDGE_TEMPERATURE,
                          "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeResponseError(ValueError):
    """A judge replied, but not with a valid judgement."""


class JudgeCallError(RuntimeError):
    """A judge request raised before any reply was obtained."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JudgeResponseError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise JudgeResponseError(f"Nonfinite JSON constant: {value}")


def _judgement(content):
    if not isinstance(content, str) or not content.strip():
        raise JudgeResponseError("Judge returned empty or non-text content")
    try:
        value = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"Judge returned invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise JudgeResponseError("Judge JSON must be an object")
    return value


def _credit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in CREDIT_VALUES:
        raise JudgeResponseError(f"{name} must be 0, 0.5, or 1")
    return float(value)


def parse_diagnosis_judgement(content):
    """Return the judge's 0/0.5/1 score; extra keys are kept only in the raw reply."""
    judgement = _judgement(content)
    if "score" not in judgement:
        raise JudgeResponseError("Diagnosis judgement requires score")
    return {"score": _credit(judgement["score"], "score")}


def parse_plan_judgement(content, reference_count):
    """Validate one credit per reference action, a 0.5-step deduction, and mismatch items."""
    judgement = _judgement(content)
    missing = [key for key in ("credits", "deduction", "strength_mismatch_items") if key not in judgement]
    if missing:
        raise JudgeResponseError("Plan judgement requires " + ", ".join(missing))
    credits = judgement["credits"]
    if not isinstance(credits, list) or len(credits) != reference_count:
        raise JudgeResponseError(f"credits must list exactly {reference_count} reference actions")
    credits = [_credit(value, "Each credit") for value in credits]
    deduction = judgement["deduction"]
    if (isinstance(deduction, bool) or not isinstance(deduction, (int, float))
            or not math.isfinite(deduction) or deduction < 0
            or (deduction / DEDUCTION_PER_UNSUPPORTED_EXTRA) % 1):
        raise JudgeResponseError("deduction must be a nonnegative multiple of 0.5")
    items = judgement["strength_mismatch_items"]
    if (not isinstance(items, list) or len(set(items)) != len(items)
            or any(type(item) is not int or not 1 <= item <= reference_count for item in items)):
        raise JudgeResponseError("strength_mismatch_items must be distinct 1-based reference item numbers")
    return {"credits": credits, "deduction": float(deduction), "strength_mismatch_items": items}


def plan_score(median_credits, median_deduction):
    """max(0, (sum of median item credits - median deduction) / number of reference actions)."""
    return max(0.0, (math.fsum(median_credits) - median_deduction) / len(median_credits))


class JudgeCache:
    """Append-only JSONL of completed judge calls, durable before each reply is used.

    Every reply is kept for auditing; only entries marked valid are reused.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.entries = {}
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        # A crash can leave a partial final line; drop it so appends stay line-aligned.
        if data and not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n") + 1]
            with self.path.open("r+b") as handle:
                handle.truncate(len(data))
        for number, line in enumerate(data.decode("utf-8").split("\n"), 1):
            if line.strip():
                entry = json.loads(line)
                if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                    raise ValueError(f"{self.path}:{number}: invalid judge response entry")
                if entry.get("valid") is True:
                    self.entries[entry["key"]] = entry

    def get(self, key):
        """The saved valid judgement for this request, or None to request it."""
        return self.entries.get(key)

    def add(self, entry):
        line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        if entry.get("valid") is True:
            self.entries[entry["key"]] = entry


class CorrectnessJudges:
    """Score diagnoses and Plans with the selected models other than each generator.

    complete(model_id, messages) sends one judge request and returns the
    provider response. Valid judgements are reused by request. Failed calls and
    malformed or refused replies are saved in run.json errors (once per patient,
    request, and message) and requested again when evaluation resumes.
    """

    def __init__(self, complete, model_ids, cache_path):
        self.complete = complete
        self.model_ids = tuple(model_ids)
        self.cache = JudgeCache(cache_path)
        self.calls = 0
        self.cache_hits = 0

    def panel(self, generator):
        judges = [model for model in self.model_ids if model != generator]
        return judges if generator in self.model_ids and len(judges) == PANEL_SIZE else None

    def metadata(self, root):
        return {
            "method": "independent three-model judge panel; median of the three judgements",
            "judge_selection": "selected_models other than the generating model; exactly three required",
            "selected_models": list(self.model_ids),
            "judge_temperature": JUDGE_TEMPERATURE,
            "response_format": "json_object unless the model's output_mode is prompt",
            "hidden_from_judges": ["generating_model", "evidence_variant", "rank"],
            "responses_file": JUDGE_FILE,
            "resumption": ("valid judgements are reused by request key; failed calls and malformed or "
                           "refused replies are kept in responses_file but requested again"),
            "diagnosis": {
                "scale": DIAGNOSIS_SCALE,
                "aggregation": "median of three judge scores",
                "shown_to_judges": ["reference_condition", "generated_diagnosis"],
                "prompt_template": DIAGNOSIS_JUDGE_TEMPLATE,
                "prompt_template_sha256": hashlib.sha256(DIAGNOSIS_JUDGE_TEMPLATE.encode()).hexdigest(),
            },
            "plan": {
                "rubric_file": repo_path(PLAN_RUBRIC, root),
                "rubric_sha256": hashlib.sha256(PLAN_RUBRIC.read_bytes()).hexdigest(),
                "credit_values": list(CREDIT_VALUES),
                "deduction_per_unsupported_extra": DEDUCTION_PER_UNSUPPORTED_EXTRA,
                "aggregation": "median credit per reference action and median total deduction",
                "formula": "max(0, (sum of median item credits - median deduction) / number of reference actions)",
                "actions": "reference and generated actions stay separate, numbered in order",
                "shown_to_judges": ["labeled_condition", "subjective", "objective",
                                    "reference_actions", "generated_actions"],
                "unavailable_references": ['["none recorded"]', "missing or empty plan",
                                           f"any action containing {RECORD_ARTIFACT_MARKER!r} "
                                           "(any case or spacing)"],
                "empty_generated_plan": "scores 0 without judge calls when a reference exists",
                "prompt_template": PLAN_JUDGE_TEMPLATE,
                "prompt_template_sha256": hashlib.sha256(PLAN_JUDGE_TEMPLATE.encode()).hexdigest(),
            },
            "rank_weighting": {
                "formula": "1 / rank",
                "weights": {str(rank): rank_weight(rank) for rank in (1, 2, 3)},
                "applied_to": ["weighted_diagnosis_score = diagnosis_score / rank",
                               "weighted_plan_score = plan_score / rank"],
                "reported_aggregate": ("rank-weighted mean = average of score / rank over judged ranks 1-3; "
                                       "not normalized, maximum (1 + 1/2 + 1/3) / 3 = 0.6111"),
                "combined_score": None,
            },
        }

    def _ask(self, judge, task, messages, parse):
        """Return one judge record, reusing a saved valid judgement for this request."""
        key = request_key(judge, messages)
        record = {"judge_model": judge, "status": "pending", "request_key": key,
                  "raw_response": None, "model_response": None, "error": None}
        entry = self.cache.get(key)
        try:
            parsed = parse(entry["content"]) if entry is not None else None
        except JudgeResponseError:
            parsed = None  # Saved under an older parser; request it again.
        if parsed is not None:
            self.cache_hits += 1
            record.update(status="scored", raw_response=entry["content"],
                          model_response=entry["model_response"], **parsed)
            return record
        label = f"{task} judge {judge} (request {key[:12]})"
        try:
            response = self.complete(judge, messages)
        except Exception as exc:
            record.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            self._record(JudgeCallError(f"{label} failed: {type(exc).__name__}: {exc}"), exc)
            return record
        except BaseException as exc:
            record.update(status="interrupted", error={"type": type(exc).__name__,
                                                        "message": "Judge call interrupted"})
            raise
        self.calls += 1
        choices = getattr(response, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        content, refusal = getattr(message, "content", None), getattr(message, "refusal", None)
        entry = {"key": key, "task": task, "judge_model": judge,
                 "recorded_at": pacific_now().isoformat(), "messages": messages,
                 "content": content if isinstance(content, str) else None,
                 "refusal": refusal if isinstance(refusal, str) else None,
                 "model_response": model_response_metadata(response), "valid": False, "error": None}
        record.update(raw_response=entry["content"], model_response=entry["model_response"])
        try:
            if entry["refusal"] or (refusal is not None and not isinstance(refusal, str)):
                raise JudgeResponseError("Judge refused the request")
            parsed = parse(entry["content"])
        except JudgeResponseError as exc:
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.cache.add(entry)
            record.update(status="invalid_response", error=entry["error"])
            self._record(JudgeResponseError(f"{label} replied invalidly: {exc}"), exc)
            return record
        entry["valid"] = True
        self.cache.add(entry)
        record.update(status="scored", **parsed)
        return record

    @staticmethod
    def _record(error, cause):
        """Save a judge failure under the active patient and scoring stage, once."""
        error.__cause__ = cause
        record_error(error, unique=True)

    @contextmanager
    def _keep_interrupted(self, entry, judges):
        """On interruption, keep the judge records obtained so far, then re-raise."""
        try:
            yield
        except BaseException as exc:
            if not isinstance(exc, Exception):
                entry.update(status="interrupted", judges=judges,
                             error={"type": type(exc).__name__, "message": "Judging interrupted"})
            raise

    def judge_diagnosis(self, entry, generator):
        """Fill a pending diagnosis_entry in place with the panel median and its weight."""
        if entry["status"] != "pending":
            return entry
        panel = self.panel(generator)
        if panel is None:
            entry.update(status="unavailable", reason="panel_unavailable")
            return entry
        messages = diagnosis_messages(entry["reference"], entry["generated"])
        judges = []
        with self._keep_interrupted(entry, judges):
            for judge in panel:
                judges.append(self._ask(judge, "diagnosis", messages, parse_diagnosis_judgement))
        entry["judges"] = judges
        if any(record["status"] != "scored" for record in judges):
            entry.update(status="error", reason="judge_failed", error={
                "type": "JudgePanelIncomplete", "message": "Not every judge returned a valid score"})
            return entry
        score = float(median(record["score"] for record in judges))
        entry.update(status="scored", score=score, weighted_score=score * entry["rank_weight"])
        return entry

    def judge_plan(self, entry, generator, condition, patient):
        """Fill a pending plan_entry in place from per-action panel medians."""
        if entry["status"] != "pending":
            return entry
        panel = self.panel(generator)
        if panel is None:
            entry.update(status="unavailable", reason="panel_unavailable")
            return entry
        reference = entry["reference_actions"]
        messages = plan_messages(condition, patient["subjective"], patient["objective"],
                                 reference, entry["generated_actions"])
        judges = []
        with self._keep_interrupted(entry, judges):
            for judge in panel:
                judges.append(self._ask(judge, "plan", messages,
                                        lambda content: parse_plan_judgement(content, len(reference))))
        entry["judges"] = judges
        if any(record["status"] != "scored" for record in judges):
            entry.update(status="error", reason="judge_failed", error={
                "type": "JudgePanelIncomplete", "message": "Not every judge returned a valid judgement"})
            return entry
        credits = [float(median(record["credits"][index] for record in judges))
                   for index in range(len(reference))]
        deduction = float(median(record["deduction"] for record in judges))
        score = plan_score(credits, deduction)
        entry.update(status="scored", score=score, weighted_score=score * entry["rank_weight"],
                     median_credits=credits, median_deduction=deduction)
        return entry


def diagnosis_entry(rank, reference, generated, problem=None):
    """A pending diagnosis-correctness entry, or a final one when judging cannot apply."""
    entry = {"status": "pending", "reason": None, "score": None, "weighted_score": None,
             "rank_weight": rank_weight(rank), "reference": reference, "generated": generated,
             "judges": [], "error": None}
    if problem is not None:
        entry.update(status="unavailable", reason=problem)
    elif generated is None or not generated.strip():
        entry.update(status="unavailable", reason="missing_generation")
    return entry


def plan_entry(rank, reference, generated, problem=None):
    """A pending plan-correctness entry; unavailable references are never judged.

    An explicitly empty generated Plan scores 0 without judge calls when the
    reference is available, as the rubric specifies.
    """
    entry = {"status": "pending", "reason": None, "score": None, "weighted_score": None,
             "rank_weight": rank_weight(rank), "reference_actions": list(reference),
             "generated_actions": list(generated) if isinstance(generated, list) else None,
             "median_credits": None, "median_deduction": None, "judges": [], "error": None}
    reason = unavailable_plan_reason(reference)
    if reason is not None:
        entry.update(status="unavailable", reason=reason)
    elif problem is not None:
        entry.update(status="unavailable", reason=problem)
    elif generated is None:
        entry.update(status="unavailable", reason="missing_generation")
    elif not generated:
        entry.update(status="scored", reason="empty_generated_plan", score=0.0, weighted_score=0.0,
                     median_credits=[0.0] * len(reference), median_deduction=0.0)
    return entry
