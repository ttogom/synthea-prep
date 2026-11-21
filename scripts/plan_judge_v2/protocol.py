"""Plan judging protocol 2: one model call per rank-1 Plan; scores are computed in code.

The judge labels each reference action full/partial/none, cites the generated
action numbers that provide the match, and lists harmful generated actions.
Code validates the reply and computes:

- coverage (primary): mean credit over reference actions (full 1, partial 0.5, none 0)
- harmful_count: reported separately, never subtracted from coverage
- penalized (secondary): max(0, (sum of credits - 0.5 x harmful_count) / reference count)

Version 1 (scripts/med_copilot_judges.py: one call per reference action plus an
extras call, all three ranks) is a different protocol. Version-2 outputs live in
their own files and record PROTOCOL_VERSION, so the two can never be mixed.
Changing the prompt, schema, credits, or scoring requires a new version.
"""

import hashlib
import json
import math
import re


PROTOCOL_NAME = "plan_judge_single_call"
PROTOCOL_VERSION = 2
JUDGED_RANK = 1
MATCH_CREDITS = {"full": 1.0, "partial": 0.5, "none": 0.0}
HARMFUL_PENALTY = 0.5
MAX_WHY_CHARS = 300
# Greedy decoding; generous for 3 reference items plus a few harmful reasons.
DECODING = {"temperature": 0.0, "max_new_tokens": 384}

RECORD_ARTIFACT = re.compile(r"\(\s*record\s+artifact\s*\)", re.IGNORECASE)

PROMPT_TEMPLATE = """You are grading a generated treatment plan against a reference plan for one patient.

Task 1. For each numbered reference action, decide how well the generated plan covers it:
- "full": a generated action is the same action, a valid more specific version, or an interchangeable substitute for this patient.
- "partial": a generated action is relevant but less specific, a partial substitute, or a reasonable fallback. For the same drug, use "partial" if the reference states a strength and the generated action omits it or states a different strength. A conditional generated action is "full" only if the reference has the same prerequisite or every offered option is a full match.
- "none": the action is missing, has the wrong target, or is a different action. Placeholders such as "further assessment needed" and mere relevance to the condition are not coverage. Asking about a behavior does not count as counseling.
If the reference does not specify a drug strength, do not lower the match only because the generated action also omits strength.
A generated item may bundle several actions; cite its number if any part of it matches. For "full" or "partial", list the generated item numbers that provide the match. For "none", give an empty list.

Task 2. List generated items that are harmful for this patient: unsafe, contraindicated by the notes, or clearly aimed at the wrong condition. An action is not harmful merely because the reference omits it. Symptom relief, hydration, rest, monitoring, follow-up, education, and confirmatory testing are not harmful unless the notes contraindicate them. Give a short reason for each harmful item. Return an empty list when none apply.

Use the Subjective and Objective notes only as patient context. Do not assume findings that are not reported. Treat the generated plan as the answer being graded, not as evidence about the patient.

Return only one JSON object in this form:
{{"reference_items": [{{"ref": <reference number>, "match": "full" | "partial" | "none", "generated": [<generated item numbers>]}}], "harmful_actions": [{{"generated": <generated item number>, "why": "<short reason>"}}]}}
Include every reference number from 1 to {reference_count} exactly once, in order.

Labeled condition:
<labeled_condition>
{condition}
</labeled_condition>

Subjective:
<subjective>
{subjective}
</subjective>

Objective:
<objective>
{objective}
</objective>

Reference plan ({reference_count} actions):
<reference_plan>
{reference_plan}
</reference_plan>

Generated plan ({generated_count} items):
<generated_plan>
{generated_plan}
</generated_plan>"""

PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def protocol_identity():
    """Everything that defines comparable scores; recorded once in manifest.json."""
    return {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "judged_rank": JUDGED_RANK,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "match_credits": dict(MATCH_CREDITS),
        "harmful_penalty": HARMFUL_PENALTY,
        "decoding": dict(DECODING),
        "primary_score": "coverage = mean reference credit (full 1, partial 0.5, none 0)",
        "secondary_score": "penalized = max(0, (sum of credits - 0.5 x harmful_count) / reference count)",
        "harmful_count": "reported separately; never subtracted from coverage",
    }


def _numbered(actions):
    return "\n".join(f"{number}. {action}" for number, action in enumerate(actions, 1))


def render_messages(condition, subjective, objective, reference_actions, generated_actions):
    """A single user message: some chat templates (Gemma) have no separate system role."""
    prompt = PROMPT_TEMPLATE.format(
        condition=condition, subjective=subjective, objective=objective,
        reference_count=len(reference_actions), generated_count=len(generated_actions),
        reference_plan=_numbered(reference_actions), generated_plan=_numbered(generated_actions))
    return [{"role": "user", "content": prompt}]


def messages_sha256(messages):
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def response_schema(reference_count, generated_count):
    """JSON schema for guided decoding; parse_response still enforces order and citations."""
    generated_number = {"type": "integer", "minimum": 1, "maximum": generated_count}
    return {
        "type": "object",
        "properties": {
            "reference_items": {
                "type": "array", "minItems": reference_count, "maxItems": reference_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "integer", "minimum": 1, "maximum": reference_count},
                        "match": {"type": "string", "enum": list(MATCH_CREDITS)},
                        "generated": {"type": "array", "items": generated_number,
                                      "maxItems": generated_count},
                    },
                    "required": ["ref", "match", "generated"],
                    "additionalProperties": False,
                },
            },
            "harmful_actions": {
                "type": "array", "maxItems": generated_count,
                "items": {
                    "type": "object",
                    "properties": {"generated": generated_number,
                                   "why": {"type": "string", "minLength": 1, "maxLength": MAX_WHY_CHARS}},
                    "required": ["generated", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reference_items", "harmful_actions"],
        "additionalProperties": False,
    }


class JudgeResponseError(ValueError):
    """The judge replied, but not with a valid protocol-2 judgment."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JudgeResponseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise JudgeResponseError(f"nonfinite JSON constant: {value}")


_FENCE = re.compile(r"\A```(?:json)?\s*\n(.*)\n```\Z", re.DOTALL)


def _strict_int(value):
    return type(value) is int


def parse_response(content, reference_count, generated_count):
    """Validate a reply and return {"reference_items": [...], "harmful_actions": [...]}.

    Accepts bare JSON or one whole-reply ```json fence (unconstrained backends).
    Extra top-level keys are ignored; they remain in the raw reply.
    """
    if not isinstance(content, str) or not content.strip():
        raise JudgeResponseError("empty or non-text reply")
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise JudgeResponseError("reply must be a JSON object")
    for key in ("reference_items", "harmful_actions"):
        if not isinstance(value.get(key), list):
            raise JudgeResponseError(f"{key} must be a list")

    items = value["reference_items"]
    if len(items) != reference_count:
        raise JudgeResponseError(f"reference_items must have {reference_count} entries, got {len(items)}")
    parsed_items = []
    for expected, item in enumerate(items, 1):
        if not isinstance(item, dict) or set(item) != {"ref", "match", "generated"}:
            raise JudgeResponseError("each reference item needs exactly ref, match, generated")
        if not _strict_int(item["ref"]) or item["ref"] != expected:
            raise JudgeResponseError(f"reference items must be numbered 1..{reference_count} in order")
        if item["match"] not in MATCH_CREDITS:
            raise JudgeResponseError("match must be full, partial, or none")
        cited = item["generated"]
        if (not isinstance(cited, list) or not all(_strict_int(n) and 1 <= n <= generated_count for n in cited)
                or len(set(cited)) != len(cited)):
            raise JudgeResponseError(f"generated citations must be distinct integers 1..{generated_count}")
        if item["match"] == "none" and cited:
            raise JudgeResponseError("a none match must not cite generated items")
        if item["match"] != "none" and not cited:
            raise JudgeResponseError("a full or partial match must cite a generated item")
        parsed_items.append({"ref": expected, "match": item["match"], "generated": list(cited)})

    harmful, seen = [], set()
    for entry in value["harmful_actions"]:
        if not isinstance(entry, dict) or set(entry) != {"generated", "why"}:
            raise JudgeResponseError("each harmful action needs exactly generated, why")
        number, why = entry["generated"], entry["why"]
        if not _strict_int(number) or not 1 <= number <= generated_count or number in seen:
            raise JudgeResponseError(f"harmful generated numbers must be distinct integers 1..{generated_count}")
        if not isinstance(why, str) or not why.strip():
            raise JudgeResponseError("harmful action why must be nonblank text")
        seen.add(number)
        harmful.append({"generated": number, "why": why.strip()[:MAX_WHY_CHARS]})
    return {"reference_items": parsed_items, "harmful_actions": harmful}


def score(parsed):
    """Primary coverage, separate harmful count, and the secondary penalized score."""
    credits = [MATCH_CREDITS[item["match"]] for item in parsed["reference_items"]]
    harmful_count = len(parsed["harmful_actions"])
    total = math.fsum(credits)
    return {
        "credits": credits,
        "coverage": total / len(credits),
        "harmful_count": harmful_count,
        "penalized": max(0.0, (total - HARMFUL_PENALTY * harmful_count) / len(credits)),
    }


def unavailable_reference_reason(plan):
    """Why a reference Plan cannot be judged, or None. Matches protocol 1's rules."""
    if not isinstance(plan, list) or not plan or not all(isinstance(item, str) for item in plan) \
            or not any(item.strip() for item in plan):
        return "missing_reference"
    if plan == ["none recorded"]:
        return "no_recorded_plan"
    if any(RECORD_ARTIFACT.search(item) for item in plan):
        return "record_artifact_plan"
    return None
