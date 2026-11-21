"""Read-only access to a base or agentic results.jsonl plus the patient S/O and label files.

Every input is opened read-only. results.jsonl (hundreds of MB) is streamed once,
hashed in the same pass, and only the rank-1 Plan of each selected variant/model
is kept. S/O text is looked up from the patient file when prompts are rendered.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from scripts.plan_judge_v2.protocol import JUDGED_RANK, unavailable_reference_reason


CHUNK = 1 << 20


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path):
    """Cheap change detector used alongside the content hash."""
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def describe_input(path, sha256):
    path = Path(path)
    return {"path": str(path), "name": path.name, "sha256": sha256, "bytes": path.stat().st_size}


def _jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: each line must be a JSON object")
                yield number, value


def load_by_id(path, required):
    """Index a JSONL file by id; duplicate ids or missing fields are errors."""
    rows = {}
    for number, row in _jsonl(path):
        patient_id = row.get("id")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError(f"{path}:{number}: missing id")
        if patient_id in rows:
            raise ValueError(f"{path}:{number}: duplicate id {patient_id}")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"{path}:{number}: missing {', '.join(missing)}")
        rows[patient_id] = row
    return rows


# Why a rank-1 Plan has no judgment. Variant/generation reasons come first, then the reference.
VARIANT_NOT_COMPLETED = "variant_not_completed"      # variant blocked or absent before generation
GENERATION_FAILED = "generation_failed"              # slot status failed_incomplete
GENERATION_NOT_PROCESSED = "generation_not_processed"  # slot never ran although the variant completed
MISSING_PLAN = "missing_plan"                        # completed slot without a rank-1 Plan
INVALID_GENERATION = "invalid_generation"            # malformed slot or Plan
EMPTY_GENERATED_PLAN = "empty_generated_plan"        # judged without a call: coverage 0
GENERATION_REASONS = (VARIANT_NOT_COMPLETED, GENERATION_FAILED, GENERATION_NOT_PROCESSED, MISSING_PLAN,
                      INVALID_GENERATION, EMPTY_GENERATED_PLAN)
REFERENCE_REASONS = ("missing_reference", "no_recorded_plan", "record_artifact_plan")


@dataclass
class PlanItem:
    """One rank-1 Plan to judge (or to record as unavailable) for a variant/generator pair.

    reason is the first entry of reasons; reasons keeps every applicable cause, so a failed
    generation for a patient whose reference is also unusable stays visible as both.
    """

    variant: str
    patient_id: str
    generator_model: str
    status: str  # "pending" | "unavailable" | "empty"
    reason: str | None
    condition: str | None
    reference_actions: list | None
    generated_actions: list | None
    reasons: tuple = ()
    source: dict | None = None  # variant_status, generation_status, generation_error_type

    @property
    def key(self):
        return self.patient_id, self.generator_model


def _status(value):
    return value.get("status") if isinstance(value, dict) else None


def _error_type(value):
    error = value.get("error") if isinstance(value, dict) else None
    return error.get("type") if isinstance(error, dict) else None


def _generated_plan(output, variant_status):
    """(reason or None, actions) for the rank-1 Plan of one final-model slot."""
    if output is None:
        return (MISSING_PLAN if variant_status == "completed" else VARIANT_NOT_COMPLETED), None
    if not isinstance(output, dict):
        return INVALID_GENERATION, None
    status = output.get("status")
    if status == "failed_incomplete":
        return GENERATION_FAILED, None
    if status != "completed":
        # The runner leaves every slot not_processed when evidence failed and blocked the variant.
        return (GENERATION_NOT_PROCESSED if variant_status == "completed" else VARIANT_NOT_COMPLETED), None
    predictions = output.get("predictions")
    if not isinstance(predictions, list) or len(predictions) < JUDGED_RANK:
        return MISSING_PLAN, None
    prediction = predictions[JUDGED_RANK - 1]
    if not isinstance(prediction, dict):
        return INVALID_GENERATION, None
    plan = prediction.get("plan")
    if plan is None:
        return MISSING_PLAN, None
    if not isinstance(plan, list) or not all(isinstance(action, str) for action in plan):
        return INVALID_GENERATION, None
    actions = [action.strip() for action in plan if action.strip()]
    if not actions:
        return EMPTY_GENERATED_PLAN, []
    return None, actions


@dataclass
class ResultsScan:
    sha256: str
    signature: tuple
    variants_found: list
    items: dict  # variant -> list[PlanItem]
    generator_models: list  # expected final-model aliases (the denominator)
    patients: int  # results.jsonl records
    patients_not_in_results: int  # S/O records the run never wrote; reported, not judged


def scan_results(results_path, labels, patients, variants, generator_models=None):
    """Stream results.jsonl once; return one item per patient x selected variant x generator.

    variants: list of names, or ["all"] for every variant present in the file.
    generator_models: optional allow-list of final-model aliases. Without it the expected
    generators are every model slot seen anywhere in the file, so a variant that was blocked
    or absent for a patient still contributes explicit unavailable rows to the denominator.
    """
    results_path = Path(results_path)
    before = file_signature(results_path)
    digest = hashlib.sha256()
    select_all = list(variants) == ["all"]
    wanted = None if select_all else list(dict.fromkeys(variants))
    found, items, seen_ids, order = [], {}, set(), []
    models_seen = []
    covered = {}  # (variant, patient_id) -> (variant_status, variant_error_type, set of models)
    with open(results_path, "rb") as handle:
        for number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"{results_path}:{number}: invalid JSON") from exc
            patient_id = record.get("id") if isinstance(record, dict) else None
            if not isinstance(patient_id, str):
                raise ValueError(f"{results_path}:{number}: record has no id")
            if patient_id in seen_ids:
                raise ValueError(f"{results_path}:{number}: duplicate patient {patient_id}")
            seen_ids.add(patient_id)
            order.append(patient_id)
            record_variants = record.get("variants")
            if not isinstance(record_variants, dict):
                raise ValueError(f"{results_path}:{number}: expected a schema-4 record with variants "
                                 "(base or agentic run_experiments.py output)")
            label = labels.get(patient_id)
            if label is None:
                raise ValueError(f"No label for results patient {patient_id}")
            if patient_id not in patients:
                raise ValueError(f"No patient S/O record for results patient {patient_id}")
            for name, body in record_variants.items():
                if name not in found:
                    found.append(name)
                plans = body.get("assessment_plans") if isinstance(body, dict) else None
                for model in plans if isinstance(plans, dict) else ():
                    if model not in models_seen:
                        models_seen.append(model)
            for variant in (record_variants if select_all else wanted):
                body = record_variants.get(variant)
                variant_status = _status(body)
                plans = body.get("assessment_plans") if isinstance(body, dict) else None
                present = set()
                for model, output in (plans.items() if isinstance(plans, dict) else ()):
                    if generator_models and model not in generator_models:
                        continue
                    present.add(model)
                    items.setdefault(variant, []).append(
                        _item(variant, patient_id, model, label, output, variant_status, _error_type(body)))
                covered[variant, patient_id] = (variant_status, _error_type(body), present)
    if before != file_signature(results_path):
        raise ValueError("results.jsonl changed while it was being read")
    if not select_all:
        missing = [variant for variant in wanted if variant not in found]
        if missing:
            raise ValueError(f"Variant(s) not in {results_path.name}: {', '.join(missing)}; "
                             f"available: {', '.join(found) or 'none'}")
    if generator_models:
        absent = sorted(set(generator_models) - set(models_seen))
        if absent:
            raise ValueError(f"Generator model(s) not in {results_path.name}: {', '.join(absent)}")
        expected = list(dict.fromkeys(generator_models))
    else:
        expected = models_seen
    selected = found if select_all else wanted
    for variant in selected:
        bucket = items.setdefault(variant, [])
        for patient_id in order:
            variant_status, error_type, present = covered.get((variant, patient_id), (None, None, set()))
            for model in expected:
                if model not in present:
                    bucket.append(_item(variant, patient_id, model, labels[patient_id], None,
                                        variant_status, error_type))
    items = {variant: items[variant] for variant in selected}
    return ResultsScan(digest.hexdigest(), before, found, items, list(expected), len(order),
                       len(set(patients) - seen_ids))


def _item(variant, patient_id, model, label, output, variant_status, variant_error_type):
    condition = str(label.get("condition", "")).replace("_", " ")
    reference = label.get("plan")
    generation_reason, generated = _generated_plan(output, variant_status)
    reference_reason = unavailable_reference_reason(reference)
    reasons = tuple(reason for reason in (generation_reason, reference_reason) if reason)
    source = {"variant_status": variant_status,
              "generation_status": _status(output) if output is not None else None,
              "generation_error_type": _error_type(output) or (
                  variant_error_type if generation_reason == VARIANT_NOT_COMPLETED else None)}
    if generation_reason == EMPTY_GENERATED_PLAN and reference_reason is None:
        status = "empty"
    elif reasons:
        status = "unavailable"
        # An empty Plan against an unusable reference cannot be scored, so the reference reason leads.
        if generation_reason == EMPTY_GENERATED_PLAN:
            reasons = (reference_reason, generation_reason)
    else:
        status = "pending"
    usable = reference_reason is None
    return PlanItem(variant, patient_id, model, status, reasons[0] if reasons else None, condition,
                    list(reference) if usable else None, generated if usable else None, reasons, source)
