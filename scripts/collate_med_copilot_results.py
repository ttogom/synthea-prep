"""Create an offline, human-readable report from saved MED-COPILOT results.

Usage: python scripts/collate_med_copilot_results.py RUN_DIRECTORY

Writes only RUN_DIRECTORY/results_summary.html. No model calls, scoring, runner
imports, or configuration changes are involved. Historical single-model and
current multi-model records are supported. An explicit evidence_variant on a
record (or run.experiment) is recognized; absent variants are never guessed.
"""

import argparse
from collections import Counter
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, pstdev
import sys
import tempfile


# Standalone execution must find the lightweight registry from any directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from scripts.med_copilot_variants import VARIANTS as REGISTERED_VARIANTS


REPORT_FILE = "results_summary.html"
# Match the saved evaluator contract without importing its model dependencies.
METRICS = ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1")
DISPLAY_METRICS = ("BERTScore_F1", "ROUGE-L", "BLEU", "METEOR", "ROUGE-1", "ROUGE-2")
VARIANTS = {**{key: definition.description for key, definition in REGISTERED_VARIANTS.items()},
            "not_recorded": "Evidence variant not recorded"}
SCORE_REASONS = {
    "scored": "Scored",
    "skipped_no_recorded_plan": "No reference plan recorded; scoring skipped",
    "missing_reference": "Reference unavailable",
    "missing_generation": "Prediction unavailable",
    "error": "Scoring failed",
    "interrupted": "Scoring interrupted",
    "pending": "Scoring not attempted",
    "not_scored": "Scoring not recorded",
    "not_processed": "Generation not completed; scoring unavailable",
    "invalid_metrics": "Saved scores are incomplete or invalid",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json_object(text, location):
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc


def _json_lines(text, location):
    return [_json_object(line, f"{location}:{number}")
            for number, line in enumerate(text.split("\n"), 1) if line.strip()]


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _error_type(error):
    # Raw exception messages can contain credentials or internal run details.
    name = error.get("type") if isinstance(error, dict) else None
    return name if isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) else None


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _score(entry, generated):
    if not isinstance(entry, dict):
        status = "not_scored" if generated else "not_processed"
        return {"status": status, "metrics": dict.fromkeys(METRICS),
                "reason": SCORE_REASONS[status]}
    status = entry.get("status", "not_scored")
    if not isinstance(status, str):
        status = "invalid_metrics"
    metrics = entry.get("metrics")
    if status == "scored":
        valid = (isinstance(metrics, dict) and set(metrics) == set(METRICS)
                 and all(_finite_number(metrics[key]) for key in METRICS))
        if not valid:
            status = "invalid_metrics"
    values = {key: metrics[key] for key in METRICS} if status == "scored" else dict.fromkeys(METRICS)
    reason = SCORE_REASONS.get(status, "Scores unavailable")
    error_type = _error_type(entry.get("error"))
    if error_type:
        reason += f" ({error_type})"
    return {"status": status, "metrics": values, "reason": reason}


def _reported_model(responses):
    if isinstance(responses, list):
        for response in reversed(responses):
            if isinstance(response, dict) and _text(response.get("reported_model")):
                return response["reported_model"]
    return None


def _legacy_model(row):
    responses = row.get("model_responses")
    return _reported_model(responses.get("assessment_plan")) if isinstance(responses, dict) else None


def _variant_labels(metadata):
    # Saved identities remain reportable after a variant is changed or removed
    # from the current registry. Only the description is copied into the report.
    labels = {**{key: definition.description for key, definition in REGISTERED_VARIANTS.items()},
              "not_recorded": VARIANTS["not_recorded"]}
    definitions = metadata.get("variant_definitions")
    if definitions is None:
        return labels
    if not isinstance(definitions, dict):
        raise ValueError("run.json.variant_definitions must be an object")
    for variant, definition in definitions.items():
        if not _text(variant) or variant == "not_recorded" or not isinstance(definition, dict):
            raise ValueError("run.json.variant_definitions must contain named variant objects")
        description = definition.get("description", labels.get(variant, variant))
        if not _text(description):
            raise ValueError(f"Saved description for variant {variant!r} must be a nonempty string")
        labels[variant] = description
    return labels


def _variant(row, metadata, labels):
    experiment = metadata.get("experiment") or {}
    value = row.get("evidence_variant", experiment.get("evidence_variant"))
    if value is None:
        return "not_recorded"
    if not isinstance(value, str) or value not in labels:
        raise ValueError(f"Unsupported evidence_variant for patient {row.get('id')!r}: {value!r}")
    return value


def _output(slot, model_id, model_name, variant):
    if not isinstance(slot, dict):
        raise ValueError(f"Model output {model_id!r} must be an object")
    status = slot.get("status")
    if status not in ("completed", "failed_incomplete", "not_processed"):
        raise ValueError(f"Model output {model_id!r} has an invalid generation status")
    predictions = slot.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 3:
        predictions = [None] * 3
    evaluations = slot.get("evaluation")
    if evaluations is None:
        evaluations = []
    if not isinstance(evaluations, list):
        raise ValueError(f"Model output {model_id!r} evaluation must be an array or null")
    by_rank = {}
    for item in evaluations:
        if (not isinstance(item, dict) or type(item.get("rank")) is not int
                or item["rank"] not in (1, 2, 3) or item["rank"] in by_rank):
            raise ValueError(f"Model output {model_id!r} has invalid or duplicate score ranks")
        by_rank[item["rank"]] = item
    pairs = []
    for rank, prediction in enumerate(predictions, 1):
        prediction = prediction if isinstance(prediction, dict) else {}
        condition = prediction.get("condition")
        condition = condition if _text(condition) else None
        plan = prediction.get("plan")
        if not (isinstance(plan, list) and len(plan) <= 3 and all(_text(step) for step in plan)):
            plan = None
        evaluation = by_rank.get(rank, {})
        pairs.append({"rank": rank, "condition": condition, "plan": plan,
                      "scores": {field: _score(evaluation.get(field), status == "completed")
                                 for field in ("condition", "plan")}})
    responses = slot.get("model_responses")
    if isinstance(responses, dict):
        responses = responses.get("assessment_plan")
    error_type = _error_type(slot.get("error"))
    problem = "Generation failed" if status == "failed_incomplete" else "Generation not attempted"
    if error_type:
        problem += f" ({error_type})"
    return {"model_id": model_id, "model_name": _reported_model(responses) or model_name,
            "variant": variant, "status": status,
            "generation_note": None if status == "completed" else problem,
            "predictions": pairs}


def _summarize(patients):
    groups = {}
    scores = {}
    for patient in patients:
        for output in patient["outputs"]:
            for prediction in output["predictions"]:
                for field, entry in prediction["scores"].items():
                    key = (output["model_id"], output["variant"], prediction["rank"], field)
                    group = groups.setdefault(key, {
                        "model_id": key[0], "variant": key[1], "rank": key[2], "field": key[3],
                        "total": 0, "counts": Counter(), "values": [],
                    })
                    group["total"] += 1
                    group["counts"][entry["status"]] += 1
                    if entry["status"] == "scored":
                        group["values"].append(entry["metrics"])
                        scores[(*key, patient["id"])] = entry["metrics"]
    summary = []
    comparisons = []
    for key, group in groups.items():
        values = group.pop("values")
        group["counts"] = dict(group["counts"])
        group["means"] = {metric: math.fsum(row[metric] / len(values) for row in values)
                          if values else None for metric in METRICS}
        summary.append(group)
        model, variant, rank, field = key
        if variant in ("llm_only", "not_recorded"):
            continue
        pairs = [(scores[(*key, patient["id"])], scores[(model, "llm_only", rank, field, patient["id"])])
                 for patient in patients if (*key, patient["id"]) in scores
                 and (model, "llm_only", rank, field, patient["id"]) in scores]
        comparisons.append({"model_id": model, "variant": variant, "rank": rank, "field": field,
                            "paired_patients": len(pairs),
                            "differences": {metric: math.fsum(
                                left[metric] / len(pairs) - right[metric] / len(pairs)
                                for left, right in pairs) if pairs else None for metric in METRICS}})
    return summary, comparisons


def load_report_data(run_dir, *, root=ROOT):
    """Read saved outputs and verified labels, returning only reportable results.

    Missing model slots and selected patients are shown as unprocessed rather
    than silently excluded. Input artifacts and external services are untouched.
    """
    run_dir = Path(run_dir).resolve()
    root = Path(root).resolve()
    metadata = _json_object((run_dir / "run.json").read_text(encoding="utf-8"), run_dir / "run.json")
    if metadata.get("experiment") is not None and not isinstance(metadata["experiment"], dict):
        raise ValueError("run.json.experiment must be an object")
    variant_labels = _variant_labels(metadata)
    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict) or not _text(evaluation.get("labels_path")):
        raise ValueError("run.json.evaluation.labels_path is required to join the reference labels")
    labels_path = root / evaluation["labels_path"]
    source = labels_path.read_bytes()
    expected_hash = evaluation.get("labels_sha256")
    if expected_hash is not None and hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError(f"Reference labels no longer match the saved labels_sha256: {labels_path}")
    labels = {}
    for label in _json_lines(source.decode("utf-8"), labels_path):
        patient_id = label.get("id")
        if not _text(patient_id) or patient_id in labels:
            raise ValueError(f"Invalid or duplicate reference patient ID: {patient_id!r}")
        if not _text(label.get("condition")) or not (
                isinstance(label.get("plan"), list) and all(isinstance(step, str) for step in label["plan"])):
            raise ValueError(f"Invalid reference condition or plan for patient {patient_id!r}")
        labels[patient_id] = label
    rows = _json_lines((run_dir / "results.jsonl").read_text(encoding="utf-8"), run_dir / "results.jsonl")
    selected_ids = evaluation.get("selected_patient_ids", [])
    if (not isinstance(selected_ids, list) or not all(_text(value) for value in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("run.json selected_patient_ids must be unique, nonempty strings")
    # Schemas 3 and 4 store explicit variant/model slots in one patient row;
    # component-scoped evidence in schema 4 does not change final-score handling.
    # Normalize it into the same report records used for historical schemas.
    selected_variants = (metadata.get("experiment") or {}).get("selected_variants", [])
    if (not isinstance(selected_variants, list) or any(not isinstance(v, str) or v not in variant_labels or v == "not_recorded"
            for v in selected_variants) or len(set(selected_variants)) != len(selected_variants)):
        raise ValueError("run.json selected_variants must be unique supported variant IDs")
    if metadata.get("schema_version") in (3, 4):
        schema = metadata["schema_version"]
        if not selected_variants:
            raise ValueError(f"Schema v{schema} requires selected_variants")
        existing = set()
        for row in rows:
            patient_id = row.get("id")
            if not _text(patient_id) or patient_id in existing:
                raise ValueError(f"Schema v{schema} requires unique, nonempty patient IDs")
            if "variants" not in row:
                raise ValueError(f"Schema v{schema} requires variants in every patient row")
            existing.add(patient_id)
        rows.extend({"id": patient_id, "status": "not_processed", "variants": {}}
                    for patient_id in selected_ids if patient_id not in existing)
    expanded = []
    for row in rows:
        if "variants" not in row:
            expanded.append(row)
            continue
        variants = row["variants"]
        if not isinstance(variants, dict):
            raise ValueError("variants must be an object")
        for variant in dict.fromkeys([*selected_variants, *variants]):
            if variant not in variant_labels or variant == "not_recorded":
                raise ValueError(f"Unsupported evidence_variant: {variant!r}")
            entry = variants.get(variant, {})
            if not isinstance(entry, dict) or not isinstance(entry.get("assessment_plans", {}), dict):
                raise ValueError("Variant and assessment_plans must be objects")
            expanded.append({"id": row.get("id"), "status": entry.get("status", "not_processed"),
                             "evidence_variant": variant, "assessment_plans": entry.get("assessment_plans", {})})
    rows = expanded
    for row in rows:
        if "assessment_plans" in row and not isinstance(row["assessment_plans"], dict):
            raise ValueError(f"assessment_plans must be an object for patient {row.get('id')!r}")
    models_metadata = metadata.get("assessment_plan_models") or {}
    if not isinstance(models_metadata, dict):
        raise ValueError("run.json.assessment_plan_models must be an object")
    configured = models_metadata.get("models") or {}
    if not isinstance(configured, dict):
        raise ValueError("run.json.assessment_plan_models.models must be an object")
    model_ids = models_metadata.get("selected_models", [])
    if not isinstance(model_ids, list) or not all(_text(model) for model in model_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("run.json selected model IDs must be unique, nonempty strings")
    legacy_names = list(dict.fromkeys(name for row in rows if (name := _legacy_model(row))))
    legacy_default = legacy_names[0] if len(legacy_names) == 1 else "Model not recorded"
    existing_ids = {row.get("id") for row in rows if _text(row.get("id"))}
    recorded_model_ids = list(dict.fromkeys(model for row in rows
                                           for model in row.get("assessment_plans", {})))
    multi_model = bool(model_ids) or any("assessment_plans" in row for row in rows)
    rows.extend({"id": patient_id, "status": "not_processed",
                 **({"assessment_plans": {model: {"status": "not_processed"}
                                           for model in model_ids or recorded_model_ids}} if multi_model else {})}
                for patient_id in selected_ids if patient_id not in existing_ids)
    patients = {}
    seen = set()
    for row in rows:
        patient_id = row.get("id")
        if not _text(patient_id) or patient_id not in labels:
            raise ValueError(f"No reference label for result patient {patient_id!r}")
        variant = _variant(row, metadata, variant_labels)
        if (patient_id, variant) in seen:
            raise ValueError(f"Duplicate result for patient {patient_id!r}, variant {variant!r}")
        seen.add((patient_id, variant))
        label = labels[patient_id]
        patient = patients.setdefault(patient_id, {
            "id": patient_id, "reference_condition": label["condition"],
            "reference_plan": label["plan"], "outputs": [],
        })
        if "assessment_plans" in row:
            slots = row["assessment_plans"]
            if not isinstance(slots, dict):
                raise ValueError(f"assessment_plans must be an object for patient {patient_id!r}")
            for model_id in dict.fromkeys([*model_ids, *slots]):
                if not _text(model_id):
                    raise ValueError("Model IDs must be nonempty strings")
                settings = configured.get(model_id, {})
                if not isinstance(settings, dict) or not isinstance(settings.get("config", {}), dict):
                    raise ValueError(f"Recorded settings for model {model_id!r} must be objects")
                name = settings.get("config", {}).get("model") or model_id
                slot = slots.get(model_id, {"status": "not_processed"})
                patient["outputs"].append(_output(slot, model_id, name, variant))
        else:
            name = _legacy_model(row) or legacy_default
            patient["outputs"].append(_output(row, name, name, variant))
    patients = list(patients.values())
    summary, comparisons = _summarize(patients)
    notices = []
    if any(output["variant"] == "not_recorded" for patient in patients for output in patient["outputs"]):
        notices.append("Evidence variants were not recorded for some outputs. They are shown as “not recorded”; "
                       "no variant has been inferred and those outputs are excluded from baseline comparisons.")
    if expected_hash is None:
        notices.append("Reference labels have no saved fingerprint; their original contents could not be verified.")
    if not patients:
        notices.append("No patient results were saved or selected. There are no scores to summarize.")
    return {"patients": patients, "summary": summary, "comparisons": comparisons, "notices": notices,
            "selected_models": model_ids, "selected_variants": selected_variants,
            "variant_labels": variant_labels}


def _e(value):
    return escape(str(value), quote=True)


def _number(value, *, signed=False):
    if value is None:
        return '<span class="na">N/A</span>'
    formatted = f"{value:+.4f}" if signed else f"{value:.4f}"
    return f'<span title="{_e(value)}">{formatted}</span>'


def _plan(plan, *, reference=False):
    if plan is None:
        return '<p class="unavailable">Plan unavailable.</p>'
    if reference and plan == ["none recorded"]:
        return '<p class="unavailable">No plan recorded in label.</p>'
    if not plan:
        return '<p class="unavailable">No plan steps recorded.</p>'
    return '<ol class="plan">' + ''.join(f'<li>{_e(step)}</li>' for step in plan) + '</ol>'


def _metric_header():
    return ''.join(f'<th scope="col">{_e(metric)}</th>' for metric in DISPLAY_METRICS)


def _patient_html(patient, index, variant_labels):
    outputs = []
    for output in patient["outputs"]:
        status = output["status"]
        note = output["generation_note"] or "Generation completed"
        pairs = []
        for pair in output["predictions"]:
            rows = []
            for field, entry in pair["scores"].items():
                rank = (f'<th class="rank-cell" scope="rowgroup" rowspan="2">{pair["rank"]}</th>'
                        if field == "condition" else '')
                prediction = (_e(pair["condition"]) if pair["condition"] else '<span class="unavailable">Condition unavailable</span>') if field == "condition" else _plan(pair["plan"])
                rows.append(f'<tr data-field="{field}">{rank}<th class="prediction-field" scope="row">'
                            f'<span class="field-kind">{field.title()}</span>{prediction}'
                            f'<span class="score-reason">{_e(entry["reason"])}</span></th>'
                            + ''.join(f'<td>{_number(entry["metrics"][metric])}</td>' for metric in DISPLAY_METRICS)
                            + '</tr>')
            pairs.append(f'<tbody class="prediction" data-rank="{pair["rank"]}">{"".join(rows)}</tbody>')
        reported = f'<p class="model-name">Reported model: {_e(output["model_name"])}</p>'
        outputs.append(f'<details class="model-output" data-model="{_e(output["model_id"])}" '
                       f'data-variant="{_e(output["variant"])}"><summary><strong>{_e(output["model_id"])}</strong>'
                       f'<span class="variant-name">{_e(variant_labels[output["variant"]])}'
                       f'<span class="variant-id">{_e(output["variant"])}</span></span>'
                       f'<span class="badge {status}">{_e(note)}</span></summary>'
                       f'<div class="model-body">{reported}<div class="table-scroll"><table class="score-table">'
                       '<caption class="sr-only">All three ranked condition and plan predictions with all six scores</caption>'
                       f'<thead><tr><th scope="col">Rank</th><th scope="col">Prediction / scoring</th>{_metric_header()}</tr></thead>'
                       f'{"".join(pairs)}</table></div></div></details>')
    search = (patient["id"] + ' ' + patient["reference_condition"] + ' '
              + patient["reference_condition"].replace('_', ' ')).lower()
    return (f'<details class="patient" data-search="{_e(search)}" id="patient-{index}">'
            f'<summary><span class="patient-heading"><strong>{_e(patient["reference_condition"].replace("_", " "))}</strong>'
            f'<span class="patient-id" title="{_e(patient["id"])}">{_e(patient["id"])}</span></span>'
            '<span class="expand-hint" aria-hidden="true">Expand / collapse <span>+</span></span></summary>'
            '<div class="patient-body"><section class="reference"><div><span class="eyebrow">Reference condition</span>'
            f'<h3>{_e(patient["reference_condition"])}</h3></div><div><span class="eyebrow">Reference plan</span>'
            f'{_plan(patient["reference_plan"], reference=True)}</div></section>{"".join(outputs)}</div></details>')


def _top_rank_groups(data):
    groups = {model: {variant: {"condition": [], "plan": []}
                      for variant in data.get("selected_variants", [])}
              for model in data.get("selected_models", [])}
    for patient in data["patients"]:
        for output in patient["outputs"]:
            fields = groups.setdefault(output["model_id"], {}).setdefault(
                output["variant"], {"condition": [], "plan": []})
            first = output["predictions"][0]
            for field in fields:
                score = first["scores"][field]
                if score["status"] == "scored":
                    fields[field].append(score["metrics"]["BERTScore_F1"])
    return groups


def _best_variant(variants):
    scored = [variant for variant, fields in variants.items() if fields["condition"]]
    return max(scored, key=lambda variant: mean(variants[variant]["condition"])) if scored else None


def format_top_rank_summary(data):
    """Summarize top-1 BERTScore only; choose best variant by condition mean.

    Missing/error/skipped scores are excluded, never replaced by zero. Ties
    retain report/config order. Plan scores are independent, without a composite.
    """
    groups = _top_rank_groups(data)
    lines = ["Top-ranked condition/Plan BERTScore_F1 (higher is better).",
             "Best variant: highest mean CONDITION BERTScore_F1; ties retain configured order.",
             "Scored subsets can differ; compare counts. No condition/Plan composite is calculated."]
    for model, variants in groups.items():
        lines.append(f"\nModel: {model}")
        for variant, fields in variants.items():
            values = []
            for field, scores in fields.items():
                values.append(f"{field}: {mean(scores):.4f} (n={len(scores)})" if scores else f"{field}: N/A (n=0)")
            lines.append(f"  {variant}: " + "; ".join(values))
        best = _best_variant(variants)
        if best is None:
            lines.append("  Best variant: unavailable (no scored top-ranked conditions)")
            continue
        lines.append(f"  Best variant: {best}")
        for field, scores in variants[best].items():
            lines.append(f"    Top-ranked {field}: " + (
                f"mean={mean(scores):.4f}, population_sd={pstdev(scores):.4f}, "
                f"min={min(scores):.4f}, max={max(scores):.4f}, n={len(scores)}"
                if scores else "N/A (n=0)"))
    return "\n".join(lines)


def _display_variant_labels(data):
    labels = dict(data.get("variant_labels", VARIANTS))
    if labels.get("patient_retrieval_guidelines") == "S/O + both evidence sources":
        labels["patient_retrieval_guidelines"] = "S/O + similar-patient example + clinical guidelines"
    return labels


def _statistics_groups(data):
    groups = {}
    for patient in data["patients"]:
        for output in patient["outputs"]:
            ranks = groups.setdefault((output["model_id"], output["variant"]), {})
            for prediction in output["predictions"]:
                fields = ranks.setdefault(prediction["rank"], {})
                for field, score in prediction["scores"].items():
                    entry = fields.setdefault(field, {"counts": Counter(), "values": {metric: [] for metric in METRICS}})
                    entry["counts"][score["status"]] += 1
                    if score["status"] == "scored":
                        for metric in METRICS:
                            entry["values"][metric].append(score["metrics"][metric])
    return groups


def _variant_statistics_html(ranks, model, variant):
    groups = []
    for rank in (1, 2, 3):
        rows = []
        for field in ("condition", "plan"):
            entry = ranks.get(rank, {}).get(field, {})
            counts = entry.get("counts", {})
            scored, total = counts.get("scored", 0), sum(counts.values())
            unavailable = [f'{count} {SCORE_REASONS.get(status, "unavailable").lower()}'
                           for status, count in counts.items() if status != "scored" and count]
            coverage = f'{scored}/{total} scored' + ('; ' + '; '.join(unavailable) if unavailable else '')
            for index, metric in enumerate(DISPLAY_METRICS):
                scores = entry.get("values", {}).get(metric, [])
                values = (mean(scores), pstdev(scores), min(scores), max(scores)) if scores else (None,) * 4
                rank_cell = f'<th scope="rowgroup" rowspan="12" class="rank-cell">{rank}</th>' if field == "condition" and index == 0 else ''
                field_cell = (f'<th scope="rowgroup" rowspan="6" class="stat-field">{field.title()}'
                              f'<span class="score-reason">{_e(coverage)}</span></th>') if index == 0 else ''
                rows.append(f'<tr data-field="{field}" data-metric="{_e(metric)}">{rank_cell}{field_cell}'
                            f'<th scope="row">{_e(metric)}</th>'
                            + ''.join(f'<td>{_number(value)}</td>' for value in values)
                            + f'<td>{len(scores)}</td></tr>')
        groups.append(f'<tbody data-rank="{rank}">{"".join(rows)}</tbody>')
    return ('<div class="table-scroll"><table class="variant-statistics-table">'
            f'<caption class="sr-only">All ranks and six-metric statistics for {_e(model)} / {_e(variant)}</caption>'
            '<thead><tr><th scope="col">Rank</th><th scope="col">Field / coverage</th><th scope="col">Metric</th>'
            '<th scope="col">Mean</th><th scope="col">Population SD</th><th scope="col">Min</th>'
            '<th scope="col">Max</th><th scope="col">Scored patients</th></tr></thead>'
            + ''.join(groups) + '</table></div>')


def _top_rank_html(data, labels):
    statistics = _statistics_groups(data)
    cards = []
    for model, variants in _top_rank_groups(data).items():
        best = _best_variant(variants)
        if best is None:
            best_label = '<p class="top-rank-unavailable">No scored top-ranked conditions; best variant unavailable.</p>'
        else:
            best_label = (f'<div class="top-rank-best"><span class="eyebrow">Best evidence variant</span>'
                          f'<strong>{_e(labels.get(best, best))}</strong>'
                          f'<span class="top-rank-variant-id">{_e(best)}</span></div>')
        rows = []
        configured_order = {variant: index for index, variant in enumerate(variants)}
        ordered = sorted(variants, key=lambda variant: (not bool(variants[variant]["condition"]),
                         -mean(variants[variant]["condition"]) if variants[variant]["condition"] else 0))
        for variant in ordered:
            fields = variants[variant]
            cells = []
            for scores in fields.values():
                cells.append(f'<td><span class="top-rank-value">{_number(mean(scores) if scores else None)}</span>'
                             f'<span class="top-rank-count">n={len(scores)}</span></td>')
            winning = variant == best
            marker = '<span class="top-rank-winner">Best</span>' if winning else ''
            scores = {field: mean(values) if values else None for field, values in fields.items()}
            attributes = ' '.join(f'data-{field}-score="{scores[field] if scores[field] is not None else ""}"' for field in ("condition", "plan"))
            table = _variant_statistics_html(statistics.get((model, variant), {}), model, variant)
            rows.append(f'<tbody class="top-rank-variant" data-variant="{_e(variant)}" data-order="{configured_order[variant]}" {attributes}>'
                        f'<tr data-variant="{_e(variant)}" data-best="{str(winning).lower()}">'
                        f'<th scope="row"><span class="top-rank-variant-name">{_e(labels.get(variant, variant))}{marker}</span>'
                        f'<span class="top-rank-variant-id">{_e(variant)}</span></th>{"".join(cells)}</tr>'
                        '<tr class="variant-details-row"><td colspan="3">'
                        f'<details class="variant-statistics" data-variant="{_e(variant)}">'
                        f'<summary>All three ranks · all six metrics <span class="sr-only">for {_e(labels.get(variant, variant))}</span></summary>'
                        f'{table}</details></td></tr></tbody>')
        table = ('<div class="table-scroll"><table class="top-rank-table" data-sort-field="condition" data-sort-direction="descending">'
                 f'<caption class="sr-only">Top-ranked BERTScore by evidence variant for {_e(model)}</caption>'
                 '<thead><tr><th scope="col">Evidence variant</th><th scope="col" aria-sort="descending">'
                 '<button type="button" class="score-sort" data-field="condition">Condition <span class="sort-indicator" aria-hidden="true">↓</span></button></th>'
                 '<th scope="col" aria-sort="none"><button type="button" class="score-sort" data-field="plan">Plan <span class="sort-indicator" aria-hidden="true">↕</span></button></th></tr></thead>'
                 f'{"".join(rows)}</table></div>') if rows else '<p class="unavailable">No evidence variants available.</p>'
        cards.append(f'<article class="top-rank-card" data-model="{_e(model)}">'
                     f'<h3>{_e(model)}</h3>{best_label}{table}</article>')
    return ''.join(cards) if cards else '<p class="empty">No model results available.</p>'


def render_report(data):
    """Render a self-contained HTML page; all source text is escaped, never executed."""
    patients = data["patients"]
    variant_labels = _display_variant_labels(data)
    outputs = [output for patient in patients for output in patient["outputs"]]
    models = list(dict.fromkeys(output["model_id"] for output in outputs))
    variants = list(dict.fromkeys(output["variant"] for output in outputs))
    entries = [entry for output in outputs for pair in output["predictions"] for entry in pair["scores"].values()]
    completed = sum(bool(patient["outputs"]) and all(output["status"] == "completed"
                    for output in patient["outputs"]) for patient in patients)
    scored = sum(entry["status"] == "scored" for entry in entries)
    skipped = sum(entry["status"] in ("skipped_no_recorded_plan", "missing_reference") for entry in entries)
    unavailable = len(entries) - scored - skipped
    invalid_predictions = sum(pair["condition"] is None or pair["plan"] is None or pair["plan"] == []
                              for output in outputs if output["status"] == "completed" for pair in output["predictions"])
    attention = len(patients) - completed + unavailable + invalid_predictions
    health = ("Results need attention" if attention else "All patients evaluated" if patients else "No results available")
    health_detail = (f'{completed}/{len(patients)} patients completed generation for all selected models and evidence variants. '
                     f'{scored} condition/plan fields scored; '
                     f'{skipped} skipped; {unavailable} unavailable.')
    notices = ''.join(f'<p class="notice">{_e(notice)}</p>' for notice in data["notices"])
    options = lambda values, labels: ''.join(f'<option value="{_e(value)}">{_e(labels.get(value, value))}</option>' for value in values)
    content = {
        "STYLE": _STYLE,
        "SCRIPT": _SCRIPT,
        "HEALTH": _e(health), "HEALTH_CLASS": "attention" if attention else "healthy",
        "HEALTH_DETAIL": _e(health_detail), "PATIENTS": str(len(patients)), "MODELS": str(len(models)),
        "NOTICES": notices, "MODEL_OPTIONS": options(models, {}), "VARIANT_OPTIONS": options(variants, variant_labels),
        "TOP_RANK_SUMMARY": _top_rank_html(data, variant_labels),
        "METRIC_HEADER": _metric_header(),
        "DETAILS": ''.join(_patient_html(patient, index, variant_labels) for index, patient in enumerate(patients)),
        # Escape script boundaries, including malicious labels and Unicode separators.
        "DATA": json.dumps(data, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026'),
    }
    return re.sub(r"@@([A-Z_]+)@@", lambda match: content[match[1]], _PAGE)


def generate_report(run_dir, *, root=ROOT):
    """Atomically write results_summary.html, preserving any old report on failure."""
    run_dir = Path(run_dir).resolve()
    html = render_report(load_report_data(run_dir, root=root))
    destination = run_dir / REPORT_FILE
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir,
                                         prefix=".results_summary-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(html)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main(argv=None):
    """Standalone entry point; the runner also generates this report after a batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Directory containing run.json and results.jsonl")
    args = parser.parse_args(argv)
    try:
        destination = generate_report(args.run_directory)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot generate results summary: {exc}\n")
    print(f"Results summary written to {destination}")
    return 0


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MED-COPILOT · Results summary</title><style>@@STYLE@@</style></head>
<body><header><div class="brand"><span class="brand-mark" aria-hidden="true">M</span> MED-COPILOT <span class="brand-rule"></span> EVALUATION</div>
<div class="hero"><div><p class="eyebrow">Predictions &amp; reference comparison</p><h1>Results summary<span>.</span></h1>
<p class="intro">Review model predictions, their corresponding plans, and six evaluation metrics for every patient.</p></div>
<div class="health @@HEALTH_CLASS@@"><strong>@@HEALTH@@</strong><p>@@HEALTH_DETAIL@@</p></div></div>
<div class="stats"><div><span>Patients</span><strong>@@PATIENTS@@</strong></div><div><span>Models</span><strong>@@MODELS@@</strong></div>
</div></header>
<main><noscript><p class="notice">JavaScript is disabled. All patients and expandable details remain available. Sorting and filtering require JavaScript.</p></noscript>
@@NOTICES@@
<section aria-labelledby="top-rank-heading"><div class="section-heading"><h2 id="top-rank-heading">Best evidence variant by model</h2></div>
<p class="section-note">Selected by the highest mean condition BERTScore_F1 for the first-ranked prediction. Higher is better. Plan scores are shown separately.</p>
<div class="top-rank-cards">@@TOP_RANK_SUMMARY@@</div>
<p class="metric-note">Click Condition or Plan to sort rank-1 means. Missing scores stay last. Counts (n) may differ; ties follow configured order. Expand any variant for all six metrics at ranks 1–3, including mean, population SD, range, and scored patient counts.</p></section>
<section aria-labelledby="patient-heading"><div class="section-heading"><div><h2 id="patient-heading">Patient predictions</h2></div>
<button id="expand-all" type="button">Expand shown patients</button></div>
<div class="filters"><label>Model<select id="model-filter"><option value="">All models</option>@@MODEL_OPTIONS@@</select></label>
<label>Evidence<select id="variant-filter"><option value="">All recorded outputs</option>@@VARIANT_OPTIONS@@</select></label>
<p>Filters apply to patient details.</p><button id="reset" type="button">Reset filters</button></div>
<div class="patient-tools"><label>Find a patient<input id="patient-search" type="search" placeholder="Search patient ID or reference condition…"></label>
<span id="patient-count" aria-live="polite">@@PATIENTS@@ patients</span></div>
<p class="section-note">The first 10 matching patients are shown initially. Search checks all patients. Expand a patient, then a model/variant row, to see all predictions and scores. Sticky headers and the floating collapse controls remain available while scrolling.</p>
<div id="patients">@@DETAILS@@</div><p id="patients-empty" class="empty" hidden>No patients match these filters.</p>
<div class="patient-pagination"><button id="show-more" type="button" hidden>Show 10 more patients</button>
<button id="show-less" type="button" hidden>Show first 10 patients</button></div></section>
<div id="collapse-controls" class="collapse-controls" role="group" aria-label="Collapse patient details" hidden>
<button id="collapse-current" type="button">Collapse current patient</button><button id="collapse-all" type="button">Collapse all patients</button></div>
<footer>Saved predictions and scores only. No generation or scoring is performed by this report. Scores measure text similarity, not clinical correctness. Values display to four decimal places; hover for full precision. N/A means unavailable, not zero.</footer></main>
<script id="report-data" type="application/json">@@DATA@@</script><script>@@SCRIPT@@</script></body></html>'''


_STYLE = r'''
:root{--ink:#1b302e;--muted:#647570;--accent:#197266;--pale:#eaf3ef;--line:#dbe4de;--paper:#fbfcf9;--amber:#8d5f18}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header,main{max-width:1280px;margin:auto;padding:24px 32px}header{padding-bottom:10px}main{padding-top:8px;padding-bottom:85px}
.brand{display:flex;align-items:center;gap:10px;font-size:10px;letter-spacing:1.7px;font-weight:700;color:var(--muted)}.brand-mark{display:grid;place-items:center;width:28px;height:28px;background:var(--accent);color:white;font-size:15px;border-radius:7px;letter-spacing:0}.brand-rule{height:14px;width:1px;background:var(--line);margin:0 4px}
.hero{display:flex;justify-content:space-between;gap:28px;align-items:center;margin:28px 0 22px}.eyebrow{display:block;font-size:9px;letter-spacing:1.1px;font-weight:700;text-transform:uppercase;color:var(--muted);margin:0 0 6px}h1{font-size:36px;font-weight:600;letter-spacing:-1.4px;line-height:1.15;margin:0 0 9px}h1 span{color:var(--accent)}.intro{max-width:560px;color:var(--muted);margin:0;font-size:13px}
.health{max-width:310px;padding:12px 14px;background:var(--pale);border-radius:8px;border-left:3px solid var(--accent);flex-shrink:0}.health strong{font-size:12px}.health p{font-size:11px;color:var(--muted);margin:4px 0 0}.health.attention{background:#fff5e5;border-color:var(--amber)}.stats{display:grid;grid-template-columns:repeat(2,1fr);border:1px solid var(--line);border-radius:9px;background:white;overflow:hidden}.stats>div{padding:12px 18px;border-right:1px solid var(--line)}.stats>div:last-child{border:0}.stats span{display:block;font-size:11px;color:var(--muted);margin-bottom:4px}.stats strong{font-size:23px;line-height:1;font-weight:600}
main>section{margin-top:26px}.section-heading{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:9px}h2{font-size:21px;line-height:1.3;letter-spacing:-.5px;font-weight:600;margin:0}.section-note{font-size:11px;color:var(--muted);margin:0 0 12px;line-height:1.6}.metric-note{font-size:11px;color:var(--muted);margin:10px 0 0}.notice{border:1px solid #e8dfca;background:#fffaf0;border-radius:7px;color:#74613d;font-size:11px;padding:9px 12px;margin:12px 0}
button{border:1px solid #cbdad0;border-radius:5px;background:white;color:var(--accent);font:600 11px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:7px 10px;cursor:pointer;white-space:nowrap}button:hover{background:var(--pale)}:focus-visible{outline:3px solid #7fc3af;outline-offset:2px}label{font-size:10px;color:var(--muted);font-weight:600;display:block}select,input{display:block;margin-top:4px;padding:7px 25px 7px 9px;border:1px solid #ccd9d0;border-radius:5px;background:white;color:var(--ink);font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:32px;max-width:100%}select{min-width:150px}input{width:340px;padding-right:8px}.filters{display:flex;align-items:end;gap:12px;padding:8px 0 12px;border-bottom:1px solid var(--line)}.filters p{font-size:10px;color:var(--muted);margin:0 auto 8px 0}.patient-tools{display:flex;align-items:end;justify-content:space-between;gap:14px;margin:10px 0 8px}.patient-tools>span{font-size:11px;color:var(--muted);margin-bottom:7px}
.table-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:white}table{border-collapse:collapse;width:100%;font-size:11px;text-align:left}th,td{padding:6px 8px;border-bottom:1px solid var(--line)}thead th{font-size:9px;color:var(--muted);background:#f3f6f1;font-weight:600;white-space:nowrap}tbody th{font-weight:500}td{text-align:right;font-variant-numeric:tabular-nums}thead th:not(:first-child){text-align:right}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}.na{color:#89948d;font-size:10px}.empty{padding:16px;text-align:center;color:var(--muted);font-size:11px;margin:0}
.top-rank-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:14px}.top-rank-card{min-width:0;padding:14px;background:white;border:1px solid var(--line);border-radius:8px}.top-rank-card h3{font-size:15px;font-weight:600;margin:0;overflow-wrap:anywhere}.top-rank-best{padding:8px 10px;margin:10px 0;background:var(--pale);border-radius:6px}.top-rank-best .eyebrow{font-size:9px;margin-bottom:3px}.top-rank-best strong{display:block;font-size:12px;font-weight:600;overflow-wrap:anywhere}.top-rank-variant-id{display:block;font-size:9px;font-weight:400;color:var(--muted);overflow-wrap:anywhere;margin-top:2px}.top-rank-variant-name{font-weight:500;overflow-wrap:anywhere}.top-rank-winner{display:inline-block;margin-left:6px;border-radius:3px;padding:1px 4px;background:#d9ece1;color:var(--accent);font-size:9px;font-weight:700}.top-rank-unavailable{font-size:11px;color:var(--muted);margin:10px 0}
.top-rank-table>thead>tr>th,.top-rank-table>tbody>tr>th,.top-rank-table>tbody>tr>td{padding:7px 10px}.top-rank-table>tbody>tr>th{min-width:135px;font-size:11px}.top-rank-table>tbody>tr>td{min-width:92px;white-space:nowrap}.top-rank-table tr[data-best="true"]{background:#f0f7f2}.top-rank-table tr[data-best="true"]>th{border-left:3px solid var(--accent);padding-left:7px}.top-rank-value{display:inline-block;font-size:12px;font-weight:600}.top-rank-count{display:block;font-size:9px;color:var(--muted);margin-top:2px}.score-sort{border:0;background:transparent;padding:2px 0;font-size:10px;color:var(--muted)}.score-sort:hover{background:transparent;color:var(--accent)}.sort-indicator{display:inline-block;min-width:12px}.top-rank-table>tbody>tr.variant-details-row>td{padding:0;white-space:normal}.top-rank-variant{border-bottom:1px solid var(--line)}.top-rank-variant:last-child{border-bottom:0}
.top-rank-table{table-layout:fixed}.top-rank-table>thead>tr>th:first-child{width:55%}.top-rank-table>tbody>tr>td{min-width:0}
.variant-statistics>summary{cursor:pointer;color:var(--accent);font-size:10px;padding:5px 10px;background:#fbfcfa}.variant-statistics>.table-scroll{margin:0 6px 7px;border-radius:4px}.variant-statistics-table{font-size:10px}.variant-statistics-table th,.variant-statistics-table td{padding:4px 6px;min-width:55px;white-space:nowrap}.variant-statistics-table th[scope="row"]{text-align:left}.variant-statistics-table .rank-cell{min-width:28px;vertical-align:top;text-align:center;background:var(--pale)}.variant-statistics-table .stat-field{min-width:100px;max-width:145px;vertical-align:top;text-align:left;white-space:normal}.variant-statistics-table tbody{border-bottom:2px solid var(--line)}.variant-statistics-table tbody:last-child{border-bottom:0}.variant-statistics-table thead th{font-size:9px}
.patient{--patient-header-height:36px;background:white;border:1px solid var(--line);border-radius:5px;margin-bottom:4px;overflow:visible}.patient>summary{height:var(--patient-header-height);position:sticky;top:0;z-index:20;cursor:pointer;padding:6px 10px;display:flex;justify-content:space-between;align-items:center;gap:10px;list-style:none;background:white;border-radius:5px}.patient[open]>summary{box-shadow:0 2px 4px #1b302e12}.patient>summary::-webkit-details-marker{display:none}.patient-heading{display:grid;grid-template-columns:140px minmax(0,1fr);align-items:center;gap:10px;min-width:0;flex:1}.patient>summary strong{text-transform:capitalize;font-size:12px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.patient-id{display:block;color:var(--muted);font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.expand-hint{color:var(--accent);font-size:9px;white-space:nowrap}.expand-hint span{display:inline-block;font-size:16px;margin-left:6px}.patient[open]>summary .expand-hint span{transform:rotate(45deg)}.patient-body{padding:0 8px 8px}.patient-pagination{display:flex;justify-content:center;gap:8px;margin-top:10px}
.reference{display:grid;grid-template-columns:1fr 2fr;gap:16px;background:var(--pale);border-radius:4px;padding:7px 9px;margin:6px 0}.reference .eyebrow{font-size:8px;margin-bottom:3px}.reference h3{font-size:12px;margin:0;font-weight:600;overflow-wrap:anywhere}.plan{padding-left:16px;margin:2px 0;font-size:11px;line-height:1.35}.plan li{margin:1px 0;padding-left:1px}.unavailable{color:var(--muted);font-size:10px;margin:3px 0}.score-reason{display:block;font-size:9px;line-height:1.3;color:var(--muted);margin-top:2px;font-weight:400;white-space:normal}
.model-output{border:1px solid var(--line);border-radius:4px;margin:3px 0}.model-output>summary{display:grid;grid-template-columns:10px minmax(105px,1fr) minmax(150px,2fr) auto;align-items:center;gap:7px;position:sticky;top:var(--patient-header-height);z-index:10;cursor:pointer;list-style:none;padding:5px 7px;background:#f8faf6;border-radius:4px;font-size:11px}.model-output>summary::-webkit-details-marker{display:none}.model-output>summary::before{content:"▸";color:var(--accent)}.model-output[open]>summary::before{content:"▾"}.model-output>summary>strong{font-size:11px;overflow-wrap:anywhere}.variant-name{color:var(--accent);font-size:10px;overflow-wrap:anywhere}.variant-id{display:block;font-size:9px;color:var(--muted)}.model-body{padding:3px 5px 5px}.model-name{font-size:9px;color:var(--muted);overflow-wrap:anywhere;margin:3px 0}.badge{font-size:9px;padding:2px 4px;max-width:165px;border:1px solid #d0e4d8;color:var(--accent);border-radius:3px;background:#f1f8f3;overflow-wrap:anywhere}.badge.failed_incomplete,.badge.not_processed{background:#fff5e5;color:var(--amber);border-color:#eaddc4}
.score-table th,.score-table td{padding:4px 6px}.score-table .rank-cell{min-width:26px;width:30px;vertical-align:top;background:var(--pale);text-align:center;font-size:11px}.score-table .prediction-field{min-width:215px;max-width:330px;text-align:left;vertical-align:top;font-size:11px;line-height:1.35;font-weight:400;overflow-wrap:anywhere}.score-table td{min-width:65px;vertical-align:top;font-size:10px}.score-table thead th{font-size:9px}.score-table .prediction{border-bottom:2px solid var(--line)}.score-table .prediction:last-child{border-bottom:0}.field-kind{display:block;font-size:8px;letter-spacing:.4px;font-weight:700;text-transform:uppercase;color:var(--accent);margin-bottom:2px}
.collapse-controls{position:fixed;right:14px;bottom:14px;z-index:100;display:flex;flex-wrap:wrap;gap:6px;max-width:calc(100vw - 28px);padding:7px;border:1px solid #c1d3c8;background:#fffffff5;box-shadow:0 3px 15px #1b302e22;border-radius:7px}.collapse-controls button{font-size:11px;padding:7px 9px}
footer{font-size:10px;color:var(--muted);border-top:1px solid var(--line);margin-top:24px;padding:15px 0}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}[hidden]{display:none!important}
@media(max-width:1100px){.top-rank-cards{grid-template-columns:minmax(0,1fr)}}
@media(max-width:850px){header,main{padding-left:18px;padding-right:18px}.hero{align-items:start;flex-direction:column;gap:15px;margin-top:22px}.health{max-width:none;width:100%}h1{font-size:32px}.filters{flex-wrap:wrap}.filters p{display:none}.section-heading{align-items:start;flex-wrap:wrap}.patient-tools input{width:100%}.patient-tools label{flex:1}.expand-hint{font-size:0}.expand-hint span{font-size:17px}}
@media(max-width:600px){header,main{padding-left:12px;padding-right:12px}.top-rank-card{padding:10px}.patient{--patient-header-height:46px}.patient-heading{grid-template-columns:minmax(0,1fr);gap:1px}.patient-tools{align-items:start;flex-direction:column;gap:5px}.patient-tools label{width:100%}.patient-tools>span{margin:0}.patient-pagination{flex-wrap:wrap}.reference{grid-template-columns:1fr;gap:5px}.model-output>summary{grid-template-columns:10px minmax(100px,1fr) auto}.model-output>summary>.variant-name{grid-column:2/4}.model-output>summary>.badge{grid-column:3;grid-row:1}.collapse-controls{right:8px;bottom:8px}}
@media print{body{background:white}header,main{max-width:none;padding:12px}.filters,.patient-tools,button,.expand-hint,.collapse-controls{display:none!important}.patient>summary,.model-output>summary{position:static}.table-scroll{overflow:visible}table th,table td{padding:4px;min-width:0!important}.top-rank-card{break-inside:avoid}.stats{break-inside:avoid}}
'''

_SCRIPT = r'''
"use strict";
const byId = id => document.getElementById(id);
const model = byId("model-filter"), variant = byId("variant-filter");
const search = byId("patient-search");
const patients = [...document.querySelectorAll("details.patient")];
const pageSize = 10;
let patientLimit = pageSize;
function matches(element) {
  return (!model.value || element.dataset.model === model.value) &&
         (!variant.value || element.dataset.variant === variant.value);
}
function update() {
  let visiblePatients = 0;
  let matchingPatients = 0;
  const query = search.value.trim().toLowerCase();
  for (const patient of patients) {
    let visibleOutputs = 0;
    for (const output of patient.querySelectorAll(".model-output")) {
      output.hidden = !matches(output);
      if (!output.hidden) visibleOutputs++;
    }
    const match = !!visibleOutputs && patient.dataset.search.includes(query);
    if (match) matchingPatients++;
    patient.hidden = !match || matchingPatients > patientLimit;
    if (!patient.hidden) visiblePatients++;
  }
  byId("patient-count").textContent = `Showing ${visiblePatients} of ${matchingPatients} matching patients`;
  byId("patients-empty").hidden = !!matchingPatients;
  const remaining = matchingPatients - visiblePatients;
  byId("show-more").hidden = !remaining;
  byId("show-more").textContent = `Show ${Math.min(pageSize, remaining)} more patients (${remaining} remaining)`;
  byId("show-less").hidden = visiblePatients <= pageSize;
  updateExpandLabel();
}
function updateExpandLabel() {
  const visible = patients.filter(patient => !patient.hidden);
  byId("expand-all").textContent = visible.length && visible.every(patient => patient.open)
    ? "Collapse shown patients" : "Expand shown patients";
  byId("collapse-controls").hidden = !visible.some(patient => patient.open);
}
function resetPagination() { patientLimit = pageSize; update(); }
for (const select of [model, variant]) select.addEventListener("change", resetPagination);
search.addEventListener("input", resetPagination);
for (const patient of patients) patient.addEventListener("toggle", updateExpandLabel);
byId("expand-all").addEventListener("click", () => {
  const visible = patients.filter(patient => !patient.hidden);
  const expand = !visible.every(patient => patient.open);
  for (const patient of visible) patient.open = expand;
  updateExpandLabel();
});
byId("reset").addEventListener("click", () => {
  model.value = variant.value = search.value = "";
  resetPagination();
});
byId("show-more").addEventListener("click", () => { patientLimit += pageSize; update(); });
byId("show-less").addEventListener("click", resetPagination);
function closePatient(patient) {
  patient.open = false;
  for (const output of patient.querySelectorAll("details.model-output")) output.open = false;
}
byId("collapse-all").addEventListener("click", () => {
  for (const patient of patients) closePatient(patient);
  updateExpandLabel();
});
byId("collapse-current").addEventListener("click", () => {
  const open = patients.filter(patient => !patient.hidden && patient.open);
  const current = open.find(patient => {
    const bounds = patient.getBoundingClientRect();
    return bounds.top < window.innerHeight && bounds.bottom > 0;
  }) || open[0];
  if (current) {
    closePatient(current);
    current.scrollIntoView({block: "start"});
  }
  updateExpandLabel();
});
function sortScores(table, field, direction) {
  const key = field + "Score";
  const groups = [...table.querySelectorAll("tbody.top-rank-variant")];
  groups.sort((left, right) => {
    const a = left.dataset[key], b = right.dataset[key];
    const aMissing = a === "" || a === undefined, bMissing = b === "" || b === undefined;
    if (aMissing !== bMissing) return aMissing ? 1 : -1;
    if (!aMissing && Number(a) !== Number(b))
      return (Number(a) < Number(b) ? -1 : 1) * (direction === "ascending" ? 1 : -1);
    return Number(left.dataset.order) - Number(right.dataset.order);
  });
  for (const group of groups) table.appendChild(group);
  table.dataset.sortField = field; table.dataset.sortDirection = direction;
  for (const button of table.querySelectorAll("button.score-sort")) {
    const active = button.dataset.field === field;
    button.parentElement.setAttribute("aria-sort", active ? direction : "none");
    button.querySelector(".sort-indicator").textContent = active ? (direction === "descending" ? "↓" : "↑") : "↕";
  }
}
for (const table of document.querySelectorAll("table.top-rank-table")) {
  for (const button of table.querySelectorAll("button.score-sort")) {
    button.addEventListener("click", () => {
      const field = button.dataset.field;
      const direction = table.dataset.sortField === field && table.dataset.sortDirection === "descending"
        ? "ascending" : "descending";
      sortScores(table, field, direction);
    });
  }
}
update();
'''


if __name__ == "__main__":
    raise SystemExit(main())
