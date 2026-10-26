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
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = "results_summary.html"
# Match the saved evaluator contract without importing its model dependencies.
METRICS = ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1")
DISPLAY_METRICS = ("BERTScore_F1", "ROUGE-L", "BLEU", "METEOR", "ROUGE-1", "ROUGE-2")
VARIANTS = {
    "llm_only": "Patient S/O only",
    "patient_retrieval": "S/O + similar-patient example",
    "guidelines": "S/O + clinical guidelines",
    "patient_retrieval_guidelines": "S/O + both evidence sources",
    "not_recorded": "Evidence variant not recorded",
}
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


def _variant(row, metadata):
    experiment = metadata.get("experiment") or {}
    value = row.get("evidence_variant", experiment.get("evidence_variant"))
    if value is None:
        return "not_recorded"
    if not isinstance(value, str) or value not in VARIANTS:
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
    selected_ids = evaluation.get("selected_patient_ids", [])
    if (not isinstance(selected_ids, list) or not all(_text(value) for value in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("run.json selected_patient_ids must be unique, nonempty strings")
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
        variant = _variant(row, metadata)
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
    return {"patients": patients, "summary": summary, "comparisons": comparisons, "notices": notices}


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


def _overview_rows(data, *, comparison=False):
    # One model/evidence cell spans all ranks. Each rank gets its own cell
    # beside the two condition/plan subrows it describes.
    models = {}
    for row in data["comparisons" if comparison else "summary"]:
        variants = models.setdefault(row["model_id"], {})
        ranks = variants.setdefault(row["variant"], {})
        ranks.setdefault(row["rank"], {})[row["field"]] = row
    rendered = []
    for model_id, variants in models.items():
        for variant, ranks in variants.items():
            row_count = sum(len(fields) for fields in ranks.values())
            identity = (f'<th class="group-label" scope="rowgroup" rowspan="{row_count}">'
                        f'<strong>{_e(model_id)}</strong><span class="row-detail">{_e(variant)}</span></th>')
            subrows = []
            for rank in sorted(ranks):
                fields = ranks[rank]
                rank_cell = (f'<th class="rank-cell" scope="row" rowspan="{len(fields)}">'
                             f'<span class="rank-label">Rank {rank}</span></th>')
                first_field = True
                for field in ("condition", "plan"):
                    if field not in fields:
                        continue
                    row = fields[field]
                    if comparison:
                        count = f'{row["paired_patients"]} paired patients'
                        values = row["differences"]
                    else:
                        counts = row["counts"]
                        skipped = counts.get("skipped_no_recorded_plan", 0) + counts.get("missing_reference", 0)
                        errors = sum(counts.get(status, 0) for status in ("error", "interrupted", "invalid_metrics"))
                        scored = counts.get("scored", 0)
                        missing = row["total"] - scored - skipped - errors
                        count = f'{scored}/{row["total"]} scored · {skipped} skipped · {missing} missing · {errors} errors'
                        values = row["means"]
                    subrows.append(f'<tr data-rank="{rank}" data-field="{field}">'
                                   f'{identity if not subrows else ""}{rank_cell if first_field else ""}'
                                   f'<th class="field-label" scope="row">{field.title()}'
                                   f'<span class="row-detail counts">{_e(count)}</span></th>'
                                   + ''.join(f'<td>{_number(values[metric], signed=comparison)}</td>'
                                             for metric in DISPLAY_METRICS) + '</tr>')
                    first_field = False
            attributes = f'data-model="{_e(model_id)}" data-variant="{_e(variant)}"'
            rendered.append(f'<tbody class="model-group" {attributes}>{"".join(subrows)}</tbody>')
    return ''.join(rendered)


def _patient_html(patient, index):
    outputs = []
    for output in patient["outputs"]:
        heading = f'<h4>{_e(output["model_id"])}</h4>'
        if output["model_name"] != output["model_id"]:
            heading += f'<p class="model-name">{_e(output["model_name"])}</p>'
        heading += f'<p class="variant-name">{_e(VARIANTS[output["variant"]])}</p>'
        status = output["status"]
        note = output["generation_note"] or "Generation completed"
        pairs = []
        for pair in output["predictions"]:
            rows = []
            for field, entry in pair["scores"].items():
                rows.append(f'<tr><th scope="row">{field.title()}<span class="score-reason">'
                            f'{_e(entry["reason"])}</span></th>'
                            + ''.join(f'<td>{_number(entry["metrics"][metric])}</td>' for metric in DISPLAY_METRICS)
                            + '</tr>')
            condition = _e(pair["condition"]) if pair["condition"] else '<span class="unavailable">Condition unavailable</span>'
            pairs.append(f'<li class="prediction"><div class="pair-heading"><span class="rank">{pair["rank"]}</span>'
                         f'<h5>{condition}</h5></div><div class="plan-block"><span class="eyebrow">Corresponding plan</span>'
                         f'{_plan(pair["plan"])}</div><div class="table-scroll"><table class="score-table">'
                         f'<caption class="sr-only">Rank {pair["rank"]} condition and plan scores</caption>'
                         f'<thead><tr><th scope="col">Evaluation</th>{_metric_header()}</tr></thead>'
                         f'<tbody>{"".join(rows)}</tbody></table></div></li>')
        outputs.append(f'<section class="model-output" data-model="{_e(output["model_id"])}" '
                       f'data-variant="{_e(output["variant"])}"><div class="output-heading"><div>{heading}</div>'
                       f'<span class="badge {status}">{_e(note)}</span></div>'
                       f'<ol class="predictions">{"".join(pairs)}</ol></section>')
    search = (patient["id"] + ' ' + patient["reference_condition"] + ' '
              + patient["reference_condition"].replace('_', ' ')).lower()
    return (f'<details class="patient" data-search="{_e(search)}" id="patient-{index}">'
            f'<summary><span class="patient-heading"><strong>{_e(patient["reference_condition"].replace("_", " "))}</strong>'
            f'<span class="patient-id" title="{_e(patient["id"])}">{_e(patient["id"])}</span></span>'
            '<span class="expand-hint" aria-hidden="true">View predictions <span>+</span></span></summary>'
            '<div class="patient-body"><section class="reference"><div><span class="eyebrow">Reference condition</span>'
            f'<h3>{_e(patient["reference_condition"])}</h3></div><div><span class="eyebrow">Reference plan</span>'
            f'{_plan(patient["reference_plan"], reference=True)}</div></section>{"".join(outputs)}</div></details>')


def render_report(data):
    """Render a self-contained HTML page; all source text is escaped, never executed."""
    patients = data["patients"]
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
        "NOTICES": notices, "MODEL_OPTIONS": options(models, {}), "VARIANT_OPTIONS": options(variants, VARIANTS),
        "SUMMARY": _overview_rows(data), "COMPARISONS": _overview_rows(data, comparison=True),
        "METRIC_HEADER": _metric_header(),
        "DETAILS": ''.join(_patient_html(patient, index) for index, patient in enumerate(patients)),
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
    """Standalone entry point; runner integration is deliberately separate."""
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
<main><noscript><p class="notice">JavaScript is disabled. All results and all overview ranks are shown below; filters require JavaScript.</p></noscript>
@@NOTICES@@
<div class="filters"><label>Model<select id="model-filter"><option value="">All models</option>@@MODEL_OPTIONS@@</select></label>
<label>Evidence<select id="variant-filter"><option value="">All recorded outputs</option>@@VARIANT_OPTIONS@@</select></label>
<p>Filters apply to the overview and patient predictions.</p><button id="reset" type="button">Reset filters</button></div>
<section aria-labelledby="overview-heading"><div class="section-heading"><div><p class="eyebrow">01 / Aggregate results</p><h2 id="overview-heading">Performance overview</h2></div>
</div>
<p class="section-note">Arithmetic means across scored patients, separately for each model, evidence variant, field, and rank. Skipped and unavailable scores are excluded; zero scores are included. Overview uses all patients.</p>
<div class="table-scroll overview"><table id="summary-table"><caption class="sr-only">Mean scores by model, evidence variant, rank, and field</caption>
<thead><tr><th scope="col">Model / evidence</th><th scope="col">Rank</th><th scope="col">Field / coverage</th>@@METRIC_HEADER@@</tr></thead>@@SUMMARY@@</table>
<p class="empty" id="summary-empty" hidden>No results match these filters.</p></div>
<p class="metric-note">Scores measure text similarity, not clinical correctness. Metrics retain their saved scales and are not combined. Values display to four decimal places; hover for full precision. N/A means unavailable, not zero.</p></section>
<section aria-labelledby="comparison-heading"><div class="section-heading"><div><p class="eyebrow">02 / Evidence comparison</p><h2 id="comparison-heading">Change from patient S/O only</h2></div><span class="chip">vs. llm_only</span></div>
<p class="section-note">Mean difference (evidence variant − llm_only) for the same model, rank, field, and metric. Only patients scored in both variants are paired. All three ranks show condition and plan scores together.</p>
<div class="table-scroll overview"><table id="comparison-table"><caption class="sr-only">Paired score differences from llm_only</caption><thead><tr><th scope="col">Model / evidence</th><th scope="col">Rank</th><th scope="col">Field / paired coverage</th>@@METRIC_HEADER@@</tr></thead>
@@COMPARISONS@@</table><p class="empty" id="comparison-empty">No paired baseline comparison is available for these patients.</p></div></section>
<section aria-labelledby="patient-heading"><div class="section-heading"><div><p class="eyebrow">03 / Individual results</p><h2 id="patient-heading">Patient predictions</h2></div>
<button id="expand-all" type="button">Expand shown patients</button></div><div class="patient-tools"><label>Find a patient<input id="patient-search" type="search" placeholder="Search patient ID or reference condition…"></label>
<span id="patient-count" aria-live="polite">@@PATIENTS@@ patients</span></div>
<p class="section-note">The first 10 matching patients are shown initially. Search checks all patients and filters this section only. Expand a patient to see the label and all three condition / plan pairs per model and evidence variant.</p>
<div id="patients">@@DETAILS@@</div><p id="patients-empty" class="empty" hidden>No patients match these filters.</p>
<div class="patient-pagination"><button id="show-more" type="button" hidden>Show 10 more patients</button>
<button id="show-less" type="button" hidden>Show first 10 patients</button></div></section>
<footer>Saved predictions and scores only. No generation or scoring is performed by this report.</footer></main>
<script id="report-data" type="application/json">@@DATA@@</script><script>@@SCRIPT@@</script></body></html>'''


_STYLE = r'''
:root{--ink:#1b302e;--muted:#647570;--accent:#197266;--pale:#eaf3ef;--line:#dbe4de;--paper:#fbfcf9;--amber:#8d5f18}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header,main{max-width:1280px;margin:auto;padding:32px 44px}header{padding-bottom:12px}.brand{display:flex;align-items:center;gap:10px;font-size:11px;letter-spacing:1.7px;font-weight:700;color:var(--muted)}
.brand-mark{display:grid;place-items:center;width:30px;height:30px;background:var(--accent);color:white;font-size:16px;border-radius:8px;letter-spacing:0}.brand-rule{height:15px;width:1px;background:var(--line);margin:0 4px}
.hero{display:flex;justify-content:space-between;gap:36px;align-items:center;margin:38px 0 30px}.eyebrow{font-size:10px;letter-spacing:1.4px;font-weight:700;text-transform:uppercase;color:var(--muted);margin:0 0 8px;display:block}
h1{font-size:42px;font-weight:600;letter-spacing:-1.8px;line-height:1.15;margin:0 0 12px}h1 span{color:var(--accent)}.intro{max-width:560px;color:var(--muted);margin:0;font-size:15px}.health{max-width:310px;padding:16px 18px;background:var(--pale);border-radius:10px;border-left:3px solid var(--accent);flex-shrink:0}
.health strong{font-size:13px}.health p{font-size:12px;color:var(--muted);margin:5px 0 0}.health.attention{background:#fff5e5;border-color:var(--amber)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);border:1px solid var(--line);border-radius:12px;background:white;overflow:hidden}.stats>div{padding:18px 24px;border-right:1px solid var(--line)}.stats>div:last-child{border:0}.stats span{display:block;font-size:12px;color:var(--muted);margin-bottom:7px}.stats strong{font-size:27px;line-height:1;font-weight:600;letter-spacing:-.8px}
main{padding-top:10px}.notice{border:1px solid #e8dfca;background:#fffaf0;border-radius:8px;color:#74613d;font-size:12px;padding:12px 16px;margin:14px 0}.filters{display:flex;align-items:end;gap:16px;padding:18px 0 22px;border-bottom:1px solid var(--line);margin-top:4px}
label{font-size:11px;color:var(--muted);font-weight:600;display:block}select,input{display:block;margin-top:5px;padding:9px 30px 9px 11px;border:1px solid #ccd9d0;border-radius:6px;background:white;color:var(--ink);font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:38px;max-width:100%}select{min-width:165px}input{width:390px;padding-right:10px}.filters p{font-size:11px;color:var(--muted);margin:0 auto 10px 0;max-width:200px}
button{border:1px solid #cbdad0;border-radius:6px;background:white;color:var(--accent);font:600 12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:10px 13px;cursor:pointer;white-space:nowrap}button:hover{background:var(--pale)}:focus-visible{outline:3px solid #7fc3af;outline-offset:3px}
main>section{margin-top:38px}.section-heading{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:12px}h2{font-size:23px;line-height:1.3;letter-spacing:-.6px;font-weight:600;margin:0}.section-note{font-size:12px;color:var(--muted);max-width:970px;margin:0 0 16px;line-height:1.7}
.table-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:9px;background:white}table{border-collapse:collapse;width:100%;font-size:12px;text-align:left}table th,table td{padding:14px 16px;border-bottom:1px solid var(--line)}thead th{font-size:10px;color:var(--muted);letter-spacing:.4px;background:#f3f6f1;font-weight:600;white-space:nowrap}tbody th{font-weight:500;min-width:240px}tbody th strong{font-size:12px;display:block;overflow-wrap:anywhere}td{text-align:right;font-variant-numeric:tabular-nums;min-width:94px}thead th:not(:first-child){text-align:right}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}.row-detail{display:block;font-size:10px;color:var(--muted);margin-top:4px;overflow-wrap:anywhere}.counts{font-size:10px;letter-spacing:.05px}.na{color:#89948d;font-size:11px}.metric-note{font-size:11px;color:var(--muted);margin:12px 0 0}.chip{font:11px ui-monospace,monospace;background:var(--pale);border:1px solid #d3e5dc;padding:6px 10px;border-radius:20px;color:var(--accent)}
.overview .group-label{min-width:190px;max-width:230px;background:#f8faf6;vertical-align:middle;text-align:left}.overview .rank-cell{min-width:76px;width:76px;vertical-align:middle;text-align:left}.overview .field-label{min-width:165px;max-width:195px;text-align:left;font-weight:600}.overview th,.overview td{padding:10px 12px}.overview td{min-width:82px;font-size:11px}.overview thead th:nth-child(2),.overview thead th:nth-child(3){text-align:left}.overview .model-group{border-bottom:2px solid var(--line)}.overview .model-group:last-child{border-bottom:0}.overview tr[data-field="condition"]:not(:first-child)>th,.overview tr[data-field="condition"]:not(:first-child)>td{border-top:2px solid var(--line)}.rank-label{display:inline-block;padding:3px 7px;border-radius:4px;background:var(--pale);color:var(--accent);font-size:10px;font-weight:700;white-space:nowrap}
.empty{padding:22px;text-align:center;color:var(--muted);font-size:12px;margin:0}.patient-tools{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:18px 0 10px}.patient-tools>span{font-size:12px;color:var(--muted);margin-bottom:10px}.patient{background:white;border:1px solid var(--line);border-radius:6px;margin-bottom:5px;overflow:hidden}.patient>summary{cursor:pointer;padding:8px 12px;display:flex;justify-content:space-between;align-items:center;gap:12px;list-style:none}.patient>summary::-webkit-details-marker{display:none}.patient-heading{display:grid;grid-template-columns:150px minmax(0,1fr);align-items:center;gap:12px;min-width:0;flex:1}.patient>summary strong{text-transform:capitalize;font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.patient-id{display:block;color:var(--muted);font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.expand-hint{color:var(--accent);font-size:10px;white-space:nowrap}.expand-hint span{display:inline-block;font-size:17px;margin-left:8px}.patient[open]>summary .expand-hint span{transform:rotate(45deg)}.patient-pagination{display:flex;justify-content:center;gap:10px;margin-top:14px}.patient-body{padding:0 16px 16px}.reference{display:grid;grid-template-columns:1fr 1.5fr;gap:30px;background:var(--pale);border-radius:7px;padding:18px 20px}.reference h3{font-size:14px;margin:0;font-weight:600;overflow-wrap:anywhere}.plan{padding-left:20px;margin:7px 0 0;font-size:13px}.plan li{margin:5px 0;padding-left:3px}.unavailable{color:var(--muted);font-size:12px;margin:6px 0}.output-heading{display:flex;justify-content:space-between;align-items:start;gap:15px;margin:24px 0 16px}.output-heading h4{font-size:15px;margin:0;overflow-wrap:anywhere}.model-name,.variant-name{font-size:11px;color:var(--muted);margin:4px 0 0;overflow-wrap:anywhere}.variant-name{color:var(--accent)}.badge{font-size:10px;padding:5px 9px;border:1px solid #d0e4d8;color:var(--accent);border-radius:5px;background:#f1f8f3;white-space:nowrap}.badge.failed_incomplete,.badge.not_processed{background:#fff5e5;color:var(--amber);border-color:#eaddc4}
.predictions{list-style:none;padding:0;margin:0}.prediction{border:1px solid var(--line);border-radius:8px;padding:17px;margin-bottom:14px}.pair-heading{display:flex;align-items:baseline;gap:11px}.rank{display:inline-grid;place-items:center;background:var(--pale);color:var(--accent);width:25px;height:25px;font-size:12px;border-radius:5px;flex-shrink:0;font-weight:700}h5{font-size:14px;line-height:1.5;font-weight:600;margin:0;overflow-wrap:anywhere}.plan-block{margin:14px 0 17px;padding-left:36px}.plan-block .eyebrow{font-size:9px;margin:0}.score-table{font-size:11px}.score-table td,.score-table th{padding:10px 12px}.score-table tbody th{min-width:215px}.score-reason{display:block;font-size:10px;color:var(--muted);margin-top:4px;font-weight:400}.score-table .na{font-size:10px}.prediction .table-scroll{border-radius:6px}footer{font-size:11px;color:var(--muted);border-top:1px solid var(--line);margin-top:35px;padding:20px 0 10px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}[hidden]{display:none!important}
@media(max-width:850px){header,main{padding-left:20px;padding-right:20px}.hero{align-items:start;flex-direction:column;gap:20px;margin-top:28px}.health{max-width:none;width:100%}h1{font-size:35px}.filters{flex-wrap:wrap}.filters p{display:none}.section-heading{align-items:start;flex-wrap:wrap}.reference{grid-template-columns:1fr;gap:20px}.expand-hint{font-size:0}.expand-hint span{font-size:20px}.patient-body{padding:0 12px 12px}.output-heading{flex-wrap:wrap}.patient-tools input{width:100%}.patient-tools label{flex:1}.stats>div{padding:17px 20px}.plan-block{padding-left:0}}
@media(max-width:600px){.patient-heading{grid-template-columns:minmax(0,1fr);gap:1px}.patient>summary{padding:7px 10px}.patient-tools{align-items:start;flex-direction:column;gap:5px}.patient-tools label{width:100%}.patient-tools>span{margin:0}.patient-pagination{flex-wrap:wrap}}
@media print{body{background:white}header,main{max-width:none;padding:15px}.filters,.patient-tools,button,.expand-hint{display:none}.patient,.prediction,.reference{break-inside:avoid}.table-scroll{overflow:visible}table{font-size:9px}table th,table td{padding:6px;min-width:0!important}.health{border:1px solid #aaa}.stats{break-inside:avoid}}
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
  for (const name of ["summary", "comparison"]) {
    const table = byId(name + "-table");
    let visible = 0;
    for (const group of table.querySelectorAll("tbody.model-group")) {
      group.hidden = !matches(group);
      if (!group.hidden) visible++;
    }
    table.hidden = !visible;
    byId(name + "-empty").hidden = !!visible;
  }
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
update();
'''


if __name__ == "__main__":
    raise SystemExit(main())
