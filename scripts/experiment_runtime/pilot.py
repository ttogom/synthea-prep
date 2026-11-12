"""Deterministic runtime sampling and population-weighted pilot projections.

The strata use input-size/structure and reference action counts for sampling
only. Patient content, prompts, model settings, and reference labels are unchanged.
"""

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from random import Random
import re
from statistics import median
from urllib.parse import urlsplit


STRATA = ("short", "typical", "long", "complex")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_EVENT = re.compile(r"\s*(?:[-*•]|\d+[.)]|\d{4}-\d{2}-\d{2}\b)")
_ARTIFACT = re.compile(r"\(\s*record\s+artifact\s*\)", re.IGNORECASE)


def _features(patient, label):
    if any(not isinstance(patient.get(field), str) for field in ("subjective", "objective")):
        raise ValueError("Pilot patients must contain string subjective/objective fields")
    actions = label.get("plan")
    if not isinstance(actions, list) or any(not isinstance(action, str) for action in actions):
        raise ValueError("Pilot labels must contain a list of reference action strings")
    text = patient["subjective"] + "\n" + patient["objective"]
    lines = [line for line in text.splitlines() if line.strip()]
    dates = len(_DATE.findall(text))
    events = sum(bool(_EVENT.match(line)) for line in lines)
    return {
        "so_characters": len(patient["subjective"]) + len(patient["objective"]),
        "date_occurrences": dates, "event_lines": events, "nonempty_lines": len(lines),
        "reference_action_count": len(actions),
        "complexity_score": dates + events + len(lines) + 2 * len(actions),
        "judgeable_reference": bool(actions and any(action.strip() for action in actions)
                                   and actions != ["none recorded"]
                                   and not any(_ARTIFACT.search(action) for action in actions)),
    }


def select_runtime_pilot(patients, labels, count=64, seed=42):
    """Select equal, disjoint size/complexity strata without changing patient data.

    patients must already be the eligible population. labels can be a mapping
    from patient ID to label or a superset list of labels. Returned patients retain
    their original input order and objects. A 64-patient pilot contains 16 per
    stratum; 48 contains 12. The manifest retains population weights so equal
    sampling does not bias a full-cohort runtime projection.
    """
    if type(count) is not int or count < 16 or count % 4:
        raise ValueError("Pilot count must be a multiple of four, at least sixteen")
    if type(seed) is not int:
        raise ValueError("Pilot seed must be an integer")
    patients = list(patients)
    ids = [patient.get("id") if isinstance(patient, dict) else None for patient in patients]
    if (not patients or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(ids) != len(set(ids)) or count > len(ids)):
        raise ValueError("Pilot population must contain enough uniquely identified patients")
    if isinstance(labels, list):
        label_ids = [label.get("id") if isinstance(label, dict) else None for label in labels]
        if any(not isinstance(value, str) for value in label_ids) or len(label_ids) != len(set(label_ids)):
            raise ValueError("Pilot labels must have unique patient IDs")
        labels = dict(zip(label_ids, labels))
    if not isinstance(labels, dict) or any(value not in labels or not isinstance(labels[value], dict) for value in ids):
        raise ValueError("Every pilot patient must have a reference label")
    features = {patient["id"]: _features(patient, labels[patient["id"]]) for patient in patients}
    complexity_order = sorted(ids, key=lambda value: (features[value]["complexity_score"],
                                                      features[value]["so_characters"], value))
    complex_count = max(1, math.ceil(len(ids) * 0.1))
    complex_ids = complexity_order[-complex_count:]
    complex_set = set(complex_ids)
    length_order = sorted((value for value in ids if value not in complex_set),
                          key=lambda value: (features[value]["so_characters"], value))
    first, third = len(length_order) // 4, 3 * len(length_order) // 4
    groups = {"short": length_order[:first], "typical": length_order[first:third],
              "long": length_order[third:], "complex": complex_ids}
    per_stratum = count // 4
    if any(len(values) < per_stratum for values in groups.values()):
        raise ValueError("Eligible population is too small for equal samples in every pilot stratum")
    random = Random(seed)
    chosen = set()
    strata = {}
    for name in STRATA:
        values = groups[name]
        if name == "complex":
            ordered = sorted(values, key=lambda value: (features[value]["complexity_score"],
                                                         features[value]["so_characters"], value))
            anchors = [ordered[-1], ordered[0]]
        elif name == "short":
            anchors = [values[0], values[-1]]
        elif name == "long":
            anchors = [values[-1], values[0]]
        else:
            anchors = [values[len(values) // 2], values[0], values[-1]]
        selected = list(dict.fromkeys(anchors))[:per_stratum]
        forced = list(selected)
        random_probability = ((per_stratum - len(forced)) / (len(values) - len(forced))
                              if len(values) > len(forced) else 1.0)
        selected.extend(random.sample([value for value in values if value not in selected],
                                      per_stratum - len(selected)))
        chosen.update(selected)
        strata[name] = {"population_count": len(values), "sample_count": per_stratum,
                        "weight": len(values) / len(ids), "selected_patient_ids": selected,
                        "forced_patient_ids": forced,
                        "inclusion_probabilities": {value: 1.0 if value in forced else random_probability
                                                    for value in selected},
                        "so_characters_min": min(features[value]["so_characters"] for value in values),
                        "so_characters_max": max(features[value]["so_characters"] for value in values)}
    selected = [patient for patient in patients if patient["id"] in chosen]
    manifest = {
        "method": "runtime-stratified-v1", "seed": seed, "population_count": len(ids),
        "sample_count": len(selected), "selected_patient_ids": [patient["id"] for patient in selected],
        "definitions": {
            "complexity_score": "date_occurrences + event_lines + nonempty_lines + 2 * reference_action_count",
            "event_lines": "nonempty lines beginning with a bullet, numbered item, or YYYY-MM-DD",
            "complex": "highest ceil(10% of eligible population) complexity scores; ties by S/O characters then ID",
            "length_strata": "remaining patients ranked by S/O character count: bottom quarter, middle half, top quarter",
            "sampling": "equal counts; deterministic boundary/extreme anchors plus seeded random remainder",
            "label_use": "reference action count and reference judgeability are selection/projection features only",
        },
        "strata": strata,
        "patient_strata": {value: name for name, values in groups.items() for value in values},
        "population_features": features,
    }
    return selected, manifest


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative runtime timing: {name}")
    return float(value)


def _wall(start, finish):
    if not start or not finish:
        return None
    return max(0.0, (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds())


def _sample_total(times, stratum):
    selected = stratum["selected_patient_ids"]
    observed = [patient for patient in selected if patient in times]
    if not observed:
        return None, observed, {}
    probabilities = stratum.get("inclusion_probabilities")
    inverse = {patient: 1 / probabilities[patient] if probabilities else 1.0 for patient in observed}
    estimate = sum(times[patient] * inverse[patient] for patient in observed)
    # Forced extremes have probability one; random cases represent the rest.
    # Complete samples use a Horvitz–Thompson total, not a naive sample mean.
    if not probabilities or len(observed) < len(selected):
        estimate *= stratum["population_count"] / sum(inverse.values())
    return estimate, observed, inverse


def _weighted(times, manifest, warnings, label):
    total = 0.0
    counts = {}
    for name in STRATA:
        stratum = manifest["strata"][name]
        estimate, observed, _ = _sample_total(times, stratum)
        counts[name] = len(observed)
        if estimate is None:
            warnings.append(f"Cannot project {label}: no timing observations in {name} stratum")
            return None, counts
        if len(observed) < len(stratum["selected_patient_ids"]):
            warnings.append(f"{label}: {name} timing coverage is {len(observed)}/{len(stratum['selected_patient_ids'])} patients")
        total += estimate
    return total, counts


def _band(seconds):
    return None if seconds is None else {"seconds": seconds, "hours": seconds / 3600,
        "planning_range_hours": [seconds * 0.75 / 3600, seconds * 1.25 / 3600]}


def _local_model(model, metadata):
    config = (metadata.get("assessment_plan_models", {}).get("models", {}).get(model, {}).get("config") or {})
    endpoint = config.get("api_base")
    return bool(endpoint and urlsplit(endpoint).hostname in ("localhost", "127.0.0.1", "::1"))


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _outputs(record):
    if "variants" in record:
        return [output for variant in record["variants"].values()
                for output in variant["assessment_plans"].values()]
    if "assessment_plans" in record:
        return list(record["assessment_plans"].values())
    return [record]


def _batch_fields(batch, final_field_counts):
    measured = batch.get("newly_scored_fields_by_patient")
    if measured is None:
        return {patient: max(1, final_field_counts.get(patient, 0)) for patient in batch["patient_ids"]}
    if (not isinstance(measured, dict) or set(measured) != set(batch["patient_ids"])
            or any(type(count) is not int or count < 0 for count in measured.values())):
        raise ValueError("Scoring batch field counts must be nonnegative integers for every batch patient")
    # A no-op snapshot still does some bookkeeping. Attribute that small cost
    # evenly rather than treating it as new model evaluation work.
    return measured if any(measured.values()) else {patient: 1 for patient in batch["patient_ids"]}


def project_runtime(run_dir, *, root):
    """Project full-population wall time from a completed stratified pilot.

    Bands are ±25% planning sensitivities, not statistical confidence intervals.
    Service times already include prompt processing, queueing, and retries seen
    by the client. Non-first local phases calibrate concurrency utilization;
    preparation overlap is counted once. Missing observations are explicit.
    """
    run_dir = Path(run_dir).resolve()
    root = Path(root).resolve()
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("resumptions"):
        raise ValueError("Runtime projection requires a fresh uninterrupted pilot: resumed timing history "
                         "omits prior preparation/scoring phases and wall time includes downtime")
    manifest = metadata.get("pilot_selection")
    if not isinstance(manifest, dict) or any(name not in manifest.get("strata", {}) for name in STRATA):
        raise ValueError("Runtime projection requires the pilot_selection stratification manifest")
    if sum(manifest["strata"][name]["population_count"] for name in STRATA) != manifest["population_count"]:
        raise ValueError("Pilot stratum populations do not sum to the full cohort")
    timings = metadata.get("runtime_timings") or {}
    generation = timings.get("generation") or []
    if not generation:
        raise ValueError("Runtime projection requires measured generation timings")
    warnings = []
    experiment = metadata.get("experiment") or {}
    execution = experiment.get("execution") or {}
    local_concurrency = execution.get("local_concurrency", 4)
    hosted_concurrency = execution.get("hosted_concurrency", 4)
    preparation_workers = execution.get("preparation_workers", 2)
    if any(type(value) is not int or value < 1 for value in (
            local_concurrency, hosted_concurrency, preparation_workers)):
        raise ValueError("Recorded execution concurrency must contain positive integers")
    variants = experiment.get("selected_variants") or sorted({row["variant"] for row in generation})
    models = experiment.get("selected_models") or sorted({row["model"] for row in generation})
    service = defaultdict(lambda: defaultdict(float))
    completed = defaultdict(set)
    selected_ids = set(manifest["selected_patient_ids"])
    for row in generation:
        if row["patient_id"] not in selected_ids:
            raise ValueError("Generation timing names a patient outside the pilot")
        key = (row["model"], row["variant"])
        service[key][row["patient_id"]] += _seconds(row["seconds"], "generation")
        if row.get("status") == "completed":
            completed[key].add(row["patient_id"])
    phases = {row["model"]: _seconds(row["seconds"], "model phase")
              for row in timings.get("model_phases", [])}
    local = [model for model in models if _local_model(model, metadata)]
    hosted = [model for model in models if model not in local]
    first_local = next(iter(phases), local[0] if local else None)
    factors = []
    for model in local:
        pilot_service = sum(sum(service[(model, variant)].values()) for variant in variants)
        if model != first_local and model in phases and pilot_service > 0:
            factors.append(max(1.0, phases[model] / (pilot_service / local_concurrency)))
    first_factor = median(factors) if factors else 1.0
    projected_models = {}
    pilot_model_seconds = {}
    for model in models:
        full_service = 0.0
        corrected_service = 0.0
        coverage = {}
        success_count = observed_count = 0
        missing = False
        pilot_service = sum(sum(service[(model, variant)].values()) for variant in variants)
        concurrency = local_concurrency if model in local else hosted_concurrency
        for variant in variants:
            estimate, counts = _weighted(service[(model, variant)], manifest, warnings, f"{model}/{variant}")
            coverage[variant] = counts
            if estimate is None:
                missing = True
            else:
                full_service += estimate
                for name in STRATA:
                    total, observed, inverse = _sample_total(service[(model, variant)], manifest["strata"][name])
                    valid_fraction = sum(inverse[patient] for patient in observed
                                         if patient in completed[(model, variant)]) / sum(inverse.values())
                    if valid_fraction == 0:
                        missing = True
                        warnings.append(f"Cannot project successful {model}/{variant}: all timed {name} cases failed")
                    else:
                        corrected_service += total / valid_fraction
            observed_count += len(service[(model, variant)])
            success_count += len(completed[(model, variant)])
        success_fraction = success_count / observed_count if observed_count else 0.0
        if success_fraction < 1:
            warnings.append(f"{model}: {observed_count - success_count}/{observed_count} timed generation slots failed; "
                            "projection assumes retry behavior remains similar")
        factor = (max(1.0, phases[model] / (pilot_service / concurrency))
                  if model in local and model != first_local and model in phases and pilot_service > 0
                  else first_factor if model in local else 1.0)
        projected = None if missing or success_fraction == 0 else corrected_service / concurrency * factor
        pilot_model_seconds[model] = pilot_service / concurrency * factor
        projected_models[model] = {"local": model in local, "concurrency": concurrency,
            "timing_observations": observed_count, "completed_slots": success_count,
            "success_fraction": success_fraction, "stratum_coverage": coverage,
            "weighted_service_seconds": None if missing else full_service,
            "retry_adjusted_weighted_service_seconds": None if missing else corrected_service,
            "phase_utilization_factor": factor, "wall_time": _band(projected)}

    preparation_times = {row["patient_id"]: _seconds(row["seconds"], "preparation")
                         for row in timings.get("preparation", [])}
    prep_service, prep_counts = _weighted(preparation_times, manifest, warnings, "preparation")
    prep_full = None if prep_service is None else prep_service / preparation_workers
    prep_pilot = sum(preparation_times.values()) / preparation_workers
    records = _read_records(run_dir / "results.jsonl")
    for model in models:
        outputs = [record.get("variants", {}).get(variant, {}).get("assessment_plans", {}).get(model, {})
                   for record in records for variant in variants]
        coverage = {"expected_ranked_pairs": len(selected_ids) * len(variants) * 3,
                    "available_conditions": 0, "valid_plans": 0, "nonempty_plans": 0,
                    "complete_nonempty_slots": 0}
        for output in outputs:
            predictions = output.get("predictions") or []
            nonempty = 0
            for pair in predictions:
                condition, plan = pair.get("condition"), pair.get("plan")
                coverage["available_conditions"] += isinstance(condition, str) and bool(condition.strip())
                valid = (isinstance(plan, list) and len(plan) <= 3
                         and all(isinstance(action, str) and action.strip() for action in plan))
                coverage["valid_plans"] += valid
                coverage["nonempty_plans"] += valid and bool(plan)
                nonempty += bool(isinstance(condition, str) and condition.strip() and valid and plan)
            coverage["complete_nonempty_slots"] += len(predictions) == 3 and nonempty == 3
        projected_models[model]["output_coverage"] = coverage
    field_counts = {}
    for record in records:
        field_counts[record["id"]] = sum(
            sum(entry.get(field, {}).get("status") in ("scored", "error", "interrupted")
                for field in ("condition", "plan"))
            for output in _outputs(record) for entry in output.get("evaluation") or [])
    scoring_batches = timings.get("scoring") or []
    scoring_cost = defaultdict(float)
    overlap_cost = {patient: 0.0 for patient in selected_ids}
    post_cost = {patient: 0.0 for patient in selected_ids}
    overlap_scoring = execution.get("overlap_scoring", False)
    batch_size = execution.get("scoring_batch_patients", 8)
    if type(overlap_scoring) is not bool or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Recorded scoring settings must contain a boolean overlap and positive batch size")
    # Every fresh final score() visits all patients once. Earlier passes live
    # inside prepare_and_generate, but their last pass can outlast the GPU.
    final_batches = math.ceil(len(selected_ids) / batch_size)
    inside_prepare_batches = max(0, len(scoring_batches) - final_batches) if overlap_scoring else 0
    generation_ends = [_seconds(row["end_seconds"], "generation end") for row in generation
                       if row["model"] in (local or models) and "end_seconds" in row]
    generation_end = max(generation_ends) if generation_ends else None
    cold_overlap = cold_post = pilot_overlap = pilot_post_inside = 0.0
    overlap_batches = timestamped_batches = 0
    per_field = []
    for batch in scoring_batches[1:]:
        fields = sum(_batch_fields(batch, field_counts).values())
        if fields:
            per_field.append(_seconds(batch["seconds"], "scoring") / fields)
    cold_score = 0.0
    for index, batch in enumerate(scoring_batches):
        seconds = _seconds(batch["seconds"], "scoring")
        fields = _batch_fields(batch, field_counts)
        total = sum(fields.values())
        if index == 0 and per_field and not overlap_scoring:
            cold_score = max(0.0, seconds - median(per_field) * total)
            seconds -= cold_score
        fraction = float(index < inside_prepare_batches)
        if overlap_scoring and generation_end is not None and "start_seconds" in batch and "end_seconds" in batch:
            start = _seconds(batch["start_seconds"], "scoring start")
            end = _seconds(batch["end_seconds"], "scoring end")
            if end < start:
                raise ValueError("Scoring batch end precedes its start")
            fraction = (min(1.0, max(0.0, (generation_end - start) / (end - start)))
                        if end > start else float(end <= generation_end))
            timestamped_batches += 1
        overlap_batches += fraction > 0
        if index == 0:
            cold_overlap = cold_score * fraction
            cold_post = cold_score * (1 - fraction)
        measured = _seconds(batch["seconds"], "scoring")
        pilot_overlap += measured * fraction
        if index < inside_prepare_batches:
            pilot_post_inside += measured * (1 - fraction)
        for patient, count in fields.items():
            cost = seconds * count / total
            scoring_cost[patient] += cost
            overlap_cost[patient] += cost * fraction
            post_cost[patient] += cost * (1 - fraction)
    score_service, score_counts = _weighted(scoring_cost, manifest, warnings, "scoring")
    score_full = None if score_service is None else score_service + cold_score
    overlap_service, _ = _weighted(overlap_cost, manifest, warnings, "overlapped scoring")
    post_service, _ = _weighted(post_cost, manifest, warnings, "post-generation scoring")
    overlap_full = overlap_service + cold_overlap
    post_full = post_service + cold_post

    def model_wall(model):
        value = projected_models[model]["wall_time"]
        return None if value is None else value["seconds"]

    local_full = [model_wall(model) for model in local]
    hosted_full = [model_wall(model) for model in hosted]
    pilot_first = pilot_model_seconds.get(first_local, 0.0)
    pilot_hosted = sum(pilot_model_seconds.get(model, 0) for model in hosted)
    pilot_remaining = sum(phases.get(model, pilot_model_seconds.get(model, 0))
                          for model in local if model != first_local)
    measured_prep_gen = _seconds(timings.get("preparation_and_generation_seconds", 0), "preparation/generation")
    pilot_start = max(prep_pilot, pilot_first)
    pilot_lanes = max(pilot_start + pilot_remaining, pilot_hosted,
                      pilot_start + pilot_overlap if overlap_scoring else 0.0)
    fixed = max(0.0, measured_prep_gen - pilot_lanes - pilot_post_inside)
    pipeline = conservative = None
    if prep_full is not None and score_full is not None and all(value is not None for value in local_full + hosted_full):
        first_full = model_wall(first_local) if first_local is not None else 0.0
        score_start = max(prep_full, first_full)
        local_lane = score_start + sum(model_wall(model) for model in local if model != first_local)
        generation_lanes = max(local_lane, sum(hosted_full))
        pipeline = max(generation_lanes, score_start + overlap_full if overlap_scoring else 0.0) + post_full + fixed
        # Scoring snapshots can arrive later than the earliest possible start.
        # Full serialization supplies an explicit conservative comparison.
        conservative = generation_lanes + score_full + fixed
    measured_wall = _wall(metadata.get("started_at"), metadata.get("finished_at"))
    scoring_seconds = _seconds(timings.get("scoring_seconds", 0), "scoring total")
    run_overhead = max(0.0, measured_wall - measured_prep_gen - scoring_seconds) if measured_wall is not None else 0.0
    if pipeline is not None:
        pipeline += run_overhead
        conservative += run_overhead
    warnings.append("The ±25% band is a planning sensitivity, not a statistical confidence interval.")
    warnings.append("Hosted estimates assume similar API quota, retry behavior, and provider load.")
    measured_fields = all("newly_scored_fields_by_patient" in batch for batch in scoring_batches)
    warnings.append("Scoring batch time is allocated by scored field count; token-length effects are not isolated.")
    if overlap_scoring:
        warnings.append("Overlap estimate assumes scoring keeps pace after the first local phase; the conservative "
                        "bound serializes scoring to cover late snapshots and batch tails.")
        if timestamped_batches < len(scoring_batches):
            warnings.append("Scoring batches without timestamps use stage-order overlap fallback; late scoring "
                            "inside preparation/generation may be underestimated.")
        if timestamped_batches:
            warnings.append("Scoring batches crossing the last local generation end are split proportionally "
                            "by elapsed time; work within a batch may not be uniform.")
        if not measured_fields:
            warnings.append("Legacy scoring snapshots omit newly scored field counts; repeated-pass service "
                            "allocation is approximate.")

    result = {"method": "stratum-weighted-runtime-v1", "run_directory": str(run_dir),
        "population_count": manifest["population_count"], "pilot_count": manifest["sample_count"],
        "measured_pilot": {"status": metadata.get("status"), "wall_seconds": measured_wall,
            "preparation_and_generation_seconds": measured_prep_gen,
            "scoring_seconds": timings.get("scoring_seconds"), "model_phases": phases},
        "generation_by_model": projected_models,
        "preparation": {"workers": preparation_workers, "stratum_coverage": prep_counts,
                        "wall_time": _band(prep_full)},
        "scoring": {"stratum_coverage": score_counts, "fixed_cold_start_seconds": cold_score,
                    "wall_time": _band(score_full), "overlapped_service_wall_time": _band(overlap_full),
                    "final_service_wall_time": _band(post_full), "overlapped_batch_count": overlap_batches,
                    "post_generation_service_wall_time": _band(post_full),
                    "generation_end_seconds": generation_end,
                    "timestamped_batch_count": timestamped_batches,
                    "measured_post_generation_service_inside_prepare_seconds": pilot_post_inside,
                    "allocation": "newly_scored_fields" if measured_fields else "final_field_count_fallback"},
        "pipeline": {"wall_time": _band(pipeline), "fixed_preparation_generation_overhead_seconds": fixed,
            "fixed_startup_report_overhead_seconds": run_overhead,
            "conservative_serial_scoring_bound": _band(conservative),
            "overlap_formula": "max(max(preparation, first local) + remaining locals, hosted work, "
                               "max(preparation, first local) + overlapped scoring) + post-generation scoring + fixed overhead"},
        "warnings": warnings, "plan_judging": None}
    result["plan_judging"] = _project_plan_stage(metadata, records, manifest, models, variants, warnings)
    judge = result["plan_judging"]
    if pipeline is not None and judge is not None and judge["observed_cache_reuse"] is not None:
        result["pipeline"]["including_plan_judging"] = _band(pipeline + judge["observed_cache_reuse"]["seconds"])
    return result


def _project_plan_stage(metadata, records, manifest, models, variants, warnings):
    stage = metadata.get("plan_evaluation") or {}
    if not stage.get("finished_at"):
        return None
    logical = {}
    scalar_status_counts = defaultdict(int)
    requests_complete = True
    for record in records:
        scalars = []
        for output in _outputs(record):
            for entry in output.get("evaluation") or []:
                plan = entry.get("plan_correctness") or {}
                judges = plan.get("judges") or []
                reference, generated = plan.get("reference_actions"), plan.get("generated_actions")
                expects_calls = bool(generated) and plan.get("status") in ("scored", "error", "pending", "interrupted")
                if expects_calls:
                    requests_complete &= (isinstance(reference, list)
                        and len(judges) == len(reference) + 1
                        and [judge.get("call") for judge in judges] == ["action"] * len(reference) + ["extras"])
                elif judges:
                    requests_complete = False
                scalars.extend(judges)
        logical[record["id"]] = len(scalars)
        for scalar in scalars:
            scalar_status_counts[scalar.get("status", "unknown")] += 1
    full_logical, coverage = _weighted(logical, manifest, warnings, "Plan judging")
    pilot_logical = sum(logical.values())
    calls, hits = stage.get("calls", 0), stage.get("cache_hits", 0)
    if any(type(value) is not int or value < 0 for value in (calls, hits)):
        raise ValueError("Plan-stage calls and cache hits must be nonnegative integers")
    wall = _wall(stage.get("started_at"), stage.get("finished_at"))
    generation_finished = metadata.get("finished_at")
    after_generation = bool(generation_finished and stage.get("started_at")
                            and datetime.fromisoformat(stage["started_at"]) >= datetime.fromisoformat(generation_finished))
    entry_counts = stage.get("counts")
    if entry_counts is not None and (not isinstance(entry_counts, dict) or any(
            type(value) is not int or value < 0 for value in entry_counts.values())):
        raise ValueError("Plan-stage entry counts must be nonnegative integers")
    no_pending = not any((entry_counts or {}).get(status, 0) for status in ("pending", "interrupted"))
    full_coverage = (full_logical is not None and calls > 0 and pilot_logical > 0
                     and wall is not None and wall > 0 and calls + hits >= pilot_logical
                     and after_generation)
    usable = (full_coverage and stage.get("status") == "completed" and no_pending
              and not (entry_counts or {}).get("error", 0)
              and requests_complete and scalar_status_counts.get("scored", 0) == pilot_logical)
    all_attempted = not set(scalar_status_counts) - {"scored", "error", "invalid_response"}
    attempted = (full_coverage and stage.get("status") in ("completed", "failed_incomplete")
                 and entry_counts is not None and no_pending and all_attempted and requests_complete)
    observed = wall * full_logical / (calls + hits) if usable else None
    without_cache = wall * full_logical / calls if usable else None
    attempted_observed = wall * full_logical / (calls + hits) if attempted else None
    attempted_without_cache = wall * full_logical / calls if attempted else None
    features = manifest.get("population_features") or {}
    theoretical = (len(models) * len(variants) * 3 * sum(
        feature["reference_action_count"] + 1 for feature in features.values()
        if feature.get("judgeable_reference"))) if features else None
    if not usable and not attempted:
        warnings.append("Plan timing is incomplete, cache-only, predates generation completion, or is a partial "
                        "resume or has unresolved judge errors; fresh full-stage throughput is unverified.")
    if attempted and not usable:
        warnings.append("The entire Plan stage was attempted but judge errors remain. Attempted-stage estimates "
                        "exclude the extra work needed to obtain fully valid judgements.")
    return {"status": stage.get("status"), "measured_wall_seconds": wall,
        "actual_calls": calls, "cache_hits": hits,
        "observed_cache_hit_fraction": hits / (calls + hits) if calls + hits else None,
        "pilot_logical_scalar_requests": pilot_logical, "weighted_full_logical_scalar_requests": full_logical,
        "theoretical_full_requests_before_empty_generation_skips": theoretical,
        "plan_entry_counts": entry_counts, "scalar_status_counts": dict(scalar_status_counts),
        "all_required_scalar_requests_present": bool(requests_complete),
        "stratum_coverage": coverage, "observed_cache_reuse": _band(observed),
        "without_cache_reuse": _band(without_cache),
        "attempted_stage_with_observed_cache_reuse": _band(attempted_observed),
        "attempted_stage_without_cache_reuse": _band(attempted_without_cache)}


def project_plan_runtime(run_dir, *, root):
    """Project a fresh Plan stage independently of resumed generation timings.

    This reads only the Plan stage, canonical judged outputs, and selection
    weights. It preserves generation resumption history in provenance and does
    not manufacture a combined generation/Plan estimate. Source hashes let a
    caller explicitly compose it with an archived uninterrupted forecast.
    """
    run_dir = Path(run_dir).resolve()
    root = Path(root).resolve()
    metadata_path, results_path = run_dir / "run.json", run_dir / "results.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = metadata.get("pilot_selection")
    if (not isinstance(manifest, dict) or any(name not in manifest.get("strata", {}) for name in STRATA)
            or sum(manifest["strata"][name]["population_count"] for name in STRATA) != manifest.get("population_count")):
        raise ValueError("Plan runtime projection requires a valid pilot_selection stratification manifest")
    records = _read_records(results_path)
    experiment = metadata.get("experiment") or {}
    variants = experiment.get("selected_variants") or sorted({variant for record in records
                                                             for variant in record.get("variants", {})})
    models = experiment.get("selected_models") or sorted({model for record in records
        for variant in record.get("variants", {}).values() for model in variant.get("assessment_plans", {})})
    warnings = ["The ±25% band is a planning sensitivity, not a statistical confidence interval."]
    projection = _project_plan_stage(metadata, records, manifest, models, variants, warnings)
    if projection is None:
        warnings.append("No finished Plan-judging stage is recorded.")
    return {"method": "stratum-weighted-plan-runtime-v1", "run_directory": str(run_dir),
        "population_count": manifest["population_count"], "pilot_count": manifest["sample_count"],
        "plan_judging": projection, "warnings": warnings,
        "provenance": {"generation_timings_used": False,
            "generation_resumption_count": len(metadata.get("resumptions") or []),
            "plan_stage_started_at": (metadata.get("plan_evaluation") or {}).get("started_at"),
            "source_artifacts": {path.name: {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (metadata_path, results_path)}}}
