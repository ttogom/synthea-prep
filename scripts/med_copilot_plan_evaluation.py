"""Complete only saved Plan judgements, with bounded concurrency and durable reuse.

The scalar response cache is the incremental checkpoint. Exporting results once
per invocation avoids rewriting the entire growing run after every patient.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from contextvars import copy_context
from copy import deepcopy
import hashlib
from itertools import islice
import json
import math
from numbers import Real
import os
from pathlib import Path
import time

from scripts.med_copilot_config import PIPELINE_DEFAULTS
from scripts.med_copilot_helpers import load_patient_records
from scripts.experiment_runtime.ollama import plan_judge_settings
from scripts.experiment_runtime.plan_judging import validate_profile
from scripts.experiment_runtime.memory_supervisor import MemoryPressureStop
from scripts.med_copilot_judges import (
    DEDUCTION_PER_UNSUPPORTED_EXTRA, JUDGE_FILE, CorrectnessJudges,
    parse_action_judgement, parse_extras_judgement, plan_action_messages,
    plan_entry, plan_evaluation_protocol, plan_extras_messages, plan_score, request_key,
)
from scripts.med_copilot_run_file import (
    RUN_FILE, RunFile, RunMetadataError, _fsync_directory, error_scope,
    pacific_now, pipeline_stage,
)
from scripts.med_copilot_run_lock import exclusive_run_lock


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite JSON number: {text}")
    return value


def _read_object(text):
    value = json.loads(text, object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError("Saved run/result must be a JSON object")
    return value


def _outputs(record):
    if "variants" in record:
        variants = record["variants"]
        if not isinstance(variants, dict) or not variants:
            raise ValueError("Result variants must be a nonempty object")
        for variant, body in variants.items():
            if not isinstance(body, dict) or not isinstance(body.get("assessment_plans"), dict):
                raise ValueError("Result variant must contain assessment_plans")
            for model, output in body["assessment_plans"].items():
                yield variant, model, output
    elif "assessment_plans" in record:
        if not isinstance(record["assessment_plans"], dict):
            raise ValueError("Result assessment_plans must be an object")
        for model, output in record["assessment_plans"].items():
            yield None, model, output
    else:
        yield None, None, record


class _SavedResults:
    """Replay a validated file without retaining the decoded cohort.

    The run lock excludes cooperating writers. Identity checks and a streaming
    digest also reject edits by other writers between validation and export.
    """

    def __init__(self, path):
        self.path = path
        self._signature = self._stat_signature()
        self._sha256 = None

    def _stat_signature(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def assert_unchanged(self):
        if self._stat_signature() != self._signature:
            raise ValueError("Saved results changed after validation; rerun before judging")

    def __iter__(self):
        self.assert_unchanged()
        digest = hashlib.sha256()
        with self.path.open(encoding="utf-8", newline="") as handle:
            for line in handle:
                digest.update(line.encode("utf-8"))
                if line.strip():
                    yield _read_object(line)
        self.assert_unchanged()
        checksum = digest.hexdigest()
        if self._sha256 is not None and checksum != self._sha256:
            raise ValueError("Saved results changed after validation; rerun before judging")
        self._sha256 = checksum


def _records(run_dir, metadata, scorer):
    if metadata.get("status") not in ("completed", "completed_with_errors"):
        raise ValueError("Generation must finish before Plan evaluation")
    selected = metadata.get("evaluation", {}).get("selected_patient_ids")
    if selected != [patient["id"] for patient in scorer.patients]:
        raise ValueError("The scorer's cohort does not match the run's selected_patient_ids")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_patient_ids contains duplicate patients")
    records = _SavedResults(run_dir / "results.jsonl")
    selected_ids, seen = set(selected), set()
    expected_models = metadata.get("assessment_plan_models", {}).get("selected_models")
    expected_variants = metadata.get("experiment", {}).get("selected_variants")
    for record in records:
        patient_id = record.get("id")
        if not isinstance(patient_id, str) or patient_id not in selected_ids or patient_id in seen:
            raise ValueError("Saved results must contain every selected patient exactly once")
        seen.add(patient_id)
        if record.get("status") not in ("completed", "failed_incomplete"):
            raise ValueError(f"Generation is incomplete for patient {record['id']}")
        if "variants" in record and expected_variants and set(record["variants"]) != set(expected_variants):
            raise ValueError("Saved result variants do not match the recorded experiment")
        groups = {}
        for variant, model, output in _outputs(record):
            groups.setdefault(variant, []).append(model)
            if not isinstance(output, dict) or output.get("status") not in (
                    "completed", "failed_incomplete", "not_processed"):
                raise ValueError(f"Invalid generation slot for patient {record['id']}")
            if output["status"] != "completed":
                continue
            predictions = output.get("predictions")
            if predictions is not None and (not isinstance(predictions, list) or len(predictions) != 3):
                raise ValueError("Generated predictions must contain exactly three ranked objects or null")
            if predictions is not None:
                for prediction in predictions:
                    if (not isinstance(prediction, dict) or set(prediction) - {"condition", "plan"}
                            or prediction.get("condition") is not None
                            and not isinstance(prediction["condition"], str)
                            or prediction.get("plan") is not None and (
                                not isinstance(prediction["plan"], list) or len(prediction["plan"]) > 3
                                or any(not isinstance(action, str) for action in prediction["plan"]))):
                        raise ValueError("Invalid saved diagnosis/Plan prediction")
            evaluation = output.get("evaluation")
            if evaluation is not None and (not isinstance(evaluation, list) or len(evaluation) != 3
                    or any(not isinstance(entry, dict) or type(entry.get("rank")) is not int
                           or entry["rank"] != rank
                           for rank, entry in enumerate(evaluation, 1))):
                raise ValueError("Saved evaluation must contain exactly ranks 1, 2 and 3")
        if not groups or any(not values for values in groups.values()):
            raise ValueError("Saved result has no generation slots")
        if expected_models and any(None not in values and set(values) != set(expected_models)
                                   for values in groups.values()):
            raise ValueError("Saved generation models do not match the recorded experiment")
    if seen != selected_ids:
        raise ValueError("Saved results must contain every selected patient exactly once")
    return records


def _verified_sources(metadata, scorer, root):
    for recorded, label in ((metadata["input"], "Patient input"),
                            ({"file": metadata["evaluation"]["labels_path"],
                              "sha256": metadata["evaluation"]["labels_sha256"]}, "Labels")):
        path = Path(root) / recorded["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded["sha256"]:
            raise ValueError(f"{label} changed since the run: {path}")
    if scorer._labels_sha256 != metadata["evaluation"]["labels_sha256"]:
        raise ValueError("The scorer's labels do not match the recorded labels")
    inputs = {patient["id"]: patient for patient in load_patient_records(Path(root) / metadata["input"]["file"])}
    labels = {_read_object(line)["id"]: _read_object(line) for line in (
        Path(root) / metadata["evaluation"]["labels_path"]).read_text(encoding="utf-8").split("\n") if line.strip()}
    for patient in scorer.patients:
        source = inputs.get(patient["id"])
        if source is None or any(patient.get(key) != source[key] for key in ("subjective", "objective")):
            raise ValueError("The scorer's patient S/O does not match the recorded input")
        if any(scorer._labels[patient["id"]][key] != labels[patient["id"]][key]
               for key in ("condition", "plan")):
            raise ValueError("The scorer's labels do not match the recorded labels")


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _validate_protocol(saved, current):
    if (not isinstance(saved, dict) or type(saved.get("version")) is not int
            or saved["version"] != current["version"]):
        raise ValueError("Cannot resume: missing or unsupported Plan evaluation protocol version")
    for group in ("plan", "rank_weighting"):
        recorded = saved.get(group)
        for key, value in current[group].items():
            if not isinstance(recorded, dict) or recorded.get(key) != value:
                raise ValueError(f"Cannot resume: Plan evaluation protocol.{group}.{key} changed")


def _evaluation_protocol(run_dir, metadata, records, root, judge):
    """Initialize only untouched evaluations; validate all existing judging provenance."""
    current = plan_evaluation_protocol(root, plan_judge=judge)
    correctness = metadata.get("evaluation", {}).get("correctness")
    if correctness is not None and not isinstance(correctness, dict):
        raise ValueError("Generation correctness provenance must be an object")
    previous = metadata.get("plan_evaluation")
    history = metadata.get("plan_evaluation_history", [])
    if previous is not None and not isinstance(previous, dict):
        raise ValueError("plan_evaluation must be an object")
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("plan_evaluation_history must be a list of objects")
    stages = [*history, *([previous] if previous else [])]
    versioned = [item for item in stages if "protocol" in item]
    for item in versioned:
        _validate_protocol(item["protocol"], current)
        if "judge_settings" in item:
            _validate_protocol({**item["protocol"], "plan": item["judge_settings"]}, current)
    if previous is not None and "protocol" in previous:
        return current
    if versioned:
        raise ValueError("Cannot resume: current Plan evaluation protocol is missing")

    cache = run_dir / JUDGE_FILE
    started = bool(stages) or (cache.exists() and cache.stat().st_size > 0)
    for record in records:
        for _, _, output in _outputs(record):
            if output.get("status") != "completed":
                continue
            for evaluation in output.get("evaluation") or []:
                entry = evaluation.get("plan_correctness") or {}
                if not isinstance(entry, dict):
                    raise ValueError("Saved Plan correctness must be an object")
                automatic_zero = (entry.get("status") == "scored" and entry.get("reason") == "empty_generated_plan"
                                  and entry.get("generated_actions") == [] and entry.get("score") == 0
                                  and entry.get("weighted_score") == 0)
                if (entry.get("judges") or entry.get("error") or entry.get("status") in ("error", "interrupted")
                        or entry.get("status") == "scored" and not automatic_zero
                        or not automatic_zero and any(entry.get(key) is not None for key in (
                            "score", "weighted_score", "credits", "unsupported_extra_count", "deduction"))):
                    started = True
    if started:
        if not metadata.get("evaluation", {}).get("correctness"):
            raise ValueError("Cannot resume legacy Plan judging without correctness provenance")
        CorrectnessJudges.validate_resume(metadata, root, plan_judge=judge)
        for item in stages:
            if "judge_settings" in item:
                legacy = {"version": current["version"], "plan": item["judge_settings"],
                          "rank_weighting": metadata["evaluation"]["correctness"].get("rank_weighting")}
                _validate_protocol(legacy, current)
    return current


def _saved_plan(entry, expected, judge, condition, patient):
    """Trust completed aggregates only with matching inputs and scalar evidence."""
    if not isinstance(entry, dict) or entry.get("status") != "scored":
        return False
    for key in ("method", "validation", "rank_weight", "reference_actions", "generated_actions"):
        if entry.get(key) != expected[key]:
            return False
    if not all(_finite(entry.get(key)) for key in ("score", "weighted_score")):
        return False
    if expected["status"] == "scored":
        return all(entry.get(key) == expected[key] for key in (
            "reason", "score", "weighted_score", "credits", "unsupported_extra_count", "deduction"))
    if expected["status"] != "pending":
        return False
    reference, generated = expected["reference_actions"], expected["generated_actions"]
    records = entry.get("judges")
    skipped_extras = entry.get("reason") == "zero_reference_coverage"
    if not isinstance(records, list) or len(records) != len(reference) + (not skipped_extras):
        return False
    context = (condition, patient["subjective"], patient["objective"])
    requests = [("action", index, plan_action_messages(*context, action, generated))
                for index, action in enumerate(reference, 1)]
    if not skipped_extras:
        requests.append(("extras", None, plan_extras_messages(*context, reference, generated)))
    credits = []
    extras = None
    try:
        for record, (kind, index, messages) in zip(records, requests):
            if (not isinstance(record, dict) or record.get("status") != "scored"
                    or record.get("judge_model") != judge or record.get("call") != kind
                    or record.get("reference_item") != index
                    or record.get("request_key") != request_key(judge, messages)):
                return False
            if kind == "action":
                credit = parse_action_judgement(record.get("raw_response"))["credit"]
                if record.get("credit") != credit:
                    return False
                credits.append(credit)
            else:
                extras = parse_extras_judgement(record.get("raw_response"))["unsupported_extra_count"]
                if record.get("unsupported_extra_count") != extras:
                    return False
    except (ValueError, TypeError, KeyError):
        return False
    if skipped_extras and any(credits):
        return False
    deduction = None if skipped_extras else DEDUCTION_PER_UNSUPPORTED_EXTRA * extras
    score = 0.0 if skipped_extras else plan_score(credits, deduction)
    return (entry.get("credits") == credits and entry.get("unsupported_extra_count") == extras
            and entry.get("deduction") == deduction and math.isclose(entry["score"], score)
            and math.isclose(entry["weighted_score"], score * expected["rank_weight"]))


def _jobs(records, scorer, judge, changes):
    for record in records:
        patient = scorer._patients[record["id"]]
        label = scorer._labels[record["id"]]
        condition = label["condition"].replace("_", " ")
        for _, _, output in _outputs(record):
            if output["status"] != "completed":
                continue
            if output.get("evaluation") is None:
                output["evaluation"] = [{"rank": rank} for rank in (1, 2, 3)]
                changes["any"] = True
            for index, evaluation in enumerate(output["evaluation"]):
                predictions = output.get("predictions")
                generated = predictions[index].get("plan") if predictions is not None else None
                expected = plan_entry(index + 1, label["plan"], generated)
                saved = evaluation.get("plan_correctness")
                if _saved_plan(saved, expected, judge, condition, patient):
                    continue
                if saved != expected:
                    changes["any"] = True
                evaluation["plan_correctness"] = expected
                if expected["status"] == "pending":
                    changes["any"] = True
                    yield record["id"], expected, condition, patient


def _write_results(run_dir, records, *, changes=None, source=None):
    temporary = run_dir / ".results.jsonl.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                # Release the writer's reference before the next window loads.
                del record
            if changes is None or changes["any"]:
                handle.flush()
                os.fsync(handle.fileno())
        if changes is None or changes["any"]:
            if source is not None:
                source.assert_unchanged()
            os.replace(temporary, run_dir / "results.jsonl")
            _fsync_directory(run_dir)
    finally:
        temporary.unlink(missing_ok=True)


class PlanJudgeUnavailableError(RuntimeError):
    """The configured judge endpoint/model cannot serve any Plan requests."""


def _run_job(run_file, judges, patient_id, entry, condition, patient):
    with error_scope(run_file, patient_id=patient_id), pipeline_stage("scoring"):
        result = judges.judge_plan(entry, condition, patient)
        for record in result.get("judges", []):
            if (record.get("error") or {}).get("type") in (
                    "NotFoundError", "AuthenticationError", "PermissionDeniedError"):
                raise PlanJudgeUnavailableError(
                    "Plan judge model or credentials are unavailable; fix the endpoint/model "
                    "and rerun. Completed judge responses have been saved.")
        return result


def _parallel(jobs, judges, run_file, concurrency, stop_requested=None, *, executor):
    pending = set()
    iterator = iter(jobs)
    exhausted = False
    try:
        while pending or not exhausted:
            if stop_requested is not None and stop_requested():
                raise MemoryPressureStop("Memory guard requested a stop")
            while not exhausted and len(pending) < concurrency * 2:
                if stop_requested is not None and stop_requested():
                    raise MemoryPressureStop("Memory guard requested a stop")
                try:
                    job = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(copy_context().run, _run_job, run_file, judges, *job))
            if pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED,
                                         timeout=1 if stop_requested is not None else None)
                for future in finished:
                    future.result()
    finally:
        for future in pending:
            future.cancel()
        # Running requests save replies in the cache before we export any shared
        # entries. A future exception or Ctrl-C stops new work, not durable saves.
        if pending:
            wait(pending)
        # A stopped jobs generator can otherwise retain its current patient
        # through an exception traceback while the unchanged tail is copied.
        iterator.close()


def _counts(records):
    counts = dict.fromkeys(("scored", "unavailable", "pending", "error", "interrupted"), 0)
    for record in records:
        for _, _, output in _outputs(record):
            if output.get("status") != "completed":
                continue
            for entry in output.get("evaluation") or ({}, {}, {}):
                plan = entry.get("plan_correctness")
                status = plan.get("status", "pending") if isinstance(plan, dict) else "pending"
                if not isinstance(status, str):
                    status = "pending"
                counts[status] = counts.get(status, 0) + 1
    return counts


def _evaluated_records(records, scorer, judges, run_file, concurrency, stop_requested, state, executor):
    """Drain two patients at a time, then export them before loading the next two.

    A stopped window is exported only after its workers finish. The untouched
    tail is copied as well; failure before replacement leaves the original file
    intact, with durable scalar replies available for reconstruction on restart.
    """
    iterator = iter(records)
    while batch := list(islice(iterator, 2)):
        if state["interrupted"] is None:
            try:
                _parallel(_jobs(batch, scorer, judges.plan_judge, state), judges, run_file,
                          concurrency, stop_requested, executor=executor)
            except BaseException as exc:
                state["interrupted"] = exc
        for record in batch:
            for status, count in _counts((record,)).items():
                state["counts"][status] = state["counts"].get(status, 0) + count
            yield record
            del record
        batch.clear()


def resume_plan_evaluation(run_dir, scorer, models, *, root, concurrency=2, context_length=None,
                           runtime_profile=None, runtime_verifier=None, _lock_held=False,
                           stop_requested=None, guard_metadata=None, client_workers=None):
    """Finish only Plan fields; valid scalar replies survive crashes and failed exports.

    models contains only the recorded judge. _lock_held is reserved for the CLI,
    which must hold the same lock while reconstructing its input/model objects.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("plan_judge_concurrency must be an integer from 1 to 64")
    if context_length is not None and (type(context_length) is not int or not 512 <= context_length <= 262144):
        raise ValueError("Plan judge context_length must be an integer from 512 to 262144")
    run_dir = Path(run_dir).resolve()
    with nullcontext() if _lock_held else exclusive_run_lock(run_dir):
        invocation_started = time.monotonic()
        metadata = _read_object((run_dir / RUN_FILE).read_text(encoding="utf-8"))
        judge = metadata.get("experiment", {}).get("plan_judge", PIPELINE_DEFAULTS["plan_judge"])
        _verified_sources(metadata, scorer, root)
        records = _records(run_dir, metadata, scorer)
        protocol = _evaluation_protocol(run_dir, metadata, records, root, judge)
        if judge not in models.model_ids:
            raise ValueError(f"Recorded Plan judge {judge} is unavailable; no Plan scores were changed")
        previous_settings = plan_judge_settings(metadata)
        settings = plan_judge_settings(metadata, concurrency=concurrency, context_length=context_length)
        if runtime_profile is not None:
            native = validate_profile(runtime_profile["settings"])
            if any(native[key] != settings[key] for key in settings):
                raise ValueError("Plan judge profile differs from effective stage settings")
            if client_workers is None:
                client_workers = native.get("client_workers", concurrency)
            elif "client_workers" in native and client_workers != native["client_workers"]:
                raise ValueError("Plan judge client workers differ from the runtime profile")
        if client_workers is None:
            client_workers = concurrency
        if type(client_workers) is not int or not 1 <= client_workers <= 64:
            raise ValueError("Plan judge client_workers must be an integer from 1 to 64")
        history = metadata.get("plan_evaluation_history", [])
        if not isinstance(history, list):
            raise ValueError("plan_evaluation_history must be a list")
        if metadata.get("plan_evaluation"):
            previous = {**metadata["plan_evaluation"]}
            if previous.get("context_length") is None:
                previous["context_length"] = previous_settings["context_length"]
            history = [*history, previous]
        runtime_provenance = runtime_verifier() if runtime_verifier is not None else None
        records.assert_unchanged()
        run_file = RunFile(run_dir, metadata)
        run_file.redact = models.redact
        def complete(model, messages):
            # Running Plans must not start their next action/extras call while
            # the scheduler drains after a stop. Saved replies remain reusable.
            if stop_requested is not None and stop_requested():
                raise MemoryPressureStop("Memory guard requested a stop")
            return models.judge(model, messages)

        judges = CorrectnessJudges(
            complete, models.model_ids, run_dir / JUDGE_FILE, plan_judge=judge,
            generator_models=metadata.get("assessment_plan_models", {}).get("selected_models"),
            judge_model_config=models.metadata(root) if hasattr(models, "metadata") else None,
            redact=models.redact,
        )
        started = pacific_now().isoformat()
        stage = {"status": "running", "started_at": started, "finished_at": None,
                 "judge_model": judge, **settings, "concurrency": client_workers,
                 "client_workers": client_workers, "gpu_slots": settings["concurrency"], "protocol": protocol,
                 "judge_settings": {**judges.metadata(root)["plan"], **protocol["plan"]},
                 "responses_file": JUDGE_FILE, "calls": 0, "cache_hits": 0,
                 "implementation_file": str(Path(__file__).resolve().relative_to(Path(root).resolve()))
                 if Path(__file__).resolve().is_relative_to(Path(root).resolve()) else str(Path(__file__).resolve()),
                 "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if runtime_profile is not None:
            stage["runtime_profile"] = deepcopy(runtime_profile)
        if runtime_provenance is not None:
            stage["runtime_provenance"] = runtime_provenance
        if guard_metadata is not None:
            stage["memory_guard"] = deepcopy(guard_metadata)
        run_file.update(plan_evaluation=stage, plan_evaluation_history=history)
        changes = {"any": False, "interrupted": None, "counts": _counts(())}
        try:
            # One executor for the invocation; only two decoded patients are
            # resident, regardless of cohort size or the completed prefix.
            with ThreadPoolExecutor(max_workers=client_workers, thread_name_prefix="plan-judge") as executor:
                _write_results(run_dir, _evaluated_records(
                    records, scorer, judges, run_file, client_workers, stop_requested, changes, executor),
                    changes=changes, source=records)
            interrupted = changes["interrupted"]
            counts = changes["counts"]
            stage.update(status=("interrupted" if isinstance(interrupted, KeyboardInterrupt)
                                 else "failed_incomplete" if interrupted is not None
                                 or any(counts.get(key, 0) for key in ("pending", "error", "interrupted"))
                                 else "completed"), finished_at=pacific_now().isoformat(),
                         calls=judges.calls, cache_hits=judges.cache_hits, counts=counts)
            stage.update(elapsed_seconds=time.monotonic() - invocation_started,
                         request_timing=judges.request_timing())
            stage["error"] = None if interrupted is None else {
                "type": type(interrupted).__name__, "message": models.redact(str(interrupted))}
            run_file.update(plan_evaluation=stage)
        except BaseException:
            # Never claim completed when the canonical export failed. Scalar
            # replies already fsynced to the cache remain available next time.
            stage.update(status="failed_incomplete", finished_at=pacific_now().isoformat(),
                         calls=judges.calls, cache_hits=judges.cache_hits,
                         elapsed_seconds=time.monotonic() - invocation_started,
                         request_timing=judges.request_timing())
            run_file.update(plan_evaluation=stage)
            raise
        if interrupted is not None:
            raise interrupted
        return judges
