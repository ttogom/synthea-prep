"""Validate the production Plan judge on temporary copies, never on the source run.

Run: python scripts/validate_plan_judging.py RUN_DIRECTORY NEW_OUTPUT_DIRECTORY
This makes live local judge requests and intentionally interrupts one invocation.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from random import Random
import sys
from threading import Lock
import time

# Direct execution starts in scripts/; initialize project/vendor paths before adapters.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))

import judge_treatment_plans as command
from scripts.experiment_runtime.judge_provenance import save_verified_runtime
from scripts.experiment_runtime.ollama import ollama_for_saved_run
from scripts.experiment_runtime.pilot import select_runtime_pilot
from scripts.experiment_runtime.plan_judging import load_judge_profile
from scripts.med_copilot_helpers import load_patient_records
from scripts.med_copilot_judges import CorrectnessJudges, plan_entry
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.med_copilot_plan_evaluation import _outputs, _read_object, resume_plan_evaluation
from scripts.med_copilot_run_lock import exclusive_run_lock


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _records(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [_read_object(line) for line in handle if line.strip()]


def without_plan_scores(value):
    """Remove only Plan correctness fields for an exact non-Plan preservation check."""
    if isinstance(value, dict):
        return {key: without_plan_scores(item) for key, item in value.items() if key != "plan_correctness"}
    if isinstance(value, list):
        return [without_plan_scores(item) for item in value]
    return value


def prepare_validation(source, output, *, root=ROOT):
    """Copy one complete patient workload per populated size/reference-count cell."""
    source, output, root = Path(source).resolve(), Path(output).resolve(), Path(root).resolve()
    if output == source or source in output.parents:
        raise ValueError("Validation output must be outside the source run")
    if output.exists():
        raise ValueError("Validation output must be a new directory")
    metadata = _read_object((source / "run.json").read_text(encoding="utf-8"))
    if metadata.get("plan_evaluation"):
        raise ValueError("Validation requires a generation-only source run")
    input_path = command._verified(root, metadata["input"], "Patient input")
    labels_path = command._verified(root, {"file": metadata["evaluation"]["labels_path"],
                                          "sha256": metadata["evaluation"]["labels_sha256"]}, "Labels")
    selected = set(metadata["evaluation"]["selected_patient_ids"])
    patients = [patient for patient in load_patient_records(input_path) if patient["id"] in selected]
    labels = {label["id"]: label for label in _records(labels_path)}
    _, population = select_runtime_pilot(patients, labels, count=16)
    records = _records(source / "results.jsonl")
    if len(records) != len(selected) or {record["id"] for record in records} != selected:
        raise ValueError("Source results do not match the recorded cohort")
    cells = defaultdict(list)
    for record in records:
        reference = labels[record["id"]]["plan"]
        if any(output_slot.get("status") == "completed" and any(
                plan_entry(1, reference, prediction.get("plan"))["status"] == "pending"
                for prediction in output_slot.get("predictions") or [])
               for _, _, output_slot in _outputs(record)):
            cell = f"{population['patient_strata'][record['id']]}:R{len(reference)}"
            cells[cell].append(record["id"])
    if not cells:
        raise ValueError("Source run has no judgeable Plan workloads")
    random = Random(20261112)
    chosen = {random.choice(sorted(ids)): cell for cell, ids in sorted(cells.items())}
    subset = [record for record in records if record["id"] in chosen]
    metadata = deepcopy(metadata)
    metadata["evaluation"]["selected_patient_ids"] = [record["id"] for record in subset]
    paths = (source / "run.json", source / "results.jsonl", input_path, labels_path)
    manifest = {"source_files": {str(path): _sha(path) for path in paths},
                "prompt_sha256": _sha(root / "med_copilot/prompts/correctness_judges.py"),
                "selected_cells": chosen, "cell_populations": {cell: len(ids) for cell, ids in cells.items()},
                "resident_patient_count": len(records)}
    output.mkdir(parents=True)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in subset), encoding="utf-8")
    (output / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records, subset, manifest


def _controls(models, judge, output):
    judges = CorrectnessJudges(models.judge, models.model_ids, output / "control_judge_responses.jsonl",
                               plan_judge=judge, redact=models.redact)
    cases = [("essential hypertension", "Follow-up hypertension. No medication allergies.", "BP 160/100.",
              "amlodipine 5 mg oral daily"),
             ("streptococcal sore throat", "Acute sore throat and fever.", "Tonsillar exudates.", "Obtain throat culture"),
             ("congestive heart failure", "Dyspnea and bilateral leg swelling.", "Bilateral pitting edema.",
              "furosemide 20 mg oral daily")]
    scores = []
    for condition, subjective, objective, action in cases:
        for kind, generated in (("correct", [action]), ("off_target", ["Schedule appendectomy"]),
                                ("placeholder", ["Further assessment needed"])):
            entry = plan_entry(1, [action], generated)
            judges.judge_plan(entry, condition, {"subjective": subjective, "objective": objective})
            scores.append({"condition": condition, "kind": kind, "evaluation": entry})
    (output / "control_results.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    if any(item["evaluation"]["status"] != "scored" or (
            item["evaluation"]["score"] < .75 if item["kind"] == "correct"
            else item["evaluation"]["score"] > .1) for item in scores):
        raise ValueError("Judge failed the small known-answer controls; inspect control_results.json")
    return {"plans": len(scores), "request_timing": judges.request_timing()}


def validate(source, output, *, root=ROOT, models_factory=AssessmentPlanModels):
    resident, original, manifest = prepare_validation(source, output, root=root)
    output = Path(output).resolve()
    summary = {"manifest": manifest, "stages": []}
    try:
        scorer, _, models = command._prepare(output, root, models_factory)
        scorer.patients = [patient for patient in scorer.patients if patient["id"] in manifest["selected_cells"]]
        metadata = _read_object((output / "run.json").read_text(encoding="utf-8"))
        profile = load_judge_profile(root / "experiments/configs/plan_judging.yaml", metadata)
        settings = {key: profile["settings"][key] for key in ("concurrency", "context_length")}
        judge = metadata["experiment"]["plan_judge"]
        with exclusive_run_lock(output), ollama_for_saved_run(output, **settings, runtime_profile=profile) as server:
            verify = lambda: save_verified_runtime(server, models, judge, output, root)
            verify()
            summary["controls"] = _controls(models, judge, output)
            actual_judge, counter, lock = models.judge, [0], Lock()
            def interrupt(model, messages):
                with lock:
                    counter[0] += 1
                    number = counter[0]
                if number == 9:
                    raise KeyboardInterrupt()
                return actual_judge(model, messages)
            models.judge = interrupt
            try:
                resume_plan_evaluation(output, scorer, models, root=root, **settings, runtime_profile=profile,
                                       runtime_verifier=verify, _lock_held=True)
                raise AssertionError("Validation did not reach its intentional interruption")
            except KeyboardInterrupt:
                summary["interrupted_stage"] = _read_object((output / "run.json").read_text())["plan_evaluation"]
                assert any(row.get("valid") for row in _records(output / "judge_responses.jsonl"))
            finally:
                models.judge = actual_judge
            completed_hashes = None
            for invocation in range(2):
                started = time.monotonic()
                judges = resume_plan_evaluation(output, scorer, models, root=root, **settings, runtime_profile=profile,
                                                runtime_verifier=verify, _lock_held=True)
                stage = _read_object((output / "run.json").read_text())["plan_evaluation"]
                summary["stages"].append({"wall_seconds": time.monotonic() - started, **stage})
                assert stage["status"] == "completed", "Plan evaluation contains failed judgments"
                if invocation:
                    assert judges.calls == 0, "Completed resume made new model requests"
                    assert completed_hashes == {name: _sha(output / name) for name in completed_hashes}, \
                        "Completed resume changed scores or durable responses"
                else:
                    completed_hashes = {name: _sha(output / name) for name in
                                        ("results.jsonl", "judge_responses.jsonl")}
            assert len(resident) == manifest["resident_patient_count"]  # Retain full-corpus residency through inference.
            assert without_plan_scores(_records(output / "results.jsonl")) == without_plan_scores(original)
            final_metadata = _read_object((output / "run.json").read_text())
            assert all(final_metadata[key] == metadata[key] for key in
                       ("experiment", "evaluation", "input", "assessment_plan_models"))
            summary["non_plan_preserved"] = True
    finally:
        summary["source_unchanged"] = all(_sha(path) == fingerprint for path, fingerprint in manifest["source_files"].items())
        summary["prompt_unchanged"] = _sha(Path(root) / "med_copilot/prompts/correctness_judges.py") == manifest["prompt_sha256"]
        (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    assert summary["source_unchanged"] and summary["prompt_unchanged"], "Validation sources changed during verification"
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_directory, args.output_directory), indent=2))
