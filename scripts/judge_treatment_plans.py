"""Finish deferred Plan judging without repeating generation or completed metrics.

Usage: python scripts/judge_treatment_plans.py RUN_DIRECTORY

Reads the recorded cohort and loads only its recorded Plan judge. Completed
Plan judgements and scalar replies are reused. Text metrics, diagnosis scores,
generation, and evidence are preserved. Concurrent requests are bounded by the
run's recorded experiment.plan_judge_concurrency (default 2 for older runs).

Exit status: 0 when every Plan entry finished; 1 when the run directory or
its inputs cannot be used, or when some Plan entries still failed (rerun
to retry them); 130 when interrupted (completed judge calls are kept for the next run).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

# Standalone execution must resolve the project and its vendored dependencies
# before importing the model and scoring adapters below.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
load_dotenv(ROOT / ".env", override=False)

from scripts.collate_med_copilot_results import generate_report
from scripts.med_copilot_config import PIPELINE_DEFAULTS
from scripts.med_copilot_helpers import load_patient_records
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.med_copilot_plan_evaluation import PlanJudgeUnavailableError, resume_plan_evaluation
from scripts.med_copilot_run_file import RunMetadataError
from scripts.med_copilot_run_lock import exclusive_run_lock
from scripts.med_copilot_scoring import PatientScorer


def _verified(root, recorded, label):
    """Resolve a recorded file and refuse it if its contents changed since the run."""
    path = root / recorded["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != recorded["sha256"]:
        raise ValueError(f"{label} changed since the run: {path}")
    return path


def _failed_entries(run_dir):
    """Count unfinished Plan entries, which a later invocation will retry."""
    stage = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("plan_evaluation", {})
    if "counts" in stage:
        return sum(stage["counts"].get(status, 0) for status in ("error", "interrupted", "pending"))
    failed = 0
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            for variant in json.loads(line).get("variants", {}).values():
                for slot in variant.get("assessment_plans", {}).values():
                    for entry in slot.get("evaluation") or []:
                        failed += entry.get("plan_correctness", {}).get("status") in (
                            "error", "interrupted", "pending")
    return failed


def _prepare(run_dir, root, models_factory):
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    evaluation, experiment = metadata["evaluation"], metadata["experiment"]
    models_metadata = metadata["assessment_plan_models"]
    patient_input = _verified(root, metadata["input"], "Patient input")
    labels = _verified(root, {"file": evaluation["labels_path"], "sha256": evaluation["labels_sha256"]}, "Labels")
    scorer = PatientScorer(
        labels, load_patient_records(patient_input), evaluation["presenting_evidence_only"],
        max_patients=evaluation["max_patients"],
        exclude_none_recorded_plans=evaluation["exclude_none_recorded_plans"],
        metric_evaluator=object(),  # Cohort selection only; this command never scores text.
    )
    judge = experiment.get("plan_judge", PIPELINE_DEFAULTS["plan_judge"])
    # Generator clients/credentials are unnecessary: their predictions are saved.
    judge_models = models_factory(
        root / models_metadata["config_file"], [judge],
        temperature=experiment["temperature"], max_retries=experiment["max_retries"],
    )
    return scorer, judge_models, judge_models


def main(argv=None, *, root=ROOT, models_factory=AssessmentPlanModels):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_directory", type=Path, help="Directory containing run.json and results.jsonl")
    run_dir = parser.parse_args(argv).run_directory.resolve()
    try:
        # Lock before reading run.json, so model/cohort reconstruction cannot race
        # a resumed generator and later overwrite its newer results.
        with exclusive_run_lock(run_dir):
            scorer, _, judge_models = _prepare(run_dir, root, models_factory)
            metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            concurrency = metadata["experiment"].get("plan_judge_concurrency", 2)
            judges = resume_plan_evaluation(run_dir, scorer, judge_models, root=root,
                                            concurrency=concurrency, _lock_held=True)
            print(f"Judge calls made: {judges.calls}; reused from {run_dir.name}/judge_responses.jsonl: {judges.cache_hits}")
            print(f"HTML results: {generate_report(run_dir, root=root)}")
            failed = _failed_entries(run_dir)
    except KeyboardInterrupt:
        print("Evaluation interrupted. Completed judge calls are saved; rerun this command to resume.",
              file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RunMetadataError, PlanJudgeUnavailableError) as exc:
        detail = f"missing run.json field {exc}" if isinstance(exc, KeyError) else exc
        print(f"Cannot resume evaluation of {run_dir}: {detail}", file=sys.stderr)
        return 1
    if failed:
        print(f"{failed} Plan correctness entries failed (see run.json errors); rerun this command to retry them.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
