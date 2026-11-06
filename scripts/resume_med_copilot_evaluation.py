"""Finish or repeat evaluation of a saved run without repeating completed judge calls.

Usage: python scripts/resume_med_copilot_evaluation.py RUN_DIRECTORY

Rebuilds the run's scorer and models from run.json, re-scores every saved
prediction (text metrics and judge panels), reuses valid judgements from
judge_responses.jsonl, then rewrites results.jsonl and results_summary.html.
Generation is never repeated.

Exit status: 0 when every judge panel finished; 1 when the run directory or its
inputs cannot be used, or when some judge panels still failed (rerun to retry
them); 130 when interrupted (completed judge calls are kept for the next run).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
load_dotenv(ROOT / ".env", override=False)

from scripts.collate_med_copilot_results import generate_report
from scripts.med_copilot_helpers import load_patient_records
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.med_copilot_scoring import PatientScorer, resume_evaluation


def _verified(root, recorded, label):
    """Resolve a recorded file and refuse it if its contents changed since the run."""
    path = root / recorded["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != recorded["sha256"]:
        raise ValueError(f"{label} changed since the run: {path}")
    return path


def _failed_panels(run_dir):
    """Count judge entries left in error, which a later resume will retry."""
    failed = 0
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            for variant in json.loads(line).get("variants", {}).values():
                for slot in variant.get("assessment_plans", {}).values():
                    for entry in slot.get("evaluation") or []:
                        failed += sum(entry.get(kind, {}).get("status") == "error"
                                      for kind in ("diagnosis_correctness", "plan_correctness"))
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
    )
    models = models_factory(
        root / models_metadata["config_file"], models_metadata["selected_models"],
        temperature=experiment["temperature"], max_retries=experiment["max_retries"],
    )
    return scorer, models


def main(argv=None, *, root=ROOT, models_factory=AssessmentPlanModels):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_directory", type=Path, help="Directory containing run.json and results.jsonl")
    run_dir = parser.parse_args(argv).run_directory.resolve()
    try:
        scorer, models = _prepare(run_dir, root, models_factory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        detail = f"missing run.json field {exc}" if isinstance(exc, KeyError) else exc
        print(f"Cannot resume evaluation of {run_dir}: {detail}", file=sys.stderr)
        return 1
    try:
        judges = resume_evaluation(run_dir, scorer, models, root=root)
    except KeyboardInterrupt:
        print("Evaluation interrupted. Completed judge calls are saved; rerun this command to resume.",
              file=sys.stderr)
        return 130
    print(f"Judge calls made: {judges.calls}; reused from {run_dir.name}/judge_responses.jsonl: {judges.cache_hits}")
    print(f"HTML results: {generate_report(run_dir, root=root)}")
    failed = _failed_panels(run_dir)
    if failed:
        print(f"{failed} judge panels failed (see run.json errors); rerun this command to retry them.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
