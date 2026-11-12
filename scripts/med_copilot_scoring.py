"""Score generated fields with the unchanged upstream MED-COPILOT evaluator."""

import argparse
import hashlib
from importlib import metadata as package_metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
from random import Random

from med_copilot.upstream.evaluate_metrics import Evaluator
from scripts.med_copilot_config import PIPELINE_DEFAULTS
from scripts.med_copilot_judges import JUDGE_FILE, CorrectnessJudges, diagnosis_entry, plan_entry
from scripts.med_copilot_run_file import (
    RUN_FILE, RunFile, RunMetadataError, checkpoint, error_scope, pipeline_stage, record_error,
)


METRIC_NAMES = ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1")
METRIC_PACKAGES = ("sacrebleu", "nltk", "rouge-score", "bert-score", "transformers", "torch")
UPSTREAM_EVALUATOR = Path(__file__).resolve().parents[1] / "med_copilot/upstream/evaluate_metrics.py"
PATIENT_SAMPLING_SEED = 42


def _boolean_argument(value):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_scoring_args(argv=None):
    """Parse the cohort option; experiment/model settings come only from config files."""
    parser = argparse.ArgumentParser(description="Run MED-COPILOT on prepared patient records.")
    parser.add_argument(
        "--presenting-evidence-only", nargs="?", const=True, default=None,
        type=_boolean_argument, metavar="true|false",
        help="Override the experiment cohort filter (otherwise use experiment.yaml).",
    )
    parser.add_argument("--resume", type=Path, default=None, metavar="RUN_DIRECTORY",
                        help="Continue an existing run without repeating completed work.")
    return parser.parse_args(argv)


def _relative_path(path, root):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _empty_entry(status, error=None):
    return {"status": status, "metrics": dict.fromkeys(METRIC_NAMES), "error": error}


def _exception_data(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _scored_text_entry(entry):
    """Reuse only complete, finite metric payloads from a saved successful field."""
    if not isinstance(entry, dict) or entry.get("status") != "scored":
        return False
    metrics = entry.get("metrics")
    return (isinstance(metrics, dict) and set(metrics) == set(METRIC_NAMES)
            and all(not isinstance(value, bool) and isinstance(value, Real)
                    and math.isfinite(value) for value in metrics.values()))


def _scored_correctness_entry(entry, expected):
    """Keep successful correctness only when it belongs to the same prediction."""
    if (not isinstance(entry, dict) or entry.get("status") != "scored"
            or expected["status"] == "unavailable"):
        return False
    identity = ("reference", "generated", "normalized_reference", "normalized_generated") if (
        "normalized_reference" in expected) else ("reference_actions", "generated_actions")
    return (all(entry.get(key) == expected.get(key) for key in (*identity, "method", "rank_weight"))
            and all(not isinstance(entry.get(key), bool) and isinstance(entry.get(key), Real)
                    and math.isfinite(entry[key]) for key in ("score", "weighted_score"))
            and 0 <= entry["score"] <= 1
            and math.isclose(entry["weighted_score"], entry["score"] * expected["rank_weight"]))


class PatientScorer:
    """Validate gold labels, select a cohort, and score three ranked condition/Plan pairs.

    Label files may contain additional patient IDs so a shared label file can
    serve several input subsets. Every label row is validated, and every input
    ID must have exactly one label before the presenting-evidence filter runs.
    Sample from the filtered cohort when max_patients limits it, keeping input order.
    """

    def __init__(self, labels_path, patients, presenting_evidence_only=True, *, max_patients=None,
                 exclude_none_recorded_plans=False):
        if type(presenting_evidence_only) is not bool:
            raise TypeError("presenting_evidence_only must be a bool")
        if max_patients is not None and (type(max_patients) is not int or max_patients < 1):
            raise ValueError("max_patients must be null or a positive integer")
        if type(exclude_none_recorded_plans) is not bool:
            raise TypeError("exclude_none_recorded_plans must be a bool")
        self.exclude_none_recorded_plans = exclude_none_recorded_plans
        self.labels_path = Path(labels_path).resolve()
        self.presenting_evidence_only = presenting_evidence_only
        self.max_patients = max_patients
        source_bytes = self.labels_path.read_bytes()
        self._labels_sha256 = hashlib.sha256(source_bytes).hexdigest()
        labels = {}
        # JSONL uses physical newlines; Unicode separators may occur inside text.
        for line_number, line in enumerate(source_bytes.decode("utf-8").split("\n"), 1):
            if not line.strip():
                continue
            location = f"{self.labels_path}:{line_number}"
            try:
                label = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid label JSON: {exc.msg}") from exc
            if not isinstance(label, dict):
                raise ValueError(f"{location}: label must be a JSON object")
            patient_id = label.get("id")
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError(f"{location}: id must be a nonempty string")
            if patient_id in labels:
                raise ValueError(f"{location}: duplicate label id {patient_id!r}")
            condition = label.get("condition")
            if not isinstance(condition, str) or not condition.replace("_", " ").strip():
                raise ValueError(f"{location}: condition must normalize to a nonempty string")
            plan = label.get("plan")
            if not isinstance(plan, list) or any(not isinstance(item, str) for item in plan):
                raise ValueError(f"{location}: plan must be a list of strings")
            if type(label.get("has_presenting_evidence")) is not bool:
                raise ValueError(f"{location}: has_presenting_evidence must be a JSON boolean")
            labels[patient_id] = label

        original_patients = list(patients)
        input_ids = []
        seen = set()
        for patient in original_patients:
            if not isinstance(patient, dict):
                raise ValueError("Each input patient must be a JSON object")
            patient_id = patient.get("id")
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError("Each input patient id must be a nonempty string")
            if patient_id in seen:
                raise ValueError(f"Duplicate input patient id {patient_id!r}")
            seen.add(patient_id)
            input_ids.append(patient_id)
        missing = [patient_id for patient_id in input_ids if patient_id not in labels]
        if missing:
            raise ValueError(f"Missing labels for input patient IDs: {', '.join(missing)}")
        self.exclusion_reasons = {}
        evidence_patients = []
        for patient in original_patients:
            patient_id = patient["id"]
            if presenting_evidence_only and not labels[patient_id]["has_presenting_evidence"]:
                self.exclusion_reasons[patient_id] = ["missing_presenting_evidence"]
            else:
                evidence_patients.append(patient)
        self.after_presenting_evidence_count = len(evidence_patients)
        eligible = []
        for patient in evidence_patients:
            patient_id = patient["id"]
            if exclude_none_recorded_plans and labels[patient_id]["plan"] == ["none recorded"]:
                self.exclusion_reasons[patient_id] = ["no_recorded_plan"]
            else:
                eligible.append(patient)
        self.eligible_patient_count = len(eligible)
        self.patients = eligible
        if max_patients is not None and max_patients < len(eligible):
            indices = Random(PATIENT_SAMPLING_SEED).sample(range(len(eligible)), max_patients)
            self.patients = [eligible[index] for index in sorted(indices)]
        if not self.patients:
            raise ValueError("No patients selected for evaluation")
        self._selected_ids = {patient["id"] for patient in self.patients}
        for patient in eligible:
            patient_id = patient["id"]
            if patient_id not in self._selected_ids:
                self.exclusion_reasons[patient_id] = ["max_patients"]
        self._input_ids = input_ids
        self._labels = labels
        self._patients = {patient["id"]: patient for patient in self.patients}
        self.judges = None

    def attach_judges(self, judges):
        """Also score diagnosis/Plan correctness (preliminary single-model midterm metric)."""
        self.judges = judges

    def metadata(self, root):
        """Describe the label snapshot, cohort, and configured upstream defaults."""
        versions = {}
        for name in METRIC_PACKAGES:
            try:
                versions[name] = package_metadata.version(name)
            except package_metadata.PackageNotFoundError:
                versions[name] = None
        return {
            "labels_path": _relative_path(self.labels_path, root),
            "labels_sha256": self._labels_sha256,
            "evaluator_path": _relative_path(UPSTREAM_EVALUATOR, root),
            "evaluator_sha256": hashlib.sha256(UPSTREAM_EVALUATOR.read_bytes()).hexdigest(),
            "presenting_evidence_only": self.presenting_evidence_only,
            "max_patients": self.max_patients,
            "patient_sampling_seed": PATIENT_SAMPLING_SEED,
            "exclude_none_recorded_plans": self.exclude_none_recorded_plans,
            "excluded_plan_value": ["none recorded"],
            "filter_order": ["presenting_evidence_only", "exclude_none_recorded_plans", "max_patients"],
            "after_presenting_evidence_count": self.after_presenting_evidence_count,
            "eligible_patient_count": self.eligible_patient_count,
            "exclusion_reasons": {key: list(value) for key, value in self.exclusion_reasons.items()},
            "input_patient_count": len(self._input_ids),
            "selected_patient_count": len(self.patients),
            "selected_patient_ids": [patient["id"] for patient in self.patients],
            "excluded_patient_ids": [
                patient_id for patient_id in self._input_ids if patient_id not in self._selected_ids
            ],
            "label_count": len(self._labels),
            "metric_packages": versions,
            **({"correctness": self.judges.metadata(root)} if self.judges is not None else {}),
            "settings": {
                "aggregation": "per_patient_per_rank_per_field_singleton",
                "prediction_count": 3,
                "plan_item_limit": 3,
                "rank_weighting": None,
                "lang": "en",
                "condition_reference": "replace underscores with spaces",
                "plan_reference": "join original items with comma and space",
                "generated_plan": "join original generated items with comma and space",
                "defaults_provenance": "configured defaults for the recorded library versions",
                "bleu": {
                    "max_ngram_order": 4, "lowercase": False, "tokenize": "13a",
                    "smooth_method": "exp", "use_effective_order": False,
                    "scale": "upstream score divided by 100",
                },
                "bert_score": {
                    "model_type": "roberta-large", "num_layers": 17,
                    "idf": False, "rescale_with_baseline": False,
                    "use_fast_tokenizer": False, "batch_size": 64,
                    "expected_tokenizer_max_length": 512,
                    "model_revision": None,
                    "model_revision_status": "not verified; upstream selects the model by name",
                },
            },
        }

    def score_result(self, result, *, rescore=False):
        """Complete unfinished fields; rescore=True deliberately recalculates saved scores."""
        if (not isinstance(result, dict) or not isinstance(result.get("id"), str)
                or result["id"] not in self._selected_ids):
            raise ValueError("Result must name a selected input patient")
        if "variants" in result:
            for variant in result["variants"].values():
                self.score_result({"id": result["id"], "assessment_plans": variant["assessment_plans"]},
                                  rescore=rescore)
        elif "assessment_plans" in result:
            for output in result["assessment_plans"].values():
                if output["status"] == "completed":
                    self._score_predictions(result["id"], output, rescore=rescore)
        else:
            self._score_predictions(result["id"], result, rescore=rescore)
        return result

    def _score_predictions(self, patient_id, result, *, rescore=False):
        label = self._labels[patient_id]
        saved = result.get("evaluation") if not rescore else None
        if (not isinstance(saved, list) or len(saved) != 3
                or any(not isinstance(entry, dict) or entry.get("rank") != index + 1
                       for index, entry in enumerate(saved))):
            saved = [{} for _ in range(3)]
        result["evaluation"] = [
            {"rank": rank, **{field: saved[rank - 1][field] if _scored_text_entry(
                saved[rank - 1].get(field)) else _empty_entry("pending") for field in ("condition", "plan")}}
            for rank in range(1, 4)
        ]
        if self.judges is not None:
            # Text metrics fill every rank first; correctness scoring then runs per rank.
            judged = [self._correctness_entries(label, result.get("predictions"), index)
                      for index in range(3)]
            for evaluation, previous, (diagnosis, plan) in zip(result["evaluation"], saved, judged):
                for key, expected in (("diagnosis_correctness", diagnosis), ("plan_correctness", plan)):
                    evaluation[key] = previous[key] if _scored_correctness_entry(
                        previous.get(key), expected) else expected
        references = {
            "condition": label["condition"].replace("_", " "),
            "plan": ", ".join(label["plan"]),
        }
        predictions = result.get("predictions")
        for index, evaluation in enumerate(result["evaluation"]):
            for field in ("condition", "plan"):
                if evaluation[field]["status"] == "scored":
                    continue
                if field == "plan" and label["plan"] == ["none recorded"]:
                    evaluation[field] = _empty_entry("skipped_no_recorded_plan")
                    checkpoint()
                    continue
                reference = references[field]
                if not reference.strip():
                    evaluation[field] = _empty_entry("missing_reference")
                    checkpoint()
                    continue
                try:
                    if predictions is None:
                        candidate = None
                    else:
                        if not isinstance(predictions, list) or len(predictions) != 3:
                            raise TypeError("Generated predictions must be a list of exactly three pairs or null")
                        prediction = predictions[index]
                        if not isinstance(prediction, dict) or set(prediction) - {"condition", "plan"}:
                            raise TypeError("Generated prediction must be a condition/plan object")
                        candidate = prediction.get(field)
                    if candidate is None:
                        evaluation[field] = _empty_entry("missing_generation")
                        continue
                    if field == "plan":
                        if (not isinstance(candidate, list) or len(candidate) > 3
                                or any(not isinstance(item, str) for item in candidate)):
                            raise TypeError("Generated plan must be a list of at most three strings or null")
                        candidate = ", ".join(candidate)
                    elif not isinstance(candidate, str):
                        raise TypeError("Generated condition must be a string or null")
                    if not candidate.strip():
                        evaluation[field] = _empty_entry("missing_generation")
                        continue
                    metrics = Evaluator.evaluate_metrics([reference], [candidate], lang="en")
                    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
                        raise ValueError("Evaluator must return exactly the six upstream metric keys")
                    values = {}
                    for name in METRIC_NAMES:
                        value = metrics[name]
                        if isinstance(value, bool) or not isinstance(value, Real):
                            raise ValueError(f"Evaluator metric {name} must be a finite real number")
                        value = float(value)
                        if not math.isfinite(value):
                            raise ValueError(f"Evaluator metric {name} must be a finite real number")
                        values[name] = value
                    evaluation[field] = {"status": "scored", "metrics": values, "error": None}
                except Exception as exc:
                    evaluation[field] = _empty_entry("error", _exception_data(exc))
                except BaseException as exc:
                    evaluation[field] = _empty_entry("interrupted", _exception_data(exc))
                    raise
                finally:
                    checkpoint()
        if self.judges is not None:
            patient = self._patients[patient_id]
            for evaluation in result["evaluation"]:
                if evaluation["diagnosis_correctness"]["status"] == "pending":
                    self.judges.score_diagnosis(evaluation["diagnosis_correctness"])
                    checkpoint()
                if evaluation["plan_correctness"]["status"] == "pending":
                    self.judges.judge_plan(evaluation["plan_correctness"],
                                           label["condition"].replace("_", " "), patient)
                    checkpoint()
        return result

    @staticmethod
    def _correctness_entries(label, predictions, index):
        """Pending judge entries for one rank; malformed generations are unavailable."""
        rank = index + 1
        condition = plan = problem = None
        if predictions is not None:
            prediction = predictions[index] if isinstance(predictions, list) and len(predictions) == 3 else None
            if not isinstance(prediction, dict) or set(prediction) - {"condition", "plan"}:
                problem = "invalid_generation"
            else:
                condition, plan = prediction.get("condition"), prediction.get("plan")
        condition_problem = problem or (None if condition is None or isinstance(condition, str)
                                        else "invalid_generation")
        plan_problem = problem or (None if plan is None or (
            isinstance(plan, list) and len(plan) <= 3 and all(isinstance(item, str) for item in plan))
            else "invalid_generation")
        return (diagnosis_entry(rank, label["condition"].replace("_", " "),
                                condition if condition_problem is None else None, condition_problem),
                plan_entry(rank, label["plan"], plan if plan_problem is None else None, plan_problem))


def _write_results(run_dir, records):
    """Atomically replace results.jsonl so an interruption keeps the previous file."""
    temporary = run_dir / ".results.jsonl.tmp"
    try:
        temporary.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                                     for record in records), encoding="utf-8")
        os.replace(temporary, run_dir / "results.jsonl")
    finally:
        temporary.unlink(missing_ok=True)


def resume_evaluation(run_dir, scorer, models, *, root, judge_models=None):
    """Re-score a saved run, reusing every completed judge call in judge_responses.jsonl.

    `scorer` must select the run's recorded cohort. `models` (the run's generating
    models) provides model_ids and redact(); `judge_models` (default `models`)
    provides judge() and model_ids for the Plan judge. results.jsonl is replaced after each patient, so an
    interruption keeps finished patients; their judge calls are never repeated.
    Ordinary scoring failures are saved in run.json errors under the scoring stage.
    """
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    recorded = (metadata.get("evaluation") or {}).get("selected_patient_ids")
    if [patient["id"] for patient in scorer.patients] != recorded:
        raise ValueError("The scorer's cohort does not match the run's selected_patient_ids")
    run_file = RunFile(run_dir, metadata)
    run_file.redact = models.redact
    judge_models = judge_models or models
    judges = CorrectnessJudges(judge_models.judge, judge_models.model_ids, run_dir / JUDGE_FILE,
                               plan_judge=(metadata.get("experiment") or {}).get(
                                   "plan_judge", PIPELINE_DEFAULTS["plan_judge"]),
                               generator_models=models.model_ids,
                               judge_model_config=(judge_models.metadata(root)
                                                   if hasattr(judge_models, "metadata") else None))
    scorer.attach_judges(judges)
    text = (run_dir / "results.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.split("\n") if line.strip()]
    for record in records:
        with error_scope(run_file, patient_id=record["id"]), pipeline_stage("scoring"):
            try:
                scorer.score_result(record, rescore=True)
            except RunMetadataError:
                raise
            except Exception as exc:
                record_error(exc)
        _write_results(run_dir, records)
    run_file.update(evaluation=scorer.metadata(root))
    return judges
