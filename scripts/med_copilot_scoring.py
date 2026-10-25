"""Score generated fields with the unchanged upstream MED-COPILOT evaluator."""

import argparse
import hashlib
from importlib import metadata as package_metadata
import json
import math
from numbers import Real
from pathlib import Path

from med_copilot.upstream.evaluate_metrics import Evaluator


METRIC_NAMES = ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1")
METRIC_PACKAGES = ("sacrebleu", "nltk", "rouge-score", "bert-score", "transformers", "torch")
UPSTREAM_EVALUATOR = Path(__file__).resolve().parents[1] / "med_copilot/upstream/evaluate_metrics.py"


def _boolean_argument(value):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_scoring_args(argv=None):
    """Parse the cohort filter, enabled by default and when the flag is bare."""
    parser = argparse.ArgumentParser(description="Run MED-COPILOT on prepared patient records.")
    parser.add_argument(
        "--presenting-evidence-only", nargs="?", const=True, default=True,
        type=_boolean_argument, metavar="true|false",
        help="Select only patients with presenting evidence (default: true).",
    )
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


class PatientScorer:
    """Validate gold labels, select a cohort, and score each generated A/P pair.

    Label files may contain additional patient IDs so a shared label file can
    serve several input subsets. Every label row is validated, and every input
    ID must have exactly one label before the presenting-evidence filter runs.
    """

    def __init__(self, labels_path, patients, presenting_evidence_only=True):
        if type(presenting_evidence_only) is not bool:
            raise TypeError("presenting_evidence_only must be a bool")
        self.labels_path = Path(labels_path).resolve()
        self.presenting_evidence_only = presenting_evidence_only
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
        self.patients = [
            patient for patient in original_patients
            if not presenting_evidence_only or labels[patient["id"]]["has_presenting_evidence"]
        ]
        if not self.patients:
            raise ValueError("No patients selected for evaluation")
        self._selected_ids = {patient["id"] for patient in self.patients}
        self._input_ids = input_ids
        self._labels = labels

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
            "input_patient_count": len(self._input_ids),
            "selected_patient_count": len(self.patients),
            "selected_patient_ids": [patient["id"] for patient in self.patients],
            "excluded_patient_ids": [
                patient_id for patient_id in self._input_ids if patient_id not in self._selected_ids
            ],
            "label_count": len(self._labels),
            "metric_packages": versions,
            "settings": {
                "aggregation": "per_patient_per_field_singleton",
                "lang": "en",
                "condition_reference": "replace underscores with spaces",
                "plan_reference": "join original items with comma and space",
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

    def score_result(self, result):
        """Attach independent field scores without changing generation status."""
        if (not isinstance(result, dict) or not isinstance(result.get("id"), str)
                or result["id"] not in self._selected_ids):
            raise ValueError("Result must name a selected input patient")
        label = self._labels[result["id"]]
        result["evaluation"] = {
            field: _empty_entry("pending") for field in ("condition", "plan")
        }
        references = {
            "condition": label["condition"].replace("_", " "),
            "plan": ", ".join(label["plan"]),
        }
        for field, candidate_field in (("condition", "assessment"), ("plan", "plan")):
            if field == "plan" and label["plan"] == ["none recorded"]:
                result["evaluation"][field] = _empty_entry("skipped_no_recorded_plan")
                continue
            reference = references[field]
            candidate = result.get(candidate_field)
            if not reference.strip():
                result["evaluation"][field] = _empty_entry("missing_reference")
                continue
            if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
                result["evaluation"][field] = _empty_entry("missing_generation")
                continue
            try:
                if not isinstance(candidate, str):
                    raise TypeError(f"Generated {candidate_field} must be a string or null")
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
                result["evaluation"][field] = {
                    "status": "scored", "metrics": values, "error": None,
                }
            except Exception as exc:
                result["evaluation"][field] = _empty_entry("error", _exception_data(exc))
            except BaseException as exc:
                result["evaluation"][field] = _empty_entry("interrupted", _exception_data(exc))
                raise
        return result
