"""Input helpers for the project-owned MED-COPILOT runner."""

import json
from pathlib import Path


def load_patient_records(patient_input):
    """Load and validate all prepared S/O JSONL records, preserving their contents."""
    patient_input = Path(patient_input)
    patients = []
    with patient_input.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{patient_input}:{line_number}: invalid patient JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{patient_input}:{line_number}: patient must be a JSON object")
            for field in ("subjective", "objective"):
                if not isinstance(record.get(field), str):
                    raise ValueError(
                        f"{patient_input}:{line_number}: {field} must be a string"
                    )
            patients.append(record)
    if not patients:
        raise ValueError(f"{patient_input}: no patient records found")
    return patients
