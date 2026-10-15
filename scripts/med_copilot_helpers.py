"""Helpers for the project-owned MED-COPILOT runner."""

import errno
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


def validate_guideline_tables(guideline_output):
    """Require local-search tables directly in the selected output directory."""
    guideline_output = Path(guideline_output)
    for name in ("communities", "community_reports", "text_units", "relationships", "entities"):
        table = guideline_output / f"{name}.parquet"
        if not table.is_file():
            raise FileNotFoundError(errno.ENOENT, "Required guideline table not found", str(table))


class JSONKeywordExtractor:
    """Keep the upstream extraction prompt, with enforced JSON and stable ordering."""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def extract(self, text):
        prompt = f"""
You are a clinical NLP expert.
From the following clinical note, extract medically meaningful entities.

IMPORTANT:
- Each item MUST be a verbatim span copied from the text.
- Do NOT paraphrase or infer.
Return ONLY valid JSON:
{{
  "symptoms": [],
  "diagnoses": [],
  "medications": []
}}

Text:
{text}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        if not getattr(response, "choices", None):
            raise ValueError("Keyword extraction returned no choices")
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise ValueError("Keyword extraction was refused")
        if choice.finish_reason != "stop":
            raise ValueError(
                f"Keyword extraction did not complete (finish_reason={choice.finish_reason})"
            )
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Keyword extraction returned empty or non-text content")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Keyword extraction returned invalid JSON: {exc.msg}") from exc
        if not isinstance(result, dict):
            raise ValueError("Keyword extraction must return a JSON object")
        keywords = []
        for category in ("symptoms", "diagnoses", "medications"):
            terms = result.get(category)
            if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
                raise ValueError(f"Keyword extraction {category} must be a list of strings")
            keywords.extend(term.strip() for term in terms)
        return list(dict.fromkeys(keywords))
