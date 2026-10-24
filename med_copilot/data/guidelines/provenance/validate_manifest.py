"""Validate the guideline provenance manifest against local indexed documents."""

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
GUIDELINES = ROOT / "med_copilot/data/guidelines"


def flat_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def validate():
    manifest = json.loads((GUIDELINES / "guideline_manifest.json").read_text())
    table_path = GUIDELINES / "output/documents.parquet"
    table = pd.read_parquet(table_path)
    rows = manifest["documents"]
    assert len(rows) == len(table) == 226, "Document count differs from 226."
    assert manifest["document_table_sha256"] == hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert len({r["document_id"] for r in rows}) == 226, "Duplicate manifest document ID."
    assert {r["document_id"] for r in rows} == set(table.id), "Document ID coverage differs."
    assert {r["document_numeric_id"] for r in rows} == set(range(226))
    evidence = json.loads((GUIDELINES / "provenance/verification_evidence.json").read_text())["documents"]
    assert len(evidence) == 226
    assert {r["document_id"] for r in evidence} == set(table.id)
    evidence_by_id = {r["document_id"]: r for r in evidence}
    source_by_id = table.set_index("id")
    for row in rows:
        source = source_by_id.loc[row["document_id"]]
        proof = evidence_by_id[row["document_id"]]
        assert row["document_numeric_id"] == source.human_readable_id
        assert row["local_filename"] == source.title
        assert row["text_unit_count"] == len(source.text_unit_ids)
        path = ROOT / row["local_file"]
        text = path.read_text(encoding="utf-8")
        assert text == source.text, f"Local source text changed: {path}"
        assert hashlib.sha512(text.encode()).hexdigest() == row["document_id"]
        assert hashlib.sha256(text.encode()).hexdigest() == row["local_text_sha256"]
        for field in ["identity", "title", "organization", "publication_date"]:
            assert row[field + "_verification_status"] == "verified", (row["local_filename"], field)
        assert row["original_title"] and row["issuing_organizations"]
        assert row["source_url"] and urlparse(row["source_url"]).scheme == "https"
        assert row["metadata_verification_urls"]
        assert row["publication_date_precision"] in {"day", "month"}
        date_format = "%Y-%m-%d" if row["publication_date_precision"] == "day" else "%Y-%m"
        datetime.strptime(row["publication_date"], date_format)
        for field in ["official_last_updated_date", "online_publication_date"]:
            if row[field]:
                datetime.strptime(row[field], "%Y-%m-%d")
        assert row["exact_original_pdf_snapshot_verified"] is False
        assert {"exact_original_pdf_snapshot", "original_acquisition_date", "original_acquisition_url"} <= set(row["unverified_fields"])
        for isbn in row["local_isbns"] + row["official_pdf_isbns"]:
            assert re.fullmatch(r"\d{13}", isbn)
            assert sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(isbn)) % 10 == 0
        if row["source_type"] == "NICE_guideline":
            assert row["guideline_identifier"] == row["local_filename"].split("_")[0]
            assert proof["overview"]["http_status"] == 200
            assert "exact_local_revision_date" in row["unverified_fields"]
            pub = [t["date"] for t in proof["overview"]["times"] if t["parent_text"].startswith("Published:")]
            assert pub == [row["publication_date"]]
            local_changes = proof["local_update_history_dates"]
            expected_month = max((entry["month"] for entry in local_changes), default=None)
            assert row["local_latest_documented_change_month"] == expected_month
            for entry in local_changes:
                offset = entry["document_character_offset"]
                assert text[offset:].lstrip().startswith(entry["date_label_raw"])
            if "isbn_matches_current_pdf" in row["version_verification_status"]:
                assert set(row["local_isbns"]) == set(row["official_pdf_isbns"])
                assert proof["official_pdf"]["http_status"] == 200
            elif "historical_isbn_differs" in row["version_verification_status"]:
                assert set(row["local_isbns"]) != set(row["official_pdf_isbns"])
                assert row["official_pdf_isbns"]
            else:
                assert row["guideline_identifier"] == "NG51" and row["source_status"] == "superseded"
        else:
            assert row["source_type"] == "journal_guideline"
            assert proof["pubmed"]["http_status"] == proof["crossref"]["http_status"] == 200
            assert proof["crossref"]["DOI"].lower() == row["doi"].lower()
            assert proof["pubmed"]["article_ids"]["doi"].lower() == row["doi"].lower()
            assert proof["pubmed"]["journal_publication_date"] == row["publication_date"]
    with (GUIDELINES / "guideline_manifest.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        csv_rows = list(reader)
        fields = reader.fieldnames
    assert len(csv_rows) == 226 and len({r["document_id"] for r in csv_rows}) == 226
    for expected, actual in zip(rows, csv_rows):
        assert all(actual[field] == flat_value(expected.get(field)) for field in fields), expected["local_filename"]
    counts = Counter(row["source_type"] for row in rows)
    assert counts == {"NICE_guideline": 221, "journal_guideline": 5}
    print("PASS: 226/226 document identities, source mappings, bibliographic fields, evidence records, uncertainty flags, and CSV/JSON rows validated offline.")


if __name__ == "__main__":
    validate()
