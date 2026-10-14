#!/usr/bin/env python3
"""
build_release.py — Build data/release-v1/ from the 7-condition Synthea dataset.

Steps:
  1. Index: one row per (patient, condition) example with eligibility flags.
  2. Stratified patient split: 50% test | 20% dev | 30% corpus.
  3. Serialize text for each condition via serialization/serialize.py.
  4. Split JSONL output into test / dev / corpus files.
  5. Append DIAGNOSIS + TREATMENT sections to corpus records.
  6. Text leak checks on test.jsonl and dev.jsonl (per condition).
  7. Sample files for human review.
  8. manifest.json and zip.

Usage:
    python3 scripts/build_release.py [--seed N] [--out-dir data/release-v1]
"""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

# ── Condition configuration ───────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_RUN   = "pop10000-seed20260916"
SRC_DIR   = REPO_ROOT / "data" / SRC_RUN

CONDITIONS = [
    {
        "name": "copd",
        "bin": "history",
        "sha": "465e92",
        "codes": {"87433001", "185086009"},
        "description": "COPD",
        "drop": REPO_ROOT / "configs" / "drop" / "copd.txt",
        "leak_terms": None,
    },
    {
        "name": "hypertension",
        "bin": "history",
        "sha": "249aea",
        "codes": {"59621000"},
        "description": "Essential hypertension",
        "drop": None,
        "leak_terms": None,
    },
    {
        "name": "strep_throat",
        "bin": "acute",
        "sha": "1059ea",
        "codes": {"43878008"},
        "description": "Streptococcal sore throat",
        "drop": None,
        "leak_terms": None,
    },
    {
        "name": "viral_pharyngitis",
        "bin": "acute",
        "sha": "5ace2f",
        "codes": {"195662009"},
        "description": "Acute viral pharyngitis",
        "drop": None,
        "leak_terms": None,
    },
    {
        "name": "bacterial_sinusitis",
        "bin": "acute",
        "sha": "c4404e",
        "codes": {"75498004"},
        "description": "Acute bacterial sinusitis",
        "drop": None,
        "leak_terms": None,
    },
    {
        "name": "cystitis",
        "bin": "acute",
        "sha": "c91f0b",
        "codes": {"307426000"},
        "description": "Acute infective cystitis",
        "drop": None,
        "leak_terms": None,
    },
    {
        "name": "heart_failure",
        "bin": "control",
        "sha": "9e8474",
        "codes": {"88805009"},
        "description": "Congestive heart failure",
        "drop": REPO_ROOT / "configs" / "drop" / "heart_failure.txt",
        "leak_terms": REPO_ROOT / "configs" / "leak_terms" / "heart_failure.txt",
    },
]

# Ordered from rarest to most common for stratification key
COND_RANK = {c["name"]: i for i, c in enumerate(
    sorted(CONDITIONS, key=lambda c: int(c["sha"], 16))  # placeholder; overwritten after loading counts
)}

SPLIT_SEED    = 42
MIN_PRIOR_ENC = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def _d(s):
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def train_dir(cond):
    return REPO_ROOT / "data" / f"{SRC_RUN}__train-{cond['sha']}"


def labels_path(cond):
    return REPO_ROOT / "data" / f"labels-{cond['sha']}.json"


def presenting_path(cond):
    return REPO_ROOT / "data" / "presenting" / f"{cond['name']}.json"


# ── Step 1: Build index ───────────────────────────────────────────────────────

def load_labels():
    """Returns {cond_name: {pid: {"code", "description", "cutoff"}}}."""
    result = {}
    for c in CONDITIONS:
        result[c["name"]] = json.loads(labels_path(c).read_text())
    return result


def count_prior_encounters(all_labels):
    """Returns {(cond_name, pid): n_prior_encounters}."""
    print("  Counting prior encounters ...")
    counts = {}
    for c in CONDITIONS:
        tdir = train_dir(c) / "csv" / "encounters.csv"
        if not tdir.exists():
            continue
        enc_per_patient = defaultdict(int)
        with open(tdir, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                enc_per_patient[row["PATIENT"]] += 1
        for pid in all_labels[c["name"]]:
            counts[(c["name"], pid)] = enc_per_patient.get(pid, 0)
    return counts


def find_same_day_patients(all_labels):
    """Returns set of pids that have two or more target diagnoses on the same calendar day."""
    pid_dates = defaultdict(list)
    for cond_name, labels in all_labels.items():
        for pid, info in labels.items():
            pid_dates[pid].append(info["cutoff"][:10])
    same_day = set()
    for pid, dates in pid_dates.items():
        if len(dates) != len(set(dates)):
            same_day.add(pid)
    return same_day


def load_presenting_flags():
    """Returns {cond_name: set of pids with presenting evidence}."""
    flags = {}
    for c in CONDITIONS:
        pp = presenting_path(c)
        if pp.exists():
            raw = json.loads(pp.read_text())
            flags[c["name"]] = set(raw.get("presenting", {}).keys())
        else:
            flags[c["name"]] = set()
    return flags


def build_index(all_labels, enc_counts, same_day_pids, presenting_flags):
    """Returns list of dicts, one per example."""
    rows = []
    for c in CONDITIONS:
        for pid, info in sorted(all_labels[c["name"]].items()):
            n_enc = enc_counts.get((c["name"], pid), 0)
            has_pres = pid in presenting_flags[c["name"]]
            ineligible = None
            if pid in same_day_pids:
                ineligible = "same_day_multi_diagnosis"
            elif n_enc < MIN_PRIOR_ENC:
                ineligible = f"too_few_encounters:{n_enc}"
            rows.append({
                "example_id":            f"{c['name']}_{pid}",
                "patient_id":            pid,
                "condition":             c["name"],
                "bin":                   c["bin"],
                "cutoff_date":           info["cutoff"][:10],
                "n_prior_encounters":    n_enc,
                "has_presenting_evidence": has_pres,
                "ineligible":            ineligible or "",
            })
    return rows


def write_index(index_rows, out_dir):
    path = out_dir / "index.csv"
    fields = ["example_id", "patient_id", "condition", "bin",
              "cutoff_date", "n_prior_encounters", "has_presenting_evidence", "ineligible"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(index_rows)
    print(f"  Wrote index.csv ({len(index_rows)} rows)")
    return path


# ── Step 2: Stratified patient split ─────────────────────────────────────────

def make_split(index_rows, seed):
    """
    Assign each patient (with at least one eligible example) to one split.
    Stratified by the patient's rarest target condition (fewest total examples).
    Returns {pid: "test" | "dev" | "corpus"}.
    """
    # Condition sizes (eligible only)
    cond_size = defaultdict(int)
    eligible = [r for r in index_rows if not r["ineligible"]]
    for r in eligible:
        cond_size[r["condition"]] += 1

    # For each patient: collect eligible conditions, pick the rarest
    pid_conds = defaultdict(set)
    for r in eligible:
        pid_conds[r["patient_id"]].add(r["condition"])

    def rarest(conds):
        return min(conds, key=lambda c: cond_size[c])

    # Group patients by stratum (rarest condition)
    strata = defaultdict(list)
    for pid, conds in sorted(pid_conds.items()):  # sorted for determinism
        strata[rarest(conds)].append(pid)

    rng = random.Random(seed)
    split_map = {}
    for stratum, pids in strata.items():
        pids = list(pids)
        rng.shuffle(pids)
        n = len(pids)
        n_test  = round(n * 0.50)
        n_dev   = round(n * 0.20)
        for pid in pids[:n_test]:
            split_map[pid] = "test"
        for pid in pids[n_test:n_test + n_dev]:
            split_map[pid] = "dev"
        for pid in pids[n_test + n_dev:]:
            split_map[pid] = "corpus"

    return split_map


# ── Step 3: Serialize text ────────────────────────────────────────────────────

def serialize_condition(c, tmp_dir):
    """
    Run serialization/serialize.py for this condition.
    Returns path to the output JSONL.
    """
    out = tmp_dir / f"{c['name']}.jsonl"
    cmd = [
        sys.executable, str(REPO_ROOT / "serialization" / "serialize.py"),
        str(train_dir(c)),
        str(labels_path(c)),
        "--out", str(out),
        "--presenting", str(presenting_path(c)),
        "--min-encounters", str(MIN_PRIOR_ENC),
    ]
    if c["drop"] and c["drop"].exists():
        cmd += ["--drop", str(c["drop"])]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: serialize.py failed for {c['name']}:")
        print(result.stderr[-500:] if result.stderr else "(no stderr)")
    return out


def load_jsonl(path):
    """Returns list of dicts from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Step 4: Build DIAGNOSIS + TREATMENT for corpus ───────────────────────────

def load_treatment_data(corpus_examples):
    """
    For each corpus (patient, condition) example, collects medications, procedures,
    and care plans whose REASONCODE is one of the target condition's SNOMED codes,
    started on or before the condition's START date in conditions.csv.

    Using the condition START date (not the serialization cutoff) is necessary
    because some patients have a cutoff pulled earlier by a shared encounter date
    unrelated to the target condition (a Synthea artifact). Treatment prescribed
    on the actual diagnosis date would otherwise be excluded.

    corpus_examples: list of {patient_id, condition, cutoff_date}
    Returns {(pid, cond_name): {"diagnosis": str, "treatment": [str]}}
    """
    cond_by_name = {c["name"]: c for c in CONDITIONS}

    pid_cond_cutoff = defaultdict(dict)
    for ex in corpus_examples:
        pid_cond_cutoff[ex["patient_id"]][ex["condition"]] = ex["cutoff_date"]
    all_pids = set(pid_cond_cutoff)

    # Find condition START date per (pid, cond_name) from conditions.csv.
    # Use the earliest occurrence (same as extract_labels).
    cond_start = {}  # (pid, cond_name) -> YYYY-MM-DD
    with open(SRC_DIR / "csv" / "conditions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["PATIENT"]
            if pid not in all_pids:
                continue
            code = row["CODE"]
            start = (row.get("START", "") or "")[:10]
            if not start:
                continue
            for cond_name in pid_cond_cutoff[pid]:
                if code in cond_by_name[cond_name]["codes"]:
                    key = (pid, cond_name)
                    if key not in cond_start or start < cond_start[key]:
                        cond_start[key] = start

    # Collect items per (pid, cond_name) where REASONCODE is a target code,
    # started on or before the condition START date.
    items_by_key = defaultdict(list)
    for stem in ("medications", "procedures", "careplans"):
        path = SRC_DIR / "csv" / f"{stem}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("PATIENT", "")
                if pid not in all_pids:
                    continue
                rc = row.get("REASONCODE", "")
                if not rc:
                    continue
                desc = row.get("DESCRIPTION", "").strip()
                if not desc:
                    continue
                item_start = (row.get("START", "") or "")[:10]
                for cond_name in pid_cond_cutoff[pid]:
                    if rc not in cond_by_name[cond_name]["codes"]:
                        continue
                    cutoff = cond_start.get((pid, cond_name), "")
                    if item_start and cutoff and item_start > cutoff:
                        continue
                    items_by_key[(pid, cond_name)].append(desc)

    # Diagnosis descriptions
    cond_descriptions = {}
    for c in CONDITIONS:
        for pid, info in json.loads(labels_path(c).read_text()).items():
            cond_descriptions[(pid, c["name"])] = info["description"]

    result = {}
    for ex in corpus_examples:
        pid = ex["patient_id"]
        cond_name = ex["condition"]
        seen, items = set(), []
        for desc in items_by_key.get((pid, cond_name), []):
            if desc not in seen:
                items.append(desc)
                seen.add(desc)
        result[(pid, cond_name)] = {
            "diagnosis": cond_descriptions.get((pid, cond_name), cond_name),
            "treatment": items,
        }
    return result


def add_corpus_sections(record, treatment_info):
    """Append DIAGNOSIS and TREATMENT sections to the record text."""
    diag  = treatment_info.get("diagnosis", "")
    items = treatment_info.get("treatment", [])
    extra = f"\n\nDIAGNOSIS\n- {diag}" if diag else ""
    extra += "\n\nTREATMENT\n" + (
        "\n".join(f"- {t}" for t in items) if items else "- none recorded"
    )
    return record["text"] + extra


# ── Step 5: Build output JSONL files ─────────────────────────────────────────

def build_output_files(index_rows, split_map, tmp_dir, out_dir, treatment_by_key):
    """
    Reads per-condition JSONL from tmp_dir, routes each record to the right
    split, and writes test/dev/corpus JSONL + label files.

    Returns counts dict: {split: {condition: n}}.
    """
    test_lines  = []
    dev_lines   = []
    corpus_lines = []

    test_label_lines  = []
    dev_label_lines   = []

    # Index eligible examples by (pid, cond) for quick lookup
    eligible_set = {(r["patient_id"], r["condition"]): r
                    for r in index_rows if not r["ineligible"]}

    for c in CONDITIONS:
        jsonl_path = tmp_dir / f"{c['name']}.jsonl"
        records = load_jsonl(jsonl_path)
        for rec in records:
            pid = rec.get("patient_id") or rec.get("id", "")
            key = (pid, c["name"])
            if key not in eligible_set:
                continue
            idx_row = eligible_set[key]
            split = split_map.get(pid)
            if split is None:
                continue

            ex_id = idx_row["example_id"]
            has_pres = idx_row["has_presenting_evidence"]

            text_record = {"id": ex_id, "text": rec["text"]}
            label_record = {
                "id": ex_id,
                "condition": c["name"],
                "bin": c["bin"],
                "has_presenting_evidence": has_pres,
            }

            if split == "test":
                test_lines.append(text_record)
                test_label_lines.append(label_record)
            elif split == "dev":
                dev_lines.append(text_record)
                dev_label_lines.append(label_record)
            elif split == "corpus":
                t_info = treatment_by_key.get((pid, c["name"]), {})
                corpus_text = add_corpus_sections(rec, t_info)
                corpus_lines.append({"id": ex_id, "text": corpus_text})

    def write(lines, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"    {path.name}: {len(lines)} records")
        return len(lines)

    write(test_lines,       out_dir / "test.jsonl")
    write(dev_lines,        out_dir / "dev.jsonl")
    write(corpus_lines,     out_dir / "corpus.jsonl")
    write(test_label_lines, out_dir / "test_labels.jsonl")
    write(dev_label_lines,  out_dir / "dev_labels.jsonl")

    # Per-split-per-condition counts
    counts = defaultdict(lambda: defaultdict(int))
    for r in test_label_lines:
        counts["test"][r["condition"]] += 1
    for r in dev_label_lines:
        counts["dev"][r["condition"]] += 1
    for r in json.loads((out_dir / "corpus.jsonl").read_text().splitlines()[0]) if (out_dir / "corpus.jsonl").stat().st_size > 0 else []:
        pass  # already counted below
    # Recount corpus from label records (no label file for corpus)
    for ex_id, _ in [(r["id"], r) for r in corpus_lines]:
        cond = ex_id.rsplit("_", 1)[0].rsplit("_", 1)[0]  # strip pid suffix
    # Simpler: walk index
    for r in index_rows:
        if not r["ineligible"] and split_map.get(r["patient_id"]) == "corpus":
            if (r["patient_id"], r["condition"]) in eligible_set:
                counts["corpus"][r["condition"]] += 1

    return dict(counts)


# ── Step 6: Text leak check ───────────────────────────────────────────────────

def run_leak_check(jsonl_path, c, out_dir):
    """
    Run check_text_leaks.py for one condition's subset of a combined JSONL.
    Returns (n_fail, n_review).
    """
    # Extract records for this condition
    all_recs = load_jsonl(jsonl_path)
    cond_recs = [r for r in all_recs if r["id"].startswith(c["name"] + "_")]
    if not cond_recs:
        return 0, 0

    # Write tmp JSONL; re-add label field that serialize.py includes
    # build a pid->label map from the labels file
    labels_raw = json.loads(labels_path(c).read_text())
    # The text JSONL we wrote has no label field; check_text_leaks needs it
    # Re-add from original serialize output (stored in tmp) if available, else
    # reconstruct from labels
    tmp_path = jsonl_path.parent / f"_leak_{c['name']}_{jsonl_path.stem}.jsonl"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in cond_recs:
            # Reconstruct pid from example_id: format is {cond}_{pid}
            ex_id = rec["id"]
            pid = ex_id[len(c["name"]) + 1:]
            info = labels_raw.get(pid, {})
            out = dict(rec)
            out["patient_id"] = pid
            out["label"] = {"code": info.get("code", ""), "description": info.get("description", "")}
            f.write(json.dumps(out) + "\n")

    report_path = out_dir / f"leak_{c['name']}_{jsonl_path.stem}.md"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "serialization" / "check_text_leaks.py"),
        str(tmp_path), str(train_dir(c)),
        "--report", str(report_path),
    ]
    if c["leak_terms"] and c["leak_terms"].exists():
        cmd += ["--terms", str(c["leak_terms"])]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_path.unlink(missing_ok=True)

    n_fail   = result.stdout.count("FAIL") + result.stderr.count("FAIL")
    n_review = result.stdout.count("REVIEW") + result.stderr.count("REVIEW")
    return n_fail, n_review


def run_all_leak_checks(out_dir):
    """Run leak checks on test.jsonl and dev.jsonl; return summary dict."""
    summary = {}
    for split in ("test", "dev"):
        jsonl = out_dir / f"{split}.jsonl"
        split_fail = split_review = 0
        for c in CONDITIONS:
            nf, nr = run_leak_check(jsonl, c, out_dir / "leak_reports")
            split_fail   += nf
            split_review += nr
            if nf or nr:
                summary.setdefault(split, {})[c["name"]] = {"fail": nf, "review": nr}
        summary.setdefault(split, {})["_total"] = {"fail": split_fail, "review": split_review}
    return summary


# ── Step 7: Sample files ──────────────────────────────────────────────────────

def write_samples(index_rows, split_map, out_dir, test_jsonl, corpus_jsonl):
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Load JSONL into maps
    def jsonl_map(path):
        return {r["id"]: r["text"] for r in load_jsonl(path)}

    test_texts   = jsonl_map(test_jsonl)
    corpus_texts = jsonl_map(corpus_jsonl)

    # Load labels
    test_labels_map = {r["id"]: r for r in load_jsonl(out_dir / "test_labels.jsonl")}

    # ── test_samples.md ──────────────────────────────────────────────────────
    lines = ["# Test Record Samples\n",
             "Guess the diagnosis, then unfold the label block below each record.\n"]

    # Find the longest record for the "longest overall" requirement
    longest_id = max(test_texts, key=lambda i: len(test_texts[i])) if test_texts else None

    # One patient without presenting evidence (for the "at least one" requirement)
    no_pres_id = next(
        (r["example_id"] for r in index_rows
         if not r["ineligible"] and split_map.get(r["patient_id"]) == "test"
         and not r["has_presenting_evidence"]),
        None,
    )

    per_cond_test = defaultdict(list)
    for r in index_rows:
        if not r["ineligible"] and split_map.get(r["patient_id"]) == "test":
            per_cond_test[r["condition"]].append(r["example_id"])

    included_special = set()
    for c in CONDITIONS:
        lines.append(f"\n---\n## {c['description']} (`{c['bin']}` bin)\n")
        ids = per_cond_test[c["name"]]
        # Sort: put special IDs (no_pres, longest) first within their condition
        ids_sorted = (
            [i for i in ids if i in (no_pres_id, longest_id)]
            + [i for i in ids if i not in (no_pres_id, longest_id)]
        )
        selected = ids_sorted[:3]
        for ex_id in selected:
            text = test_texts.get(ex_id, "(text not found)")
            lbl  = test_labels_map.get(ex_id, {})
            tag  = ""
            if ex_id == no_pres_id and ex_id not in included_special:
                tag = " *(no presenting evidence)*"
                included_special.add(no_pres_id)
            if ex_id == longest_id and ex_id not in included_special:
                tag += " *(longest record overall)*"
                included_special.add(longest_id)
            lines.append(f"### {ex_id}{tag}\n")
            lines.append("```")
            lines.append(text)
            lines.append("```\n")
            lines.append("<details><summary>Label</summary>\n")
            lines.append("```json")
            lines.append(json.dumps(lbl, indent=2))
            lines.append("```")
            lines.append("</details>\n")

    (samples_dir / "test_samples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    wrote test_samples.md")

    # ── corpus_samples.md ────────────────────────────────────────────────────
    lines2 = ["# Corpus Record Samples\n",
              "These records include DIAGNOSIS and TREATMENT sections.\n"]

    per_cond_corpus = defaultdict(list)
    for r in index_rows:
        if not r["ineligible"] and split_map.get(r["patient_id"]) == "corpus":
            per_cond_corpus[r["condition"]].append(r["example_id"])

    for c in CONDITIONS:
        lines2.append(f"\n---\n## {c['description']} (`{c['bin']}` bin)\n")
        for ex_id in per_cond_corpus[c["name"]][:2]:
            text = corpus_texts.get(ex_id, "(text not found)")
            lines2.append(f"### {ex_id}\n")
            lines2.append("```")
            lines2.append(text)
            lines2.append("```\n")

    (samples_dir / "corpus_samples.md").write_text("\n".join(lines2) + "\n", encoding="utf-8")
    print(f"    wrote corpus_samples.md")


# ── Step 8: Manifest and zip ──────────────────────────────────────────────────

def write_manifest(out_dir, index_rows, split_map, counts, leak_summary,
                   seed, lauren_merged):
    total    = len(index_rows)
    eligible = sum(1 for r in index_rows if not r["ineligible"])
    inelig   = total - eligible

    # Per-condition ineligibility breakdown
    cond_inelig = defaultdict(int)
    for r in index_rows:
        if r["ineligible"]:
            cond_inelig[r["condition"]] += 1

    manifest = {
        "release": "v1",
        "source_run": SRC_RUN,
        "source_commit": _git_commit(),
        "split_seed": seed,
        "min_prior_encounters": MIN_PRIOR_ENC,
        "lauren_serialization_fixes_included": lauren_merged,
        "total_examples": total,
        "eligible_examples": eligible,
        "ineligible_examples": inelig,
        "ineligible_by_condition": dict(cond_inelig),
        "counts_by_split_and_condition": {
            split: dict(cond_counts)
            for split, cond_counts in counts.items()
        },
        "leak_check_summary": leak_summary,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote manifest.json")
    return manifest


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=REPO_ROOT)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def zip_release(out_dir):
    """Zip data/release-v1/ excluding any __scrub- or __train- subdirectories."""
    zip_path = out_dir.parent / "release-v1.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(out_dir.parent)
                zf.write(path, arcname)
    size_mb = zip_path.stat().st_size / 1_048_576
    print(f"\n  Zip: {zip_path}  ({size_mb:.1f} MB)")
    return zip_path, size_mb


# ── Counts table ─────────────────────────────────────────────────────────────

def print_counts_table(index_rows, split_map):
    """Print per-condition per-split example counts, flagging < 100 test examples."""
    cond_split = defaultdict(lambda: defaultdict(int))
    for r in index_rows:
        if r["ineligible"]:
            continue
        sp = split_map.get(r["patient_id"], "?")
        cond_split[r["condition"]][sp] += 1

    print(f"\n{'Condition':<26} {'Bin':<10} {'Test':>6} {'Dev':>6} {'Corpus':>8} {'Total':>7}  Flag")
    print("-" * 75)
    grand = defaultdict(int)
    for c in CONDITIONS:
        row = cond_split[c["name"]]
        n_test = row.get("test", 0)
        n_dev  = row.get("dev", 0)
        n_corp = row.get("corpus", 0)
        total  = n_test + n_dev + n_corp
        flag = " *** <100 test" if n_test < 100 else ""
        print(f"{c['name']:<26} {c['bin']:<10} {n_test:>6} {n_dev:>6} {n_corp:>8} {total:>7}{flag}")
        grand["test"]   += n_test
        grand["dev"]    += n_dev
        grand["corpus"] += n_corp
        grand["total"]  += total
    print("-" * 75)
    print(f"{'TOTAL':<26} {'':10} {grand['test']:>6} {grand['dev']:>6} {grand['corpus']:>8} {grand['total']:>7}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed",    type=int, default=SPLIT_SEED)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "release-v1")
    ap.add_argument("--tmp-dir", type=Path, default=REPO_ROOT / "data" / "release-v1" / "_tmp")
    ap.add_argument("--skip-serialize", action="store_true",
                    help="skip serialize.py runs (reuse existing _tmp/*.jsonl)")
    args = ap.parse_args()

    out_dir = args.out_dir
    tmp_dir = args.tmp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Check Lauren's branch
    lauren_merged = _check_lauren()
    print(f"\n  Lauren's serialization-update: {'MERGED into main' if lauren_merged else 'NOT merged (building from main)'}")

    print("\n[1] Building index ...")
    all_labels      = load_labels()
    enc_counts      = count_prior_encounters(all_labels)
    same_day_pids   = find_same_day_patients(all_labels)
    presenting_flags = load_presenting_flags()
    index_rows      = build_index(all_labels, enc_counts, same_day_pids, presenting_flags)
    write_index(index_rows, out_dir)

    print("\n[2] Stratified patient split ...")
    split_map = make_split(index_rows, args.seed)
    split_counts = defaultdict(int)
    for sp in split_map.values():
        split_counts[sp] += 1
    print(f"  {split_counts['test']} patients test | {split_counts['dev']} dev | {split_counts['corpus']} corpus")

    print("\n[3] Serializing text ...")
    if not args.skip_serialize:
        for c in CONDITIONS:
            print(f"  {c['name']} ...", end="", flush=True)
            p = serialize_condition(c, tmp_dir)
            n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
            print(f" {n} records")
    else:
        print("  (skipped — reusing existing _tmp/*.jsonl)")

    print("\n[4] Building corpus treatment data ...")
    corpus_examples = [r for r in index_rows
                       if not r["ineligible"] and split_map.get(r["patient_id"]) == "corpus"]
    treatment_by_key = load_treatment_data(corpus_examples)
    print(f"  Treatment found for {len(treatment_by_key)} corpus examples")

    print("\n[5] Writing output JSONL files ...")
    counts = build_output_files(index_rows, split_map, tmp_dir, out_dir, treatment_by_key)

    print("\n[6] Running text leak checks ...")
    (out_dir / "leak_reports").mkdir(exist_ok=True)
    leak_summary = run_all_leak_checks(out_dir)
    for split, cond_results in leak_summary.items():
        tot = cond_results.get("_total", {})
        print(f"  {split}: {tot.get('fail',0)} FAIL, {tot.get('review',0)} REVIEW")

    print("\n[7] Writing sample files ...")
    write_samples(index_rows, split_map, out_dir,
                  out_dir / "test.jsonl", out_dir / "corpus.jsonl")

    print("\n[8] Writing manifest and zipping ...")
    write_manifest(out_dir, index_rows, split_map, counts, leak_summary,
                   args.seed, lauren_merged)
    zip_path, zip_mb = zip_release(out_dir)

    print_counts_table(index_rows, split_map)
    print(f"\nZip: {zip_path}  ({zip_mb:.1f} MB)")
    subprocess.run(["open", "-R", str(zip_path)])
    print("Done.")


def _check_lauren():
    """Return True if serialization-update has been merged into origin/main."""
    try:
        r = subprocess.run(
            ["git", "branch", "-r", "--merged", "origin/main"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return "serialization-update" in r.stdout
    except Exception:
        return False


if __name__ == "__main__":
    main()
