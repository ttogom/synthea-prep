#!/usr/bin/env python3
"""
serialize.py — Turn a clean training directory into one free-text summary per
patient for an LLM.

Input is the output of scripts/extract_labels.py:
  <run>__train-<sha6>/   scrubbed CSVs, notes and symptoms (no target names)
  labels JSON            patient id -> {code, description, cutoff}

Output is JSONL, one line per patient:
  {"patient_id", "cutoff", "label": {code, description}, "text", "chars", "est_tokens"}
plus a <out>.stats.json with length statistics.

The label is kept in its own field and never written into "text".

Compression (all configurable, see --help):
  * Long-lived facts (conditions, medications, immunizations) cover the whole
    history but are collapsed: one line per item with a count and dates.
  * Detailed facts (visits, procedures, vitals/labs, chief complaints) cover
    only the --window-years up to the patient's last visit (or the cutoff,
    with --window-from cutoff). Identical visits are collapsed.
  * Vitals/labs keep the latest --max-values readings per test.
  * Synthea's simulation scores (DALY, QALY, QOLS) and identifying fields
    (names, address, SSN) are never written.
  * --no-compress disables the window and caps, for comparison.

Dates are written relative to the cutoff ("3 years ago"), so the text never
states the diagnosis date.

Usage:
    python3 serialization/serialize.py <train-dir> <labels.json> --out <file.jsonl>

Standard library only.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Synthea-internal disease-burden scores: not clinical data, and they encode
# illness. The address survey item is identifying. The rest repeat the
# demographics line.
DROP_OBSERVATIONS = {
    "DALY", "QALY", "QOLS", "Address", "Race", "Hispanic or Latino",
    "Preferred language", "Primary insurance",
}
# Social-history and survey items shown in the Patient section (latest answer),
# matched as substrings. Other survey items with a numeric score (fall risk,
# pain, MMSE, ...) are listed with vitals/labs; the remaining screening
# questions are left out.
SOCIAL_ITEMS = (
    "Tobacco smoking status", "Pregnancy status", "Employment status",
    "Highest level of education", "Housing status", "Stress level",
    "total income of all family members", "How many people are living",
)
SOCIAL_CATEGORIES = {"social-history", "survey"}
CHARS_PER_TOKEN = 4  # rough estimate until a model/tokenizer is chosen
NOTE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_date(value):
    """Date from a Synthea date or datetime string; None if empty."""
    value = (value or "").strip()
    return date.fromisoformat(value[:10]) if value else None


def ago(d, cutoff):
    """Human-readable time before the cutoff, e.g. '3 years ago'."""
    days = (cutoff - d).days
    if days < 31:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def age_on(birth, on):
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def fmt_value(value, units):
    """'142.0' + 'mm[Hg]' -> '142 mm[Hg]'; unit braces like {score} dropped."""
    try:
        number = float(value)
        value = f"{number:.1f}".rstrip("0").rstrip(".")
    except ValueError:
        pass
    units = (units or "").strip()
    if units.startswith("{") and units.endswith("}"):
        units = ""
    return f"{value} {units}".strip()


def times(n):
    return "once" if n == 1 else f"{n} times"


def read_terms(path):
    """Lower-case terms, one per line; '#' starts a comment."""
    if not path:
        return []
    terms = []
    for line in path.read_text().splitlines():
        term = line.split("#", 1)[0].strip().lower()
        if term:
            terms.append(term)
    return terms


def matches(row, drop):
    text = f"{row.get('DESCRIPTION', '')} {row.get('REASONDESCRIPTION', '')}".lower()
    return any(term in text for term in drop)


def read_rows(path, patients, drop):
    """Rows of a CSV grouped by PATIENT, restricted to the given patients,
    without rows whose description or reason contains a --drop term."""
    grouped = defaultdict(list)
    if not path.exists():
        return grouped
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("PATIENT") in patients and not matches(row, drop):
                grouped[row["PATIENT"]].append(row)
    return grouped


def read_chief_complaints(notes_dir, patients):
    """patient id -> list of (date, complaint) from note Chief Complaint sections."""
    result = defaultdict(list)
    if not notes_dir.exists():
        return result
    for path in notes_dir.glob("*.txt"):
        patient = path.stem.rsplit("_", 1)[-1]
        if patient not in patients:
            continue
        entry_date, in_complaint = None, False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if NOTE_DATE.match(line):
                entry_date, in_complaint = date.fromisoformat(line), False
            elif line.startswith("#"):
                in_complaint = line == "# Chief Complaint"
            elif in_complaint and line.startswith("- ") and entry_date:
                result[patient].append((entry_date, line[2:].strip()))
    return result


# ── Sections ──────────────────────────────────────────────────────────────────

def section(title, lines):
    return f"{title}\n" + "\n".join(f"- {line}" for line in lines) if lines else ""


def demographics(p, cutoff):
    sex = {"M": "male", "F": "female"}.get(p["GENDER"], p["GENDER"])
    ethnicity = "Hispanic" if p["ETHNICITY"] == "hispanic" else "non-Hispanic"
    marital = {"M": "married", "S": "single", "D": "divorced", "W": "widowed"}.get(p["MARITAL"])
    line = f"{age_on(parse_date(p['BIRTHDATE']), cutoff)}-year-old {sex}, {p['RACE']}, {ethnicity}"
    if marital:
        line += f", {marital}"
    return line


def conditions(rows, cutoff):
    """Split into active problems, past problems and other findings, collapsed."""
    by_desc = defaultdict(list)
    for r in rows:
        by_desc[r["DESCRIPTION"]].append(r)
    active, past, findings = [], [], []
    for desc, group in by_desc.items():
        starts = sorted(parse_date(r["START"]) for r in group)
        ongoing = any(not r["STOP"] for r in group)
        n = len(group)
        if not desc.endswith("(disorder)"):
            findings.append((starts[-1], f"{desc} (last noted {ago(starts[-1], cutoff)})"))
        elif ongoing:
            active.append((starts[0], f"{desc}, since {ago(starts[0], cutoff)}"))
        else:
            detail = f"{times(n)}, most recent {ago(starts[-1], cutoff)}" if n > 1 else ago(starts[-1], cutoff)
            past.append((starts[-1], f"{desc} ({detail})"))
    return ([t for _, t in sorted(active)],
            [t for _, t in sorted(past, reverse=True)],
            [t for _, t in sorted(findings, reverse=True)])


def medications(rows, cutoff):
    by_desc = defaultdict(list)
    for r in rows:
        by_desc[r["DESCRIPTION"]].append(r)
    current, past = [], []
    for desc, group in by_desc.items():
        starts = sorted(parse_date(r["START"]) for r in group)
        reasons = sorted({r["REASONDESCRIPTION"] for r in group if r["REASONDESCRIPTION"]})
        reason = f"; for {', '.join(reasons)}" if reasons else ""
        if any(not r["STOP"] for r in group):
            current.append((starts[0], f"{desc} (started {ago(starts[0], cutoff)}{reason})"))
        else:
            n = len(group)
            detail = f"{times(n)}, last {ago(starts[-1], cutoff)}" if n > 1 else ago(starts[-1], cutoff)
            past.append((starts[-1], f"{desc} ({detail}{reason})"))
    return [t for _, t in sorted(current)], [t for _, t in sorted(past, reverse=True)]


def counted(rows, date_col, cutoff, since=None):
    """Collapse rows by DESCRIPTION: 'X (3 times, last 2 years ago)'."""
    by_desc = defaultdict(list)
    for r in rows:
        d = parse_date(r[date_col])
        if since is None or d >= since:
            by_desc[r["DESCRIPTION"]].append(d)
    items = []
    for desc, dates in by_desc.items():
        last = max(dates)
        detail = f"{times(len(dates))}, last {ago(last, cutoff)}" if len(dates) > 1 else ago(last, cutoff)
        items.append((last, f"{desc} ({detail})"))
    return [t for _, t in sorted(items, reverse=True)]


def observations(rows, cutoff, since, max_values):
    """Latest readings per test (vitals/labs) and latest answer per social item."""
    by_desc = defaultdict(list)
    is_social = {}
    for r in rows:
        desc = r["DESCRIPTION"]
        if desc in DROP_OBSERVATIONS:
            continue
        social_item = any(item in desc for item in SOCIAL_ITEMS)
        if r["CATEGORY"] in SOCIAL_CATEGORIES and not social_item and r["TYPE"] != "numeric":
            continue  # yes/no screening questions
        d = parse_date(r["DATE"])
        if since is not None and d < since and not social_item:
            continue
        # Keep the full timestamp: several readings of one test on the same
        # day (e.g. during a hospital stay) must be ordered by time.
        by_desc[desc].append((r["DATE"], d, fmt_value(r["VALUE"], r["UNITS"])))
        is_social[desc] = social_item
    measures, social = [], []
    for desc in sorted(by_desc):
        readings = sorted(by_desc[desc], key=lambda x: x[0], reverse=True)
        if is_social[desc]:
            _, d, value = readings[0]
            social.append(f"{desc}: {value} ({ago(d, cutoff)})")
            continue
        if max_values:
            readings = readings[:max_values]
        latest = f"{readings[0][2]} ({ago(readings[0][1], cutoff)})"
        earlier = ", ".join(f"{v} ({ago(d, cutoff)})" for _, d, v in readings[1:])
        measures.append(f"{desc}: {latest}" + (f"; earlier {earlier}" if earlier else ""))
    return measures, social


def complaints(entries, cutoff, since):
    by_text = defaultdict(list)
    for d, text in entries:
        if since is None or d >= since:
            by_text[text.lower()].append(d)
    items = [(max(ds), f"{text} (reported {times(len(ds))}, last {ago(max(ds), cutoff)})")
             for text, ds in by_text.items()]
    return [t for _, t in sorted(items, reverse=True)]


def encounters(rows, cutoff, since, max_encounters):
    """Identical visits (same class, type and reason) collapsed, newest first."""
    by_kind = defaultdict(list)
    for r in rows:
        d = parse_date(r["START"])
        if since is None or d >= since:
            by_kind[(r["ENCOUNTERCLASS"], r["DESCRIPTION"], r["REASONDESCRIPTION"])].append(d)
    items = []
    for (cls, desc, reason), dates in by_kind.items():
        last = max(dates)
        reason = f"; reason: {reason}" if reason else ""
        count = f" ({len(dates)} visits)" if len(dates) > 1 else ""
        items.append((last, f"{ago(last, cutoff)}: {cls} visit, {desc}{reason}{count}"))
    items.sort(reverse=True)
    if max_encounters:
        items = items[:max_encounters]
    return [t for _, t in items]


# ── Patient ──────────────────────────────────────────────────────────────────

def serialize_patient(pid, label, data, args):
    cutoff = date.fromisoformat(label["cutoff"])
    since = None
    if args.window_years:
        # Anchor on the last visit, not the cutoff: some patients (e.g. 270 of
        # 848 type 2 diabetes patients at 10k) have no visit in the 5 years
        # before diagnosis, so a cutoff-anchored window would be empty.
        visits = [parse_date(r["START"]) for r in data["encounters"][pid]]
        anchor = max(visits) if visits and args.window_from == "last-visit" else cutoff
        since = date(anchor.year - args.window_years, anchor.month, min(anchor.day, 28))
    window = (f"{args.window_years} years up to the last visit" if args.window_from == "last-visit"
              else f"last {args.window_years} years") if args.window_years else "full history"

    active, past, findings = conditions(data["conditions"][pid], cutoff)
    current_meds, past_meds = medications(data["medications"][pid], cutoff)
    measures, social = observations(data["observations"][pid], cutoff, since, args.max_values)
    allergies = [r["DESCRIPTION"] + (f" (reaction: {r['DESCRIPTION1']})" if r["DESCRIPTION1"] else "")
                 for r in data["allergies"][pid] if not r["STOP"]]
    careplans = [r["DESCRIPTION"] + (f" (for {r['REASONDESCRIPTION']})" if r["REASONDESCRIPTION"] else "")
                 for r in data["careplans"][pid] if not r["STOP"]]
    devices = sorted({r["DESCRIPTION"] for r in data["devices"][pid] if not r["STOP"]})

    presenting_ep = data.get("presenting", {}).get(pid)
    presenting_lines = []
    if presenting_ep:
        for s in presenting_ep.get("symptoms", []):
            presenting_lines.append(f"{s['name']} (severity: {s['severity']})")
        vitals = presenting_ep.get("vitals", [])
        if vitals:
            vital_str = ", ".join(
                f"{v['name'].lower()} {v['value']} {v['unit']}" for v in vitals
            )
            presenting_lines.append(f"Vital signs at this visit: {vital_str}")

    if args.no_history:
        # "Remove history" ablation (CONDITION_SELECTION.md): demographics and
        # the presenting complaint only, nothing from earlier visits.
        parts = [
            section("PATIENT", [demographics(data["patients"][pid], cutoff)]),
            section("PRESENTING COMPLAINT", presenting_lines),
        ]
        return "\n\n".join(p for p in parts if p)

    parts = [
        section("PATIENT", [demographics(data["patients"][pid], cutoff)] + social),
        section("PRESENTING COMPLAINT", presenting_lines),
        section(f"CHIEF COMPLAINTS AT VISITS ({window})",
                complaints(data["complaints"][pid], cutoff, since)),
        section("ACTIVE PROBLEMS", active),
        section("PAST PROBLEMS", past),
        section("OTHER FINDINGS", findings),
        section("ALLERGIES", allergies),
        section("CURRENT MEDICATIONS", current_meds),
        section("PAST MEDICATIONS", past_meds),
        section("ACTIVE CARE PLANS", careplans),
        section("DEVICES IN USE", devices),
        section(f"VITALS AND LABS ({window}, latest first)", measures),
        section(f"PROCEDURES ({window})",
                counted(data["procedures"][pid], "START", cutoff, since)),
        section("IMMUNIZATIONS", counted(data["immunizations"][pid], "DATE", cutoff)),
        section(f"RECENT VISITS ({window}, newest first)",
                encounters(data["encounters"][pid], cutoff, since, args.max_encounters)),
    ]
    return "\n\n".join(p for p in parts if p)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("train_dir", type=Path, help="<run>__train-<sha6> directory")
    ap.add_argument("labels", type=Path, help="labels JSON from extract_labels.py")
    ap.add_argument("--out", type=Path, required=True, help="output .jsonl path")
    ap.add_argument("--window-years", type=int, default=5,
                    help="history window for visits, procedures, vitals/labs, complaints (default 5)")
    ap.add_argument("--window-from", choices=("last-visit", "cutoff"), default="last-visit",
                    help="where the history window ends (default: the patient's last visit)")
    ap.add_argument("--max-values", type=int, default=3,
                    help="readings kept per vital/lab test (default 3)")
    ap.add_argument("--max-encounters", type=int, default=15,
                    help="most recent visits listed (default 15)")
    ap.add_argument("--min-encounters", type=int, default=0,
                    help="skip patients with fewer pre-cutoff visits (default 0: keep all)")
    ap.add_argument("--no-compress", action="store_true",
                    help="full history, no caps (for comparison)")
    ap.add_argument("--drop", type=Path,
                    help="file of terms; rows whose description or reason contains one are left out "
                         "(condition-specific, e.g. configs/drop/heart_failure.txt)")
    ap.add_argument("--presenting", type=Path, default=None,
                    help="presenting.json from extract_presenting.py; adds a PRESENTING COMPLAINT "
                         "section to each record that has an episode")
    ap.add_argument("--no-history", action="store_true",
                    help="'remove history' ablation: write only demographics and the presenting "
                         "complaint (use with --presenting)")
    args = ap.parse_args()
    if args.presenting and not args.presenting.exists():
        sys.exit(f"ERROR: --presenting file not found: {args.presenting}")
    if args.no_history and not args.presenting:
        print("WARNING: --no-history without --presenting writes demographics only", file=sys.stderr)
    if args.no_compress:
        args.window_years = args.max_values = args.max_encounters = 0

    csv_dir = args.train_dir / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"ERROR: {csv_dir} not found; pass a __train- directory")
    if args.out.resolve().is_relative_to(args.train_dir.resolve()):
        sys.exit("ERROR: --out must be outside the training directory")
    labels = json.loads(args.labels.read_text())
    patients = set(labels)
    drop = read_terms(args.drop)

    data = {name: read_rows(csv_dir / f"{name}.csv", patients, drop) for name in (
        "conditions", "medications", "observations", "allergies", "careplans",
        "devices", "procedures", "immunizations", "encounters")}
    data["patients"] = {r["Id"]: r for r in csv.DictReader(open(csv_dir / "patients.csv", encoding="utf-8"))
                        if r["Id"] in patients}
    data["complaints"] = {pid: [(d, c) for d, c in entries if not any(t in c.lower() for t in drop)]
                          for pid, entries in read_chief_complaints(args.train_dir / "notes", patients).items()}
    data["complaints"] = defaultdict(list, data["complaints"])
    data["presenting"] = {}
    if args.presenting:
        raw = json.loads(args.presenting.read_text())
        data["presenting"] = raw.get("presenting", {})

    missing = patients - set(data["patients"])
    if missing:
        sys.exit(f"ERROR: {len(missing)} labelled patients not in patients.csv; "
                 "labels and training dir don't match")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lengths, skipped = [], 0
    with open(args.out, "w", encoding="utf-8") as out:
        for pid in sorted(patients):
            if len(data["encounters"][pid]) < args.min_encounters:
                skipped += 1
                continue
            label = labels[pid]
            text = serialize_patient(pid, label, data, args)
            tokens = round(len(text) / CHARS_PER_TOKEN)
            lengths.append(tokens)
            out.write(json.dumps({
                "patient_id": pid,
                "cutoff": label["cutoff"],
                "label": {"code": label["code"], "description": label["description"]},
                "text": text,
                "chars": len(text),
                "est_tokens": tokens,
            }) + "\n")

    if not lengths:
        sys.exit("ERROR: no patients written")
    stats = {
        "train_dir": args.train_dir.name,
        "patients_written": len(lengths),
        "patients_skipped_min_encounters": skipped,
        "settings": {k: getattr(args, k) for k in (
            "window_years", "window_from", "max_values", "max_encounters",
            "min_encounters", "no_compress", "no_history")} | {
            "drop": str(args.drop) if args.drop else None,
            "drop_terms": drop,
            "presenting": str(args.presenting) if args.presenting else None},
        "est_tokens": {
            "min": min(lengths),
            "median": round(statistics.median(lengths)),
            "p90": sorted(lengths)[int(0.9 * (len(lengths) - 1))],
            "max": max(lengths),
        },
        "chars_per_token": CHARS_PER_TOKEN,
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    t = stats["est_tokens"]
    print(f"Wrote {len(lengths)} patients to {args.out} (skipped {skipped})")
    print(f"Est. tokens: min {t['min']}, median {t['median']}, p90 {t['p90']}, max {t['max']}")


if __name__ == "__main__":
    main()
