#!/usr/bin/env python3
"""
scrub.py — Diagnosis-based filtering & scrubbing of a Synthea run.

Builds a "pre-diagnosis" dataset: for each patient who has a target condition,
cut their record at the diagnosis and keep only what came before. See
docs/SCRUBBING.md for usage and decisions, docs/SCRUBBING_PLAN.md for the design
record.

Usage:
    python3 scripts/scrub.py <source-run-dir> --codes <codes.txt> [--out <dir>] [--force]

Uses only the Python standard library (no pandas).
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SCRUB_SCRIPT_VERSION = "1.3"

# Patient-scoped, dated CSVs → the single column whose value decides the cut.
# A row of a scrubbed patient is dropped when this column's local date is >= the
# cutoff, or when it belongs to an encounter that was dropped.
FILTER_COL = {
    "conditions": "START",   # also the source of every patient's cutoff
    "encounters": "START",
    "medications": "START",
    "observations": "DATE",
    "procedures": "START",
    "allergies": "START",
    "careplans": "START",
    "devices": "START",
    "imaging_studies": "DATE",
    "immunizations": "DATE",
    "supplies": "DATE",
    "payer_transitions": "START_DATE",
}
# End-date columns. On kept rows of scrubbed patients, a value on/after the
# cutoff is blanked (the item is "still ongoing" as of the cutoff).
END_COL = {
    "conditions": "STOP",
    "encounters": "STOP",
    "medications": "STOP",
    "procedures": "STOP",
    "allergies": "STOP",
    "careplans": "STOP",
    "devices": "STOP",
    "payer_transitions": "END_DATE",
}
# medications.csv totals accumulated up to STOP (or the end of the simulation
# when STOP is empty). Blanked on every emitted row whose STOP is empty after
# scrubbing.
MED_RUNNING_COLS = ("DISPENSES", "TOTALCOST")
# patients.csv lifetime totals. Blanked for every emitted patient (they can't be
# recomputed as of the cutoff: claims are not exported).
PATIENT_LIFETIME_COLS = ("HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE")
# Global reference tables (no PATIENT column) → copied byte-for-byte.
REFERENCE_CSVS = {"organizations", "providers"}
PATIENTS_STEM = "patients"

# A note entry starts with a bare date line preceded by a blank line.
NOTE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Synthea's JVM zone. generate_synthea.sh pins it; runs from any other zone are
# rejected (see check_tz).
TZ_NAME = "UTC"

# Minimum fraction of note entry dates that must equal the local date (in UTC)
# of one of the patient's encounters. The right zone scores 100%; a zone one
# hour off already drops below this on pop10 (Denver 99.3%).
TZ_NOTES_MIN = 0.999
# Fallback when the run has no notes: conditions.START vs its encounter's local
# date. Weaker (some START values are late in the data itself).
TZ_CONDITIONS_MIN = 0.9

WARNINGS: list = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


# ── Dates ───────────────────────────────────────────────────────────────────
# Synthea writes datetimes in UTC ("...Z") but date-only fields (conditions.START,
# allergies.START, careplans.START, supplies.DATE, note headers, DEATHDATE) as
# the date in the generating JVM's timezone, which generate_synthea.sh pins to
# UTC. All comparisons are made on dates in that zone.
def local_date(s: str, tz):
    """Synthea date or datetime → date in `tz`. None for empty/unparseable.

    Date-only values are already local dates and are returned as-is.
    """
    s = (s or "").strip()
    if not s:
        return None
    try:
        if len(s) == 10:
            return date.fromisoformat(s)
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).date()


def age_on(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def parse_int(s: str):
    try:
        return int(s)
    except ValueError:
        return None


# ── CSV IO (byte-stable: values read/written as-is, '\n' terminators) ─────────
def load_csv(path: Path):
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        if header:
            writer.writerow(header)
        writer.writerows(rows)


# ── Codes file ────────────────────────────────────────────────────────────────
def read_codes(path: Path) -> list:
    """One SNOMED code per line. Blank lines and '#' comments (full-line or
    trailing) are ignored. Codes must be numeric."""
    codes = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0].strip()
        if not code:
            continue
        if not code.isdigit():
            sys.exit(f"ERROR: {path}:{n}: {code!r} is not a numeric SNOMED code")
        codes.append(code)
    return sorted(set(codes))


def codes_sha6(codes: list) -> str:
    joined = "\n".join(sorted(set(codes)))
    return hashlib.sha1(joined.encode()).hexdigest()[:6]


# ── Notes ─────────────────────────────────────────────────────────────────────
def split_note(path: Path):
    """Split a note file into (preamble_lines, [(date_str, entry_lines), ...]).

    A note file is a sequence of entries, one per encounter, each starting with
    a bare YYYY-MM-DD line (local date) preceded by a blank line.
    """
    lines = path.read_text().splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines)
              if NOTE_DATE_RE.match(ln.strip()) and (i == 0 or not lines[i - 1].strip())]
    preamble = lines[:starts[0]] if starts else list(lines)
    entries = []
    for n, s in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        entries.append((lines[s].strip(), lines[s:end]))
    return preamble, entries


def patient_id_from_filename(path: Path) -> str:
    """Synthea per-patient files are named <Names...>_<patientId>.txt."""
    return path.stem.rsplit("_", 1)[-1]


# ── Source index ──────────────────────────────────────────────────────────────
def load_encounters(csv_dir: Path):
    """{encounter_id: (patient_id, START)}."""
    header, rows = load_csv(csv_dir / "encounters.csv")
    e = {c: i for i, c in enumerate(header)}
    return {r[e["Id"]]: (r[e["PATIENT"]], r[e["START"]]) for r in rows}


def check_tz(src_run: Path, csv_dir: Path, encounters: dict, tz, tz_name: str):
    """Verify the data was generated in `tz`; exit if not. Returns (basis, rate)."""
    enc_dates = defaultdict(Counter)   # pid -> Counter(local encounter dates)
    for pid, start in encounters.values():
        d = local_date(start, tz)
        if d is not None:
            enc_dates[pid][d.isoformat()] += 1

    notes_dir = src_run / "notes"
    notes = sorted(notes_dir.glob("*.txt")) if notes_dir.is_dir() else []
    if notes:
        total = miss = 0
        for note in notes:
            _, entries = split_note(note)
            nd = Counter(d for d, _ in entries)
            total += sum(nd.values())
            miss += sum((nd - enc_dates[patient_id_from_filename(note)]).values())
        basis, rate, minimum = "note dates", (total - miss) / total if total else 1.0, TZ_NOTES_MIN
    else:
        header, rows = load_csv(csv_dir / "conditions.csv")
        c = {h: i for i, h in enumerate(header)}
        total = agree = 0
        for row in rows:
            enc = encounters.get(row[c["ENCOUNTER"]])
            if enc:
                total += 1
                agree += local_date(enc[1], tz) == local_date(row[c["START"]], tz)
        basis, rate, minimum = "conditions.START", agree / total if total else 1.0, TZ_CONDITIONS_MIN
    if rate < minimum:
        sys.exit(f"ERROR: only {rate:.1%} of {basis} match the {tz_name} date of an encounter "
                 f"(need {minimum:.1%}). The run was generated in a different timezone — "
                 f"regenerate it with scripts/generate_synthea.sh, which pins {tz_name}.")
    return basis, rate


# ── Cutoff discovery ────────────────────────────────────────────────────────
def find_cutoffs(csv_dir: Path, target_codes: set, encounters: dict, tz):
    """Return (cutoffs, matched_codes, target_descriptions, bad_start_pids).

    cutoffs: {patient_id: local date}. For each target row the diagnosis date is
    the earlier of its START and the local date of its linked encounter (some
    START values in Synthea output are a day, or more, later than the visit that
    recorded the condition); the cutoff is the earliest over the patient's rows.
    bad_start_pids: patients with a target row whose START doesn't parse.
    """
    header, rows = load_csv(csv_dir / "conditions.csv")
    idx = {c: i for i, c in enumerate(header)}
    pid_i, code_i, start_i = idx["PATIENT"], idx["CODE"], idx["START"]
    desc_i, enc_i = idx["DESCRIPTION"], idx["ENCOUNTER"]

    cutoffs, matched_codes, target_descs, bad = {}, set(), set(), {}
    for row in rows:
        if row[code_i] not in target_codes:
            continue
        matched_codes.add(row[code_i])
        target_descs.add(row[desc_i])
        pid = row[pid_i]
        d = local_date(row[start_i], tz)
        if d is None:
            bad.setdefault(pid, f"conditions.csv: unparseable target START {row[start_i]!r}")
            continue
        enc = encounters.get(row[enc_i])
        enc_d = local_date(enc[1], tz) if enc else None
        if enc_d is not None and enc_d < d:
            d = enc_d
        if pid not in cutoffs or d < cutoffs[pid]:
            cutoffs[pid] = d
    return cutoffs, matched_codes, target_descs, bad


def pull_cutoffs_by_reason(csv_dir: Path, target_codes: set, cutoffs: dict,
                           encounters: dict, tz) -> set:
    """Move cutoffs earlier to rows done *for* a target condition.

    An earlier episode can lack a conditions.csv row while encounters,
    medications, etc. still name it in REASONCODE; those rows would otherwise be
    kept. Each such row's date is the earlier of its filter column and its
    linked encounter's local date. Only patients already in `cutoffs` are
    affected (a REASONCODE alone doesn't add a patient). Updates `cutoffs` in
    place; returns the patients whose cutoff moved. Unparseable dates are
    skipped here — find_unparseable excludes those patients.
    """
    moved = set()
    for stem, col in FILTER_COL.items():
        path = csv_dir / f"{stem}.csv"
        if not path.is_file():
            continue
        header, rows = load_csv(path)
        idx = {c: i for i, c in enumerate(header)}
        if "REASONCODE" not in idx:
            continue
        pid_i, col_i, reason_i = idx["PATIENT"], idx[col], idx["REASONCODE"]
        enc_i = idx.get("ENCOUNTER")
        for row in rows:
            pid = row[pid_i]
            if row[reason_i] not in target_codes or pid not in cutoffs:
                continue
            d = local_date(row[col_i], tz)
            if d is None:
                continue
            enc = encounters.get(row[enc_i]) if enc_i is not None else None
            enc_d = local_date(enc[1], tz) if enc else None
            if enc_d is not None and enc_d < d:
                d = enc_d
            if d < cutoffs[pid]:
                cutoffs[pid] = d
                moved.add(pid)
    return moved


def find_unparseable(src_run: Path, csv_dir: Path, patients: set,
                     encounters: dict, tz) -> dict:
    """Scan every date a scrubbed patient's cut depends on.

    Returns {patient_id: first problem}. Such patients can't be scrubbed
    reliably and are excluded from the output.
    """
    bad = {}

    def flag(pid, msg):
        bad.setdefault(pid, msg)

    for stem, col in FILTER_COL.items():
        path = csv_dir / f"{stem}.csv"
        if not path.is_file():
            continue
        header, rows = load_csv(path)
        idx = {c: i for i, c in enumerate(header)}
        pid_i, col_i = idx["PATIENT"], idx[col]
        end_i = idx.get(END_COL.get(stem))
        for row in rows:
            pid = row[pid_i]
            if pid not in patients:
                continue
            if local_date(row[col_i], tz) is None:
                flag(pid, f"{stem}.csv: unparseable {col} {row[col_i]!r}")
            if end_i is not None and row[end_i] and local_date(row[end_i], tz) is None:
                flag(pid, f"{stem}.csv: unparseable {END_COL[stem]} {row[end_i]!r}")

    header, rows = load_csv(csv_dir / "patients.csv")
    idx = {c: i for i, c in enumerate(header)}
    for row in rows:
        pid = row[idx["Id"]]
        if pid not in patients:
            continue
        if local_date(row[idx["BIRTHDATE"]], tz) is None:
            flag(pid, f"patients.csv: unparseable BIRTHDATE {row[idx['BIRTHDATE']]!r}")
        death = row[idx["DEATHDATE"]]
        if death and local_date(death, tz) is None:
            flag(pid, f"patients.csv: unparseable DEATHDATE {death!r}")

    sym = src_run / "symptoms" / "csv" / "symptoms.csv"
    if sym.is_file():
        header, rows = load_csv(sym)
        idx = {c: i for i, c in enumerate(header)}
        for row in rows:
            pid = row[idx["PATIENT"]]
            if pid not in patients:
                continue
            if parse_int(row[idx["AGE_BEGIN"]]) is None:
                flag(pid, f"symptoms.csv: unparseable AGE_BEGIN {row[idx['AGE_BEGIN']]!r}")
            if row[idx["AGE_END"]] and parse_int(row[idx["AGE_END"]]) is None:
                flag(pid, f"symptoms.csv: unparseable AGE_END {row[idx['AGE_END']]!r}")

    notes_dir = src_run / "notes"
    if notes_dir.is_dir():
        n_enc = Counter(pid for pid, _ in encounters.values())
        for note in sorted(notes_dir.glob("*.txt")):
            pid = patient_id_from_filename(note)
            if pid not in patients:
                continue
            _, entries = split_note(note)
            for d, _ in entries:
                if local_date(d, tz) is None:
                    flag(pid, f"notes/{note.name}: unparseable entry date {d!r}")
            # One entry per encounter; a mismatch means a header wasn't
            # recognised and entries would be merged into their neighbours.
            if len(entries) != n_enc[pid]:
                flag(pid, f"notes/{note.name}: {len(entries)} dated entries but "
                          f"{n_enc[pid]} encounters")
    return bad


# ── Per-file filtering ────────────────────────────────────────────────────────
def filter_event_csv(src: Path, dst: Path, stem: str, emitted: set,
                     cutoffs: dict, dropped_encs: set, tz) -> dict:
    """Filter one patient-scoped CSV. Returns per-file stats."""
    header, rows = load_csv(src)
    stats = {"source": len(rows), "written": 0, "dropped_by_date": 0,
             "dropped_by_encounter": 0, "end_dates_blanked": 0}
    if not header:
        write_csv(dst, header, [])
        return stats
    idx = {c: i for i, c in enumerate(header)}
    pid_i, col_i = idx["PATIENT"], idx[FILTER_COL[stem]]
    enc_i = idx.get("ENCOUNTER")
    end_i = idx.get(END_COL.get(stem))
    run_is = [idx[c] for c in MED_RUNNING_COLS] if stem == "medications" else []
    if run_is:
        stats["running_totals_blanked"] = 0

    kept = []
    for row in rows:
        pid = row[pid_i]
        if pid not in emitted:
            continue
        cutoff = cutoffs.get(pid)
        if cutoff is not None:
            if local_date(row[col_i], tz) >= cutoff:
                stats["dropped_by_date"] += 1
                continue
            if enc_i is not None and row[enc_i] in dropped_encs:
                stats["dropped_by_encounter"] += 1
                continue
            if end_i is not None and row[end_i] and local_date(row[end_i], tz) >= cutoff:
                row = list(row)
                row[end_i] = ""
                stats["end_dates_blanked"] += 1
        if run_is and not row[idx["STOP"]]:
            row = list(row)
            for i in run_is:
                row[i] = ""
            stats["running_totals_blanked"] += 1
        kept.append(row)
    write_csv(dst, header, kept)
    stats["written"] = len(kept)
    return stats


def filter_patients_csv(src: Path, dst: Path, emitted: set, cutoffs: dict, tz):
    """Keep emitted patients; blank their lifetime totals and a DEATHDATE
    on/after the cutoff. Returns (stats, births)."""
    header, rows = load_csv(src)
    idx = {c: i for i, c in enumerate(header)}
    id_i, death_i, birth_i = idx["Id"], idx["DEATHDATE"], idx["BIRTHDATE"]
    lifetime_is = [idx[c] for c in PATIENT_LIFETIME_COLS if c in idx]

    kept, deaths_blanked, births = [], 0, {}
    for row in rows:
        pid = row[id_i]
        if pid not in emitted:
            continue
        row = list(row)
        births[pid] = row[birth_i]
        cutoff = cutoffs.get(pid)
        if cutoff is not None and row[death_i] and local_date(row[death_i], tz) >= cutoff:
            row[death_i] = ""
            deaths_blanked += 1
        for i in lifetime_is:
            row[i] = ""
        kept.append(row)
    write_csv(dst, header, kept)
    return {"source": len(rows), "written": len(kept),
            "deathdates_blanked": deaths_blanked}, births


def filter_symptoms_csv(src: Path, dst: Path, emitted: set, cutoffs: dict,
                        births: dict, target_descs: set) -> dict:
    """symptoms.csv has no date column; AGE_BEGIN/AGE_END (whole years) are the
    only time signal. For scrubbed patients, drop rows whose PATHOLOGY is a
    target condition or whose AGE_BEGIN >= the patient's age on the cutoff date,
    and blank AGE_END >= that age on kept rows."""
    header, rows = load_csv(src)
    idx = {c: i for i, c in enumerate(header)}
    pid_i, path_i = idx["PATIENT"], idx["PATHOLOGY"]
    begin_i, end_i = idx["AGE_BEGIN"], idx["AGE_END"]
    cutoff_age = {pid: age_on(date.fromisoformat(births[pid]), c)
                  for pid, c in cutoffs.items() if pid in emitted}

    stats = {"source": len(rows), "written": 0, "dropped_by_pathology": 0,
             "dropped_by_age": 0, "age_end_blanked": 0}
    kept = []
    for row in rows:
        pid = row[pid_i]
        if pid not in emitted:
            continue
        if pid in cutoff_age:
            if row[path_i] in target_descs:
                stats["dropped_by_pathology"] += 1
                continue
            if int(row[begin_i]) >= cutoff_age[pid]:
                stats["dropped_by_age"] += 1
                continue
            if row[end_i] and int(row[end_i]) >= cutoff_age[pid]:
                row = list(row)
                row[end_i] = ""
                stats["age_end_blanked"] += 1
        kept.append(row)
    write_csv(dst, header, kept)
    stats["written"] = len(kept)
    return stats


def scrub_note(src: Path, dst: Path, cutoff: date):
    """Keep only note entries dated before the cutoff (text before the first
    entry is kept). Returns (entries_total, entries_kept); writes nothing if
    none kept."""
    preamble, entries = split_note(src)
    out = list(preamble)
    kept = 0
    for d, entry in entries:
        if date.fromisoformat(d) < cutoff:
            out.extend(entry)
            kept += 1
    if kept:
        dst.write_text("".join(out))
    return len(entries), kept


# ── Summary helpers ───────────────────────────────────────────────────────────
def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def top_conditions(csv_dir: Path, limit: int = 20):
    header, rows = load_csv(csv_dir / "conditions.csv")
    if not header:
        return []
    idx = {c: i for i, c in enumerate(header)}
    pid_i, desc_i = idx["PATIENT"], idx["DESCRIPTION"]
    per_desc = defaultdict(set)
    for row in rows:
        per_desc[row[desc_i]].add(row[pid_i])
    counts = Counter({d: len(p) for d, p in per_desc.items()})
    return counts.most_common(limit)


def is_scrub_output(path: Path) -> bool:
    try:
        return "scrub_script_version" in json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError, TypeError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnosis-based filtering & scrubbing of a Synthea run."
    )
    ap.add_argument("source_run_dir", help="Path to data/<run-name>")
    ap.add_argument("--codes", required=True, help="Path to codes .txt (one SNOMED code per line)")
    ap.add_argument("--out", help="Explicit output dir (overrides auto-naming)")
    ap.add_argument("--force", action="store_true",
                    help="Replace an existing scrub output dir")
    args = ap.parse_args()

    # ── Validate everything before touching the output dir ───────────────────
    src_run = Path(args.source_run_dir).resolve()
    csv_dir = src_run / "csv"
    if not csv_dir.is_dir():
        sys.exit(f"ERROR: {csv_dir} not found — is that a Synthea run dir?")

    codes_path = Path(args.codes)
    codes = read_codes(codes_path)
    if not codes:
        sys.exit(f"ERROR: no codes found in {codes_path}")
    sha6 = codes_sha6(codes)

    src_manifest = None
    mf = src_run / "manifest.json"
    if mf.exists():
        src_manifest = json.loads(mf.read_text())
    # Runs without a recorded zone (older runs, or a missing manifest) are
    # accepted only if their dates fit UTC; check_tz verifies that below.
    run_tz = (src_manifest or {}).get("timezone")
    if run_tz and run_tz != TZ_NAME:
        sys.exit(f"ERROR: the run's manifest.json records timezone {run_tz}; only "
                 f"{TZ_NAME} runs are supported. Regenerate it with scripts/generate_synthea.sh.")
    tz_name = TZ_NAME
    tz = ZoneInfo(tz_name)

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        name = f"{src_run.name}__scrub-{codes_path.stem}-{sha6}"
        out_dir = src_run.parent / name
    # Never write into, or delete, the source run or a dir holding it.
    if src_run.is_relative_to(out_dir) or out_dir.is_relative_to(src_run):
        sys.exit(f"ERROR: output dir {out_dir} overlaps the source run {src_run}.")
    replace = out_dir.exists() and any(out_dir.iterdir())
    if replace:
        if not args.force:
            sys.exit(f"ERROR: {out_dir} exists and is not empty. Use --force to replace it.")
        if not is_scrub_output(out_dir):
            sys.exit(f"ERROR: {out_dir} is not a previous scrub output (no scrub "
                     f"manifest.json) — refusing to delete it, even with --force.")

    encounters = load_encounters(csv_dir)
    tz_basis, tz_rate = check_tz(src_run, csv_dir, encounters, tz, tz_name)

    target_codes = set(codes)
    cutoffs, matched_codes, target_descs, bad = find_cutoffs(
        csv_dir, target_codes, encounters, tz)
    moved_by_reason = pull_cutoffs_by_reason(csv_dir, target_codes, cutoffs, encounters, tz)
    unmatched = target_codes - matched_codes
    if not matched_codes:
        sys.exit("ERROR: no patient has any target code — nothing to scrub. Nothing written.")
    if unmatched:
        warn(f"{len(unmatched)} code(s) matched no patient: {sorted(unmatched)}")

    # Patients with the diagnosis whose dates can't all be trusted are excluded
    # rather than kept: keeping them risks emitting post-diagnosis data.
    diagnosed = set(cutoffs) | set(bad)
    for pid, msg in find_unparseable(src_run, csv_dir, diagnosed, encounters, tz).items():
        bad.setdefault(pid, msg)
    for pid in sorted(bad):
        warn(f"patient {pid} excluded — {bad[pid]}")
    cutoffs = {p: c for p, c in cutoffs.items() if p not in bad}
    excluded = set(bad)

    pat_header, pat_rows = load_csv(csv_dir / "patients.csv")
    pat_id_i = {c: i for i, c in enumerate(pat_header)}["Id"]
    all_patients = {r[pat_id_i] for r in pat_rows}
    affected = set(cutoffs)
    emitted = set(affected)
    moved_by_reason &= affected
    if not emitted:
        sys.exit("ERROR: every matched patient was excluded — dataset would be "
                 "empty. Nothing written.")

    # Encounters of scrubbed patients on/after their cutoff; rows linked to them
    # are dropped even if their own date is earlier.
    dropped_encs = {eid for eid, (pid, start) in encounters.items()
                    if pid in cutoffs and local_date(start, tz) >= cutoffs[pid]}

    # ── Write ────────────────────────────────────────────────────────────────
    if replace:
        shutil.rmtree(out_dir)
    out_csv = out_dir / "csv"
    out_csv.mkdir(parents=True, exist_ok=True)

    print(f"Source run   : {src_run.name}")
    print(f"Codes        : {codes} (sha6 {sha6})")
    print(f"Timezone     : {tz_name} ({tz_rate:.1%} of {tz_basis} match)")
    print(f"Patients     : {len(all_patients)} total, {len(affected)} affected, "
          f"{len(excluded)} excluded, {len(emitted)} emitted, "
          f"{len(moved_by_reason)} cut earlier by REASONCODE")
    print(f"Output       : {out_dir}")

    per_file = {}
    births = {}
    for csv_path in sorted(csv_dir.glob("*.csv")):
        stem = csv_path.stem
        dst = out_csv / csv_path.name
        key = f"csv/{csv_path.name}"
        if stem in FILTER_COL:
            per_file[key] = filter_event_csv(
                csv_path, dst, stem, emitted, cutoffs, dropped_encs, tz)
        elif stem == PATIENTS_STEM:
            per_file[key], births = filter_patients_csv(
                csv_path, dst, emitted, cutoffs, tz)
        elif stem in REFERENCE_CSVS:
            shutil.copyfile(csv_path, dst)  # byte-identical
            s = len(load_csv(csv_path)[1])
            per_file[key] = {"source": s, "written": s}
        else:
            warn(f"unknown CSV '{csv_path.name}' — copied unchanged (not filtered)")
            shutil.copyfile(csv_path, dst)
            s = len(load_csv(csv_path)[1])
            per_file[key] = {"source": s, "written": s}

    # symptoms/csv is filtered; symptoms/text is not carried over.
    sym_src = src_run / "symptoms" / "csv" / "symptoms.csv"
    if sym_src.is_file():
        per_file["symptoms/csv/symptoms.csv"] = filter_symptoms_csv(
            sym_src, out_dir / "symptoms" / "csv" / "symptoms.csv",
            emitted, cutoffs, births, target_descs)

    # notes/: emitted patients only, cut by entry date.
    notes_src = src_run / "notes"
    notes_written = notes_entries_total = notes_entries_kept = 0
    if notes_src.is_dir():
        notes_dst = out_dir / "notes"
        notes_dst.mkdir(exist_ok=True)
        for note in sorted(notes_src.glob("*.txt")):
            pid = patient_id_from_filename(note)
            if pid not in emitted:
                continue
            total, kept = scrub_note(note, notes_dst / note.name, cutoffs[pid])
            notes_entries_total += total
            notes_entries_kept += kept
            notes_written += bool(kept)

    # Patients with no encounter left before the cutoff → empty pre-dx history.
    enc_hdr, enc_out = load_csv(out_csv / "encounters.csv")
    enc_pid_i = enc_hdr.index("PATIENT")
    empty_history = sorted(affected - {r[enc_pid_i] for r in enc_out})

    # ── manifest.json ─────────────────────────────────────────────────────────
    manifest = {
        "scrub_script_version": SCRUB_SCRIPT_VERSION,
        "source_run": src_run.name,
        "source_manifest": src_manifest,
        "codes": codes,
        "codes_file": str(codes_path),
        "codes_sha6": sha6,
        "unmatched_codes": sorted(unmatched),
        "timezone": tz_name,
        "timezone_check": {"basis": tz_basis, "agreement": round(tz_rate, 4)},
        "cut_rule": {
            "cutoff": "earliest over target conditions rows, and over rows of "
                      "patients with such a row whose REASONCODE is a target code, "
                      "of min(own date, local date of the row's encounter START)",
            "drop": "rows whose filter column's local date >= cutoff, or whose "
                    "ENCOUNTER starts on/after the cutoff",
            "end_dates": "blank end dates >= cutoff on kept rows",
            "medications": "blank " + ", ".join(MED_RUNNING_COLS)
                           + " on every emitted row with an empty STOP",
            "symptoms": "drop target PATHOLOGY rows and AGE_BEGIN >= age at cutoff; "
                        "blank AGE_END >= age at cutoff",
            "patients": "blank DEATHDATE >= cutoff; blank "
                        + ", ".join(PATIENT_LIFETIME_COLS) + " for every emitted patient",
            "unparseable": "patients with the diagnosis and any unparseable date "
                           "are excluded",
        },
        "counts": {
            "patients_source": len(all_patients),
            "patients_affected": len(affected),
            "patients_cutoff_from_reasoncode": len(moved_by_reason),
            "patients_excluded": len(excluded),
            "patients_emitted": len(emitted),
            "patients_empty_history": len(empty_history),
            "encounters_dropped": len(dropped_encs),
            "notes_files_written": notes_written,
            "notes_entries_scrubbed_patients": {"source": notes_entries_total,
                                                "kept": notes_entries_kept},
            "files": dict(sorted(per_file.items())),
        },
        "excluded_patients": {p: bad[p] for p in sorted(bad)},
        "warnings": WARNINGS,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # ── SCRUB_SUMMARY.md (self-contained) ─────────────────────────────────────
    L = []
    L.append(f"# Scrub Summary: {out_dir.name}\n")
    L.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    L.append("## Parameters\n")
    L.append("| Field | Value |")
    L.append("|-------|-------|")
    L.append(f"| Source run | {src_run.name} |")
    L.append(f"| Codes | {', '.join(codes)} |")
    L.append(f"| Codes sha6 | {sha6} |")
    L.append(f"| Timezone | {tz_name} ({tz_rate:.1%} of {tz_basis} match) |")
    L.append("| Cut rule | drop rows dated (local) on/after the diagnosis date, or "
             "linked to an encounter on/after it |")
    if unmatched:
        L.append(f"| Unmatched codes | {', '.join(sorted(unmatched))} |")

    L.append("\n## Scrub Stats\n")
    L.append("| Metric | Count |")
    L.append("|--------|-------|")
    L.append(f"| Patients in source | {len(all_patients):,} |")
    L.append(f"| Patients affected (scrubbed) | {len(affected):,} |")
    L.append(f"| Patients cut earlier by a REASONCODE row | {len(moved_by_reason):,} |")
    L.append(f"| Patients excluded (target code, unparseable date) | {len(excluded):,} |")
    L.append(f"| Patients emitted | {len(emitted):,} |")
    L.append(f"| Patients with empty pre-dx history (no prior encounter) | {len(empty_history):,} |")
    L.append(f"| Encounters dropped | {len(dropped_encs):,} |")
    L.append(f"| Note files written | {notes_written:,} |")
    L.append(f"| Note entries kept (scrubbed patients) | "
             f"{notes_entries_kept:,} of {notes_entries_total:,} |")
    L.append(f"| Warnings | {len(WARNINGS):,} |")

    L.append("\n## Rows per File\n")
    L.append("| File | Source | Written | Dropped | Values blanked |")
    L.append("|------|--------|---------|---------|----------------|")
    for name, st in sorted(per_file.items()):
        dropped = sum(v for k, v in st.items() if k.startswith("dropped"))
        blanked = sum(v for k, v in st.items() if k.endswith("blanked"))
        L.append(f"| {name} | {st['source']:,} | {st['written']:,} | {dropped:,} | {blanked:,} |")

    if WARNINGS:
        L.append("\n## Warnings\n")
        L.extend(f"- {w}" for w in WARNINGS)

    tops = top_conditions(out_csv)
    if tops:
        L.append("\n## Top Conditions in Result (by patient count)\n")
        L.append("| # | Condition | Patients |")
        L.append("|---|-----------|----------|")
        for i, (desc, n) in enumerate(tops, 1):
            L.append(f"| {i} | {desc} | {n:,} |")

    L.append("\n## Output File Inventory\n")
    L.append("| File | Rows | Size |")
    L.append("|------|------|------|")
    for csv_path in sorted(out_dir.glob("**/*.csv")):
        _, rws = load_csv(csv_path)
        rel = csv_path.relative_to(out_dir)
        L.append(f"| {rel} | {len(rws):,} | {fmt_size(csv_path.stat().st_size)} |")

    (out_dir / "SCRUB_SUMMARY.md").write_text("\n".join(L) + "\n")

    print(f"Manifest     : {out_dir / 'manifest.json'}")
    print(f"Summary      : {out_dir / 'SCRUB_SUMMARY.md'}")
    print(f"Done with {len(WARNINGS)} warning(s)." if WARNINGS else "Done.")


if __name__ == "__main__":
    main()
