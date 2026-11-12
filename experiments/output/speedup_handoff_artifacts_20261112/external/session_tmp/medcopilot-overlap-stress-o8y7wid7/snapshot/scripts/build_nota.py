#!/usr/bin/env python3
"""
build_nota.py — Build "none of the above" (NOTA) examples for release-v2.1.

Serializes records for three non-target conditions and writes:

  data/release-v2.1/test_nota.jsonl
  data/release-v2.1/dev_nota.jsonl
  data/release-v2.1/corpus_nota.jsonl
  data/release-v2.1/test_nota_labels.jsonl
  data/release-v2.1/dev_nota_labels.jsonl

Source conditions:
  - acute_bronchitis  (SNOMED 10509002)
  - viral_sinusitis   (SNOMED 444814009) — hard case: resembles bacterial sinusitis
  - covid19           (SNOMED 840544004) — replaces otitis_media (0% presenting evidence)

Pipeline (avoids scrub.py / extract_labels.py to skip memory-intensive full-run scrub):
  1. Compute cutoffs inline from source conditions.csv
  2. Build train dirs by streaming-filtering source CSVs (affected patients only)
  3. Create minimal fake scrub dirs (manifest.json only) for extract_presenting.py
  4. Run extract_presenting.py
  5. Run serialize.py (--time-from last-record, --min-encounters 5)

Split assignment:
  - Existing v2 test/dev patients → same split in _nota files
  - Existing v2 corpus patients   → corpus_nota
  - New patients (not in v2)      → test/dev only, 50:20 ratio, seed-fixed

Caps: N_TEST_PER_COND=200, N_DEV_PER_COND=83, N_CORPUS_PER_COND=133.

IDs: nota_{source_condition}_{patient_id}
NOTE: source condition name is visible in the ID — see DATASET_SPLITS.md.

Usage:
    python3 scripts/build_nota.py [--skip-presenting] [--skip-serialize] [--force]
"""

import argparse
import csv
import datetime
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "data" / "pop10000-seed20260916"
V2_DIR    = REPO_ROOT / "data" / "release-v2"
OUT_DIR   = REPO_ROOT / "data" / "release-v2.1"
SCRATCH   = REPO_ROOT / "data" / "_nota"

TIME_FROM = "last-record"
MIN_ENC   = 5
SEED      = 42

N_TEST_PER_COND   = 200
N_DEV_PER_COND    = 83
N_CORPUS_PER_COND = 133

TARGET_CODES = frozenset({
    "87433001", "185086009",  # copd
    "59621000",               # hypertension
    "43878008",               # strep_throat
    "195662009",              # viral_pharyngitis
    "75498004",               # bacterial_sinusitis
    "307426000",              # cystitis
    "88805009",               # heart_failure
})

TARGET_COND_NAMES = sorted(
    ["copd", "hypertension", "strep_throat", "viral_pharyngitis",
     "bacterial_sinusitis", "cystitis", "heart_failure"],
    key=len, reverse=True,
)

NOTA_CONDITIONS = [
    {
        "name": "acute_bronchitis",
        "codes": ["10509002"],
        "description": "Acute bronchitis (disorder)",
    },
    {
        "name": "viral_sinusitis",
        "codes": ["444814009"],
        "description": "Viral sinusitis (disorder)",
    },
    {
        "name": "covid19",
        "codes": ["840544004"],
        "description": "Suspected disease caused by Severe acute respiratory coronavirus 2 (situation)",
    },
]

for _c in NOTA_CONDITIONS:
    code_str = "\n".join(sorted(_c["codes"]))
    _c["sha6"] = hashlib.sha1(code_str.encode()).hexdigest()[:6]

# Columns that determine row inclusion (must be < cutoff)
FILTER_COL = {
    "conditions":        "START",
    "encounters":        "START",
    "medications":       "START",
    "observations":      "DATE",
    "procedures":        "START",
    "allergies":         "START",
    "careplans":         "START",
    "devices":           "START",
    "imaging_studies":   "DATE",
    "immunizations":     "DATE",
    "supplies":          "DATE",
    "payer_transitions": "START_DATE",
}
END_COL = {
    "conditions":        "STOP",
    "encounters":        "STOP",
    "medications":       "STOP",
    "procedures":        "STOP",
    "allergies":         "STOP",
    "careplans":         "STOP",
    "devices":           "STOP",
    "payer_transitions": "END_DATE",
}
MED_RUNNING_COLS = ("DISPENSES", "TOTALCOST")


# ── Path helpers ─────────────────────────────────────────────────────────────

def fake_scrub_dir(c):
    # Named to match the real convention so extract_presenting.py parses stem correctly.
    return SCRATCH / f"{SRC_DIR.name}__scrub-{c['name']}-{c['sha6']}"


def train_dir_path(c):
    return SCRATCH / f"train-{c['name']}-{c['sha6']}"


def labels_path(c):
    return SCRATCH / f"labels-{c['name']}-{c['sha6']}.json"


def presenting_path(c):
    return SCRATCH / f"presenting-{c['name']}.json"


def serialized_path(c):
    return SCRATCH / f"serialized-{c['name']}.jsonl"


# ── Step 1: Compute cutoffs ───────────────────────────────────────────────────

def compute_cutoffs(c):
    """
    Returns {pid: cutoff_date_str} using earliest condition START date (same
    rule as extract_labels.py: min(condition_start, encounter_start)).
    """
    codes = set(c["codes"])
    src_csv = SRC_DIR / "csv"

    # encounter start per encounter id
    enc_start = {}
    with open(src_csv / "encounters.csv", newline="") as f:
        for row in csv.DictReader(f):
            enc_start[row["Id"]] = row["START"][:10]

    cutoffs = {}
    descriptions = {}
    with open(src_csv / "conditions.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["CODE"] not in codes:
                continue
            pid = row["PATIENT"]
            cond_start = row["START"][:10]
            enc_s = enc_start.get(row.get("ENCOUNTER", ""), cond_start)
            cutoff = min(cond_start, enc_s) if enc_s else cond_start
            if pid not in cutoffs or cutoff < cutoffs[pid]:
                cutoffs[pid] = cutoff
                descriptions[pid] = row["DESCRIPTION"]

    return cutoffs, descriptions


def build_labels_json(c, cutoffs, descriptions):
    """Write labels.json for this nota condition."""
    lp = labels_path(c)
    labels = {}
    for pid, cutoff in cutoffs.items():
        labels[pid] = {
            "code": c["codes"][0],
            "description": descriptions.get(pid, c["description"]),
            "cutoff": cutoff,
        }
    lp.write_text(json.dumps(labels, indent=2) + "\n")
    return lp


# ── Step 2: Build train dir (streaming, affected patients only) ──────────────

def build_train_dir(c, cutoffs, force=False):
    """
    Stream-filter source CSVs and write train dir for affected patients.
    """
    td = train_dir_path(c)
    if td.exists() and not force:
        print(f"  train dir exists, skipping: {td.name}")
        return
    if td.exists():
        shutil.rmtree(td)
    td.mkdir(parents=True)

    src_csv = SRC_DIR / "csv"
    dst_csv = td / "csv"
    dst_csv.mkdir()

    affected = set(cutoffs.keys())

    # patients.csv — copy rows for affected patients (no date filter)
    _copy_patients_csv(src_csv / "patients.csv", dst_csv / "patients.csv", affected)

    # per-patient CSVs — filter by date
    for src_path in sorted(src_csv.glob("*.csv")):
        stem = src_path.stem
        if stem == "patients":
            continue
        if stem not in FILTER_COL:
            # Reference table (organizations, providers): copy as-is
            shutil.copy2(src_path, dst_csv / src_path.name)
            continue
        _filter_csv(src_path, dst_csv / src_path.name, stem, cutoffs)

    print(f"  Built train dir: {td.name} ({len(affected)} patients)")


def _copy_patients_csv(src, dst, affected):
    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["Id"] in affected:
                writer.writerow(row)


def _filter_csv(src, dst, stem, cutoffs):
    filter_col = FILTER_COL[stem]
    end_col    = END_COL.get(stem)
    is_med     = stem == "medications"

    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            return
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            pid = row.get("PATIENT", "")
            if pid not in cutoffs:
                continue
            cutoff = cutoffs[pid]
            date_val = (row.get(filter_col) or "")[:10]
            if not date_val:
                writer.writerow(row)
                continue
            if date_val >= cutoff:
                continue
            # Blank end-date if it falls on/after the cutoff
            if end_col:
                end_val = (row.get(end_col) or "")[:10]
                if end_val and end_val >= cutoff:
                    row[end_col] = ""
                    if is_med:
                        for col in MED_RUNNING_COLS:
                            if col in row:
                                row[col] = ""
            writer.writerow(row)


# ── Step 3: Fake scrub dir (manifest.json only) ──────────────────────────────

def create_fake_scrub_dir(c, force=False):
    fsd = fake_scrub_dir(c)
    if fsd.exists() and not force:
        return
    fsd.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scrub_script_version": "1.3",
        "source_run": SRC_DIR.name,
        "codes": c["codes"],
        "codes_sha6": c["sha6"],
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (fsd / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ── Step 4: Run extract_presenting.py ────────────────────────────────────────

def run_extract_presenting(c, force=False):
    pp = presenting_path(c)
    if pp.exists() and not force:
        print(f"  presenting file exists, skipping: {pp.name}")
        return
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "extract_presenting.py"),
        str(fake_scrub_dir(c)),
        str(SRC_DIR),
        str(labels_path(c)),
        "--out", str(pp),
        "--report", str(SCRATCH / f"presenting-report-{c['name']}.md"),
    ]
    print(f"  Running extract_presenting.py for {c['name']} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1000:])
        sys.exit(f"ERROR: extract_presenting.py failed for {c['name']}")
    print(result.stdout.strip() or "  (no output)")


# ── Step 5: Run serialize.py ──────────────────────────────────────────────────

def run_serialize(c, force=False):
    sp = serialized_path(c)
    if sp.exists() and not force:
        print(f"  serialized file exists, skipping: {sp.name}")
        return
    cmd = [
        sys.executable, str(REPO_ROOT / "serialization" / "serialize.py"),
        str(train_dir_path(c)),
        str(labels_path(c)),
        "--out", str(sp),
        "--presenting", str(presenting_path(c)),
        "--min-encounters", str(MIN_ENC),
        "--time-from", TIME_FROM,
    ]
    print(f"  Running serialize.py for {c['name']} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-500:])
        sys.exit(f"ERROR: serialize.py failed for {c['name']}")
    if result.stderr:
        print("  stderr:", result.stderr[-200:])


# ── Split assignment ──────────────────────────────────────────────────────────

def load_existing_split_map():
    split_map = {}
    for split in ("test", "dev", "corpus"):
        for line in open(V2_DIR / f"{split}.jsonl"):
            rec = json.loads(line)
            ex_id = rec["id"]
            for cname in TARGET_COND_NAMES:
                if ex_id.startswith(cname + "_"):
                    pid = ex_id[len(cname) + 1:]
                    split_map[pid] = split
                    break
    return split_map


def load_target_cutoff_dates():
    target_dates = defaultdict(set)
    with open(SRC_DIR / "csv" / "conditions.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["CODE"] in TARGET_CODES:
                target_dates[row["PATIENT"]].add(row["START"][:10])
    return target_dates


def assign_splits(all_serialized, cutoffs_by_cond, existing_split_map,
                  target_cutoff_dates):
    """
    all_serialized: {(pid, cond_name): record_dict}
    cutoffs_by_cond: {cond_name: {pid: cutoff_str}}

    Returns {(pid, cond_name): split_str}.
    """
    by_group = defaultdict(list)
    for (pid, cname) in all_serialized:
        cutoff = cutoffs_by_cond[cname].get(pid, "")
        if cutoff in target_cutoff_dates.get(pid, set()):
            continue
        group = existing_split_map.get(pid, "new")
        by_group[(cname, group)].append(pid)

    rng = random.Random(SEED)
    result = {}

    for c in NOTA_CONDITIONS:
        cname = c["name"]

        corpus_pids = sorted(by_group.get((cname, "corpus"), []))
        rng.shuffle(corpus_pids)
        for pid in corpus_pids[:N_CORPUS_PER_COND]:
            result[(pid, cname)] = "corpus"

        existing_test = sorted(by_group.get((cname, "test"), []))
        for pid in existing_test[:N_TEST_PER_COND]:
            result[(pid, cname)] = "test"

        existing_dev = sorted(by_group.get((cname, "dev"), []))
        for pid in existing_dev[:N_DEV_PER_COND]:
            result[(pid, cname)] = "dev"

        new_pids = sorted(by_group.get((cname, "new"), []))
        rng.shuffle(new_pids)
        n_test_need = N_TEST_PER_COND - min(len(existing_test), N_TEST_PER_COND)
        n_dev_need  = N_DEV_PER_COND  - min(len(existing_dev),  N_DEV_PER_COND)
        for pid in new_pids[:n_test_need]:
            result[(pid, cname)] = "test"
        for pid in new_pids[n_test_need:n_test_need + n_dev_need]:
            result[(pid, cname)] = "dev"

    return result


# ── Plan computation ──────────────────────────────────────────────────────────

def compute_nota_plans(examples, cutoffs_by_cond):
    """
    examples: list of (pid, nota_cond_name) pairs
    Returns {(pid, nota_cond_name): [description, ...]} — empty → ["none recorded"].
    """
    nota_by_name = {c["name"]: c for c in NOTA_CONDITIONS}
    all_pids = {pid for pid, _ in examples}
    pid_conds = defaultdict(set)
    for pid, cname in examples:
        pid_conds[pid].add(cname)

    items_by_key = defaultdict(list)
    for stem in ("medications", "procedures", "careplans"):
        with open(SRC_DIR / "csv" / f"{stem}.csv", newline="") as f:
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
                item_start = (row.get("START") or "")[:10]
                for cname in pid_conds[pid]:
                    if rc not in nota_by_name[cname]["codes"]:
                        continue
                    cutoff = cutoffs_by_cond[cname].get(pid, "")
                    if item_start and cutoff and item_start > cutoff:
                        continue
                    items_by_key[(pid, cname)].append(desc)

    result = {}
    for pid, cname in examples:
        seen, items = set(), []
        for desc in items_by_key.get((pid, cname), []):
            if desc not in seen:
                items.append(desc)
                seen.add(desc)
        result[(pid, cname)] = items or ["none recorded"]
    return result


# ── Corpus section builder ────────────────────────────────────────────────────

def add_corpus_sections(text, cond_description, plan_items):
    extra = f"\n\nDIAGNOSIS\n- {cond_description}"
    extra += "\n\nTREATMENT\n" + (
        "\n".join(f"- {t}" for t in plan_items) if plan_items else "- none recorded"
    )
    return text + extra


# ── Text scan ────────────────────────────────────────────────────────────────

def leak_phrase(cond):
    return (cond["description"].lower()
            .replace(" (disorder)", "")
            .replace(" (situation)", ""))


def scan_text_for_leak(records, cond):
    phrase = leak_phrase(cond)
    hits = []
    for rec in records:
        # For corpus, check only the part before DIAGNOSIS section
        text = rec["text"].lower().split("\n\ndiagnosis\n")[0]
        if phrase in text:
            idx = text.index(phrase)
            hits.append((rec["id"], text[max(0, idx - 30):idx + 60]))
    return hits


def has_leak(text, cond):
    return leak_phrase(cond) in text.lower()


# ── Verification ──────────────────────────────────────────────────────────────

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-presenting", action="store_true")
    ap.add_argument("--skip-serialize",  action="store_true")
    ap.add_argument("--force",           action="store_true",
                    help="Re-run even if outputs already exist")
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    force = args.force

    print(f"Source run : {SRC_DIR.name}")
    print(f"Scratch    : {SCRATCH}")
    print(f"Output dir : {OUT_DIR}")

    # ── 1. Per-condition pipeline ─────────────────────────────────────────────
    cutoffs_by_cond = {}
    for c in NOTA_CONDITIONS:
        print(f"\n[{c['name']}]")
        cutoffs, descs = compute_cutoffs(c)
        cutoffs_by_cond[c["name"]] = cutoffs
        print(f"  {len(cutoffs)} patients with condition")

        build_labels_json(c, cutoffs, descs)
        build_train_dir(c, cutoffs, force=force)
        create_fake_scrub_dir(c, force=force)

        if not args.skip_presenting:
            run_extract_presenting(c, force=force)
        if not args.skip_serialize:
            run_serialize(c, force=force)

    # ── 2. Load serialized records ────────────────────────────────────────────
    print("\n[Loading serialized records]")
    all_serialized = {}
    for c in NOTA_CONDITIONS:
        cname = c["name"]
        sp = serialized_path(c)
        if not sp.exists():
            sys.exit(f"ERROR: {sp} not found — did serialize.py run?")
        records = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
        for rec in records:
            all_serialized[(rec["patient_id"], cname)] = rec
        print(f"  {cname}: {len(records)} serialized records")

    # ── 3. Assign splits ──────────────────────────────────────────────────────
    print("\n[Assigning splits]")
    existing_split_map  = load_existing_split_map()
    target_cutoff_dates = load_target_cutoff_dates()
    split_assign = assign_splits(
        all_serialized, cutoffs_by_cond, existing_split_map, target_cutoff_dates)

    split_counts = Counter(split_assign.values())
    print(f"  test={split_counts['test']}, dev={split_counts['dev']}, "
          f"corpus={split_counts['corpus']}")
    print(f"  excluded from nota (same-day target or beyond cap): "
          f"{len(all_serialized) - len(split_assign)}")

    # ── 4. Compute plan labels ────────────────────────────────────────────────
    print("\n[Computing plan labels]")
    test_dev_examples = [(pid, cn) for (pid, cn), sp in split_assign.items()
                         if sp in ("test", "dev")]
    corpus_examples   = [(pid, cn) for (pid, cn), sp in split_assign.items()
                         if sp == "corpus"]

    plan_map        = compute_nota_plans(test_dev_examples, cutoffs_by_cond)
    corpus_plan_map = compute_nota_plans(corpus_examples, cutoffs_by_cond)
    print(f"  Done ({len(plan_map)} test/dev + {len(corpus_plan_map)} corpus).")

    # ── 5. Build output records ───────────────────────────────────────────────
    print("\n[Building output records]")
    nota_by_name = {c["name"]: c for c in NOTA_CONDITIONS}

    test_records   = []
    dev_records    = []
    corpus_records = []
    test_labels    = []
    dev_labels     = []

    for (pid, cname), split in sorted(split_assign.items()):
        raw_rec = all_serialized[(pid, cname)]
        c       = nota_by_name[cname]
        ex_id   = f"nota_{cname}_{pid}"
        has_pres = "PRESENTING COMPLAINT" in raw_rec["text"]

        if split == "corpus":
            plan_items = corpus_plan_map.get((pid, cname), ["none recorded"])
            corpus_text = add_corpus_sections(raw_rec["text"], c["description"], plan_items)
            corpus_records.append({"id": ex_id, "text": corpus_text})

        elif split == "test":
            if has_leak(raw_rec["text"], c):
                continue  # drop: source condition name visible in pre-cutoff text
            plan = plan_map.get((pid, cname), ["none recorded"])
            test_records.append({"id": ex_id, "text": raw_rec["text"]})
            test_labels.append({
                "id":                     ex_id,
                "condition":              "none_of_the_above",
                "bin":                    "none",
                "has_presenting_evidence": has_pres,
                "plan":                   plan,
                "source_condition":       cname,
            })

        elif split == "dev":
            if has_leak(raw_rec["text"], c):
                continue  # drop: source condition name visible in pre-cutoff text
            plan = plan_map.get((pid, cname), ["none recorded"])
            dev_records.append({"id": ex_id, "text": raw_rec["text"]})
            dev_labels.append({
                "id":                     ex_id,
                "condition":              "none_of_the_above",
                "bin":                    "none",
                "has_presenting_evidence": has_pres,
                "plan":                   plan,
                "source_condition":       cname,
            })

    # ── 6. Text scan ──────────────────────────────────────────────────────────
    print("\n[Text scan — source condition leak check]")
    total_hits = 0
    for c in NOTA_CONDITIONS:
        cname = c["name"]
        all_recs = (
            [r for r in test_records   if r["id"].startswith(f"nota_{cname}_")] +
            [r for r in dev_records    if r["id"].startswith(f"nota_{cname}_")] +
            [r for r in corpus_records if r["id"].startswith(f"nota_{cname}_")]
        )
        hits = scan_text_for_leak(all_recs, c)
        n    = len(hits)
        total_hits += n
        pct  = 100.0 * n / len(all_recs) if all_recs else 0
        status = f"WARNING ({n} hits, {pct:.1f}%)" if n else "OK"
        print(f"  {cname}: {len(all_recs)} records, leak={status}")
        if hits:
            for ex_id, snip in hits[:3]:
                print(f"    {ex_id}: ...{snip}...")
            if len(hits) > 3:
                print(f"    ... and {len(hits) - 3} more")

    # ── 7. Cross-split sanity ─────────────────────────────────────────────────
    print("\n[Cross-split check]")
    pids_t = {r["id"].split("_", 2)[2] for r in test_records}
    pids_d = {r["id"].split("_", 2)[2] for r in dev_records}
    pids_c = {r["id"].split("_", 2)[2] for r in corpus_records}
    for a_name, a in [("test", pids_t), ("dev", pids_d), ("corpus", pids_c)]:
        for b_name, b in [("test", pids_t), ("dev", pids_d), ("corpus", pids_c)]:
            if a_name >= b_name:
                continue
            overlap = a & b
            print(f"  {a_name} ∩ {b_name} = {len(overlap)}"
                  + (" WARNING" if overlap else " OK"))

    # ── 8. Verify existing v2.1 files unchanged ───────────────────────────────
    print("\n[Verifying existing v2.1 files are byte-identical to v2]")
    ok = True
    for fname in ("test.jsonl", "dev.jsonl", "corpus.jsonl"):
        match = md5_file(V2_DIR / fname) == md5_file(OUT_DIR / fname)
        print(f"  {fname}: {'OK' if match else 'MISMATCH'}")
        if not match:
            ok = False
    if not ok:
        sys.exit("ERROR: byte-identity check failed.")

    # ── 9. Write output ───────────────────────────────────────────────────────
    def write_jsonl(rows, path):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Wrote {path.name} ({len(rows)} records)")

    print("\n[Writing output files]")
    write_jsonl(test_records,   OUT_DIR / "test_nota.jsonl")
    write_jsonl(dev_records,    OUT_DIR / "dev_nota.jsonl")
    write_jsonl(corpus_records, OUT_DIR / "corpus_nota.jsonl")
    write_jsonl(test_labels,    OUT_DIR / "test_nota_labels.jsonl")
    write_jsonl(dev_labels,     OUT_DIR / "dev_nota_labels.jsonl")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print("\n[Counts by condition and split]")
    for c in NOTA_CONDITIONS:
        cname = c["name"]
        nt = sum(1 for r in test_records   if f"nota_{cname}_" in r["id"])
        nd = sum(1 for r in dev_records    if f"nota_{cname}_" in r["id"])
        nc = sum(1 for r in corpus_records if f"nota_{cname}_" in r["id"])
        tp = sum(1 for r in test_labels  if r["source_condition"] == cname
                 and r["has_presenting_evidence"])
        dp = sum(1 for r in dev_labels   if r["source_condition"] == cname
                 and r["has_presenting_evidence"])
        print(f"  {cname}: test={nt} (pres={tp}), dev={nd} (pres={dp}), corpus={nc}")

    print(f"\nTotals: test={len(test_records)}, dev={len(dev_records)}, "
          f"corpus={len(corpus_records)}")
    print(f"Text leaks: {total_hits} total (see WARNINGs above)")
    print("\nDone.")


if __name__ == "__main__":
    main()
