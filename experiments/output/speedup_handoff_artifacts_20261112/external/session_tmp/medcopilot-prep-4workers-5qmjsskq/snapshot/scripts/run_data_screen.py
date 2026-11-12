#!/usr/bin/env python3
"""
run_data_screen.py — orchestrate Step 4 for all candidate conditions.

For each condition: scrub → extract_labels → check_leaks (text/date only) →
condition_screen (antecedent screen + probe + symptoms) → collect JSON.

Usage:
    python3 scripts/run_data_screen.py [--force] [--skip-probe]

Writes per-condition JSON to data/screen/ and docs/CONDITION_SCREEN.md.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
SCREEN_DIR = DATA / "screen"
SOURCE_DIR = DATA / "pop10000-seed20260916"
SYMPTOMS_CSV = SOURCE_DIR / "symptoms" / "csv" / "symptoms.csv"


CONDITIONS = [
    # (stem, group)
    ("copd",               "history"),
    ("polyp_colon",        "history"),
    ("hypertension",       "history"),
    ("strep_throat",       "acute"),
    ("viral_pharyngitis",  "acute"),
    ("bacterial_sinusitis","acute"),
    ("cystitis",           "acute"),
    ("heart_failure",      "demo"),
    ("ischemic_heart",     "demo"),
    ("sleep_apnea",        "demo"),
    ("alzheimers",         "demo"),
]


def sha6(codes_file: Path) -> str:
    codes = set()
    for line in codes_file.read_text().splitlines():
        code = line.split("#", 1)[0].strip()
        if code:
            codes.add(code)
    return hashlib.sha1("\n".join(sorted(codes)).encode()).hexdigest()[:6]


def run(cmd: list, label: str):
    print(f"\n  [{label}] {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        print(f"    {line}")
    for line in r.stderr.splitlines():
        print(f"    STDERR: {line}")
    if r.returncode != 0:
        print(f"  FAILED (rc={r.returncode})")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    SCREEN_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for stem, group in CONDITIONS:
        codes_file = CONFIGS / f"{stem}.txt"
        if not codes_file.exists():
            print(f"SKIP {stem}: no config at {codes_file}")
            continue

        h = sha6(codes_file)
        scrub_dir = DATA / f"pop10000-seed20260916__scrub-{stem}-{h}"
        labels_path = DATA / f"labels-{h}.json"
        train_dir = DATA / f"pop10000-seed20260916__train-{h}"
        screen_json = SCREEN_DIR / f"{stem}.json"
        leak_report = ROOT / "docs" / f"LEAK_REPORT_{stem}.md"

        print(f"\n{'='*60}")
        print(f"  {stem.upper()} (group={group}, sha6={h})")

        # 1. Scrub
        if not scrub_dir.exists() or args.force:
            run(["python3", "scripts/scrub.py", str(SOURCE_DIR),
                 "--codes", str(codes_file),
                 "--out", str(scrub_dir), "--force"],
                "scrub")
        else:
            print(f"  [scrub] reusing {scrub_dir.name}")

        if not scrub_dir.exists():
            print(f"  ERROR: scrub dir not created, skipping {stem}")
            continue

        # 2. Extract labels
        if not labels_path.exists() or args.force:
            run(["python3", "scripts/extract_labels.py", str(scrub_dir),
                 "--labels", str(labels_path),
                 "--train-dir", str(train_dir), "--force"],
                "extract_labels")
        else:
            print(f"  [labels] reusing {labels_path.name}")

        if not labels_path.exists():
            print(f"  ERROR: labels not created, skipping {stem}")
            continue

        n_pos = len(json.loads(labels_path.read_text()))
        print(f"  Positives: {n_pos}")

        # 3. Check leaks (text + date only; probe is in condition_screen)
        run(["python3", "scripts/check_leaks.py", str(scrub_dir),
             str(SOURCE_DIR), str(labels_path),
             "--report", str(leak_report),
             "--skip-probe"],
            "check_leaks")

        # Parse leak result from the report
        leak_pass = None
        if leak_report.exists():
            txt = leak_report.read_text()
            c1 = "PASS" if "Date assertion | PASS" in txt else ("FAIL" if "Date assertion | FAIL" in txt else "?")
            c2_m = re.search(r"Text/code scan \| (PASS|FAIL)", txt)
            c2 = c2_m.group(1) if c2_m else "?"
            leak_pass = "PASS" if c1 == "PASS" and c2 == "PASS" else f"FAIL (date={c1}, text={c2})"
        print(f"  Leak check: {leak_pass}")

        # 4. Antecedent screen + probe + symptoms
        probe_flag = ["--skip-probe"] if args.skip_probe else []
        ok = run(["python3", "scripts/condition_screen.py",
                  str(scrub_dir), str(SOURCE_DIR), str(labels_path),
                  "--symptoms", str(SYMPTOMS_CSV),
                  "--out", str(screen_json)] + probe_flag,
                 "condition_screen")

        if screen_json.exists():
            r = json.loads(screen_json.read_text())
            results[stem] = {
                "group": group,
                "stem": stem,
                "leak_pass": leak_pass,
                **r,
            }
            probe = r.get("probe", {})
            if "accuracy" in probe:
                print(f"  Probe: {probe['accuracy']:.1%} vs {probe['baseline']:.1%} "
                      f"(lift {probe['lift']:+.1%})")
            screen = r.get("screen", {})
            if screen.get("top_10"):
                best = screen["top_10"][0]
                print(f"  Top antecedent: {best['description'][:50]}  z={best['z']:.1f}  "
                      f"({best['p_pos']:.0%} pos vs {best['p_neg']:.0%} neg)")

    # Write combined results
    combined_path = SCREEN_DIR / "all_results.json"
    combined_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote combined results → {combined_path}")

    # Write CONDITION_SCREEN.md
    write_markdown(results)
    print("Wrote docs/CONDITION_SCREEN.md")


def _fmt_probe(r: dict) -> str:
    p = r.get("probe", {})
    if "error" in p:
        return f"error"
    if p.get("skipped"):
        return "—"
    acc = p.get("accuracy", 0)
    base = p.get("baseline", 0)
    lift = p.get("lift", 0)
    return f"{acc:.1%} vs {base:.1%} ({lift:+.1%})"


def _fmt_strongest(r: dict) -> str:
    screen = r.get("screen", {})
    top = screen.get("top_10", [])
    if not top:
        return "—"
    best = top[0]
    sig = "*" if best.get("significant") else ""
    desc = best["description"][:45]
    return f"{desc}{sig} (z={best['z']:.1f})"


def _fmt_symptoms(r: dict) -> str:
    sym = r.get("symptoms", {})
    if sym.get("skipped") or sym.get("error"):
        return "—"
    n = sym.get("n_with_symptoms", 0)
    total = sym.get("n_positives", 0)
    pct = sym.get("pct_with_symptoms", 0)
    tops = sym.get("top_symptoms", [])[:3]
    return f"{n}/{total} ({pct}%) — {', '.join(tops)}"


def write_markdown(results: dict):
    L = []
    L.append("# Condition Data Screen")
    L.append("")
    L.append("Generated by `scripts/run_data_screen.py` from the 10 k UTC run "
              "(`pop10000-seed20260916`). Each candidate was processed through "
              "scrub → extract_labels → leak check → antecedent screen → classifier probe.")
    L.append("")

    # Summary table
    L.append("## Summary table")
    L.append("")
    L.append("| Group | Condition | Cases | Leak | Probe vs baseline | "
             "Strongest antecedent (Bonferroni sig.*) | Symptom availability |")
    L.append("|-------|-----------|------:|------|-------------------|"
             "----------------------------------------|---------------------|")

    group_order = ["history", "acute", "demo"]
    stem_order = [s for g in group_order for s, grp in CONDITIONS if grp == g]

    for stem in stem_order:
        if stem not in results:
            L.append(f"| — | {stem} | — | — | — | — | — |")
            continue
        r = results[stem]
        descs = r.get("descriptions", [])
        name = " + ".join(d.split(" (")[0] for d in descs)
        n_pos = r.get("n_pos", "?")
        group = r.get("group", "?")
        leak = r.get("leak_pass", "?")
        probe_str = _fmt_probe(r)
        strongest = _fmt_strongest(r)
        syms = _fmt_symptoms(r)
        L.append(f"| {group} | {name} | {n_pos} | {leak} | {probe_str} | "
                 f"{strongest} | {syms} |")

    L.append("")
    L.append("\\* Bonferroni-corrected at α=0.05 / N_features.")
    L.append("")

    # Per-condition sections
    for stem in stem_order:
        if stem not in results:
            continue
        r = results[stem]
        descs = r.get("descriptions", [])
        codes = r.get("codes", [])
        name = " + ".join(d.split(" (")[0] for d in descs)
        group = r.get("group", "?")
        n_pos = r.get("n_pos", "?")
        n_neg = r.get("n_neg", "?")

        L.append(f"## {name}")
        L.append("")
        L.append(f"**Group:** {group}  "
                 f"**Codes:** {', '.join(codes)}  "
                 f"**Positives:** {n_pos}  "
                 f"**Matched negatives:** {n_neg}")
        L.append("")

        # Leak check
        leak = r.get("leak_pass", "?")
        L.append(f"**Leak check (date + text):** {leak}")
        L.append("")

        # Probe
        probe = r.get("probe", {})
        if "accuracy" in probe:
            L.append(f"**Classifier probe:** {probe['accuracy']:.1%} ± {probe['accuracy_std']:.1%} "
                     f"(majority-class baseline {probe['baseline']:.1%}, "
                     f"lift {probe['lift']:+.1%})")
        elif probe.get("skipped"):
            L.append("**Classifier probe:** skipped")
        else:
            L.append(f"**Classifier probe:** {probe.get('error', '?')}")
        L.append("")

        # Antecedent screen
        screen = r.get("screen", {})
        n_feats = screen.get("n_features_tested", 0)
        n_sig = screen.get("n_significant", 0)
        bt = screen.get("bonf_threshold", 0)
        L.append(f"**Antecedent screen:** {n_feats} features tested, "
                 f"Bonferroni threshold p < {bt:.2e}, "
                 f"{n_sig} significant features.")
        L.append("")
        top10 = screen.get("top_10", [])
        if top10:
            L.append("| Rank | Feature | Description | p_pos | p_neg | z | Sig |")
            L.append("|-----:|---------|-------------|------:|------:|--:|-----|")
            for i, feat in enumerate(top10, 1):
                sig_mark = "✓" if feat.get("significant") else ""
                desc = feat.get("description", feat["feature"])[:55]
                L.append(f"| {i} | `{feat['feature'][:30]}` | {desc} | "
                         f"{feat['p_pos']:.3f} | {feat['p_neg']:.3f} | "
                         f"{feat['z']:.1f} | {sig_mark} |")
        L.append("")

        # Symptom availability
        sym = r.get("symptoms", {})
        if not sym.get("skipped") and not sym.get("error"):
            n_sym = sym.get("n_with_symptoms", 0)
            pct = sym.get("pct_with_symptoms", 0)
            tops = sym.get("top_symptoms", [])
            L.append(f"**Symptom availability (Step 5 input):** "
                     f"{n_sym}/{n_pos} positives ({pct}%) have symptom records "
                     f"tied to this condition in `symptoms.csv`.")
            if tops:
                L.append(f"Top symptoms: {', '.join(tops[:8])}")
        L.append("")

    (ROOT / "docs" / "CONDITION_SCREEN.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
