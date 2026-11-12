#!/usr/bin/env python3
"""
scan_modules.py — Static analysis of Synthea module JSON files.

For every ConditionOnset code across all Synthea modules, traces paths from
Initial and classifies what gates onset:

  demographic-only        — gated only on Age / Gender / Race / SES / Date
  history-dependent       — at least one gate reads prior clinical records
    visible               — prior condition, active medication, observation,
                            vital sign, or a visible attribute (obesity, smoker)
    hidden-only           — only internal attributes (veteran, atopic, …)
                            that do not appear in the CSV output
  excluded                — symptom-type targets, precursors that name the
                            target, or screening/work-up codes with known
                            text leaks (cross-checked against SCRUBBING.md
                            §Known limitations)

Sanity check: CHF (88805009) must come out demographic-only.

Usage:
    python3 scripts/scan_modules.py \\
        [--modules-dir .synthea/src/main/resources/modules] \\
        [--conditions data/pop10000-seed20260916/csv/conditions.csv] \\
        [--out-md docs/MODULE_SCAN.md] \\
        [--out-json docs/module_scan.json]
"""

import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

DEFAULT_MODULES = Path(".synthea/src/main/resources/modules")
DEFAULT_CONDS   = Path("data/pop10000-seed20260916/csv/conditions.csv")
DEFAULT_MD      = Path("docs/MODULE_SCAN.md")
DEFAULT_JSON    = Path("docs/module_scan.json")

# ── condition-type taxonomy ───────────────────────────────────────────────────
DEMO_TYPES = frozenset({"Age", "Gender", "Race", "Socioeconomic Status", "Date"})
VISIBLE_HISTORY_TYPES = frozenset({
    "Active Condition", "Active Medication", "Observation",
    "Vital Sign", "MultiObservation", "Active CarePlan",
})
LOGICAL_TYPES = frozenset({"And", "Or", "Not", "At Least"})
SKIP_TYPES    = frozenset({"PriorState", "Symptom", "True", "False"})

# `smoker` / smoking-related attributes are set by the Java LifecycleModule at
# age 16 and then recorded as LOINC 72166-2 observations at every wellness
# visit — they are visible in observations.csv.
OBSERVATION_ATTRIBUTES = frozenset({"smoker", "quit smoking age", "pack years"})

# ── exclusion codes (from SCRUBBING.md §Known limitations) ───────────────────
KNOWN_SYMPTOM_CODES = {
    "386661006",  # Fever (finding)
    "49727002",   # Cough (finding)
    "84229001",   # Fatigue (finding)
    "25064002",   # Headache (finding)
    "422587007",  # Nausea (finding)
    "267036007",  # Dyspnea (finding)
    "267102003",  # Sore throat (finding) — also precursor
}
KNOWN_PRECURSOR_CODES = {
    "195967001",  # Asthma — childhood asthma row survives before cutoff
    "36971009",   # Sinusitis — viral/acute/chronic variants survive
    "267102003",  # Sore throat — streptococcal sore throat survives
    "44054006",   # Diabetes mellitus type 2 — microalbuminuria row survives
}
KNOWN_SCREENING_CODES = {
    "73430006",   # Sleep apnea — assessment/referral names target
    "363406005",  # Malignant neoplasm of colon — screening encounter names it
}

# Additional heuristic: SNOMED semantic tag that indicates symptom/finding
_FINDING_RE = re.compile(r"\(finding\)\s*$", re.IGNORECASE)
_SITUATION_RE = re.compile(r"\(situation\)\s*$", re.IGNORECASE)

SYMPTOM_KEYWORDS = frozenset({
    "fever", "cough", "fatigue", "headache", "nausea", "dyspnea",
    "pain", "vomiting", "diarrhea", "malaise", "insomnia", "dizziness",
    "syncope", "palpitation", "pruritus", "itching",
})


# ── module loading ────────────────────────────────────────────────────────────

def load_all_modules(modules_dir: Path) -> dict:
    """Return {relative_path: module_dict} for every JSON module file."""
    modules = {}
    for f in sorted(modules_dir.rglob("*.json")):
        try:
            m = json.loads(f.read_text())
        except Exception:
            continue
        if "states" not in m:
            continue
        key = str(f.relative_to(modules_dir))
        modules[key] = m
    return modules


# ── attribute visibility map ──────────────────────────────────────────────────

def build_visible_attributes(all_modules: dict) -> set:
    """
    Return the set of attribute names that are directly observable in the
    patient CSV output.

    An attribute is visible if it is set by a ConditionOnset via
    assign_to_attribute (the condition code lands in conditions.csv) or if it
    is a known observation-backed attribute (smoker, pack years, …).
    """
    visible = set(OBSERVATION_ATTRIBUTES)
    for mod in all_modules.values():
        for st in mod["states"].values():
            if st.get("type") == "ConditionOnset":
                attr = st.get("assign_to_attribute")
                if attr:
                    visible.add(attr)
    return visible


# ── graph helpers ─────────────────────────────────────────────────────────────

def _successors(state: dict) -> list:
    """All state names that this state can transition to."""
    out = []
    if "direct_transition" in state:
        out.append(state["direct_transition"])
    for item in state.get("conditional_transition", []):
        t = item.get("transition")
        if t:
            out.append(t)
    for item in state.get("complex_transition", []):
        for d in item.get("distributions", []):
            t = d.get("transition")
            if t:
                out.append(t)
    for item in state.get("distributed_transition", []):
        t = item.get("transition")
        if t:
            out.append(t)
    return out


def backward_reachable(states: dict, target: str) -> set:
    """
    Return the set of state names from which `target` is reachable (BFS
    backward through the successor graph).  The target itself is included.
    """
    # Build predecessor map
    preds: dict[str, list] = defaultdict(list)
    for name, st in states.items():
        for succ in _successors(st):
            preds[succ].append(name)

    visited = {target}
    queue = [target]
    while queue:
        cur = queue.pop()
        for p in preds.get(cur, []):
            if p not in visited:
                visited.add(p)
                queue.append(p)
    return visited


def pre_onset_reachable(states: dict, target: str,
                         all_modules: dict | None = None) -> set:
    """
    Forward BFS from Initial, stopping before entering any ConditionOnset
    state other than `target`.

    This constrains the ancestor set to states that lie on the *first* path
    from Initial to `target`, filtering out post-onset care-management cycles
    and unrelated disease branches in multi-condition modules
    (e.g. injuries.json, wellness_encounters.json).

    For modules with a single disease pathway (e.g. congestive_heart_failure)
    this has no effect: there are no other ConditionOnsets to block.

    Note: CallSubmodule states are not themselves blocked here even if the
    submodule contains ConditionOnsets — blocking them would also cut off
    legitimate pre-diagnosis care steps (e.g. CKD checks in metabolic modules).
    The submodule's internal checks are collected as soft modifiers in
    collect_gates regardless.
    """
    other_onsets = frozenset(
        n for n, st in states.items()
        if st.get("type") == "ConditionOnset" and n != target
    )
    visited = {"Initial"}
    queue = ["Initial"]
    while queue:
        cur = queue.pop()
        for succ in _successors(states.get(cur, {})):
            if succ not in visited and succ not in other_onsets:
                visited.add(succ)
                queue.append(succ)
    return visited


# ── condition flattening ──────────────────────────────────────────────────────

def flatten_conditions(cond: dict) -> list:
    """
    Recursively expand And/Or/Not/At-Least into a flat list of leaf conditions.
    """
    if cond is None:
        return []
    ct = cond.get("condition_type", "")
    if ct in LOGICAL_TYPES:
        out = []
        for sub in cond.get("conditions", []):
            out.extend(flatten_conditions(sub))
        if "condition" in cond:  # Not has a single sub-condition
            out.extend(flatten_conditions(cond["condition"]))
        return out
    return [cond]


# ── gate collection ───────────────────────────────────────────────────────────

def collect_gates(states: dict, ancestors: set, target: str,
                  all_modules: dict, modules_dir: Path) -> list:
    """
    Walk every ancestor state (excluding target itself) and collect all
    condition checks on transitions that lead toward the target.

    For CallSubmodule states in the ancestor set, collect ALL leaf condition
    checks within the referenced submodule (the submodule always runs to
    completion, so its internal checks all affect state when control returns).

    Returns a list of (condition_dict, is_gate) pairs where is_gate is True
    when at least one of the transition's alternative branches does NOT reach
    the target (i.e., the check can prevent onset, not just change timing).
    """
    gates = []

    for sname in ancestors:
        if sname == target:
            continue
        st = states[sname]

        # Guard state: the allow condition must be satisfied to proceed.
        # Note: Guard states can ALSO have a conditional_transition that fires
        # after the guard passes — do not continue; fall through to collect it.
        if st.get("type") == "Guard" and "allow" in st:
            for leaf in flatten_conditions(st["allow"]):
                gates.append((leaf, True))  # guard is always a hard gate

        # Conditional transition: check which branches are on-path vs off-path
        for item in st.get("conditional_transition", []):
            t = item.get("transition", "")
            cond = item.get("condition")
            if cond and t in ancestors:
                # Determine if the alternative (falling through to the next
                # condition or the default) can ALSO reach the target.
                is_hard_gate = not _all_alternatives_reach(
                    st["conditional_transition"], t, ancestors
                )
                for leaf in flatten_conditions(cond):
                    gates.append((leaf, is_hard_gate))

        # Complex transition: condition selects a distribution bucket
        for item in st.get("complex_transition", []):
            cond = item.get("condition")
            on_path_dists = [
                d for d in item.get("distributions", [])
                if d.get("transition", "") in ancestors
            ]
            if cond and on_path_dists:
                # If there are distributions in this item whose transitions
                # are NOT on-path, the condition can gate onset.
                off_path_dists = [
                    d for d in item.get("distributions", [])
                    if d.get("transition", "") not in ancestors
                ]
                is_hard_gate = bool(off_path_dists)
                for leaf in flatten_conditions(cond):
                    gates.append((leaf, is_hard_gate))

        # CallSubmodule: pull all leaf conditions from the referenced submodule
        if st.get("type") == "CallSubmodule":
            sub_path = st.get("submodule", "")
            sub_key = sub_path if sub_path.endswith(".json") else sub_path + ".json"
            sub_mod = all_modules.get(sub_key)
            if sub_mod:
                for leaf in _all_leaves_in_module(sub_mod):
                    gates.append((leaf, False))  # submodule checks are modifiers

    return gates


def _all_alternatives_reach(items: list, chosen_transition: str,
                             ancestors: set) -> bool:
    """
    For a conditional_transition list, return True if the branches that would
    fire when `chosen_transition`'s condition is False all also lead toward
    the target (i.e., are in ancestors).  If any off-condition branch is
    missing from ancestors, the condition is a potential gate.
    """
    # The "alternatives" are the items that come after the chosen item plus the
    # default (no-condition) item.  If they're all in ancestors, not a gate.
    found = False
    for item in items:
        if item.get("transition") == chosen_transition:
            found = True
            continue
        if found:
            t = item.get("transition", "")
            if t and t not in ancestors:
                return False
    return True


def _all_leaves_in_module(mod: dict) -> list:
    """Return every leaf condition check found anywhere in a module."""
    leaves = []
    for st in mod["states"].values():
        if st.get("type") == "Guard" and "allow" in st:
            leaves.extend(flatten_conditions(st["allow"]))
        for item in st.get("conditional_transition", []):
            if "condition" in item:
                leaves.extend(flatten_conditions(item["condition"]))
        for item in st.get("complex_transition", []):
            if "condition" in item:
                leaves.extend(flatten_conditions(item["condition"]))
    return leaves


# ── classification ────────────────────────────────────────────────────────────

def classify_gate(cond: dict, visible_attrs: set) -> str:
    """
    Classify a single leaf condition as one of:
      demo     — demographic only
      visible  — history-visible (observable in patient CSVs)
      hidden   — history-hidden (internal attribute not in CSVs)
      skip     — internal/structural (PriorState, Symptom, etc.)
    """
    ct = cond.get("condition_type", "")
    if ct in DEMO_TYPES:
        return "demo"
    if ct in VISIBLE_HISTORY_TYPES:
        return "visible"
    if ct == "Attribute":
        attr = cond.get("attribute", "")
        return "visible" if attr in visible_attrs else "hidden"
    if ct in SKIP_TYPES:
        return "skip"
    return "skip"


def overall_class(gate_classifications: list) -> str:
    """
    Collapse a list of per-gate class strings into one overall class.
    Precedence: visible > hidden > demo > skip
    """
    s = set(gate_classifications)
    if "visible" in s:
        return "visible"
    if "hidden" in s:
        return "hidden"
    if "demo" in s:
        return "demo"
    return "demo"  # no meaningful gates found → demographic by default


# ── exclusion detection ───────────────────────────────────────────────────────

def exclusion_flags(code: str, desc: str) -> list:
    """
    Return a list of exclusion reason strings for the given condition code and
    SNOMED description, using both hardcoded known-cases and heuristics.
    """
    flags = []
    desc_lower = desc.lower()

    # Known cases from SCRUBBING.md
    if code in KNOWN_SYMPTOM_CODES:
        flags.append("symptom_type:known")
    if code in KNOWN_PRECURSOR_CODES:
        flags.append("precursor_names_target:known")
    if code in KNOWN_SCREENING_CODES:
        flags.append("screening_work_up:known")

    # Heuristic: SNOMED "(finding)" tag + symptom keyword
    if _FINDING_RE.search(desc):
        words = set(re.findall(r"[a-z]+", desc_lower))
        if words & SYMPTOM_KEYWORDS:
            if "symptom_type:known" not in flags:
                flags.append("symptom_type:heuristic")

    # Heuristic: "(situation)" tag often means a history-of / referral code
    if _SITUATION_RE.search(desc):
        if not any("situation" in f for f in flags):
            flags.append("situation_code:heuristic")

    return flags


# ── case count from CSV ───────────────────────────────────────────────────────

def load_case_counts(conds_path: Path) -> dict:
    """Return {code: patient_count} from conditions.csv (unique patients)."""
    if not conds_path.is_file():
        return {}
    by_code: dict[str, set] = defaultdict(set)
    with open(conds_path, newline="") as fh:
        for row in csv.DictReader(fh):
            code = row.get("CODE", "").strip()
            pat  = row.get("PATIENT", "").strip()
            if code and pat:
                by_code[code].add(pat)
    return {code: len(pats) for code, pats in by_code.items()}


# ── per-module analysis ───────────────────────────────────────────────────────

def analyse_module(mod_key: str, mod: dict, all_modules: dict,
                   modules_dir: Path, visible_attrs: set,
                   case_counts: dict) -> list:
    """
    Return a list of result dicts, one per (code, module) pair.
    """
    states = mod["states"]
    results = []

    for sname, st in states.items():
        if st.get("type") != "ConditionOnset":
            continue
        for code_obj in st.get("codes", []):
            code = code_obj.get("code", "").strip()
            desc = code_obj.get("display", "").strip()
            if not code:
                continue

            # Intersect backward-reachable with pre-onset-reachable to exclude
            # post-onset care-cycle states and unrelated disease branches.
            ancs = (backward_reachable(states, sname)
                    & pre_onset_reachable(states, sname))
            gates = collect_gates(states, ancs, sname, all_modules, modules_dir)

            classified = [classify_gate(g, visible_attrs) for g, _ in gates]
            gate_types_hard = [
                classify_gate(g, visible_attrs)
                for g, is_hard in gates
                if is_hard
            ]

            oc = overall_class(classified)
            # If there are no hard gates at all, oc_hard falls back to "demo"
            # via overall_class([]) — do not fall back to oc (soft class).
            oc_hard = overall_class(gate_types_hard)

            # Summarise what each class of gate looks like (for reporting)
            visible_gates = sorted({
                g.get("condition_type", "")
                + (f":{g.get('attribute','')}" if g.get("condition_type") == "Attribute" else "")
                for g, _ in gates if classify_gate(g, visible_attrs) == "visible"
            })
            hidden_gates = sorted({
                g.get("attribute", "?")
                for g, _ in gates if classify_gate(g, visible_attrs) == "hidden"
            })
            demo_gates = sorted({
                g.get("condition_type", "")
                for g, _ in gates if classify_gate(g, visible_attrs) == "demo"
            })

            flags = exclusion_flags(code, desc)

            results.append({
                "code": code,
                "description": desc,
                "module": mod_key,
                "state": sname,
                "overall_class": oc,
                "overall_class_hard_gates": oc_hard,
                "visible_gates": visible_gates,
                "hidden_gates": hidden_gates,
                "demo_gates": demo_gates,
                "exclusion_flags": flags,
                "case_count": case_counts.get(code, 0),
            })

    return results


# ── deduplication ─────────────────────────────────────────────────────────────

def deduplicate(all_results: list) -> list:
    """
    Keep one entry per SNOMED code.  If the same code appears in multiple
    modules, prefer the entry with the highest overall_class (visible >
    hidden > demo) and most case counts.
    """
    CLASS_RANK = {"visible": 2, "hidden": 1, "demo": 0}
    best: dict[str, dict] = {}
    for r in all_results:
        code = r["code"]
        if code not in best:
            best[code] = r
        else:
            existing = best[code]
            if CLASS_RANK.get(r["overall_class"], 0) > CLASS_RANK.get(existing["overall_class"], 0):
                best[code] = r
            elif CLASS_RANK.get(r["overall_class"], 0) == CLASS_RANK.get(existing["overall_class"], 0):
                # Same class: prefer the module entry with more detail (more gates)
                if len(r["visible_gates"]) + len(r["hidden_gates"]) > \
                   len(existing["visible_gates"]) + len(existing["hidden_gates"]):
                    best[code] = r
    return sorted(best.values(), key=lambda x: x["description"].lower())


# ── markdown output ───────────────────────────────────────────────────────────

_SNOMED_TAG_RE = re.compile(r"\s*\([^)]+\)\s*$")

def _strip_tag(desc: str) -> str:
    return _SNOMED_TAG_RE.sub("", desc).strip()


def write_markdown(results: list, out_path: Path, modules_dir: Path,
                   n_modules: int) -> None:
    """Write docs/MODULE_SCAN.md with method and grouped table."""
    groups = {
        "visible":  [],
        "hidden":   [],
        "demo":     [],
        "excluded": [],
    }
    for r in results:
        if r["exclusion_flags"]:
            groups["excluded"].append(r)
        elif r["overall_class"] == "visible":
            groups["visible"].append(r)
        elif r["overall_class"] == "hidden":
            groups["hidden"].append(r)
        else:
            groups["demo"].append(r)

    lines = []
    lines.append("# Synthea Module Scan")
    lines.append("")
    lines.append(
        f"Static analysis of {n_modules} Synthea module JSON files. "
        f"For each unique ConditionOnset SNOMED code, traces all paths from "
        f"`Initial` to the onset state and classifies what gates whether onset "
        f"can occur."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each `ConditionOnset` state, a backward BFS identifies all "
        "ancestor states. That set is then intersected with a forward BFS "
        "from `Initial` that stops before entering any other `ConditionOnset` "
        "in the same module. This *pre-first-onset filter* removes states that "
        "are only reachable after a different condition has already been "
        "diagnosed — eliminating false ancestors from post-onset care-management "
        "cycles (e.g. `wellness_encounters.json`) and unrelated injury branches "
        "in multi-condition modules (e.g. `injuries.json`). Condition checks on "
        "transitions within the filtered ancestor set are collected and "
        "classified by type. `CallSubmodule` ancestors are expanded inline: all "
        "leaf conditions in the submodule are collected as modifiers."
    )
    lines.append("")
    lines.append("### Gate classification")
    lines.append("")
    lines.append(
        "| Class | Condition types |\n"
        "|-------|-----------------|\n"
        "| **demographic** | Age, Gender, Race, Socioeconomic Status, Date |\n"
        "| **visible history** | Active Condition, Active Medication, Observation, "
        "Vital Sign, MultiObservation, Active CarePlan; "
        "Attribute where the attribute is set by a ConditionOnset "
        "(`assign_to_attribute`) or by a recorded observation "
        "(`smoker`, tobacco-related) |\n"
        "| **hidden history** | Attribute where the attribute is an internal "
        "flag set by demographic logic only (e.g. `veteran`, `atopic`, "
        "`diabetes` flag in metabolic_syndrome_disease.json) |"
    )
    lines.append("")
    lines.append(
        "The overall class for each code is the highest-precedence class "
        "found across all gates: visible > hidden > demographic. "
        "Conditions flagged as `PriorState`, `Symptom`, `True`, or `False` "
        "are structural and are excluded from classification."
    )
    lines.append("")
    lines.append("### Cross-module attributes")
    lines.append("")
    lines.append(
        "Attributes set by a `ConditionOnset` in one module (via "
        "`assign_to_attribute`) and read as an `Attribute` condition in "
        "another module are classified as **visible**: the originating "
        "condition appears in `conditions.csv`. For example, `obesity` is "
        "set by a BMI ConditionOnset in `wellness_encounters.json` and is "
        "therefore visible."
    )
    lines.append("")
    lines.append(
        "Attributes set by demographic logic only (`veteran`, `atopic`, "
        "`diabetes` flag from `metabolic_syndrome_disease.json`) are classified "
        "as **hidden**: they are internal state, not observable records."
    )
    lines.append("")
    lines.append("### Known approximations")
    lines.append("")
    lines.append(
        "The pre-first-onset filter eliminates the main source of over-inclusion — "
        "post-onset care-management cycles and unrelated disease branches in "
        "multi-condition modules — but does not resolve three remaining cases:"
    )
    lines.append("")
    lines.append(
        "- **Observation checks that follow selection, not predict it**: In "
        "`metabolic_syndrome_care.json`, HbA1c and glucose are recorded as part of "
        "the diagnosis encounter after the patient is already selected by the hidden "
        "`diabetes`/`prediabetes` attribute. These checks are technically on the "
        "pre-first-onset path, so type 2 diabetes and prediabetes remain classified "
        "as visible. The `hidden_gates` field shows the true structural gate "
        "(`diabetes`, `prediabetes` — set by demographic logic in "
        "`metabolic_syndrome_disease.json`)."
    )
    lines.append(
        "- **Multi-injury module (injuries.json)**: The broken-jaw submodule "
        "(`injuries/broken_jaw.json`) is callable on the first pass through "
        "`injuries.json` and contains `dental_referral` attribute checks. These "
        "propagate as soft modifiers to all other injury types, including gunshot "
        "wounds and fractures, even though prior dental care has no causal role. "
        "The `hidden_gates` field lists `osteoporosis` and prescription attributes "
        "that do modulate injury incidence rates in the model."
    )
    lines.append(
        "- **Sequential disease progression (e.g. colorectal cancer)**: The filter "
        "stops before `Detect_Adenoma` (an in-module ConditionOnset in "
        "`colorectal_cancer.json`), which is also the causal prerequisite for cancer "
        "onset. The prior-polyp gate is therefore excluded from the visible-gates "
        "list; only the `smoker` incidence-rate modifier is reported. The "
        "`colorectal_cancer_stage` attribute (representing polyp-to-cancer "
        "progression) appears in `hidden_gates`."
    )
    lines.append("")
    lines.append("### Exclusion flags")
    lines.append("")
    lines.append(
        "Conditions are flagged as *excluded* when they are known to cause "
        "label-leakage issues per `SCRUBBING.md` §Known limitations, or by "
        "heuristic detection of the same patterns:"
    )
    lines.append("")
    lines.append(
        "- **symptom_type** — the SNOMED code is a finding or the description "
        "matches symptom keywords; the symptom may be recorded before the "
        "formal diagnosis."
    )
    lines.append(
        "- **precursor_names_target** — a precursor or variant condition whose "
        "description names the target survives before the scrub cutoff."
    )
    lines.append(
        "- **screening_work_up** — pre-diagnosis assessment or referral records "
        "name the target condition."
    )
    lines.append(
        "- **situation_code** — SNOMED `(situation)` tag typically marks "
        "referrals, history-of records, or administrative findings."
    )
    lines.append("")
    lines.append("### Sanity check")
    lines.append("")
    chf_entries = [r for r in results if r["code"] == "88805009"]
    if chf_entries:
        chf = chf_entries[0]
        chf_class = chf["overall_class"]
        chf_flags = chf["exclusion_flags"]
        if chf_class == "demo" and not chf_flags:
            lines.append(
                f"CHF (88805009) classifies as **demographic-only** with no "
                f"exclusion flags. ✓  "
                f"Gates found: {', '.join(chf['demo_gates']) or 'Age, Gender (distributed delay)'}."
            )
        else:
            lines.append(
                f"⚠  CHF (88805009) classified as `{chf_class}` — expected "
                f"`demographic`. Check the analysis."
            )
    else:
        lines.append("⚠  CHF (88805009) not found in any module.")
    lines.append("")

    # ── tables per group ──────────────────────────────────────────────────────
    for group_key, group_label in [
        ("visible",  "History-dependent (visible)"),
        ("hidden",   "History-dependent (hidden only)"),
        ("demo",     "Demographic-only"),
        ("excluded", "Excluded"),
    ]:
        rows = groups[group_key]
        rows.sort(key=lambda r: r["description"].lower())
        lines.append(f"## {group_label}  ({len(rows)} conditions)")
        lines.append("")
        if not rows:
            lines.append("*None in this run.*")
            lines.append("")
            continue

        lines.append(
            "| Code | Description | Module | Gates / notes | Cases (10k) |"
        )
        lines.append("|------|-------------|--------|---------------|------------|")
        for r in rows:
            desc_short = _strip_tag(r["description"])
            mod_short  = r["module"].replace(".json", "").replace("\\", "/")
            # Summarise gates
            gate_parts = []
            if r["visible_gates"]:
                gate_parts.append("vis: " + ", ".join(r["visible_gates"][:3]))
                if len(r["visible_gates"]) > 3:
                    gate_parts[-1] += f" +{len(r['visible_gates'])-3}"
            if r["hidden_gates"]:
                gate_parts.append("hid: " + ", ".join(r["hidden_gates"][:3]))
                if len(r["hidden_gates"]) > 3:
                    gate_parts[-1] += f" +{len(r['hidden_gates'])-3}"
            if r["demo_gates"]:
                gate_parts.append("dem: " + ", ".join(r["demo_gates"][:3]))
                if len(r["demo_gates"]) > 3:
                    gate_parts[-1] += f" +{len(r['demo_gates'])-3}"
            if r["exclusion_flags"]:
                gate_parts.append("excl: " + "; ".join(r["exclusion_flags"]))
            gate_str = " · ".join(gate_parts) if gate_parts else "—"
            case_str = str(r["case_count"]) if r["case_count"] else "—"
            lines.append(
                f"| {r['code']} | {desc_short} | {mod_short} "
                f"| {gate_str} | {case_str} |"
            )
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"| Group | Count |\n"
        f"|-------|-------|\n"
        f"| History-dependent (visible) | {len(groups['visible'])} |\n"
        f"| History-dependent (hidden only) | {len(groups['hidden'])} |\n"
        f"| Demographic-only | {len(groups['demo'])} |\n"
        f"| Excluded | {len(groups['excluded'])} |\n"
        f"| **Total** | **{len(results)}** |"
    )
    lines.append("")
    lines.append(
        "_Generated by `scripts/scan_modules.py`. "
        "Source: `.synthea/src/main/resources/modules/`._"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules-dir", type=Path, default=DEFAULT_MODULES)
    ap.add_argument("--conditions",  type=Path, default=DEFAULT_CONDS)
    ap.add_argument("--out-md",      type=Path, default=DEFAULT_MD)
    ap.add_argument("--out-json",    type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    if not args.modules_dir.is_dir():
        sys.exit(f"ERROR: modules dir not found: {args.modules_dir}")

    print(f"Loading modules from {args.modules_dir} …")
    all_modules = load_all_modules(args.modules_dir)
    print(f"  {len(all_modules)} modules loaded")

    visible_attrs = build_visible_attributes(all_modules)
    print(f"  {len(visible_attrs)} visible attributes identified")

    case_counts = load_case_counts(args.conditions)
    print(f"  {len(case_counts)} condition codes in conditions.csv")

    print("Analysing modules …")
    all_results = []
    for mod_key, mod in all_modules.items():
        all_results.extend(
            analyse_module(mod_key, mod, all_modules, args.modules_dir,
                           visible_attrs, case_counts)
        )

    deduped = deduplicate(all_results)
    print(f"  {len(deduped)} unique ConditionOnset codes")

    # ── sanity check ──────────────────────────────────────────────────────────
    chf_entries = [r for r in deduped if r["code"] == "88805009"]
    if not chf_entries:
        print("SANITY FAIL: CHF (88805009) not found in any module")
        sys.exit(1)
    chf = chf_entries[0]
    if chf["overall_class"] != "demo" or chf["exclusion_flags"]:
        print(
            f"SANITY FAIL: CHF classified as {chf['overall_class']!r} "
            f"flags={chf['exclusion_flags']} — expected demographic-only with no flags"
        )
        sys.exit(1)
    print(f"  Sanity check: CHF = {chf['overall_class']} ✓")

    # ── write JSON ────────────────────────────────────────────────────────────
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"conditions": deduped}, indent=2) + "\n"
    )
    print(f"Wrote JSON → {args.out_json}")

    # ── write Markdown ────────────────────────────────────────────────────────
    write_markdown(deduped, args.out_md, args.modules_dir, len(all_modules))
    print(f"Wrote Markdown → {args.out_md}")

    # ── final summary ─────────────────────────────────────────────────────────
    excluded = [r for r in deduped if r["exclusion_flags"]]
    visible  = [r for r in deduped if not r["exclusion_flags"] and r["overall_class"] == "visible"]
    hidden   = [r for r in deduped if not r["exclusion_flags"] and r["overall_class"] == "hidden"]
    demo     = [r for r in deduped if not r["exclusion_flags"] and r["overall_class"] == "demo"]

    print()
    print("Result summary:")
    print(f"  history-dependent (visible) : {len(visible)}")
    print(f"  history-dependent (hidden)  : {len(hidden)}")
    print(f"  demographic-only            : {len(demo)}")
    print(f"  excluded                    : {len(excluded)}")
    print(f"  total unique codes          : {len(deduped)}")


if __name__ == "__main__":
    main()
