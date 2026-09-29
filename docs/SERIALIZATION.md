# Serialization: Training Data to Free Text

`serialization/serialize.py` turns a clean training directory (from
[`extract_labels.py`](LABELS.md)) into one free-text summary per patient for an
LLM. `serialization/check_text_leaks.py` then checks that text for anything
that names or hints at the diagnosis.

```
Synthea run ──scrub.py──▶ __scrub-… ──extract_labels.py──▶ __train-<sha6> + labels
            ──serialize.py──▶ <name>.jsonl ──check_text_leaks.py──▶ <name>.leaks.md
```

## Quick start

```bash
# Heart failure, 10k (after scrub.py and extract_labels.py)
python3 serialization/serialize.py \
    data/pop10000-seed20260916__train-9e8474 \
    data/pop10000-seed20260916__labels-9e8474.json \
    --out data/serialized/pop10000-chf.jsonl

python3 serialization/check_text_leaks.py \
    data/serialized/pop10000-chf.jsonl \
    data/pop10000-seed20260916__train-9e8474 \
    --terms configs/leak_terms/heart_failure.txt
```

Python 3.9+, standard library only.

> `extract_labels.py` names the labels file `labels-<sha6>.json` by code set
> only, so a 1k and a 10k run of the same condition collide. Pass
> `--labels data/<run>__labels-<sha6>.json` to keep them apart.

## Any condition, one command

Nothing in `serialize.py` or `check_text_leaks.py` is specific to one
condition: the target comes from each patient's label. For a new condition,
write a codes file and run the whole pipeline:

```bash
printf '195967001   # Asthma (disorder)\n' > configs/asthma.txt
serialization/run_condition.sh data/pop1000-seed20260916 configs/asthma.txt
# → data/serialized/pop1000-seed20260916-asthma.jsonl  (+ .stats.json, .leaks.md)
```

`run_condition.sh` runs scrub → labels → serialize → text leak check. It
passes any extra options to `serialize.py`, and it picks up
`configs/leak_terms/<codes-stem>.txt` if one exists. Condition-specific
knowledge lives only in that optional terms file.

Tried on pop1000:

| Condition | Patients | Leak check |
|---|---|---|
| Essential hypertension | 250 | PASS |
| Asthma | 46 | FAIL: 43 patients keep "Childhood asthma", matching SCRUBBING.md's ≥43 |

For a multi-condition dataset, run each condition and concatenate the JSONL
files. A patient can then appear once per condition, each time with its own
cutoff.

## Output

`<name>.jsonl`, one line per patient:

```json
{"patient_id": "…", "cutoff": "2014-02-14",
 "label": {"code": "88805009", "description": "Chronic congestive heart failure (disorder)"},
 "text": "PATIENT\n- 58-year-old male, …", "chars": 9720, "est_tokens": 2430}
```

The label is only in `label`, never in `text`. `<name>.stats.json` records the
settings and token statistics.

## What the text contains

Sections, in order (empty sections are omitted):

| Section | Coverage | Compression |
|---|---|---|
| PATIENT | Age at cutoff, sex, race, ethnicity, marital status, latest social items (smoking, employment, education, housing, stress, income) | Latest answer only |
| CHIEF COMPLAINTS AT VISITS | From notes, within the window | One line per complaint with count |
| ACTIVE / PAST PROBLEMS | `(disorder)` conditions, whole history | One line per condition with count and dates |
| OTHER FINDINGS | Other conditions (`(finding)`, `(situation)`), whole history | One line each |
| ALLERGIES, ACTIVE CARE PLANS, DEVICES IN USE | Active at cutoff | — |
| CURRENT / PAST MEDICATIONS | Whole history, with reason | One line per drug with count |
| VITALS AND LABS | Within the window | Latest `--max-values` readings per test |
| PROCEDURES | Within the window | One line per procedure with count |
| IMMUNIZATIONS | Whole history | One line per vaccine with count |
| RECENT VISITS | Within the window | Identical visits collapsed; newest `--max-encounters` |

Always left out:
- **Identifying data:** names, address, SSN and IDs. Note text is used only for chief complaints, because note bodies repeat the patient's name.
- **Synthea's burden scores** (DALY, QALY, QOLS). They're simulation output, not clinical data, and they were among the leak probe's top features in [LEAK_REPORT_10k.md](LEAK_REPORT_10k.md).
- **Yes/no screening questions** (PRAPARE survey items).
- **`symptoms.csv`.** Its age-based rows can pass the cutoff (SCRUBBING.md §Known limitations). Chief complaints from notes carry the same information with dates.

Dates are written relative to the cutoff ("3 years ago"), so the text never
states the diagnosis date.

## Compression options

| Option | Default | Meaning |
|---|---|---|
| `--window-years` | 5 | Window for visits, procedures, vitals/labs, complaints |
| `--window-from` | `last-visit` | Window ends at the patient's last visit, or at `cutoff` |
| `--max-values` | 3 | Readings kept per vital/lab test |
| `--max-encounters` | 15 | Visit lines kept |
| `--min-encounters` | 0 | Skip patients with fewer pre-cutoff visits |
| `--no-compress` | off | Full history, no caps (for comparison) |

The window ends at the last visit because some patients have no visits shortly
before diagnosis: 270 of 848 type 2 diabetes patients (10k) have none in the 5
years before, so a cutoff-anchored window would be empty.

Token counts assume 4 characters per token until a tokenizer is chosen.

## Leak check on the text

`check_text_leaks.py` checks each patient's text against that patient's label.
It exits with code 1 if any check fails.

| Level | Check |
|---|---|
| FAIL | Target code; full description; all distinctive description words on one line; DALY/QALY/QOLS; the patient's name |
| REVIEW | A two-word piece of the description; any term in `--terms` |

`configs/leak_terms/<condition>.txt` lists condition-specific hints: tests,
questionnaires, drugs and symptoms mostly used for that condition. REVIEW hits
are not failures. They need a human decision.

## Results (2026-09-29)

| Dataset | Patients | Est. tokens median / p90 / max | Leak check |
|---|---|---|---|
| pop1000 heart failure | 34 | 2,412 / 4,300 / 6,090 | PASS, 3 to review |
| pop10000 heart failure | 313 | 2,430 / 4,300 / 6,265 | PASS, 6 to review |
| pop10000 heart failure, `--no-compress` | 313 | 3,221 / 9,740 / 60,148 | — |
| pop10000 type 2 diabetes | 848 | 352 / 2,413 / 5,568 | FAIL (expected, see below) |

Findings for the team:

1. **Some heart-failure-related tests appear before the cutoff, but they are not a leak.** 22 of the 313 heart failure patients (10k) have at least one of these before diagnosis:
   - left ventricular ejection fraction (11 patients)
   - NT-proBNP (12 patients)
   - NYHA functional class (2 patients)
   - KCCQ-12 (2 patients)

   `check_leaks.py` doesn't flag them because they don't name the target. In Synthea they come from the work-up for aortic valve and bypass surgery, not from a heart failure diagnosis. 592 patients who never get heart failure have them too, so the rate is similar in both groups:

   | Group | With these tests |
   |---|---|
   | Heart failure patients, before diagnosis | 7% |
   | Patients who never get heart failure, at any time | 5% |

   Their values are mostly normal (for example NT-proBNP 1.3 pg/mL). They are kept as genuine history. If a later condition has a work-up that *is* specific to it, drop those tests or cut at them there.
2. **Type 2 diabetes fails as expected.** 12 patients keep "Microalbuminuria/Proteinuria due to type 2 diabetes mellitus", the known related-code limitation in SCRUBBING.md. The check catches it, which confirms the text check works.
3. **Type 2 diabetes histories are thin.** The median is 8 pre-diagnosis visits (heart failure: 36), and 198 of 848 patients have fewer than 5. `--min-encounters 5` would drop them.
4. **The presenting symptoms are missing.** As CHF_FINDINGS.md notes, CHF symptoms are scrubbed with the diagnosis. Chief complaints in the text come from earlier, unrelated visits.

## Not done yet

- **No negatives.** Every patient is a positive, as in the scrub. Matched negatives need a cutoff rule first (SCRUBBING.md D9).
- **Tokenizer.** Token counts are estimates until the model is chosen.
- **Probe on text.** `check_leaks.py`'s classifier probe runs on the CSVs. The same probe on serialized text would also cover notes.
