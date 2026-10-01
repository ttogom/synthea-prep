# Labels and Clean Training Data

`scripts/extract_labels.py` takes a scrub output directory and produces:

1. **`labels-<sha6>.json`** (sibling of the scrub dir) — per-patient ground truth:
   ```json
   { "patient-uuid": {"code": "88805009",
                      "description": "Chronic congestive heart failure (disorder)",
                      "cutoff": "2018-03-14"}, ... }
   ```
   JSON is used so the file is keyed by patient ID without parsing all rows, and
   description strings (which can contain commas) need no quoting.

2. **`<source-run>__train-<sha6>/`** (sibling of the scrub dir) — a copy of the
   scrub output with `manifest.json` and `SCRUB_SUMMARY.md` removed. Both files
   name the target condition and its code. The training directory name contains
   only the opaque sha6, not the condition name.

After building the training directory the script scans every file in it for the
target SNOMED code and description and reports any hits. See
[SCRUBBING.md](SCRUBBING.md) §Known limitations for categories of target text
that survive before the cutoff (related codes, symptoms named as text).

## Quick start

```bash
python3 scripts/extract_labels.py data/<run>__scrub-<stem>-<sha6>
# writes data/labels-<sha6>.json
#        data/<run>__train-<sha6>/
```

## Labels source

Labels come from the **source run's** `conditions.csv`, not the scrubbed output
(the diagnosis row is absent there). Cutoffs are recomputed independently using
the same rule as `scrub.py`: for each target condition row, the cutoff is the
earlier of the row's `START` and the local date of the linked encounter's
`START`; rows with the target as `REASONCODE` can move the cutoff earlier.

## What is excluded from the training directory

| File | Reason |
|------|--------|
| `manifest.json` | Lists target SNOMED codes and their sha6 |
| `SCRUB_SUMMARY.md` | Names target codes in the Parameters table heading |

`csv/`, `symptoms/`, and `notes/` are copied unchanged from the scrub output.
The labels file itself lives next to the training directory, not inside it.

## CHF cohort (pop1000, code 88805009)

Scrubbed on 2026-09-28 with `configs/heart_failure.txt` (code `88805009`,
sha6 `9e8474`). Key figures:

| Metric | Value |
|--------|-------|
| Patients in source | 1,130 |
| Patients emitted (CHF+, pre-dx records) | 34 |
| Patients excluded | 0 |
| Patients cut earlier by REASONCODE | 0 |
| Patients with empty pre-dx history | 0 |
| Patients with symptoms rows | 34/34 |
| Median prior encounters | 36 |
| Median years of history | 38.4 |
| Min years of history | 5.7 (1 patient) |
| Leak scan | 0 hits |

All 34 patients have at least 11 prior encounters and 5.7 years of history.
The one patient under 10 years (22 years old at cutoff, 5.7 years, 11 encounters)
has a meaningful pre-diagnosis record despite the short window.

**Suggested minimum-history threshold:** ≥ 5 encounters (all 34 pass) or
≥ 2 years (all 34 pass). If the downstream task needs richer context, ≥ 10
years cuts to 33/34 and ≥ 10 encounters cuts to 34/34. Revisit on the 10k run.
