# Dataset Splits (release-v1)

All files are in `data/release-v1/`. The directory is gitignored; regenerate
with the command in the Reproduce section below.

---

## Files

| File | Purpose | Format |
|------|---------|--------|
| `index.csv` | One row per (patient, condition) example; includes eligibility flags, split assignment can be derived by joining with split output | CSV |
| `test.jsonl` | Test records — text only, no answer | `{"id", "text"}` |
| `test_labels.jsonl` | Test answers — one row per test record | `{"id", "condition", "bin", "has_presenting_evidence"}` |
| `dev.jsonl` | Development records — same format as test | `{"id", "text"}` |
| `dev_labels.jsonl` | Dev answers | `{"id", "condition", "bin", "has_presenting_evidence"}` |
| `corpus.jsonl` | Corpus records — text + DIAGNOSIS + TREATMENT | `{"id", "text"}` |
| `samples/test_samples.md` | 3 test records per condition with labels (for human review) | Markdown |
| `samples/corpus_samples.md` | 2 corpus records per condition | Markdown |
| `leak_reports/` | Per-condition text leak check reports for test and dev | Markdown |
| `manifest.json` | Seed, source run, commit, Lauren's fix status, counts | JSON |
| `../release-v1.zip` | Packaged archive of the above (no intermediate scrub dirs) | ZIP |

### `test.jsonl` / `dev.jsonl`

One JSON object per line. `id` is `{condition}_{patient_id}`. `text` is the
full serialized record (history + PRESENTING COMPLAINT where available),
with the diagnosis removed. Nothing in `text` names or hints at the
target condition. Verify with `serialization/check_text_leaks.py`.

### `test_labels.jsonl` / `dev_labels.jsonl`

Same `id` as the corresponding text file. `condition` is the snake_case
condition name matching `configs/bins/`. `bin` is one of `history`,
`acute`, `control`. `has_presenting_evidence` is `true` if the record
includes a PRESENTING COMPLAINT section.

### `corpus.jsonl`

Same serialized record as test/dev, with two additional sections appended:

```
DIAGNOSIS
- <condition description>

TREATMENT
- <medication or procedure or care plan started at the diagnosing encounter>
- ...
```

Corpus records are for fine-tuning and SOAP-conversion tasks (see
`docs/SOAP_CONVERSION.md`). They must not be used as test or dev examples.

### `index.csv` columns

| Column | Description |
|--------|-------------|
| `example_id` | `{condition}_{patient_id}` |
| `patient_id` | Synthea UUID |
| `condition` | snake_case condition name |
| `bin` | `history` \| `acute` \| `control` |
| `cutoff_date` | Diagnosis date (YYYY-MM-DD) |
| `n_prior_encounters` | Encounters in the scrubbed train directory (pre-cutoff) |
| `has_presenting_evidence` | Whether `data/presenting/<condition>.json` has an entry for this patient |
| `ineligible` | Empty if eligible; otherwise `same_day_multi_diagnosis` or `too_few_encounters:<n>` |

---

## Split rules

- **Unit**: patient. A patient with multiple conditions contributes all their
  examples to the same split.
- **Eligibility filters** (applied before splitting):
  - Patient must have ≥ 5 prior encounters.
  - Patient must not have two target-condition diagnoses on the same calendar day.
- **Proportions**: 50% test | 20% dev | 30% corpus, stratified by each
  patient's rarest target condition (fewest total eligible examples in the
  dataset). Fixed seed 42.
- **Guarantee**: no patient appears in more than one split.

---

## Reproduce

```bash
# Ensure presenting files are current (run from repo root):
SRC=data/pop10000-seed20260916
for COND_SHA in copd-465e92 hypertension-249aea strep_throat-1059ea \
                viral_pharyngitis-5ace2f bacterial_sinusitis-c4404e \
                cystitis-c91f0b heart_failure-9e8474; do
    COND=${COND_SHA%-*}; SHA=${COND_SHA##*-}
    python3 scripts/extract_presenting.py \
        data/${SRC##*/}__scrub-${COND_SHA} ${SRC} data/labels-${SHA}.json
done

# Build the release (takes ~10 min — serialize.py runs 7 times):
python3 scripts/build_release.py

# Rerun without re-serializing (reuse existing _tmp/*.jsonl):
python3 scripts/build_release.py --skip-serialize
```

The split is fully determined by `--seed` (default 42) and the labels files.
The same command on the same commit produces byte-identical output.

---

## Known limitations

- **Hypertension and COPD**: 71% and 62% of positives respectively have no
  presenting evidence, because Synthea's 10-year export window excludes older
  diagnoses. Evaluate these conditions on both the full split and the
  with-presenting-evidence subset. See `docs/CONDITION_SELECTION.md` (Export
  window section) and `docs/BINS.md` for details.
- **Class imbalance**: viral pharyngitis (43%) and hypertension (23%) dominate.
  Use `class_weight=balanced` or equivalent.
- **Synthea artifacts**: record-shape shortcuts exist (balanced accuracy 33.6%
  from age, sex, and encounter count alone). Any model must be evaluated against
  this floor.
- **CHIEF COMPLAINTS section removed (v1 deliberate choice)**: Synthea populates
  the note chief complaints from the same symptom model that drives the
  presenting complaint. In a 56% of COPD records and ~10% of acute-condition
  records with presenting evidence, at least one presenting symptom also
  appeared verbatim in the windowed CHIEF COMPLAINTS history. To ensure
  presenting evidence appears in exactly one place, the CHIEF COMPLAINTS section
  (generated from note chief complaints) is omitted from all records. A future
  version could re-introduce it after excluding symptoms that match the label.
- **Lauren's serialization fixes**: check `manifest.json`
  `lauren_serialization_fixes_included` to see whether the `--no-history`
  ablation flag and heart-failure presenting-symptom fix from the
  `serialization-update` branch were available when this release was built.
