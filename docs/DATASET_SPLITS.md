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

## "None of the above" (NOTA) examples — release-v2.1

Five additional files in `data/release-v2.1/` extend the dataset with negative
examples for the "none of the above" class:

| File | Purpose | Format |
|------|---------|--------|
| `test_nota.jsonl` | NOTA test records — text only | `{"id", "text"}` |
| `dev_nota.jsonl` | NOTA dev records | `{"id", "text"}` |
| `corpus_nota.jsonl` | NOTA corpus records — text + DIAGNOSIS + TREATMENT | `{"id", "text"}` |
| `test_nota_labels.jsonl` | NOTA test answers | `{"id", "condition", "bin", "has_presenting_evidence", "plan", "source_condition"}` |
| `dev_nota_labels.jsonl` | NOTA dev answers | `{"id", "condition", "bin", "has_presenting_evidence", "plan", "source_condition"}` |

**Source conditions** (3 non-target acute conditions):

| Condition | SNOMED | test | dev | corpus | presenting |
|-----------|--------|------|-----|--------|------------|
| acute_bronchitis | 10509002 | 200 | 83 | 842 | 100% |
| viral_sinusitis  | 444814009 | 200 | 83 | 1326 | 99% |
| otitis_media     | 65363002  | 200 | 83 | 164 | 0% |
| **Total**        | | **600** | **249** | **2332** | |

`condition` is always `"none_of_the_above"`. `bin` is always `"none"`.
`source_condition` is in the labels file only — never in the record text.

**ID format**: `nota_{source_condition}_{patient_id}`.
The source condition name is embedded in the ID. Evaluation code that reads
IDs directly would leak the source condition; use only the `text` field for
inference and the `condition` field from labels for evaluation.

**Split assignment**: existing v2 patients keep their split (test→test,
dev→dev, corpus→corpus_nota). New patients are assigned to test or dev at a
50:20 ratio (seed 42); none go to corpus. The existing splits are unchanged —
`test.jsonl` and `test_nota.jsonl` are separate files; combine them for the
full evaluation set.

**Known NOTA-specific limitations**:
- **Otitis media has no Synthea symptom module**: 0% of otitis_media records
  have a PRESENTING COMPLAINT section. All 447 test+dev otitis_media records
  rely on history alone.
- **Source condition name appears in historical visit text** (~1.5% of
  acute_bronchitis records, ~0.7% of viral_sinusitis records): Synthea
  sometimes records a prior encounter for the same condition before the formal
  diagnosis date (e.g., a pre-diagnosis urgentcare visit coded with
  `REASONCODE=10509002`). The condition name then appears in the RECENT VISITS
  section. Otitis_media is clean (0% leaks).
- **NOTA TREATMENT empty in ~61% of test/dev records**: acute bronchitis and
  viral sinusitis have few REASONCODE-matched prescriptions; otitis_media has
  no Synthea prescription module.

---

## Known limitations

- **Chief complaints removed**: the CHIEF COMPLAINTS history section was dropped
  because it repeated at least one presenting symptom verbatim in ~10% of
  records (56% for COPD), creating double signal. Presenting evidence now appears
  only in the PRESENTING COMPLAINT section.
- **Corpus TREATMENT empty in 40% of records**: clinically expected for viral
  pharyngitis (Synthea generates no prescription for most cases) and for bacterial
  sinusitis (Synthea codes ~half its antibiotic prescriptions under
  `REASONCODE=444814009` Viral sinusitis rather than the bacterial sinusitis code,
  so they don't match). Records with no matched items show `- none recorded`.
- **Cystitis results are lower confidence**: Synthea links cystitis conditions to
  encounters that started 1–1000 days before the condition was formally recorded,
  pulling the serialization cutoff earlier than the diagnosis date. The presenting
  symptoms are correct, but the history window may exclude events from the
  diagnosis day.
- **Presenting complaints are near-fixed templates**: Synthea assigns the same
  small set of symptoms to every episode of a condition, so a model that sees the
  PRESENTING COMPLAINT section can identify most conditions by template-matching.
  Strep throat vs. viral pharyngitis is the main hard case (overlapping symptoms).
- **10-year export window**: Synthea's `exporter.years_of_history = 10` excludes
  observations older than 10 years before the reference date. COPD (62%) and
  hypertension (71%) positives have no presenting evidence because their diagnoses
  predate the window. Evaluate those conditions on the with-presenting-evidence
  subset as well as the full split.
- **Class imbalance**: viral pharyngitis (43%) and hypertension (23%) dominate;
  use `class_weight=balanced` or equivalent.
- **Synthea record-shape shortcuts**: balanced accuracy of 33.6% is achievable
  from age, sex, and encounter count alone. Evaluate any model against this floor.
- **Lauren's serialization fixes not included**: the `serialization-update`
  branch's "days/years ago" wording fix and heart-failure presenting-symptom fix
  are not merged into main. Check `manifest.json`
  `lauren_serialization_fixes_included`.
