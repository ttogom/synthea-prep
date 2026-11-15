# Regenerating the report tables

Run from `synthea-prep` with the `medcopilot` environment active:

```bash
# Once saved treatment-plan judging results are ready, refresh the experiment CSV.
python scripts/analysis/build_analysis_dataset.py

# Compute the three downstream statistics CSVs.
python scripts/analysis/summarize_experiment_results.py --csv-only

# Render all three tables and update their copies in the report's figures folder.
python scripts/analysis/render_tables.py
```

The default run is `experiments/output/2026-11-12_022701PST__test_so__main`.
The export step reads saved results; none of these commands runs model inference
or judging. `results.jsonl` and `run.json` are read only.

Each table has a matching statistics CSV under `analysis/`, a generated `.tex`
under `analysis/tables/`, and a report copy under
`CSCI-566-Midterm-Report/figures/`:

| Name | Contents |
| --- | --- |
| `main_results` | Four configuration means, mean confidence intervals, and baseline-improvement stars for each generator. |
| `generator_retrieval_gains` | Similar-patient and guideline gains over no retrieval, including equal-generator averages. |
| `group_retrieval_gains` | Raw no-retrieval means and retrieval deltas for History, Acute, and Control. |

For a formatting-only change, rerun the render command. To regenerate one table:

```bash
python scripts/analysis/render_tables.py --table main_results
python scripts/analysis/render_tables.py --table generator_retrieval_gains
python scripts/analysis/render_tables.py --table group_retrieval_gains
```

Rendering consumes the saved CSVs; it does not recompute statistical tests.
The default updates report copies and touches report `main.tex` so an enabled
preview watcher recompiles it. Use `--analysis-only` to write only experiment
artifacts. The scripts do not commit or push changes.

For another run, pass `--results-csv /path/to/experiment_results.csv` to the summary
script and `--analysis-dir /path/to/analysis` to the renderer. An optional
`--report-dir /path/to/report/figures` selects another report destination.
Running the summary script without `--csv-only` also renders the experiment
artifacts; report synchronization remains an explicit render command.

## Code map

- `build_analysis_dataset.py`: saved experiment JSONL and labels → experiment CSV.
- `summarize_experiment_results.py`: validated rank-1 scores, eligibility and judge
  reporting policy → the three statistics CSVs. Previously committed diagnosis-only
  helper APIs remain available, but the CLI generates only the selected tables.
- `table_statistics.py`: means, patient-cluster confidence intervals, paired tests,
  and multiplicity correction.
- `table_rendering.py`: all three LaTeX layouts, captions, rounding, bolding, and stars.
- `render_tables.py`: CSV-to-TeX export, report synchronization, and export provenance.

## Statistical and provenance details

Means give equal weight to condition examples. Repeated examples from one patient
stay together for uncertainty estimation. Stars use one-sided paired tests with
CR1 standard errors clustered by patient and Student-t degrees of freedom equal
to the number of patient clusters minus one. Display rounding does not determine
significance or the best configuration.

Holm families are separate for each outcome:

- Main results: three retrieval comparisons within each generator.
- Generator gains: both single-source configurations across all generators and
  their average (10 tests for the current four generators). Average scores weight
  the fixed generators equally within each case before clustering by patient.
- Group gains: nine comparisons (three retrieval configurations × three groups)
  within each generator. These test improvement within groups, not differences
  between group gains.

`main_results.csv` and `generator_retrieval_gains.csv` also save **exploratory
two-sided checks** for diagnosis and treatment scores. These were added after
observing decreases and test equality with No retrieval against a difference
in either direction. They reuse the same paired CR1 statistics and degrees of
freedom, with `p_value_two_sided = 2 * Student-t.sf(abs(t), df)`. Holm correction
retains each table's full family listed above, including positive and negative
effects and the generator-average comparisons; it is not restricted to the two
declines that motivated the checks. Unavailable tests retain their family place.

The supplementary columns are `p_value_two_sided`, `p_value_two_sided_holm`, and
`significant_two_sided_{raw,holm}_{0_01,0_05}`. Primary `p_value`, `p_value_holm`,
and `stars` remain one-sided improvement results. The exploratory checks do not
alter any table markers. The normal CSV-only regeneration command updates both
sets of tests, including treatment checks when eligible scores are complete.

The main-results 95% intervals describe absolute means and use a seeded patient
cluster bootstrap. Incomplete eligible arms or comparisons remain unreported;
missing judgments are never zero scores. Unavailable Plan references are excluded.
Plan History currently covers hypertension only because COPD reference plans are
unavailable. Experimental, unvalidated judge scores remain excluded.

`analysis/table_statistics_manifest.json` records source CSV and run-metadata
fingerprints, statistical settings, code hashes, and downstream CSV hashes.
`analysis/table_exports_manifest.json` connects each input CSV to its generated
TeX, report-copy hash, and renderer code. Together they provide the provenance
chain from the experiment export to the report tables.

## Offline checks

```bash
python tests/test_table_statistics.py
python tests/test_table_rendering.py
python tests/test_table_pipeline.py
```

These include independent statistical reference checks, incomplete/complete Plan
exports, report synchronization, malformed-input handling, and provenance checks.
