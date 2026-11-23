# Regenerating the report tables

## Combined seven-variant results

Run from the repository root with the dependencies in
[requirements.txt](../../requirements.txt) installed. The default command-line
workflow builds `experiments/results_all_variants/`:

```bash
python scripts/analysis/build_analysis_dataset.py
python scripts/analysis/summarize_experiment_results.py --csv-only
python scripts/analysis/render_tables.py --analysis-only
```

The committed `sources.yaml` identifies the inputs; the dataset README documents
its columns and statistical rules. The builder reads the two original experiment
JSONLs and the seven per-run `plan_judging_v2/` exports. Protocol-1 treatment
evaluations and the old experiment CSV are excluded. All three ranked predictions
and diagnosis metrics are exported; treatment judging covers rank 1 only.

`main_summary_stats.csv` replaces the main-results filename in this workflow.
Generator gains contain **only** equal-generator averages: all four generator
scores in both arms are required per record before paired patient-clustered tests.
Missing scores are excluded from means and paired tests, while genuine zeros are
retained. The new `coverage_and_failures.csv` exposes statuses, denominators and
failure reasons. Details and correction families are in the dataset
README. No new provenance manifests or midterm report copies are written.

Combined CSVs explicitly label p-values with `one_sided` or `two_sided` before
the optional `_holm` suffix, including `history_contrast_p_value_one_sided_holm`.
The historical export names described below retain their original schema.

For the six direct, two-sided agentic-versus-non-agentic comparisons, run:

```bash
python scripts/analysis/compare_agentic_variants.py
```

This writes only `analysis/agentic_vs_nonagentic_significance.csv` under the
combined results directory. It averages all four generators within each record,
uses matched records and patient-clustered tests, and applies Holm across all six
comparisons. See the [dataset README](../../experiments/results_all_variants/README.md#agentic-versus-non-agentic-significance)
for missing-score handling and column definitions.

The historical four-variant APIs retain their original defaults for reproducibility.
The instructions below describe that earlier workflow and its saved artifacts;
they should not be used to mix protocol-1 scores into the combined analysis.

## Historical four-variant workflow

The earlier APIs remain available for explicitly selected historical exports.
The dataset builder now targets the seven-variant comparison; it does not rebuild
the old protocol-1 CSV. To refresh statistics/tables from an existing old export,
run from the repository root:

```bash
# Compute the three downstream statistics CSVs.
python scripts/analysis/summarize_experiment_results.py --results-csv experiments/output/2026-11-12_022701PST__test_so__main/experiment_results.csv --csv-only

# Render all four tables and update their copies in the report's figures folder.
python scripts/analysis/render_tables.py --analysis-dir experiments/output/2026-11-12_022701PST__test_so__main/analysis
```

These commands select the saved November 12 export explicitly; none runs model
inference or judging. `results.jsonl` and `run.json` are read only.

The three statistics CSVs under `analysis/` supply four tables. Each table has a
generated `.tex` under `analysis/tables/` and a report copy under
`CSCI-566-Midterm-Report/figures/`. The History contrast table shares the group
statistics CSV instead of duplicating its data:

| Name | Contents |
| --- | --- |
| `main_results` | Four configuration means, mean confidence intervals, and baseline-improvement stars for each generator. |
| `generator_retrieval_gains` | Similar-patient and guideline gains over no retrieval, including equal-generator averages. |
| `group_retrieval_gains` | Raw no-retrieval means and retrieval deltas for History, Acute, and Control. |
| `history_gains_over_others` | Two single-column panels of History-minus-Acute/Control gain differences, with direct-test stars; reads `group_retrieval_gains.csv`. |

For a formatting-only change, rerun the render command. To regenerate one table:

```bash
python scripts/analysis/render_tables.py --analysis-dir experiments/output/2026-11-12_022701PST__test_so__main/analysis --table main_results
```

Other table names are `generator_retrieval_gains`, `group_retrieval_gains`, and
`history_gains_over_others`.

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

- `build_analysis_dataset.py`: source-configured experiment JSONLs, labels, and
  protocol-2 judge exports → the two combined datasets.
- `summarize_experiment_results.py`: validated rank-1 scores, eligibility and judge
  reporting policy → the three statistics CSVs. Previously committed diagnosis-only
  helper APIs remain available, but the CLI generates only the selected tables.
- `table_statistics.py`: means, patient-cluster confidence intervals, paired tests,
  and multiplicity correction.
- `compare_agentic_variants.py`: six paired two-sided counterpart comparisons
  with equal-generator averages and a joint Holm family.
- `table_rendering.py`: all four LaTeX layouts, captions, rounding, bolding, and stars.
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

The same `group_retrieval_gains.csv` also saves **direct History-versus-Acute and
History-versus-Control gain tests** in `history_contrast_*` columns, populated
on the Similar-patient retrieval rows for Acute and Control. The estimate is
History's mean (Similar-patient retrieval minus No retrieval) gain minus the
row group's mean gain. The one-sided alternative is a larger History gain.
Two-group OLS contrasts use CR1 errors clustered by patient jointly across both
groups, including patients present in both groups, and G−1 Student-t degrees of
freedom. The small-sample correction is G/(G−1) × (N−1)/(N−2).

`history_contrast_p_value` and `history_contrast_p_value_holm` save raw and
Holm-adjusted p-values. This separate family contains both group comparisons
across all generators per outcome (eight tests with four generators).
`history_contrast_significant_{raw,holm}_{0_01,0_05}` records the threshold
decisions. Saved effects, standard errors, pointwise 95% intervals, coverage,
and test status make the comparisons inspectable. Missing eligible pairs or
insufficient patient replication suppress inference; missing tests retain
their Holm-family places. The existing table stars still test within groups.
The separate `history_gains_over_others.tex` table uses these direct-test
p-values for its own stars, prints three decimals in left-aligned columns,
and reserves a Treatment plan column even while those comparisons are missing.
Its label is `tab:history-gains-over-others`. Default rendering and report
synchronization include it; `--table history_gains_over_others` selects it alone.
Refresh `experiment_results.csv` and rerun the same `--csv-only` command above
to populate treatment comparisons automatically once eligible scores are
complete. Treatment History remains hypertension-only while COPD references
are unavailable. No separate analysis script or output CSV is required.

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
python tests/test_combined_analysis.py
```

These include independent statistical reference checks, incomplete/complete Plan
exports, report synchronization, malformed-input handling, and provenance checks.
