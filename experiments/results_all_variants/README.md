# Results across all seven variants

Start with `experiment_results.csv`: one row per condition-example `id`, generator
`model`, and `variant`. All expected slots remain present, including failures.
`source_run` identifies the original experiment. The raw sources stay in their
original run directories; `sources.yaml` lists their repository-relative paths.

## Rebuild from saved results

Run from the repository root with the dependencies in
[requirements.txt](../../requirements.txt) installed:

```bash
python scripts/analysis/build_analysis_dataset.py
python scripts/analysis/summarize_experiment_results.py --csv-only
python scripts/analysis/render_tables.py --analysis-only
```

These commands only read saved experiments and judge exports; no inference or
judging is run. They regenerate the two datasets, four analysis CSVs, and tables.
`analysis/figures/` is created by the renderer for future plots; it needs no
placeholder file. No provenance manifests or midterm report copies are generated
by this combined workflow.

## Datasets and columns

- `experiment_results.csv`: all three ranked diagnoses, BioLORD and BERTScore-F1
  values and their statuses, all three treatment plans, labels, and rank-1
  protocol-2 judgments. `diagnosis1`/`plan1` refer to the first predicted diagnosis.
- `planN` and `plan_label` are human-readable displays. Their `_actions` columns
  preserve the original action arrays as JSON, including embedded semicolons.
  Judge action numbers refer to the corresponding array after stripping whitespace
  and dropping blank items, as in the protocol-2 reader.
- `plan1_score` is protocol-2 coverage (mean reference-action credit).
  `plan1_reference_item_credits` gives full=1, partial=0.5, none=0 credits.
  `plan1_penalized` is the secondary score; harmful flags are exploratory.
  Other `plan1_` fields retain the judging status, reasons, attempts, errors,
  model/revision and run identifier from the source CSV.
- `plan_judgments_v2.csv`: unchanged columns concatenated from seven per-variant
  judging CSVs, one consolidated outcome per patient/generator/variant.
- Ranks 2 and 3 retain their predictions and diagnosis scores, but their treatment
  status is `not_judged`; protocol 2 evaluated only rank 1.
- Protocol-1 plan evaluations, unsupported-extra penalties, and the old
  `experiment_results.csv` are never imported. All treatment scores come from
  the per-run `plan_judging_v2/` directories.

For pandas, read with `keep_default_na=False, na_values=['']` so literal model
outputs such as `None` remain text while empty score cells are treated as missing.

## Analysis and missing scores

Primary tables use rank-1 BioLORD diagnosis scores and protocol-2 coverage.
BERTScore and lower-rank diagnosis scores remain available in the record-level CSV.
`main_summary_stats.csv` gives per-generator/variant means and patient-cluster
95% bootstrap intervals (1,000 draws, seed 20261114), plus paired baseline tests.
`generator_retrieval_gains.csv` contains only equal-weight four-generator averages:
average within each record/variant, then compare with `llm_only` on matched records.
All four scores must exist in both arms; missing generators are never reweighted.
`group_retrieval_gains.csv` breaks results down by History, Acute and Control and
retains the existing direct History-vs-Acute/Control similar-patient gain contrasts.
`coverage_and_failures.csv` reports metric/status/reason counts and rates overall,
by condition group, and by condition, for each variant/generator and pooled models.

Means exclude missing scores and include genuine zeros (including empty generated
plans scored zero). Paired tests use only records with both required scores.
Unavailable references are excluded from treatment eligibility; generation and
judge failures remain explicit missing outcomes. Paired baseline/retrieval means
are reported on the same paired cohort; group tables also retain raw arm means.
Counts identify the actual scored and paired samples. Different missingness can
therefore give different cohorts; these are available-case comparisons.

Uncertainty and tests cluster by patient UUID, keeping a patient's condition
examples together. Tests use CR1 errors and G-1 Student-t degrees of freedom.
Primary tests are one-sided improvement; exploratory two-sided checks are separate.
All exported p-value names state their direction: `p_value_one_sided` and
`p_value_one_sided_holm`, or `p_value_two_sided` and `p_value_two_sided_holm`.
History contrasts use `history_contrast_p_value_one_sided` and
`history_contrast_p_value_one_sided_holm`. The `_holm` suffix indicates adjustment
for multiple comparisons.
Holm families per outcome: six baseline comparisons within each generator for the
main summary; six model-average comparisons for generator gains; eighteen tests
(six variants x three groups) within each generator for group gains. The existing
History contrast family is two contrasts across four generators. Missing tests
retain their family positions. Harmful/penalized scores are secondary and their
clinical accuracy was not established by this export. COPD references are record
artifacts, so treatment History summaries cover hypertension only.

## Agentic versus non-agentic significance

```bash
python scripts/analysis/compare_agentic_variants.py
```

This separately writes `analysis/agentic_vs_nonagentic_significance.csv`: six
two-sided tests comparing patient retrieval, guidelines, and both with their
matching non-agentic variants, for diagnosis and treatment coverage. It reads
`experiment_results.csv`; existing summary CSVs and tables are left untouched.

Each test averages the four generators within each record before pairing the
two variants. All four scores must exist in both arms; unavailable treatment
references are excluded. The CSV includes paired arm means, the signed
`mean_difference` (agentic minus non-agentic), paired record/patient counts,
test status, raw `p_value_two_sided`, and `p_value_two_sided_holm`. The tests use
patient-clustered CR1 errors with G-1 Student-t degrees of freedom and one Holm
family across **all six tests**, with raw/adjusted 0.01 and 0.05 significance flags.
Untestable comparisons retain their family positions and have blank p-values
and flags. These are exploratory comparisons. Paired means can differ from
the available-case means in `main_summary_stats.csv` because their cohorts differ.
