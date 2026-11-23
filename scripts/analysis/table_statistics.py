"""Statistics for the three report tables.

All means weight condition examples equally. Paired tests use CR1 standard
errors clustered by patient UUID and G-1 Student-t degrees of freedom.
Main-results intervals resample patients; each table's Holm family is documented
in the analysis manifest. Missing eligible judgments are never treated as zero.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


VARIANTS = ("llm_only", "patient_retrieval", "guidelines", "patient_retrieval_guidelines")


VARIANT_LABELS = {
    "llm_only": "No retrieval", "patient_retrieval": "Sim. patient retrieval",
    "guidelines": "Guideline retrieval", "patient_retrieval_guidelines": "Both",
    "agentic_patient_retrieval": "Agentic sim. patient retrieval",
    "agentic_guidelines": "Agentic guideline retrieval", "agentic_both": "Agentic both",
}


def active_variants(results):
    """Keep the historical four-arm order and append the selected agentic arms."""
    unknown = set(results["variant"]) - set(VARIANT_LABELS)
    if unknown:
        raise ValueError(f"Unknown retrieval configuration: {sorted(unknown)}")
    return VARIANTS + tuple(v for v in VARIANT_LABELS if v not in VARIANTS and v in set(results.variant))


BASELINE_TEST_VARIANTS = ("patient_retrieval", "guidelines")


SIGNIFICANCE_LEVELS = (0.01, 0.05)


MODEL_AVERAGE = "model_average"


# Explicit public CSV names; generic internal names also serve two-sided tests.
ONE_SIDED_P_VALUE_COLUMNS = {
    'p_value': 'p_value_one_sided',
    'p_value_holm': 'p_value_one_sided_holm',
    'history_contrast_p_value': 'history_contrast_p_value_one_sided',
    'history_contrast_p_value_holm': 'history_contrast_p_value_one_sided_holm',
}


def _holm_adjust(pvalues):
    """Adjust the planned family, reserving missing tests with p=1."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(np.where(np.isnan(values), 1.0, values), kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
        np.where(np.isnan(values[order]), 1.0, values[order]) * np.arange(len(values), 0, -1)
    ))
    adjusted[np.isnan(values)] = np.nan
    return adjusted


def _paired_cluster_test(differences, patients, alternative="greater"):
    if alternative not in ("greater", "two-sided"):
        raise ValueError("alternative must be greater or two-sided.")
    codes, unique = pd.factorize(patients, sort=True)
    count = len(unique)
    if count < 2:
        return np.nan, np.nan, np.nan, "insufficient_patients"
    residual_sums = np.bincount(codes, weights=differences - differences.mean())
    variance = count / (count - 1) * np.dot(residual_sums, residual_sums) / len(differences) ** 2
    # A degenerate sampling variance cannot support a reliable t reference.
    if variance <= np.finfo(float).eps ** 2:
        return np.nan, count - 1, np.nan, "zero_variance"
    statistic = float(differences.mean() / np.sqrt(variance))
    pvalue = (2 * student_t.sf(abs(statistic), count - 1) if alternative == "two-sided"
              else student_t.sf(statistic, count - 1))
    return statistic, count - 1, float(pvalue), "tested"


def _with_exploratory_two_sided_checks(summary, family_columns, effect_column):
    """Add separate post-hoc checks; never replace primary p-values or stars.

    Reuse the saved paired CR1 t statistics and degrees of freedom. Retain every
    comparison in each original Holm family, including unavailable tests, rather
    than selecting only the observed decreases. Baseline rows are not tests.
    """
    output = summary.copy()
    for column in ("p_value_two_sided", "p_value_two_sided_holm"):
        output[column] = np.nan
    comparisons = output.variant.ne("llm_only")
    tested = comparisons & output.test_status.eq("tested")
    output.loc[tested, "p_value_two_sided"] = 2 * student_t.sf(
        np.abs(output.loc[tested, "test_statistic"].to_numpy(dtype=float)),
        output.loc[tested, "test_df"].to_numpy(dtype=float),
    )
    for _, family in output.loc[comparisons].groupby(list(family_columns), sort=False):
        output.loc[family.index, "p_value_two_sided_holm"] = _holm_adjust(family.p_value_two_sided)
    for adjustment, column in (("raw", "p_value_two_sided"), ("holm", "p_value_two_sided_holm")):
        for alpha in SIGNIFICANCE_LEVELS:
            flag = f"significant_two_sided_{adjustment}_{str(alpha).replace('.', '_')}"
            significant = output[effect_column].ne(0) & output[column].lt(alpha)
            output[flag] = significant.where(output[column].notna(), pd.NA).astype("boolean")
    return output


def summarize_main_results(results, bootstrap_samples=1000, seed=20261114, *, available_scores=False):
    """Return main-results means, pointwise 95% CIs, raw/adjusted p-values, and stars.

    With available_scores=True, means/bootstrap intervals use observed scores
    and tests use observed pairs; coverage_complete remains descriptive. The
    historical default requires complete eligible cohorts.
    Input is the normalized, validated output of load_scored_results, with any
    unvalidated Plan judge scores already masked. In the historical mode,
    missing eligible scores suppress means and comparisons. The null is
    mean(retrieval - baseline) <= 0; the alternative is a positive difference.
    Separate exploratory two-sided columns reuse the same baseline-comparison
    family per model/outcome and do not determine the reported stars.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be an integer of at least two.")
    required = {"id", "patient_uuid", "model", "variant", "diagnosis", "plan", "plan_eligible"}
    if required - set(results):
        raise ValueError("Pass the normalized frame returned by load_scored_results.")
    if results.empty or results.duplicated(["id", "model", "variant"]).any():
        raise ValueError("Require nonempty, unique case/model/variant rows.")
    variants = active_variants(results)
    metadata = results.drop_duplicates("id").set_index("id").sort_index()
    rng = np.random.default_rng(seed)
    weights_by_cohort = {}
    rows = []
    for model in sorted(results["model"].unique()):
        for metric in ("diagnosis", "plan"):
            cohort = metadata if metric == "diagnosis" else metadata.loc[metadata["plan_eligible"]]
            pivot = results.loc[results["model"].eq(model)].pivot(
                index="id", columns="variant", values=metric,
            ).reindex(index=cohort.index, columns=variants)
            n_cases = len(cohort)
            n_patients = cohort["patient_uuid"].nunique()
            family = []
            for variant in variants:
                values = pivot[variant]
                n_scored = int(values.notna().sum())
                complete = n_cases > 0 and n_scored == n_cases
                n_pairs = int((values.notna() & pivot["llm_only"].notna()).sum())
                record = {
                    "model": model, "variant": variant, "metric": metric,
                    "level": "overall", "group": "all", "n_eligible": n_cases,
                    "n_scored": n_scored, "n_pairs": n_pairs, "n_patients": n_patients,
                    "coverage_complete": complete, "mean_score": np.nan,
                    "mean_ci_low": np.nan, "mean_ci_high": np.nan,
                    "baseline_delta": np.nan, "test_statistic": np.nan,
                    "test_df": np.nan, "p_value": np.nan, "p_value_holm": np.nan,
                    "test_status": "baseline" if variant == "llm_only" else "incomplete" if n_cases else "unavailable",
                    "stars": "", "family_size": len(variants) - 1,
                }
                if complete or (available_scores and n_scored):
                    used = values.dropna()
                    patients = cohort.loc[used.index, "patient_uuid"].to_numpy()
                    codes, unique = pd.factorize(patients, sort=True)
                    record["n_patients"] = len(unique)
                    array = used.to_numpy(dtype=float)
                    record["mean_score"] = float(array.mean())
                    if len(unique) >= 2:
                        key = tuple(unique)
                        if key not in weights_by_cohort:
                            weights_by_cohort[key] = rng.multinomial(
                                len(unique), np.full(len(unique), 1 / len(unique)), size=bootstrap_samples)
                        counts = np.bincount(codes, minlength=len(unique))
                        sums = np.bincount(codes, weights=array, minlength=len(unique))
                        means = (weights_by_cohort[key] @ sums) / (weights_by_cohort[key] @ counts)
                        low, high = np.quantile(means, [0.025, 0.975])
                        record.update(mean_ci_low=float(low), mean_ci_high=float(high))
                if variant != "llm_only" and n_pairs and (available_scores or n_pairs == n_cases):
                    paired = pivot[[variant, "llm_only"]].dropna()
                    delta = (paired[variant] - paired["llm_only"]).to_numpy(dtype=float)
                    paired_patients = cohort.loc[paired.index, "patient_uuid"].to_numpy()
                    statistic, df, pvalue, status = _paired_cluster_test(delta, paired_patients)
                    record.update(baseline_delta=float(delta.mean()), test_statistic=statistic,
                                  test_df=df, p_value=pvalue, test_status=status)
                family.append(record)
            for record, adjusted in zip(family[1:], _holm_adjust([r["p_value"] for r in family[1:]])):
                record["p_value_holm"] = float(adjusted)
                if record["baseline_delta"] > 0 and np.isfinite(adjusted):
                    record["stars"] = "**" if adjusted < 0.01 else "*" if adjusted < 0.05 else ""
            rows.extend(family)
    return _with_exploratory_two_sided_checks(
        pd.DataFrame(rows), ("model", "metric"), "baseline_delta",
    )


def summarize_generator_retrieval_gains(results, metrics=("diagnosis", "plan"),
                                    include_model_average=False, *, available_scores=False, average_only=False):
    """Test single-source retrieval against no retrieval for each model/outcome.

    Pass normalized rank-1 rows from load_scored_results. The null is that the
    condition-example mean retrieval-minus-baseline difference is <= 0; the
    alternative is > 0. Patient-UUID clusters preserve dependence across a
    patient's condition examples. Standard errors use G-1 Student-t df.

    Holm correction includes both requested comparisons across every model
    within each requested outcome (eight tests for four models). Incomplete
    or degenerate tests stay unavailable, not nonsignificant, and retain their
    place in the correction family. No complete-case subset is substituted.

    average_only=True selects all retrieval variants, builds within-record
    equal-generator means, and excludes individual-generator tests before Holm
    correction. available_scores=True uses only observed pairs. The historical
    defaults below retain the original two-source, complete-cohort behavior.
    With include_model_average, also test the equal-generator average gain.
    Average paired scores across the fixed generators within each case before
    clustering by patient; do not treat generators as independent replicates.
    A missing score from any generator makes that averaged case unavailable.
    Both average tests join the Holm family (ten tests for four generators).
    Separate exploratory two-sided columns retain this entire family per outcome.
    """
    if not isinstance(include_model_average, bool):
        raise ValueError("include_model_average must be a boolean.")
    if not isinstance(average_only, bool):
        raise ValueError("average_only must be a boolean.")
    variants = active_variants(results)
    comparisons = tuple((variant, "llm_only") for variant in
                        (variants[1:] if average_only else BASELINE_TEST_VARIANTS))
    summary = _summarize_paired_significance(
        results, metrics, comparisons, "greater", include_model_average=include_model_average or average_only,
        available_scores=available_scores, average_only=average_only,
    )
    return _with_exploratory_two_sided_checks(summary, ("metric",), "mean_difference")


def summarize_agentic_counterpart_comparisons(results):
    """Six two-sided agentic-minus-fixed tests on within-record model averages.

    Require all four generator slots and seven variants. Each pair uses only
    records with all four scores in both arms; Plan references must be eligible.
    Cluster by patient UUID and adjust one Holm family across all six tests.
    """
    if {'id', 'model', 'variant'} - set(results) or len(results) != results.id.nunique() * 4 * len(VARIANT_LABELS):
        raise ValueError('Require every record/model/variant slot for four models and seven variants')
    comparisons = (('agentic_patient_retrieval', 'patient_retrieval'),
                   ('agentic_guidelines', 'guidelines'),
                   ('agentic_both', 'patient_retrieval_guidelines'))
    summary = _summarize_paired_significance(
        results, ('diagnosis', 'plan'), comparisons, 'two-sided',
        include_model_average=True, available_scores=True, average_only=True,
        separate_metric_families=False,
    )
    summary.loc[summary.n_pairs.eq(0), 'n_patients'] = 0
    summary.loc[summary.n_pairs.eq(0), ['mean_baseline', 'mean_retrieval']] = np.nan
    summary['alternative'] = 'two-sided'
    renamed = {'baseline_variant': 'nonagentic_variant', 'variant': 'agentic_variant',
               'mean_baseline': 'mean_nonagentic', 'mean_retrieval': 'mean_agentic',
               'n_baseline': 'n_nonagentic', 'n_retrieval': 'n_agentic',
               'p_value': 'p_value_two_sided', 'p_value_holm': 'p_value_two_sided_holm'}
    renamed.update({column: column.replace('significant_', 'significant_two_sided_', 1)
                    for column in summary if column.startswith('significant_')})
    return summary.rename(columns=renamed)


def _summarize_paired_significance(results, metrics, comparisons, alternative,
                                 include_model_average=False, available_scores=False, average_only=False,
                                 separate_metric_families=True):
    """Shared pairing, eligibility, clustering, and outcome-wise multiplicity."""
    if isinstance(metrics, str) or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a nonempty sequence of distinct outcome names.")
    if set(metrics) - {"diagnosis", "plan"}:
        raise ValueError("Supported outcomes are diagnosis and plan.")
    required = {"id", "patient_uuid", "model", "variant", "plan_eligible", *metrics}
    if required - set(results):
        raise ValueError("Pass the normalized frame returned by load_scored_results.")
    identity = ["id", "patient_uuid", "model", "variant", "plan_eligible"]
    if results.empty or results[identity].isna().any().any():
        raise ValueError("Require nonempty rows with populated case and patient identities.")
    if results.duplicated(["id", "model", "variant"]).any():
        raise ValueError("Require unique case/model/variant rows.")
    variants = active_variants(results)
    if not results["plan_eligible"].isin([True, False]).all():
        raise ValueError("plan_eligible must contain boolean eligibility values.")
    if results.groupby("id")[["patient_uuid", "plan_eligible"]].nunique().gt(1).any().any():
        raise ValueError("Each case must have consistent patient identity and Plan eligibility.")
    for metric in metrics:
        values = results[metric].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{metric} scores must be finite and in [0, 1], or missing.")
    metadata = results.drop_duplicates("id").set_index("id").sort_index()
    models = sorted(results["model"].unique())
    n_generators = len(models)
    if average_only and (n_generators != 4 or set(results.variant) != set(VARIANT_LABELS)):
        raise ValueError('Average-only analysis requires four generators and all seven variants')
    if include_model_average:
        if MODEL_AVERAGE in models:
            raise ValueError(f"{MODEL_AVERAGE} is reserved for the generated average row.")
        case_variants = pd.MultiIndex.from_product(
            [metadata.index, variants], names=["id", "variant"],
        )
        average = case_variants.to_frame(index=False)
        average["model"] = MODEL_AVERAGE
        average["patient_uuid"] = average["id"].map(metadata["patient_uuid"])
        average["plan_eligible"] = average["id"].map(metadata["plan_eligible"])
        for metric in metrics:
            scores = results.pivot(index=["id", "variant"], columns="model", values=metric)
            scores = scores.reindex(index=case_variants, columns=models)
            average[metric] = scores.mean(axis=1, skipna=False).to_numpy()
        results = pd.concat([results, average], ignore_index=True)
        models = [MODEL_AVERAGE] if average_only else [MODEL_AVERAGE, *models]
    rows = []
    for model in models:
        for metric in metrics:
            cohort = metadata if metric == "diagnosis" else metadata.loc[metadata["plan_eligible"]]
            pivot = results.loc[results["model"].eq(model)].pivot(
                index="id", columns="variant", values=metric,
            ).reindex(index=cohort.index, columns=variants)
            n_cases = len(cohort)
            for variant, reference_variant in comparisons:
                baseline = pivot[reference_variant]
                retrieval = pivot[variant]
                n_baseline, n_retrieval = int(baseline.notna().sum()), int(retrieval.notna().sum())
                n_pairs = int((baseline.notna() & retrieval.notna()).sum())
                complete = n_cases > 0 and n_pairs == n_cases
                row = {
                    "model": model, "metric": metric, "baseline_variant": reference_variant,
                    "variant": variant, "n_eligible": n_cases, "n_baseline": n_baseline,
                    "n_retrieval": n_retrieval, "n_pairs": n_pairs,
                    "n_patients": int(cohort["patient_uuid"].nunique()),
                    "coverage_complete": complete,
                    "mean_baseline": float(baseline.mean()) if n_cases and n_baseline == n_cases else np.nan,
                    "mean_retrieval": float(retrieval.mean()) if n_cases and n_retrieval == n_cases else np.nan,
                    "mean_difference": np.nan, "test_statistic": np.nan, "test_df": np.nan,
                    "p_value": np.nan, "p_value_holm": np.nan,
                    "test_status": "incomplete" if n_cases else "unavailable",
                }
                if include_model_average:
                    row["n_generators_averaged"] = n_generators if model == MODEL_AVERAGE else 1
                if complete or (available_scores and n_pairs):
                    paired = pd.DataFrame({"baseline": baseline, "retrieval": retrieval}).dropna()
                    delta = (paired.retrieval - paired.baseline).to_numpy(dtype=float)
                    paired_patients = cohort.loc[paired.index, "patient_uuid"].to_numpy()
                    row.update(mean_baseline=float(paired.baseline.mean()),
                               mean_retrieval=float(paired.retrieval.mean()),
                               n_patients=len(set(paired_patients)))
                    statistic, df, pvalue, status = _paired_cluster_test(
                        delta, paired_patients, alternative=alternative,
                    )
                    row.update(mean_difference=float(delta.mean()), test_statistic=statistic,
                               test_df=df, p_value=pvalue, test_status=status)
                rows.append(row)
    summary = pd.DataFrame(rows)
    families = summary.groupby('metric', sort=False) if separate_metric_families else [('all', summary)]
    for _, family in families:
        summary.loc[family.index, 'p_value_holm'] = _holm_adjust(family.p_value)
        summary.loc[family.index, 'family_size'] = len(family)
    summary["family_size"] = summary["family_size"].astype(int)
    for adjustment, p_column in (("raw", "p_value"), ("holm", "p_value_holm")):
        for alpha in SIGNIFICANCE_LEVELS:
            column = f"significant_{adjustment}_{str(alpha).replace('.', '_')}"
            direction = (summary["mean_difference"].gt(0) if alternative == "greater"
                         else summary["mean_difference"].ne(0))
            flags = direction & summary[p_column].lt(alpha)
            summary[column] = flags.where(summary[p_column].notna(), pd.NA).astype("boolean")
    return summary


def summarize_group_retrieval_gains(results, *, available_scores=False):
    """Return four-arm group means and one-sided baseline tests for group.

    Three retrieval configurations times three condition groups form a nine-test
    Holm family per model/outcome. Unavailable tests retain their place. These
    test within-group improvements. Separate history_contrast_* columns test
    History-minus-Acute/Control differences in similar-patient retrieval gains;
    their Holm family spans both comparisons across generators per outcome.
    """
    required = {"id", "bin", "diagnosis", "plan"}
    if required - set(results):
        raise ValueError("Pass normalized results with group and outcome columns.")
    groups = ("history", "acute", "control")
    if results["bin"].isna().any() or set(results["bin"]) - set(groups):
        raise ValueError("Unknown or missing group condition group.")
    if results.groupby("id")["bin"].nunique().gt(1).any():
        raise ValueError("Each case must have a consistent condition group.")
    # Reuse shared identity, eligibility, and score validation; only the group
    # tests below are reported, not these overall validation results.
    validated = _summarize_paired_significance(
        results, ("diagnosis", "plan"), (("patient_retrieval", "llm_only"),), "greater",
        available_scores=available_scores,
    )
    variants = active_variants(results)
    metadata = results.drop_duplicates("id").set_index("id").sort_index()
    rows = []
    for model in sorted(validated["model"].unique()):
        for metric in ("diagnosis", "plan"):
            family = []
            for group in groups:
                cohort = metadata.loc[metadata["bin"].eq(group)]
                if metric == "plan":
                    cohort = cohort.loc[cohort["plan_eligible"]]
                pivot = results.loc[results["model"].eq(model)].pivot(
                    index="id", columns="variant", values=metric,
                ).reindex(index=cohort.index, columns=variants)
                baseline = pivot["llm_only"]
                n_eligible = len(cohort)
                n_baseline = int(baseline.notna().sum())
                for variant in variants[1:]:
                    retrieval = pivot[variant]
                    n_retrieval = int(retrieval.notna().sum())
                    n_pairs = int((baseline.notna() & retrieval.notna()).sum())
                    complete = n_eligible > 0 and n_pairs == n_eligible
                    record = {
                        "model": model, "metric": metric, "group": group,
                        "baseline_variant": "llm_only", "variant": variant,
                        "n_eligible": n_eligible, "n_baseline": n_baseline,
                        "n_retrieval": n_retrieval, "n_pairs": n_pairs,
                        "n_patients": cohort["patient_uuid"].nunique(),
                        "mean_baseline": float(baseline.mean()) if n_baseline and (available_scores or n_baseline == n_eligible) else np.nan,
                        "mean_retrieval": float(retrieval.mean()) if n_retrieval and (available_scores or n_retrieval == n_eligible) else np.nan,
                        "paired_mean_baseline": np.nan, "paired_mean_retrieval": np.nan,
                        "coverage_complete": complete, "mean_difference": np.nan,
                        "test_statistic": np.nan, "test_df": np.nan, "p_value": np.nan,
                        "p_value_holm": np.nan, "family_size": (len(variants) - 1) * len(groups), "stars": "",
                        "test_status": "incomplete" if n_eligible else "unavailable",
                    }
                    if complete or (available_scores and n_pairs):
                        paired = pd.DataFrame({"baseline": baseline, "retrieval": retrieval}).dropna()
                        differences = (paired.retrieval - paired.baseline).to_numpy(dtype=float)
                        paired_patients = cohort.loc[paired.index, "patient_uuid"].to_numpy()
                        record.update(paired_mean_baseline=float(paired.baseline.mean()),
                                      paired_mean_retrieval=float(paired.retrieval.mean()),
                                      n_patients=len(set(paired_patients)))
                        statistic, df, pvalue, status = _paired_cluster_test(
                            differences, paired_patients, alternative="greater",
                        )
                        record.update(mean_difference=float(differences.mean()), test_statistic=statistic,
                                      test_df=df, p_value=pvalue, test_status=status)
                    family.append(record)
            for record, adjusted in zip(family, _holm_adjust([row["p_value"] for row in family])):
                record["p_value_holm"] = float(adjusted)
                if record["mean_difference"] > 0:
                    record["stars"] = "**" if adjusted < 0.01 else "*" if adjusted < 0.05 else ""
            rows.extend(family)
    return _with_history_gain_contrasts(pd.DataFrame(rows), results, available_scores=available_scores)


def _with_history_gain_contrasts(summary, results, *, available_scores=False):
    """Append direct between-group tests without changing table rows or stars.

    For each generator/outcome, regress the paired case gain on two group
    indicators and contrast their means (History minus Acute or Control).
    The CR1 sandwich clusters by patient jointly across both groups, with
    G/(G-1) * (N-1)/(N-2) correction and G-1 Student-t degrees of freedom.
    The one-sided alternative is that the History mean gain is larger.
    A separate Holm family contains two contrasts per generator per outcome.
    Missing eligible pairs suppress inference, retaining their family places.
    Input validation is performed by summarize_group_retrieval_gains.
    """
    metadata = results.drop_duplicates("id").set_index("id").sort_index()
    models = sorted(results["model"].unique())
    rows = []
    for model in models:
        for metric in ("diagnosis", "plan"):
            eligible = metadata if metric == "diagnosis" else metadata.loc[metadata.plan_eligible]
            pivot = results.loc[results.model.eq(model)].pivot(
                index="id", columns="variant", values=metric,
            ).reindex(index=eligible.index, columns=("llm_only", "patient_retrieval"))
            gains = pivot.patient_retrieval - pivot.llm_only
            for group in ("acute", "control"):
                cohort = eligible.loc[eligible.bin.isin(("history", group))]
                history = cohort.bin.eq("history").to_numpy()
                values = gains.reindex(cohort.index).to_numpy(dtype=float)
                n_history, n_group = int(history.sum()), int((~history).sum())
                history_pairs = int(np.isfinite(values[history]).sum())
                group_pairs = int(np.isfinite(values[~history]).sum())
                complete = bool(n_history and n_group and
                                history_pairs == n_history and group_pairs == n_group)
                codes, patients = pd.factorize(cohort.patient_uuid, sort=True)
                n_patients = len(patients)
                record = {
                    "model": model, "metric": metric, "group": group,
                    "variant": "patient_retrieval", "reference_group": "history",
                    "history_n_eligible": n_history, "history_n_pairs": history_pairs,
                    "comparison_n_eligible": n_group, "comparison_n_pairs": group_pairs,
                    "n_patients": n_patients, "coverage_complete": complete,
                    "history_gain": float(np.nanmean(values[history])) if history_pairs and (available_scores or history_pairs == n_history) else np.nan,
                    "comparison_gain": float(np.nanmean(values[~history])) if group_pairs and (available_scores or group_pairs == n_group) else np.nan,
                    "estimate": np.nan, "standard_error": np.nan,
                    "ci_low": np.nan, "ci_high": np.nan,
                    "test_statistic": np.nan, "test_df": np.nan,
                    "p_value": np.nan, "p_value_holm": np.nan,
                    "family_size": 2 * len(models),
                    "test_status": "unavailable" if not n_history or not n_group else "incomplete",
                }
                if complete or (available_scores and history_pairs and group_pairs):
                    observed = np.isfinite(values)
                    cohort = cohort.loc[observed]
                    history, values = history[observed], values[observed]
                    n_history, n_group = int(history.sum()), int((~history).sum())
                    codes, patients = pd.factorize(cohort.patient_uuid, sort=True)
                    n_patients = len(patients)
                    record["n_patients"] = n_patients
                    estimate = record["history_gain"] - record["comparison_gain"]
                    record["estimate"] = estimate
                    # Each group needs patient replication of its own.
                    if (cohort.loc[history, "patient_uuid"].nunique() < 2 or
                            cohort.loc[~history, "patient_uuid"].nunique() < 2):
                        record["test_status"] = "insufficient_patients"
                    else:
                        residuals = values - np.where(history, record["history_gain"], record["comparison_gain"])
                        influence = residuals * np.where(history, 1 / n_history, -1 / n_group)
                        cluster_sums = np.bincount(codes, weights=influence, minlength=n_patients)
                        n_cases = len(cohort)
                        correction = n_patients / (n_patients - 1) * (n_cases - 1) / (n_cases - 2)
                        variance = correction * np.dot(cluster_sums, cluster_sums)
                        df = n_patients - 1
                        record["test_df"] = df
                        if variance <= np.finfo(float).eps ** 2:
                            record["test_status"] = "zero_variance"
                        else:
                            error = float(np.sqrt(variance))
                            statistic = float(estimate / error)
                            radius = student_t.ppf(0.975, df) * error
                            record.update(standard_error=error, test_statistic=statistic,
                                          p_value=float(student_t.sf(statistic, df)),
                                          ci_low=float(estimate - radius), ci_high=float(estimate + radius),
                                          test_status="tested")
                rows.append(record)
    contrasts = pd.DataFrame(rows)
    for _, family in contrasts.groupby("metric", sort=False):
        contrasts.loc[family.index, "p_value_holm"] = _holm_adjust(family.p_value)
    for adjustment, column in (("raw", "p_value"), ("holm", "p_value_holm")):
        for alpha in SIGNIFICANCE_LEVELS:
            flag = f"significant_{adjustment}_{str(alpha).replace('.', '_')}"
            flags = contrasts.estimate.gt(0) & contrasts[column].lt(alpha)
            contrasts[flag] = flags.where(contrasts[column].notna(), pd.NA).astype("boolean")
    keys = ["model", "metric", "group", "variant"]
    contrasts = contrasts.rename(columns={
        column: f"history_contrast_{column}" for column in contrasts if column not in keys
    })
    return summary.merge(contrasts, on=keys, how="left", sort=False, validate="one_to_one")
