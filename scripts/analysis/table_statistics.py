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


BASELINE_TEST_VARIANTS = ("patient_retrieval", "guidelines")


SIGNIFICANCE_LEVELS = (0.01, 0.05)


MODEL_AVERAGE = "model_average"


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


def summarize_main_results(results, bootstrap_samples=1000, seed=20261114):
    """Return main-results means, pointwise 95% CIs, raw/adjusted p-values, and stars.

    Input is the normalized, validated output of load_scored_results, with any
    unvalidated Plan judge scores already masked. Missing eligible scores are
    never dropped to obtain a selectively complete cohort. The null is
    mean(retrieval - baseline) <= 0; the alternative is a positive difference.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be an integer of at least two.")
    required = {"id", "patient_uuid", "model", "variant", "diagnosis", "plan", "plan_eligible"}
    if required - set(results):
        raise ValueError("Pass the normalized frame returned by load_scored_results.")
    if results.empty or results.duplicated(["id", "model", "variant"]).any():
        raise ValueError("Require nonempty, unique case/model/variant rows.")
    if set(results["variant"]) - set(VARIANTS):
        raise ValueError("Unknown retrieval configuration.")
    metadata = results.drop_duplicates("id").set_index("id").sort_index()
    rng = np.random.default_rng(seed)
    weights_by_cohort = {}
    rows = []
    for model in sorted(results["model"].unique()):
        for metric in ("diagnosis", "plan"):
            cohort = metadata if metric == "diagnosis" else metadata.loc[metadata["plan_eligible"]]
            pivot = results.loc[results["model"].eq(model)].pivot(
                index="id", columns="variant", values=metric,
            ).reindex(index=cohort.index, columns=VARIANTS)
            patients = cohort["patient_uuid"].to_numpy()
            codes, unique = pd.factorize(patients, sort=True)
            n_patients, n_cases = len(unique), len(cohort)
            counts = np.bincount(codes, minlength=n_patients)
            weights = None
            if n_patients >= 2:
                key = tuple(unique)
                if key not in weights_by_cohort:
                    weights_by_cohort[key] = rng.multinomial(
                        n_patients, np.full(n_patients, 1 / n_patients), size=bootstrap_samples,
                    )
                weights = weights_by_cohort[key]
            family = []
            for variant in VARIANTS:
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
                    "stars": "", "family_size": 3,
                }
                if complete:
                    array = values.to_numpy(dtype=float)
                    record["mean_score"] = float(array.mean())
                    if weights is not None:
                        sums = np.bincount(codes, weights=array, minlength=n_patients)
                        means = (weights @ sums) / (weights @ counts)
                        low, high = np.quantile(means, [0.025, 0.975])
                        record.update(mean_ci_low=float(low), mean_ci_high=float(high))
                    if variant != "llm_only" and n_pairs == n_cases:
                        delta = array - pivot["llm_only"].to_numpy(dtype=float)
                        statistic, df, pvalue, status = _paired_cluster_test(delta, patients)
                        record.update(baseline_delta=float(delta.mean()), test_statistic=statistic,
                                      test_df=df, p_value=pvalue, test_status=status)
                family.append(record)
            for record, adjusted in zip(family[1:], _holm_adjust([r["p_value"] for r in family[1:]])):
                record["p_value_holm"] = float(adjusted)
                if record["baseline_delta"] > 0 and np.isfinite(adjusted):
                    record["stars"] = "**" if adjusted < 0.01 else "*" if adjusted < 0.05 else ""
            rows.extend(family)
    return pd.DataFrame(rows)


def summarize_generator_retrieval_gains(results, metrics=("diagnosis", "plan"),
                                    include_model_average=False):
    """Test single-source retrieval against no retrieval for each model/outcome.

    Pass normalized rank-1 rows from load_scored_results. The null is that the
    condition-example mean retrieval-minus-baseline difference is <= 0; the
    alternative is > 0. Patient-UUID clusters preserve dependence across a
    patient's condition examples. Cmain-results standard errors use G-1 Student-t df.

    Holm correction includes both requested comparisons across every model
    within each requested outcome (eight tests for four models). Incomplete
    or degenerate tests stay unavailable, not nonsignificant, and retain their
    place in the correction family. No complete-case subset is substituted.

    With include_model_average, also test the equal-generator average gain.
    Average paired scores across the fixed generators within each case before
    clustering by patient; do not treat generators as independent replicates.
    A missing score from any generator makes that averaged case unavailable.
    Both average tests join the Holm family (ten tests for four generators).
    """
    if not isinstance(include_model_average, bool):
        raise ValueError("include_model_average must be a boolean.")
    comparisons = tuple((variant, "llm_only") for variant in BASELINE_TEST_VARIANTS)
    return _summarize_paired_significance(
        results, metrics, comparisons, "greater", include_model_average=include_model_average,
    )


def _summarize_paired_significance(results, metrics, comparisons, alternative,
                                 include_model_average=False):
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
    if set(results["variant"]) - set(VARIANTS):
        raise ValueError("Unknown retrieval configuration.")
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
    if include_model_average:
        if MODEL_AVERAGE in models:
            raise ValueError(f"{MODEL_AVERAGE} is reserved for the generated average row.")
        case_variants = pd.MultiIndex.from_product(
            [metadata.index, VARIANTS], names=["id", "variant"],
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
        models = [MODEL_AVERAGE, *models]
    rows = []
    for model in models:
        for metric in metrics:
            cohort = metadata if metric == "diagnosis" else metadata.loc[metadata["plan_eligible"]]
            pivot = results.loc[results["model"].eq(model)].pivot(
                index="id", columns="variant", values=metric,
            ).reindex(index=cohort.index, columns=VARIANTS)
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
                if complete:
                    delta = retrieval.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
                    statistic, df, pvalue, status = _paired_cluster_test(
                        delta, cohort["patient_uuid"].to_numpy(), alternative=alternative,
                    )
                    row.update(mean_difference=float(delta.mean()), test_statistic=statistic,
                               test_df=df, p_value=pvalue, test_status=status)
                rows.append(row)
    summary = pd.DataFrame(rows)
    for metric in metrics:
        mask = summary["metric"].eq(metric)
        summary.loc[mask, "p_value_holm"] = _holm_adjust(summary.loc[mask, "p_value"])
        summary.loc[mask, "family_size"] = int(mask.sum())
    summary["family_size"] = summary["family_size"].astype(int)
    for adjustment, p_column in (("raw", "p_value"), ("holm", "p_value_holm")):
        for alpha in SIGNIFICANCE_LEVELS:
            column = f"significant_{adjustment}_{str(alpha).replace('.', '_')}"
            direction = (summary["mean_difference"].gt(0) if alternative == "greater"
                         else summary["mean_difference"].ne(0))
            flags = direction & summary[p_column].lt(alpha)
            summary[column] = flags.where(summary[p_column].notna(), pd.NA).astype("boolean")
    return summary


def summarize_group_retrieval_gains(results):
    """Return four-arm group means and one-sided baseline tests for group.

    Three retrieval configurations times three condition groups form a nine-test
    Holm family per model/outcome. Unavailable tests retain their place. These
    test within-group improvements, not differences between group gains.
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
    )
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
                ).reindex(index=cohort.index, columns=VARIANTS)
                baseline = pivot["llm_only"]
                n_eligible = len(cohort)
                n_baseline = int(baseline.notna().sum())
                for variant in VARIANTS[1:]:
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
                        "mean_baseline": float(baseline.mean()) if n_eligible and n_baseline == n_eligible else np.nan,
                        "mean_retrieval": float(retrieval.mean()) if n_eligible and n_retrieval == n_eligible else np.nan,
                        "coverage_complete": complete, "mean_difference": np.nan,
                        "test_statistic": np.nan, "test_df": np.nan, "p_value": np.nan,
                        "p_value_holm": np.nan, "family_size": 9, "stars": "",
                        "test_status": "incomplete" if n_eligible else "unavailable",
                    }
                    if complete:
                        differences = retrieval.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
                        statistic, df, pvalue, status = _paired_cluster_test(
                            differences, cohort["patient_uuid"].to_numpy(), alternative="greater",
                        )
                        record.update(mean_difference=float(differences.mean()), test_statistic=statistic,
                                      test_df=df, p_value=pvalue, test_status=status)
                    family.append(record)
            for record, adjusted in zip(family, _holm_adjust([row["p_value"] for row in family])):
                record["p_value_holm"] = float(adjusted)
                if record["mean_difference"] > 0:
                    record["stars"] = "**" if adjusted < 0.01 else "*" if adjusted < 0.05 else ""
            rows.extend(family)
    return pd.DataFrame(rows)
