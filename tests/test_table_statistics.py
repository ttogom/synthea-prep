"""Offline reference checks for the three report tables' statistics."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, ttest_rel
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Direct test execution starts in tests/; resolve project modules first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analysis import table_statistics as stats
from scripts.analysis.table_statistics import (
    VARIANTS, MODEL_AVERAGE, _holm_adjust, summarize_main_results,
    summarize_group_retrieval_gains, summarize_generator_retrieval_gains,
)


def fixture():
    rows = []
    for index in range(24):
        for position, variant in enumerate(VARIANTS):
            score = 0.4 + 0.01 * (index % 7)
            change = (0, 0.06 + 0.01 * (index % 5), -0.04 - 0.01 * (index % 3),
                      0.01 * ((index % 5) - 2))[position]
            rows.append({"id": str(index), "patient_uuid": f"patient-{index}",
                         "model": "gpt_4o_mini", "variant": variant, "diagnosis": score + change,
                         "plan": score + change, "plan_eligible": True})
    return pd.DataFrame(rows)


class ExploratoryTwoSidedTests(unittest.TestCase):
    def test_main_checks_match_scipy_for_both_directions_and_outcomes(self):
        frame = fixture()
        frame["plan"] = 1 - frame.diagnosis
        output = summarize_main_results(frame, bootstrap_samples=32)
        for metric in ("diagnosis", "plan"):
            pivot = frame.pivot(index="id", columns="variant", values=metric)
            expected = []
            for variant in VARIANTS[1:]:
                reference = ttest_rel(pivot[variant], pivot.llm_only, alternative="two-sided")
                primary = ttest_rel(pivot[variant], pivot.llm_only, alternative="greater")
                row = output.loc[output.metric.eq(metric) & output.variant.eq(variant)].iloc[0]
                np.testing.assert_allclose(row.p_value_two_sided, reference.pvalue, rtol=1e-10, atol=0)
                np.testing.assert_allclose(row.p_value, primary.pvalue, rtol=1e-10, atol=0)
                expected.append(reference.pvalue)
                if row.baseline_delta < 0:
                    self.assertEqual(row.stars, "")
                    self.assertGreater(row.p_value, 0.5)
            selected = output.loc[output.metric.eq(metric) & output.variant.ne("llm_only")]
            np.testing.assert_allclose(selected.p_value_two_sided_holm,
                                       multipletests(expected, method="holm")[1])
            negative = selected.loc[selected.baseline_delta.lt(-0.01)].iloc[0]
            self.assertTrue(negative.significant_two_sided_holm_0_01)
            self.assertTrue(negative.significant_two_sided_holm_0_05)
        baseline = output.loc[output.variant.eq("llm_only")]
        self.assertTrue(baseline.p_value_two_sided.isna().all())
        self.assertTrue(baseline.significant_two_sided_holm_0_05.isna().all())

    def test_generator_checks_match_clustered_reference_and_include_average_family(self):
        frame = fixture()
        frame["patient_uuid"] = frame.id.astype(int).floordiv(2).astype(str)
        other = frame.assign(model="ollama_gemma3_4b", diagnosis=frame.diagnosis * 0.9 + 0.02,
                             plan=frame.plan * 0.9 + 0.02)
        frame = pd.concat([frame, other], ignore_index=True)
        output = summarize_generator_retrieval_gains(frame, include_model_average=True)
        for metric in ("diagnosis", "plan"):
            block = output.loc[output.metric.eq(metric)]
            self.assertTrue(block.family_size.eq(6).all())
            expected = []
            for row in block.itertuples():
                subset = frame if row.model == MODEL_AVERAGE else frame.loc[frame.model.eq(row.model)]
                pivot = subset.groupby(["id", "variant"])[metric].mean().unstack()
                patients = subset.drop_duplicates("id").set_index("id").loc[pivot.index, "patient_uuid"]
                delta = pivot[row.variant] - pivot.llm_only
                reference = sm.OLS(delta, np.ones((len(delta), 1))).fit(
                    cov_type="cluster", cov_kwds={"groups": patients, "use_correction": True}, use_t=True)
                np.testing.assert_allclose(row.p_value_two_sided, reference.pvalues.iloc[0], rtol=1e-10, atol=0)
                self.assertEqual(row.test_df, patients.nunique() - 1)
                expected.append(reference.pvalues.iloc[0])
            np.testing.assert_allclose(block.p_value_two_sided_holm,
                                       multipletests(expected, method="holm")[1], rtol=1e-9)

    def test_missing_checks_stay_unavailable_and_keep_their_holm_family_place(self):
        frame = fixture()
        frame.loc[frame.id.eq("0") & frame.variant.eq("patient_retrieval"), "plan"] = np.nan
        outputs = (
            (summarize_main_results(frame, bootstrap_samples=32).query("variant != 'llm_only'"), ("model",)),
            (summarize_generator_retrieval_gains(frame, include_model_average=True), ("metric",)),
        )
        for output, family_columns in outputs:
            plan = output.loc[output.metric.eq("plan")]
            missing = plan.variant.eq("patient_retrieval")
            self.assertTrue(plan.loc[missing, "p_value_two_sided"].isna().all())
            self.assertTrue(plan.loc[missing, "significant_two_sided_holm_0_05"].isna().all())
            for _, family in plan.groupby(list(family_columns), sort=False):
                expected = multipletests(family.p_value_two_sided.fillna(1), method="holm")[1]
                available = family.p_value_two_sided.notna().to_numpy()
                np.testing.assert_allclose(family.loc[available, "p_value_two_sided_holm"], expected[available])
            self.assertTrue(output.loc[output.metric.eq("diagnosis"), "p_value_two_sided"].notna().all())

    def test_single_cluster_and_zero_variance_do_not_invent_secondary_significance(self):
        frame = fixture()
        for variant, score in zip(VARIANTS, (0.4, 0.5, 0.3, 0.6)):
            frame.loc[frame.variant.eq(variant), ["diagnosis", "plan"]] = score
        for data in (frame, fixture().assign(patient_uuid="one-patient")):
            for output in (summarize_main_results(data, bootstrap_samples=32),
                           summarize_generator_retrieval_gains(data, include_model_average=True)):
                self.assertTrue(output.p_value_two_sided.isna().all())
                self.assertTrue(output.p_value_two_sided_holm.isna().all())
                self.assertTrue(output.significant_two_sided_holm_0_05.isna().all())


class BetweenGroupGainTests(unittest.TestCase):
    def frame(self):
        data = fixture()
        data["bin"] = data.id.astype(int).map(
            lambda index: "history" if index < 8 else "acute" if index < 19 else "control")
        # Repeated patients cross group boundaries; sizes are deliberately unequal.
        data["patient_uuid"] = data.id.astype(int).mod(6).astype(str)
        data["plan"] = 1 - data.diagnosis
        return pd.concat([data, data.assign(model="second", diagnosis=data.diagnosis * 0.9)],
                         ignore_index=True)

    def contrasts(self, frame):
        output = summarize_group_retrieval_gains(frame)
        return output.loc[output.history_contrast_reference_group.notna()]

    def test_direct_gain_contrasts_match_independent_clustered_ols_and_holm(self):
        frame = self.frame()
        output = self.contrasts(frame)
        self.assertEqual(len(output), 8)
        self.assertTrue(output.history_contrast_family_size.eq(4).all())
        for metric in ("diagnosis", "plan"):
            rows = output.loc[output.metric.eq(metric)]
            expected = []
            for row in rows.itertuples():
                subset = frame.loc[frame.model.eq(row.model) & frame.bin.isin(("history", row.group))]
                pivot = subset.pivot(index="id", columns="variant", values=metric)
                metadata = subset.drop_duplicates("id").set_index("id").loc[pivot.index]
                history = metadata.bin.eq("history").to_numpy()
                delta = pivot.patient_retrieval - pivot.llm_only
                design = np.column_stack([history, ~history]).astype(float)
                regression = sm.OLS(delta, design).fit(
                    cov_type="cluster", cov_kwds={"groups": metadata.patient_uuid, "use_correction": True}, use_t=True)
                reference = regression.t_test([1, -1])
                statistic = reference.tvalue.item()
                pvalue = student_t.sf(statistic, metadata.patient_uuid.nunique() - 1)
                np.testing.assert_allclose(row.history_contrast_estimate, reference.effect.item(), rtol=1e-10, atol=1e-15)
                np.testing.assert_allclose(row.history_contrast_standard_error, reference.sd.item(), rtol=1e-10)
                np.testing.assert_allclose(row.history_contrast_test_statistic, statistic, rtol=1e-10)
                np.testing.assert_allclose(row.history_contrast_p_value, pvalue, rtol=1e-10)
                np.testing.assert_allclose([row.history_contrast_ci_low, row.history_contrast_ci_high],
                                           reference.conf_int()[0], rtol=1e-10, atol=1e-15)
                self.assertEqual(row.history_contrast_test_df, metadata.patient_uuid.nunique() - 1)
                self.assertEqual(row.history_contrast_history_n_eligible, int(history.sum()))
                self.assertEqual(row.history_contrast_comparison_n_eligible, int((~history).sum()))
                expected.append(pvalue)
            adjusted = multipletests(expected, method="holm")[1]
            np.testing.assert_allclose(rows.history_contrast_p_value_holm, adjusted, rtol=1e-10)
            for adjustment, pvalues in (("raw", np.array(expected)), ("holm", adjusted)):
                for alpha in (0.01, 0.05):
                    flag = f"history_contrast_significant_{adjustment}_{str(alpha).replace('.', '_')}"
                    np.testing.assert_array_equal(rows[flag].to_numpy(dtype=bool),
                                                  (rows.history_contrast_estimate > 0) & (pvalues < alpha))
        shuffled = self.contrasts(frame.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(output.reset_index(drop=True), shuffled.reset_index(drop=True))

    def test_missing_pairs_suppress_only_affected_contrasts_and_keep_family_places(self):
        frame = self.frame()
        frame.loc[frame.model.eq("second") & frame.id.eq("8") & frame.variant.eq("patient_retrieval"), "plan"] = np.nan
        output = self.contrasts(frame)
        plan = output.loc[output.metric.eq("plan")]
        missing = plan.loc[plan.model.eq("second") & plan.group.eq("acute")].iloc[0]
        self.assertEqual(missing.history_contrast_test_status, "incomplete")
        self.assertEqual(missing.history_contrast_comparison_n_pairs, 10)
        self.assertTrue(pd.isna(missing.history_contrast_estimate))
        self.assertTrue(pd.isna(missing.history_contrast_significant_holm_0_05))
        expected = multipletests(plan.history_contrast_p_value.fillna(1), method="holm")[1]
        available = plan.history_contrast_p_value.notna().to_numpy()
        np.testing.assert_allclose(plan.loc[available, "history_contrast_p_value_holm"], expected[available])
        self.assertTrue(output.loc[output.metric.eq("diagnosis"), "history_contrast_p_value"].notna().all())
        # A missing History baseline invalidates both comparisons for that generator.
        frame.loc[frame.model.eq("second") & frame.id.eq("0") & frame.variant.eq("llm_only"), "plan"] = np.nan
        second = self.contrasts(frame).query("model == 'second' and metric == 'plan'")
        self.assertTrue(second.history_contrast_p_value.isna().all())

    def test_ineligible_references_and_absent_groups_are_explicit(self):
        frame = self.frame()
        frame.loc[frame.id.eq("0"), "plan_eligible"] = False
        frame.loc[frame.id.eq("0"), "plan"] = np.nan
        plan = self.contrasts(frame).query("metric == 'plan'")
        self.assertTrue(plan.history_contrast_history_n_eligible.eq(7).all())
        self.assertTrue(plan.history_contrast_p_value.notna().all())
        frame.loc[frame.bin.eq("history"), "plan_eligible"] = False
        plan = self.contrasts(frame).query("metric == 'plan'")
        self.assertTrue(plan.history_contrast_test_status.eq("unavailable").all())
        self.assertTrue(plan.history_contrast_p_value.isna().all())
        empty = self.contrasts(self.frame().assign(bin="history"))
        self.assertTrue(empty.history_contrast_test_status.eq("unavailable").all())

    def test_insufficient_patients_and_zero_contrast_variance_are_not_significant(self):
        frame = self.frame()
        insufficient = self.contrasts(frame.assign(patient_uuid="only-patient"))
        self.assertTrue(insufficient.history_contrast_test_status.eq("insufficient_patients").all())
        self.assertTrue(insufficient.history_contrast_p_value.isna().all())
        frame[["diagnosis", "plan"]] = 0.5
        degenerate = self.contrasts(frame)
        self.assertTrue(degenerate.history_contrast_test_status.eq("zero_variance").all())
        self.assertTrue(degenerate.history_contrast_significant_holm_0_05.isna().all())


class MainAndGroupStatisticsTests(unittest.TestCase):
    def test_group_tests_match_clustered_reference_and_adjust_within_outcome(self):
        frame = fixture()
        frame["bin"] = frame.id.astype(int).floordiv(8).map({0: "history", 1: "acute", 2: "control"})
        frame["patient_uuid"] = frame.id.astype(int).mod(6).astype(str)
        output = summarize_group_retrieval_gains(frame)
        self.assertEqual(len(output), 18)
        for metric in ("diagnosis", "plan"):
            rows = output.loc[output.metric.eq(metric)]
            expected_p = []
            for group in ("history", "acute", "control"):
                subset = frame.loc[frame.bin.eq(group)]
                pivot = subset.pivot(index="id", columns="variant", values=metric)
                patients = subset.drop_duplicates("id").set_index("id").loc[pivot.index, "patient_uuid"]
                for variant in VARIANTS[1:]:
                    delta = pivot[variant] - pivot.llm_only
                    reference = sm.OLS(delta, np.ones((len(delta), 1))).fit(
                        cov_type="cluster", cov_kwds={"groups": patients, "use_correction": True}, use_t=True)
                    row = rows.loc[rows.group.eq(group) & rows.variant.eq(variant)].iloc[0]
                    self.assertAlmostEqual(row.mean_baseline, pivot.llm_only.mean())
                    self.assertAlmostEqual(row.mean_retrieval, pivot[variant].mean())
                    self.assertAlmostEqual(row.mean_difference, delta.mean())
                    self.assertAlmostEqual(row.test_statistic, reference.tvalues.iloc[0])
                    pvalue = student_t.sf(reference.tvalues.iloc[0], patients.nunique() - 1)
                    self.assertAlmostEqual(row.p_value, pvalue)
                    expected_p.append(pvalue)
            np.testing.assert_allclose(rows.p_value_holm, multipletests(expected_p, method="holm")[1])

    def test_group_excludes_unavailable_plans_masks_missing_scores_and_retains_empty_groups(self):
        frame = fixture()
        frame["bin"] = "history"
        frame = frame.loc[frame.variant.isin(("llm_only", "patient_retrieval"))].copy()
        frame.loc[frame.id.eq("0"), "plan_eligible"] = False
        frame.loc[frame.id.eq("0"), "plan"] = np.nan
        frame.loc[frame.id.eq("1") & frame.variant.eq("patient_retrieval"), "plan"] = np.nan
        output = summarize_group_retrieval_gains(frame)
        plan = output.loc[output.metric.eq("plan") & output.group.eq("history")].iloc[0]
        self.assertEqual(plan.n_eligible, 23)
        self.assertFalse(plan.coverage_complete)
        self.assertTrue(pd.notna(plan.mean_baseline))
        self.assertTrue(pd.isna(plan.mean_retrieval))
        self.assertTrue(pd.isna(plan.mean_difference))
        self.assertTrue(pd.isna(plan.p_value_holm))
        empty = output.loc[output.group.ne("history")]
        self.assertTrue(empty.test_status.eq("unavailable").all())
        self.assertTrue(empty.stars.eq("").all())
        diagnosis = output.loc[output.metric.eq("diagnosis") & output.group.eq("history")].iloc[0]
        self.assertAlmostEqual(diagnosis.p_value_holm, min(1, 9 * diagnosis.p_value))

    def test_one_case_per_patient_matches_scipy_paired_t_test_for_both_outcomes(self):
        frame = fixture()
        result = summarize_main_results(frame, 128)
        for metric in ("diagnosis", "plan"):
            pivot = frame.pivot(index="id", columns="variant", values=metric)
            for variant in VARIANTS[1:]:
                reference = ttest_rel(pivot[variant], pivot["llm_only"], alternative="greater")
                row = result.loc[result.metric.eq(metric) & result.variant.eq(variant)].iloc[0]
                self.assertAlmostEqual(row.test_statistic, reference.statistic)
                self.assertAlmostEqual(row.p_value, reference.pvalue)
                self.assertEqual(row.test_df, 23)
            block = result.loc[result.metric.eq(metric)].set_index("variant")
            self.assertEqual(block.loc["patient_retrieval", "stars"], "**")
            self.assertEqual(block.loc["guidelines", "stars"], "")
            self.assertEqual(block.loc["llm_only", "stars"], "")
            self.assertTrue(pd.isna(block.loc["llm_only", "p_value"]))

    def test_repeated_patients_match_statsmodels_clustered_regression(self):
        frame = fixture()
        frame["patient_uuid"] = frame["id"].astype(int).floordiv(3).astype(str)
        result = summarize_main_results(frame, 128)
        pivot = frame.pivot(index="id", columns="variant", values="diagnosis")
        patients = frame.drop_duplicates("id").set_index("id").loc[pivot.index, "patient_uuid"]
        delta = pivot["patient_retrieval"] - pivot["llm_only"]
        fit = sm.OLS(delta, np.ones((len(delta), 1))).fit(
            cov_type="cluster", cov_kwds={"groups": patients, "use_correction": True}, use_t=True,
        )
        row = result.loc[result.metric.eq("diagnosis") & result.variant.eq("patient_retrieval")].iloc[0]
        self.assertAlmostEqual(row.test_statistic, fit.tvalues.iloc[0])
        self.assertAlmostEqual(row.p_value, fit.pvalues.iloc[0] / 2)
        self.assertEqual(row.test_df, 7)

    def test_holm_matches_reference_and_retains_missing_hypotheses(self):
        for values in ([0.009, 0.03, 0.6], [0.02, 0.001, 0.02], [0, 1, 0.05]):
            np.testing.assert_allclose(_holm_adjust(values), multipletests(values, method="holm")[1])
        np.testing.assert_allclose(_holm_adjust([0.02, np.nan, np.nan]), [0.06, np.nan, np.nan])

    def test_missing_scores_and_missing_baseline_are_never_used_for_stars(self):
        frame = fixture()
        frame.loc[frame.id.eq("0") & frame.variant.eq("patient_retrieval"), "plan"] = np.nan
        frame.loc[frame.id.eq("1") & frame.variant.eq("llm_only"), "diagnosis"] = np.nan
        result = summarize_main_results(frame, 64)
        plan = result.loc[result.metric.eq("plan") & result.variant.eq("patient_retrieval")].iloc[0]
        self.assertFalse(plan.coverage_complete)
        self.assertTrue(pd.isna(plan.mean_score))
        self.assertTrue(pd.isna(plan.p_value))
        self.assertEqual(plan.stars, "")
        diagnosis = result.loc[result.metric.eq("diagnosis")]
        self.assertTrue(diagnosis.p_value.isna().all())
        self.assertTrue(diagnosis.stars.eq("").all())
        self.assertEqual(diagnosis.loc[diagnosis.variant.eq("patient_retrieval"), "mean_score"].notna().sum(), 1)

    def test_cluster_bootstrap_is_reproducible_case_weighted_and_not_iid(self):
        frame = fixture()
        # Unequal cluster sizes and different cluster outcomes expose patient-mean reweighting.
        frame["patient_uuid"] = np.where(frame.id.astype(int) < 20, "large", "small")
        frame["diagnosis"] = np.where(frame.id.astype(int) < 20, 0.2, 0.8)
        first = summarize_main_results(frame, 256, seed=17)
        pd.testing.assert_frame_equal(first, summarize_main_results(frame, 256, seed=17))
        self.assertAlmostEqual(first.iloc[0].mean_score, 0.3)
        self.assertAlmostEqual(first.iloc[0].mean_ci_low, 0.2)
        self.assertAlmostEqual(first.iloc[0].mean_ci_high, 0.8)
        self.assertTrue(first.loc[first.metric.eq("diagnosis"), "stars"].eq("").all())

    def test_unavailable_reference_single_patient_and_degenerate_differences(self):
        frame = fixture()
        frame["plan_eligible"] = False
        frame["patient_uuid"] = "only-patient"
        result = summarize_main_results(frame, 32)
        self.assertTrue(result.loc[result.metric.eq("plan"), "mean_score"].isna().all())
        self.assertTrue(result.loc[result.metric.eq("diagnosis"), "mean_ci_low"].isna().all())
        self.assertTrue(result.stars.eq("").all())
        frame = fixture()
        frame["diagnosis"] = 0.5
        result = summarize_main_results(frame, 32)
        self.assertTrue(result.loc[result.metric.eq("diagnosis") & result.variant.ne("llm_only"),
                                   "test_status"].eq("zero_variance").all())
        for samples in (1, True, 2.5):
            with self.assertRaises(ValueError):
                summarize_main_results(frame, samples)


class BaselineSignificanceTests(unittest.TestCase):
    def test_both_outcomes_match_independent_paired_t_reference(self):
        frame = fixture()
        result = summarize_generator_retrieval_gains(frame)
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result.variant), {"patient_retrieval", "guidelines"})
        for metric in ("diagnosis", "plan"):
            pivot = frame.pivot(index="id", columns="variant", values=metric)
            block = result.loc[result.metric.eq(metric)].set_index("variant")
            for variant in block.index:
                reference = ttest_rel(pivot[variant], pivot.llm_only, alternative="greater")
                row = block.loc[variant]
                self.assertAlmostEqual(row.test_statistic, reference.statistic)
                self.assertAlmostEqual(row.p_value, reference.pvalue)
                self.assertAlmostEqual(row.mean_difference, (pivot[variant] - pivot.llm_only).mean())
                self.assertEqual((row.n_pairs, row.n_patients, row.test_df), (24, 24, 23))
            expected = multipletests(block.p_value.to_numpy(), method="holm")[1]
            np.testing.assert_allclose(block.p_value_holm, expected)
            self.assertTrue(block.loc["patient_retrieval", "significant_holm_0_01"])
            self.assertFalse(block.loc["guidelines", "significant_raw_0_05"])

    def test_patient_clusters_and_shuffled_rows_match_clustered_regression(self):
        frame = fixture()
        frame["patient_uuid"] = frame.id.astype(int).floordiv(3).astype(str)
        result = summarize_generator_retrieval_gains(frame, metrics=("diagnosis",))
        pivot = frame.pivot(index="id", columns="variant", values="diagnosis")
        patients = frame.drop_duplicates("id").set_index("id").loc[pivot.index, "patient_uuid"]
        delta = pivot.patient_retrieval - pivot.llm_only
        fit = sm.OLS(delta, np.ones((len(delta), 1))).fit(
            cov_type="cluster", cov_kwds={"groups": patients, "use_correction": True}, use_t=True,
        )
        row = result.loc[result.variant.eq("patient_retrieval")].iloc[0]
        self.assertAlmostEqual(row.test_statistic, fit.tvalues.iloc[0])
        self.assertAlmostEqual(row.p_value, fit.pvalues.iloc[0] / 2)
        self.assertEqual(row.test_df, 7)
        pd.testing.assert_frame_equal(result, summarize_generator_retrieval_gains(
            frame.sample(frac=1, random_state=91), metrics=("diagnosis",),
        ))

    def test_holm_family_includes_both_comparisons_across_models_per_outcome(self):
        first = fixture()
        second = first.assign(model="second-model")
        result = summarize_generator_retrieval_gains(pd.concat([first, second], ignore_index=True))
        self.assertTrue(result.family_size.eq(4).all())
        for metric in ("diagnosis", "plan"):
            block = result.loc[result.metric.eq(metric)]
            expected = multipletests(block.p_value.to_numpy(), method="holm")[1]
            np.testing.assert_allclose(block.p_value_holm, expected)

    def test_requested_alpha_flags_use_strict_thresholds_and_distinguish_adjustment(self):
        with patch("scripts.analysis.table_statistics._paired_cluster_test",
                   return_value=(2.0, 23, 0.02, "tested")):
            result = summarize_generator_retrieval_gains(fixture(), metrics=("diagnosis",))
        positive = result.loc[result.variant.eq("patient_retrieval")].iloc[0]
        self.assertAlmostEqual(positive.p_value_holm, 0.04)
        self.assertTrue(positive.significant_raw_0_05)
        self.assertTrue(positive.significant_holm_0_05)
        self.assertFalse(positive.significant_holm_0_01)
        with patch("scripts.analysis.table_statistics._paired_cluster_test",
                   side_effect=[(2.0, 23, 0.01, "tested"), (1.0, 23, 0.05, "tested")]):
            result = summarize_generator_retrieval_gains(fixture(), metrics=("diagnosis",))
        self.assertFalse(result.iloc[0].significant_raw_0_01)
        self.assertFalse(result.iloc[1].significant_raw_0_05)
        self.assertFalse(result.iloc[1].significant_holm_0_05)

    def test_missing_pair_suppresses_inference_without_selecting_complete_cases(self):
        frame = fixture()
        frame.loc[frame.id.eq("0") & frame.variant.eq("guidelines"), "plan"] = np.nan
        result = summarize_generator_retrieval_gains(frame)
        missing = result.loc[result.metric.eq("plan") & result.variant.eq("guidelines")].iloc[0]
        self.assertEqual((missing.n_eligible, missing.n_pairs), (24, 23))
        self.assertEqual(missing.test_status, "incomplete")
        self.assertTrue(pd.isna(missing.mean_retrieval))
        self.assertTrue(pd.isna(missing.p_value))
        self.assertTrue(pd.isna(missing.significant_holm_0_05))
        available = result.loc[result.metric.eq("plan") & result.variant.eq("patient_retrieval")].iloc[0]
        self.assertAlmostEqual(available.p_value_holm, 2 * available.p_value)
        self.assertEqual(available.family_size, 2)
        self.assertTrue(result.loc[result.metric.eq("diagnosis"), "p_value"].notna().all())
        frame.loc[frame.id.eq("1") & frame.variant.eq("llm_only"), "diagnosis"] = np.nan
        result = summarize_generator_retrieval_gains(frame, metrics=("diagnosis",))
        self.assertTrue(result.test_status.eq("incomplete").all())
        self.assertTrue(result.p_value.isna().all())

    def test_missing_arm_retains_comparison_and_ineligible_plan_references_are_excluded(self):
        frame = fixture()
        frame.loc[frame.id.eq("0"), "plan_eligible"] = False
        frame.loc[frame.id.eq("0"), "plan"] = np.nan
        result = summarize_generator_retrieval_gains(frame)
        self.assertTrue(result.loc[result.metric.eq("plan"), "n_eligible"].eq(23).all())
        self.assertTrue(result.loc[result.metric.eq("plan"), "coverage_complete"].all())
        without_guidelines = frame.loc[frame.variant.ne("guidelines")]
        result = summarize_generator_retrieval_gains(without_guidelines, metrics=("diagnosis",))
        missing = result.loc[result.variant.eq("guidelines")].iloc[0]
        self.assertEqual((missing.n_retrieval, missing.n_pairs, missing.family_size), (0, 0, 2))
        self.assertEqual(missing.test_status, "incomplete")

    def test_case_weighted_mean_and_unavailable_tests_are_explicit(self):
        frame = fixture()
        frame["patient_uuid"] = np.where(frame.id.astype(int) < 20, "large", "small")
        frame["diagnosis"] = 0.1
        frame.loc[frame.variant.eq("patient_retrieval") & frame.id.astype(int).lt(20), "diagnosis"] = 0.5
        result = summarize_generator_retrieval_gains(frame, metrics=("diagnosis",))
        row = result.loc[result.variant.eq("patient_retrieval")].iloc[0]
        self.assertAlmostEqual(row.mean_difference, 20 * 0.4 / 24)
        self.assertNotAlmostEqual(row.mean_difference, 0.2)
        frame = fixture()
        frame["patient_uuid"] = "one-patient"
        frame["plan_eligible"] = False
        result = summarize_generator_retrieval_gains(frame)
        self.assertTrue(result.loc[result.metric.eq("plan"), "test_status"].eq("unavailable").all())
        self.assertTrue(result.loc[result.metric.eq("diagnosis"), "test_status"].eq("insufficient_patients").all())
        self.assertTrue(result.significant_raw_0_05.isna().all())
        frame = fixture()
        frame["diagnosis"] = 0.5
        result = summarize_generator_retrieval_gains(frame, metrics=("diagnosis",))
        self.assertTrue(result.test_status.eq("zero_variance").all())
        self.assertTrue(result.significant_holm_0_01.isna().all())

    def test_invalid_input_is_rejected(self):
        for metrics in ((), "diagnosis", ("diagnosis", "diagnosis"), ("unknown",)):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                summarize_generator_retrieval_gains(fixture(), metrics=metrics)
        for bad in (np.inf, -0.1, 1.1):
            frame = fixture()
            frame.loc[0, "diagnosis"] = bad
            with self.subTest(score=bad), self.assertRaises(ValueError):
                summarize_generator_retrieval_gains(frame)
        for frame in (fixture().iloc[:0], fixture().drop(columns="patient_uuid"),
                      pd.concat([fixture(), fixture().iloc[:1]], ignore_index=True)):
            with self.assertRaises(ValueError):
                summarize_generator_retrieval_gains(frame)
        frame = fixture()
        frame.loc[0, "patient_uuid"] = "inconsistent-patient"
        with self.assertRaises(ValueError):
            summarize_generator_retrieval_gains(frame)


class ModelAverageSignificanceTests(unittest.TestCase):
    def test_equal_generator_case_averages_match_paired_reference_for_both_scores(self):
        first = fixture()
        second = first.assign(model="second-model").copy()
        patient = second.variant.eq("patient_retrieval")
        second.loc[patient, "diagnosis"] = 1 - second.loc[patient, "diagnosis"]
        second["plan"] = second["plan"] * 0.5 + 0.1
        frame = pd.concat([first, second], ignore_index=True)
        summary = summarize_generator_retrieval_gains(frame, include_model_average=True)
        self.assertTrue(summary.family_size.eq(6).all())
        for metric in ("diagnosis", "plan"):
            means = frame.groupby(["id", "variant"])[metric].mean().unstack()
            for variant in ("patient_retrieval", "guidelines"):
                row = summary.loc[summary.model.eq(MODEL_AVERAGE) & summary.metric.eq(metric)
                                  & summary.variant.eq(variant)].iloc[0]
                expected = ttest_rel(means[variant], means.llm_only, alternative="greater")
                self.assertAlmostEqual(row.mean_difference, (means[variant] - means.llm_only).mean())
                self.assertAlmostEqual(row.test_statistic, expected.statistic)
                self.assertAlmostEqual(row.p_value, expected.pvalue)
                self.assertEqual((row.n_pairs, row.n_patients, row.n_generators_averaged), (24, 24, 2))
                first_row = summary.loc[summary.metric.eq(metric) & summary.variant.eq(variant)].iloc[0]
                self.assertEqual(first_row.model, MODEL_AVERAGE)
            block = summary.loc[summary.metric.eq(metric)]
            np.testing.assert_allclose(block.p_value_holm, multipletests(block.p_value, method="holm")[1])

    def test_identical_generators_do_not_create_independent_patient_replicates(self):
        frame = pd.concat([fixture().assign(model=f"model-{i}") for i in range(4)], ignore_index=True)
        frame["patient_uuid"] = frame.id.astype(int).floordiv(3).astype(str)
        summary = summarize_generator_retrieval_gains(frame, metrics=("diagnosis",), include_model_average=True)
        self.assertTrue(summary.family_size.eq(10).all())
        average = summary.loc[summary.model.eq(MODEL_AVERAGE) & summary.variant.eq("patient_retrieval")].iloc[0]
        single = summary.loc[summary.model.eq("model-0") & summary.variant.eq("patient_retrieval")].iloc[0]
        self.assertAlmostEqual(average.test_statistic, single.test_statistic)
        self.assertAlmostEqual(average.p_value, single.p_value)
        self.assertEqual((average.n_pairs, average.n_patients, average.test_df), (24, 8, 7))

    def test_missing_generator_score_is_not_averaged_over_available_generators(self):
        frame = pd.concat([fixture(), fixture().assign(model="second-model")], ignore_index=True)
        frame.loc[frame.model.eq("second-model") & frame.id.eq("0")
                  & frame.variant.eq("patient_retrieval"), "plan"] = np.nan
        summary = summarize_generator_retrieval_gains(frame, include_model_average=True)
        row = summary.loc[summary.model.eq(MODEL_AVERAGE) & summary.metric.eq("plan")
                          & summary.variant.eq("patient_retrieval")].iloc[0]
        self.assertEqual((row.n_eligible, row.n_pairs), (24, 23))
        self.assertEqual(row.test_status, "incomplete")
        self.assertTrue(pd.isna(row.mean_retrieval))
        self.assertTrue(pd.isna(row.p_value))
        self.assertTrue(pd.isna(row.significant_holm_0_05))
        frame.loc[frame.id.eq("0"), "plan_eligible"] = False
        summary = summarize_generator_retrieval_gains(frame, include_model_average=True)
        row = summary.loc[summary.model.eq(MODEL_AVERAGE) & summary.metric.eq("plan")
                          & summary.variant.eq("patient_retrieval")].iloc[0]
        self.assertEqual((row.n_eligible, row.n_pairs), (23, 23))
        self.assertEqual(row.test_status, "tested")

    def test_missing_whole_generator_case_does_not_shrink_average_cohort(self):
        frame = pd.concat([fixture(), fixture().assign(model="second-model")], ignore_index=True)
        frame = frame.loc[~(frame.model.eq("second-model") & frame.id.eq("0"))]
        summary = summarize_generator_retrieval_gains(frame, metrics=("diagnosis",), include_model_average=True)
        average = summary.loc[summary.model.eq(MODEL_AVERAGE)]
        self.assertTrue(average.n_eligible.eq(24).all())
        self.assertTrue(average.n_pairs.eq(23).all())
        self.assertTrue(average.p_value.isna().all())

    def test_average_option_and_reserved_model_key_are_validated(self):
        for flag in (1, "yes", None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                summarize_generator_retrieval_gains(fixture(), include_model_average=flag)
        with self.assertRaisesRegex(ValueError, "reserved"):
            summarize_generator_retrieval_gains(fixture().assign(model=MODEL_AVERAGE), include_model_average=True)


ALL_MODELS = ("gpt_4o_mini", "ollama_gemma3_4b", "ollama_llama3_2_3b", "ollama_ministral_3_3b")

def normalized_fixture():
    rows = []
    for i in range(24):
        for m, model in enumerate(ALL_MODELS):
            for v, variant in enumerate(stats.VARIANT_LABELS):
                score = 0.2 + 0.004 * i + 0.012 * m + v * (0.012 + 0.0006 * i)
                rows.append({'id': str(i), 'patient_uuid': str(i // 2), 'model': model,
                             'variant': variant, 'bin': ("history", "acute", "control")[i % 3],
                             'diagnosis': score, 'plan': score * 0.8 + 0.03,
                             'plan_eligible': i != 23})
    return pd.DataFrame(rows)


class AvailableScoreStatisticsTests(unittest.TestCase):
    def test_average_never_silently_drops_a_generator_or_variant(self):
        frame = normalized_fixture()
        for incomplete in (frame.loc[frame.model.ne(ALL_MODELS[0])],
                           frame.loc[frame.variant.ne('agentic_both')]):
            with self.subTest(rows=len(incomplete)), self.assertRaisesRegex(ValueError, 'four generators'):
                stats.summarize_generator_retrieval_gains(incomplete, average_only=True, available_scores=True)

    def test_all_missing_outcomes_keep_schema_and_planned_families_without_inference(self):
        frame = normalized_fixture().assign(diagnosis=np.nan, plan=np.nan)
        main = stats.summarize_main_results(frame, 16, available_scores=True)
        average = stats.summarize_generator_retrieval_gains(frame, average_only=True, available_scores=True)
        group = stats.summarize_group_retrieval_gains(frame, available_scores=True)
        self.assertEqual((len(main), len(average), len(group)), (56, 12, 144))
        for output, estimate, family_size in ((main, 'mean_score', 6), (average, 'mean_difference', 6),
                                               (group, 'mean_difference', 18)):
            self.assertTrue(output[estimate].isna().all())
            self.assertTrue(output.p_value_holm.isna().all())
            self.assertTrue(output.n_pairs.eq(0).all())
            self.assertTrue(output.family_size.eq(family_size).all())
        self.assertTrue(group[['paired_mean_baseline', 'paired_mean_retrieval']].isna().all().all())

    def test_model_average_pairs_and_holm_match_independent_clustered_ols(self):
        frame = normalized_fixture()
        frame.loc[frame.id.eq('0') & frame.variant.eq('agentic_both') & frame.model.eq(ALL_MODELS[0]), 'plan'] = np.nan
        frame.loc[frame.id.eq('2') & frame.variant.eq('llm_only') & frame.model.eq(ALL_MODELS[1]), 'plan'] = np.nan
        frame = frame.loc[~(frame.id.eq('4') & frame.variant.eq('agentic_guidelines') & frame.model.eq(ALL_MODELS[2]))]
        output = stats.summarize_generator_retrieval_gains(frame, average_only=True, available_scores=True)
        self.assertEqual(len(output), 12)
        self.assertEqual(set(output.model), {'model_average'})
        for metric in ('diagnosis', 'plan'):
            expected_pvalues = []
            block = output.loc[output.metric.eq(metric)]
            for row in block.itertuples():
                eligible = frame.loc[frame.plan_eligible] if metric == 'plan' else frame
                pivot = eligible.pivot(index=['id', 'variant'], columns='model', values=metric).reindex(columns=ALL_MODELS)
                means = pivot.mean(axis=1, skipna=False).unstack('variant')
                pair = means[[row.variant, 'llm_only']].dropna()
                delta = pair[row.variant] - pair.llm_only
                patients = eligible.drop_duplicates('id').set_index('id').loc[pair.index, 'patient_uuid']
                fit = sm.OLS(delta, np.ones((len(delta), 1))).fit(
                    cov_type='cluster', cov_kwds={'groups': patients, 'use_correction': True}, use_t=True)
                self.assertEqual(row.n_pairs, len(pair))
                self.assertEqual(row.test_df, patients.nunique() - 1)
                self.assertEqual(row.n_generators_averaged, 4)
                self.assertAlmostEqual(row.mean_difference, delta.mean())
                self.assertAlmostEqual(row.p_value, student_t.sf(fit.tvalues.iloc[0], patients.nunique() - 1))
                expected_pvalues.append(row.p_value)
            np.testing.assert_allclose(block.p_value_holm, multipletests(expected_pvalues, method='holm')[1])
            self.assertTrue(block.family_size.eq(6).all())
        plan = output.loc[output.metric.eq('plan') & output.variant.eq('agentic_both')].iloc[0]
        self.assertEqual(plan.n_pairs, 21)  # 23 eligible - missing baseline - missing retrieval.


    def test_missing_comparison_retains_average_only_family(self):
        frame = normalized_fixture()
        frame.loc[frame.variant.eq('agentic_both'), 'plan'] = np.nan
        out = stats.summarize_generator_retrieval_gains(frame, average_only=True, available_scores=True)
        plan = out.loc[out.metric.eq('plan')]
        self.assertTrue(plan.family_size.eq(6).all())
        missing = plan.variant.eq('agentic_both')
        self.assertTrue(plan.loc[missing, 'p_value'].isna().all())
        expected = multipletests(plan.p_value.fillna(1), method='holm')[1]
        np.testing.assert_allclose(plan.loc[~missing, 'p_value_holm'], expected[~missing])


    def test_main_means_include_zero_and_use_only_available_pairs(self):
        frame = normalized_fixture()
        mask = frame.model.eq(ALL_MODELS[0]) & frame.variant.eq('agentic_both')
        frame.loc[mask & frame.id.eq('0'), 'plan'] = 0
        frame.loc[mask & frame.id.eq('2'), 'plan'] = np.nan
        out = stats.summarize_main_results(frame, 64, seed=17, available_scores=True)
        row = out.loc[out.model.eq(ALL_MODELS[0]) & out.variant.eq('agentic_both') & out.metric.eq('plan')].iloc[0]
        values = frame.loc[mask & frame.plan_eligible, 'plan']
        self.assertFalse(row.coverage_complete)
        self.assertAlmostEqual(row.mean_score, values.mean())
        self.assertEqual(row.n_scored, values.notna().sum())
        self.assertEqual(row.n_pairs, row.n_scored)
        self.assertTrue(np.isfinite(row.mean_ci_low))
        self.assertEqual(row.family_size, 6)
        pd.testing.assert_frame_equal(out, stats.summarize_main_results(frame, 64, seed=17, available_scores=True))


    def test_group_deltas_and_history_contrasts_use_available_pairs(self):
        frame = normalized_fixture()
        frame.loc[frame.id.eq('0') & frame.variant.eq('patient_retrieval') & frame.model.eq(ALL_MODELS[0]), 'plan'] = np.nan
        out = stats.summarize_group_retrieval_gains(frame, available_scores=True)
        row = out.loc[out.model.eq(ALL_MODELS[0]) & out.variant.eq('patient_retrieval') & out.metric.eq('plan')
                      & out.group.eq('history')].iloc[0]
        subset = frame.loc[frame.model.eq(ALL_MODELS[0]) & frame.bin.eq('history') & frame.plan_eligible]
        pair = subset.pivot(index='id', columns='variant', values='plan')[['llm_only', 'patient_retrieval']].dropna()
        self.assertFalse(row.coverage_complete)
        self.assertAlmostEqual(row.mean_difference, (pair.patient_retrieval - pair.llm_only).mean())
        self.assertAlmostEqual(row.paired_mean_retrieval - row.paired_mean_baseline, row.mean_difference)
        self.assertEqual(row.family_size, 18)
        contrast = out.loc[out.model.eq(ALL_MODELS[0]) & out.variant.eq('patient_retrieval') & out.metric.eq('plan')
                           & out.group.eq('acute')].iloc[0]
        cohort = frame.loc[frame.model.eq(ALL_MODELS[0]) & frame.plan_eligible & frame.bin.isin(['history', 'acute'])]
        pivot = cohort.pivot(index='id', columns='variant', values='plan')[['llm_only', 'patient_retrieval']].dropna()
        metadata = cohort.drop_duplicates('id').set_index('id').loc[pivot.index]
        delta = pivot.patient_retrieval - pivot.llm_only
        design = np.column_stack([metadata.bin.eq('history'), metadata.bin.eq('acute')]).astype(float)
        fit = sm.OLS(delta, design).fit(cov_type='cluster', cov_kwds={'groups': metadata.patient_uuid,
                                                                  'use_correction': True}, use_t=True)
        test = fit.t_test([1, -1])
        self.assertAlmostEqual(contrast.history_contrast_estimate, float(test.effect[0]))
        self.assertAlmostEqual(contrast.history_contrast_standard_error, float(test.sd[0, 0]))
        self.assertEqual(contrast.history_contrast_test_status, 'tested')



class AgenticCounterpartStatisticsTests(unittest.TestCase):
    def frame(self):
        data = normalized_fixture()
        data['plan'] = 1 - data.plan
        # Different effects/noise across counterparts; some effects are negative.
        for variant, scale in (('agentic_patient_retrieval', .003), ('agentic_guidelines', -.004),
                               ('agentic_both', .007)):
            mask = data.variant.eq(variant)
            data.loc[mask, 'diagnosis'] += scale * (data.loc[mask, 'id'].astype(int) % 5 - 2)
            data.loc[mask, 'plan'] += scale * (data.loc[mask, 'id'].astype(int) % 7 - 3)
        return data

    def test_six_signed_effects_and_two_sided_tests_match_independent_clustered_ols(self):
        frame = self.frame()
        # Missing one model invalidates only that case/outcome/counterpart, not the whole cohort.
        frame.loc[frame.id.eq('0') & frame.model.eq(ALL_MODELS[0])
                  & frame.variant.eq('agentic_patient_retrieval'), 'diagnosis'] = np.nan
        frame.loc[frame.id.eq('2') & frame.model.eq(ALL_MODELS[1])
                  & frame.variant.eq('guidelines'), 'plan'] = np.nan
        frame.loc[frame.id.eq('4') & frame.model.isin(ALL_MODELS[2:])
                  & frame.variant.eq('agentic_both'), 'plan'] = np.nan
        frame.loc[frame.id.eq('6') & frame.variant.eq('agentic_patient_retrieval'), 'plan'] = 0
        original = frame.copy(deep=True)
        output = stats.summarize_agentic_counterpart_comparisons(frame)
        self.assertEqual(len(output), 6)
        self.assertEqual(set(zip(output.agentic_variant, output.nonagentic_variant)), {
            ('agentic_patient_retrieval', 'patient_retrieval'), ('agentic_guidelines', 'guidelines'),
            ('agentic_both', 'patient_retrieval_guidelines')})
        self.assertTrue(output.n_generators_averaged.eq(4).all())
        self.assertTrue(output.model.eq('model_average').all())
        self.assertTrue(output.alternative.eq('two-sided').all())
        expected_pvalues = []
        for row in output.itertuples():
            eligible = frame.loc[frame.plan_eligible] if row.metric == 'plan' else frame
            scores = eligible.pivot(index=['id', 'variant'], columns='model', values=row.metric)
            means = scores.reindex(columns=ALL_MODELS).mean(axis=1, skipna=False).unstack('variant')
            pairs = means[[row.agentic_variant, row.nonagentic_variant]].dropna()
            differences = pairs[row.agentic_variant] - pairs[row.nonagentic_variant]
            patients = eligible.drop_duplicates('id').set_index('id').loc[pairs.index, 'patient_uuid']
            fit = sm.OLS(differences, np.ones((len(differences), 1))).fit(
                cov_type='cluster', cov_kwds={'groups': patients, 'use_correction': True}, use_t=True)
            self.assertEqual(row.n_eligible, eligible.id.nunique())
            self.assertEqual(row.n_pairs, len(pairs))
            self.assertEqual(row.n_patients, patients.nunique())
            self.assertEqual(row.test_df, patients.nunique() - 1)
            self.assertAlmostEqual(row.mean_agentic, pairs[row.agentic_variant].mean())
            self.assertAlmostEqual(row.mean_nonagentic, pairs[row.nonagentic_variant].mean())
            self.assertAlmostEqual(row.mean_difference, differences.mean())
            self.assertAlmostEqual(row.test_statistic, fit.tvalues.iloc[0])
            self.assertAlmostEqual(row.p_value_two_sided, fit.pvalues.iloc[0])
            expected_pvalues.append(fit.pvalues.iloc[0])
        adjusted = multipletests(expected_pvalues, method='holm')[1]
        np.testing.assert_allclose(output.p_value_two_sided_holm, adjusted, rtol=1e-10)
        self.assertTrue(output.family_size.eq(6).all())
        for adjustment, pvalues in (('raw', np.array(expected_pvalues)), ('holm', adjusted)):
            for alpha in (.01, .05):
                column = f'significant_two_sided_{adjustment}_{str(alpha).replace(".", "_")}'
                np.testing.assert_array_equal(output[column].to_numpy(dtype=bool), pvalues < alpha)
        self.assertTrue(output.mean_difference.gt(0).any())
        self.assertTrue(output.mean_difference.lt(0).any())
        pd.testing.assert_frame_equal(frame, original)

    def test_order_invariance_and_no_dependence_on_llm_only_scores(self):
        frame = self.frame()
        expected = stats.summarize_agentic_counterpart_comparisons(frame)
        frame.loc[frame.variant.eq('llm_only'), ['diagnosis', 'plan']] = np.nan
        pd.testing.assert_frame_equal(expected, stats.summarize_agentic_counterpart_comparisons(
            frame.sample(frac=1, random_state=13)))

    def test_missing_outcome_reserves_all_six_holm_places_and_never_becomes_zero(self):
        frame = self.frame().assign(plan=np.nan)
        output = stats.summarize_agentic_counterpart_comparisons(frame)
        missing = output.metric.eq('plan')
        self.assertTrue(output.loc[missing, ['mean_nonagentic', 'mean_agentic', 'mean_difference',
                                            'p_value_two_sided', 'p_value_two_sided_holm']].isna().all().all())
        self.assertTrue(output.loc[missing, 'n_pairs'].eq(0).all())
        self.assertTrue(output.loc[missing, 'n_patients'].eq(0).all())
        self.assertTrue(output.loc[missing, 'significant_two_sided_holm_0_05'].isna().all())
        adjusted = multipletests(output.p_value_two_sided.fillna(1), method='holm')[1]
        np.testing.assert_allclose(output.loc[~missing, 'p_value_two_sided_holm'], adjusted[~missing])
        self.assertTrue(output.family_size.eq(6).all())

    def test_unavailable_references_and_unpaired_arms_do_not_produce_paired_means(self):
        frame = self.frame().assign(plan_eligible=False)
        plans = stats.summarize_agentic_counterpart_comparisons(frame).query("metric == 'plan'")
        self.assertTrue(plans.n_eligible.eq(0).all())
        self.assertTrue(plans.test_status.eq('unavailable').all())
        self.assertTrue(plans.p_value_two_sided.isna().all())
        frame = self.frame()
        frame.loc[frame.variant.eq('agentic_both'), 'plan'] = np.nan
        row = stats.summarize_agentic_counterpart_comparisons(frame).query(
            "metric == 'plan' and agentic_variant == 'agentic_both'").iloc[0]
        self.assertEqual(row.n_nonagentic, 23)
        self.assertEqual(row.n_agentic, 0)
        self.assertTrue(pd.isna(row.mean_nonagentic))
        self.assertTrue(pd.isna(row.mean_agentic))

    def test_insufficient_clusters_and_constant_differences_never_claim_significance(self):
        constant = self.frame().assign(diagnosis=.4, plan=.4)
        constant.loc[constant.variant.str.startswith('agentic_'), 'diagnosis'] = .5
        for frame, status in ((constant, 'zero_variance'),
                              (self.frame().assign(patient_uuid='one-patient'), 'insufficient_patients')):
            with self.subTest(status=status):
                output = stats.summarize_agentic_counterpart_comparisons(frame)
                self.assertTrue(output.test_status.eq(status).all())
                self.assertTrue(output.p_value_two_sided.isna().all())
                self.assertTrue(output.p_value_two_sided_holm.isna().all())
                self.assertTrue(output.significant_two_sided_holm_0_05.isna().all())

    def test_incomplete_slots_duplicate_keys_bad_identity_and_bad_scores_are_rejected(self):
        frame = self.frame()
        duplicate = frame.copy()
        duplicate.iloc[1] = duplicate.iloc[0]
        inconsistent = frame.copy()
        inconsistent.loc[0, 'patient_uuid'] = 'wrong-patient'
        for invalid in (frame.iloc[1:], frame.loc[frame.model.ne(ALL_MODELS[0])],
                        frame.loc[frame.variant.ne('agentic_both')], duplicate, inconsistent,
                        frame.assign(diagnosis=np.inf), frame.assign(plan=-.1),
                        frame.drop(columns='plan_eligible')):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                stats.summarize_agentic_counterpart_comparisons(invalid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
