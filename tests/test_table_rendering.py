"""Validate report-table layouts and score availability."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

# Direct test execution starts in tests/; resolve project modules first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analysis import table_rendering as rendering
from scripts.analysis.table_statistics import VARIANT_LABELS


class RetrievalGainRenderingTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame([
            {"model": model, "metric": metric, "variant": variant,
             "baseline_variant": "llm_only", "mean_difference": gain,
             "p_value_holm": pvalue, "coverage_complete": True,
             "family_size": 6, "n_generators_averaged": 2 if model == "model_average" else 1}
            for model, gain, pvalue in (("model_average", 0.0617, 0.009),
                                        ("gpt_4o_mini", 0.1234, 0.03),
                                        ("ollama_gemma3_4b", -0.0101, 0.001))
            for variant in ("patient_retrieval", "guidelines")
            for metric in ("diagnosis", "plan")
        ])

    def test_source_panels_average_last_shared_heading_and_missing_treatment_column(self):
        text = rendering.render_generator_retrieval_gains_table(self.scores.loc[self.scores.metric.eq("diagnosis")])
        self.assertLess(text.index("Similar-patient retrieval"), text.index("Guideline retrieval"))
        for group in text.split(r"\begin{tabular*}")[1:]:
            self.assertLess(group.index("GPT-4o mini"), group.index("Gemma 3 4B"))
            self.assertLess(group.index("Gemma 3 4B"), group.index("Average across generators"))
            self.assertIn("\\addlinespace[3pt]\nAverage across generators", group)
        self.assertEqual(text.count(r"Average across generators & $+0.06^{**}$ & -- \\"), 2)
        self.assertEqual(text.count(r"$+0.12^{*}$ & --"), 2)
        self.assertEqual(text.count(r"$-0.01$ & --"), 2)
        self.assertEqual(text.count(r"\multicolumn{2}{c@{}}{$\Delta$ vs. No retrieval}"), 2)
        self.assertEqual(text.count(r"Generator & Diagnosis & Treatment plan \\"), 2)
        self.assertEqual(text.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lll@{}}"), 2)
        self.assertEqual(text.count(r"\toprule"), 2)
        self.assertEqual(text.count(r"\bottomrule"), 2)
        self.assertEqual(text.count(r"\caption{"), 1)
        self.assertIn(r"\begin{table}[t]", text)
        self.assertIn("(a) Similar-patient retrieval", text)
        self.assertIn("(b) Guideline retrieval", text)
        self.assertNotIn(r"\mathbf", text)
        self.assertNotIn(r"\textbf{Average", text)
        self.assertIn("one-sided paired tests", text)
        for command in (r"\toprule", r"\bottomrule", r"\small", r"\captionsetup{justification=raggedright",
                        r"\label{tab:retrieval-gains}"):
            self.assertIn(command, text)

    def test_future_treatment_values_and_thresholds_come_from_statistics(self):
        self.scores.loc[self.scores.metric.eq("plan"), ["mean_difference", "p_value_holm"]] = [0.2222, 0.01]
        text = rendering.render_generator_retrieval_gains_table(self.scores)
        self.assertIn(r"$+0.12^{*}$ & $+0.22^{*}$", text)
        self.assertIn(r"Average across generators & $+0.06^{**}$ & $+0.22^{*}$", text)
        self.scores.loc[self.scores.metric.eq("plan"), "p_value_holm"] = 0.05
        self.assertIn(r"$+0.12^{*}$ & $+0.22$", rendering.render_generator_retrieval_gains_table(self.scores))
        self.scores.loc[self.scores.metric.eq("plan"), "coverage_complete"] = False
        self.assertNotIn("0.22", rendering.render_generator_retrieval_gains_table(self.scores))

    def test_display_rounding_does_not_change_significance_or_emit_negative_zero(self):
        self.scores.loc[self.scores.model.eq("gpt_4o_mini"), ["mean_difference", "p_value_holm"]] = [0.0049, 0.009]
        self.scores.loc[self.scores.model.eq("ollama_gemma3_4b"), "mean_difference"] = -0.0049
        text = rendering.render_generator_retrieval_gains_table(self.scores)
        self.assertIn(r"$+0.00^{**}$", text)
        self.assertNotIn("-0.00", text)

    def test_malformed_or_wrong_family_statistics_are_rejected(self):
        invalid = (
            self.scores.drop(columns="p_value_holm"), self.scores.iloc[1:],
            pd.concat([self.scores, self.scores.iloc[[0]]]),
            self.scores.assign(family_size=4), self.scores.assign(n_generators_averaged=1),
            self.scores.assign(baseline_variant="guidelines"),
            self.scores.assign(mean_difference=np.inf), self.scores.assign(p_value_holm=-0.1),
        )
        for scores in invalid:
            with self.subTest(columns=list(scores.columns), rows=len(scores)):
                with self.assertRaises(ValueError):
                    rendering.render_generator_retrieval_gains_table(scores)


class HistoryGainRenderingTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame([
            {"model": model, "metric": metric, "group": group, "variant": "patient_retrieval",
             "history_contrast_reference_group": "history", "history_contrast_family_size": 4,
             "history_contrast_estimate": effect if metric == "diagnosis" else np.nan,
             "history_contrast_p_value_holm": pvalue if metric == "diagnosis" else np.nan,
             "history_contrast_coverage_complete": metric == "diagnosis",
             "history_contrast_test_status": "tested" if metric == "diagnosis" else "incomplete",
             "p_value_holm": 0.000001}  # Within-group stars must not leak into direct contrasts.
            for model, group, effect, pvalue in (
                ("gpt_4o_mini", "acute", -0.002574, 1.0),
                ("gpt_4o_mini", "control", 0.104419, 0.000001),
                ("ollama_gemma3_4b", "acute", 0.056952, 0.001035),
                ("ollama_gemma3_4b", "control", -0.066041, 1.0))
            for metric in rendering.METRICS
        ])

    def test_approved_panels_three_decimals_and_direct_stars(self):
        text = rendering.render_history_gains_over_others_table(self.scores)
        self.assertIn(r"\begin{table}[!hbp]", text)
        self.assertEqual(text.count(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lll@{}}"), 2)
        self.assertIn("(a) History $-$ Acute", text)
        self.assertIn("(b) History $-$ Control", text)
        self.assertIn(r"GPT-4o mini & $-0.003$ & --", text)
        self.assertIn(r"GPT-4o mini & $+0.104^{**}$ & --", text)
        self.assertIn(r"Gemma 3 4B & $+0.057^{**}$ & --", text)
        self.assertIn(r"Gemma 3 4B & $-0.066$ & --", text)
        self.assertEqual(text.count(r"Generator & Diagnosis & Treatment plan"), 2)
        self.assertEqual(text.count(r"\caption{"), 1)
        self.assertIn(r"\label{tab:history-gains-over-others}", text)
        self.assertIn("Holm-adjusted across both comparisons and all generators", text)
        self.assertNotIn("Average", text)
        self.assertNotIn(r"\mathbf", text)
        without_plan = rendering.render_history_gains_over_others_table(self.scores.query("metric == 'diagnosis'"))
        self.assertEqual(text, without_plan)

    def test_completed_treatment_uses_unrounded_effect_and_adjusted_thresholds(self):
        plans = self.scores.metric.eq("plan")
        self.scores.loc[plans, "history_contrast_coverage_complete"] = True
        self.scores.loc[plans, "history_contrast_test_status"] = "tested"
        self.scores.loc[plans, "history_contrast_estimate"] = 0.123456
        self.scores.loc[plans, "history_contrast_p_value_holm"] = 0.01
        text = rendering.render_history_gains_over_others_table(self.scores)
        self.assertEqual(text.count(r"& $+0.123^{*}$"), 4)
        self.scores.loc[plans, "history_contrast_p_value_holm"] = 0.05
        self.assertEqual(rendering.render_history_gains_over_others_table(self.scores).count(r"& $+0.123$"), 4)
        self.scores.loc[plans, "history_contrast_estimate"] = -0.0004
        self.scores.loc[plans, "history_contrast_p_value_holm"] = 0.001
        text = rendering.render_history_gains_over_others_table(self.scores)
        self.assertNotIn("-0.000", text)
        self.assertEqual(text.count(r"& $+0.000$"), 4)
        self.scores.loc[plans, "history_contrast_estimate"] = 0.0004
        self.assertEqual(rendering.render_history_gains_over_others_table(self.scores).count(r"& $+0.000^{**}$"), 4)

    def test_incomplete_stale_effects_are_masked_and_unavailable_tests_have_no_stars(self):
        plans = self.scores.metric.eq("plan")
        self.scores.loc[plans, "history_contrast_estimate"] = 0.999
        self.assertNotIn("0.999", rendering.render_history_gains_over_others_table(self.scores))
        self.scores.loc[plans, "history_contrast_coverage_complete"] = True
        self.scores.loc[plans, "history_contrast_test_status"] = "zero_variance"
        self.assertEqual(rendering.render_history_gains_over_others_table(self.scores).count(r"& $+0.999$"), 4)

    def test_bad_schema_identity_coverage_pvalues_or_family_is_rejected(self):
        for data in (
            self.scores.drop(columns="history_contrast_estimate"), self.scores.iloc[1:],
            pd.concat([self.scores, self.scores.iloc[[0]]]),
            self.scores.assign(history_contrast_family_size=2),
            self.scores.assign(history_contrast_reference_group="acute"),
            self.scores.assign(variant="guidelines"),
            self.scores.assign(history_contrast_estimate=np.inf),
            self.scores.assign(history_contrast_p_value_holm=-0.1),
            self.scores.assign(history_contrast_test_status="tested"),
            self.scores.assign(history_contrast_test_status="unknown"),
        ):
            with self.subTest(columns=list(data.columns), rows=len(data)), self.assertRaises(ValueError):
                rendering.render_history_gains_over_others_table(data)


class TableRenderingTests(unittest.TestCase):
    def test_group_table_has_raw_baseline_and_three_starred_two_decimal_deltas(self):
        scores = pd.DataFrame([
            {"model": "gpt_4o_mini", "group": group, "metric": metric, "variant": variant,
             "mean_baseline": 0.4, "mean_retrieval": mean, "mean_difference": mean - 0.4,
             "p_value_holm": pvalue, "coverage_complete": True}
            for group, value, patient_p in (("history", 0.5, 0.001), ("acute", 0.6, 0.03), ("control", 0.35, 0.001))
            for metric in ("diagnosis", "plan")
            for variant, mean, pvalue in (
                ("patient_retrieval", value, patient_p),
                ("guidelines", 0.3 if group == "control" else 0.45, 0.04),
                ("patient_retrieval_guidelines", 0.36 if group == "control" else 0.55, 0.0005),
            )
        ])
        text = rendering.render_group_retrieval_gains_table(scores)
        self.assertIn("{Diagnoses}", text)
        self.assertIn("{Treatment Plans}", text)
        self.assertEqual(text.count("+0.10^{**}"), 2)
        self.assertEqual(text.count(r"\mathbf{+0.20}^{*}"), 2)
        self.assertEqual(text.count("-0.05$"), 2)
        self.assertEqual(text.count(r"\mathbf{0.40}"), 2)
        self.assertIn("{Guideline", text)
        self.assertIn(r"\footnotesize Both", text)
        self.assertEqual(text.count(r"$\Delta$ vs. No retrieval"), 2)
        self.assertIn("(score)", text)
        for forbidden in ("$n=", "95", "CI", "History minus", "0.5000"):
            self.assertNotIn(forbidden, text)
        scores.loc[scores.metric.eq("plan") & scores.group.eq("acute") & scores.variant.eq("patient_retrieval"),
                   ["coverage_complete", "mean_retrieval", "mean_difference", "p_value_holm"]] = [False, np.nan, np.nan, np.nan]
        masked = rendering.render_group_retrieval_gains_table(scores)
        self.assertIn(r" & {\footnotesize --} & {\footnotesize $+0.05^{*}$} & {\footnotesize $\mathbf{+0.15}^{**}$}", masked)
        for invalid in (scores.iloc[1:], pd.concat([scores, scores.iloc[[0]]]),
                        scores.assign(mean_difference=0.999)):
            with self.assertRaises(ValueError):
                rendering.render_group_retrieval_gains_table(invalid)

    def test_absolute_score_table_uses_four_arms_and_masks_partial_means(self):
        coverage = pd.DataFrame([
            {"model": "gpt_4o_mini", "variant": variant, "metric": metric,
             "level": "overall", "group": "all", "mean_score": value,
             "coverage_complete": metric == "diagnosis",
             "mean_ci_low": value - 0.01, "mean_ci_high": value + 0.01,
             "baseline_delta": value - 0.72341, "p_value_holm": 0.005}
            for variant, value in zip(
                ("llm_only", "patient_retrieval", "guidelines", "patient_retrieval_guidelines"),
                (0.72341, 0.82341, 0.62341, 0.92341))
            for metric in rendering.METRICS
        ])
        text = rendering.render_main_results_table(coverage)
        self.assertIn(r"No retrieval & $0.7234$ & {\scriptsize $[0.71, 0.73]$} & -- & -- \\[2pt]", text)
        self.assertIn(r"Sim. patient retrieval & $0.8234^{**}$ & {\scriptsize $[0.81, 0.83]$} & -- & --", text)
        self.assertIn(r"Guideline retrieval & $0.6234$ & {\scriptsize $[0.61, 0.63]$} & -- & --", text)
        self.assertIn(r"Both & $\mathbf{0.9234}^{**}$ & {\scriptsize $[0.91, 0.93]$} & -- & --", text)
        self.assertIn("{Diagnoses}", text)
        self.assertIn("{Treatment Plans}", text)
        self.assertIn("95\\% CI", text)
        self.assertIn('over "No retrieval" configuration', text)
        self.assertIn(r"lllclc@{}", text)
        self.assertIn("{Generator Model}", text)
        self.assertIn("justification=raggedright", text)
        caption = text.split(r"\caption{", 1)[1].split(r"\label", 1)[0]
        self.assertIn("Both scores range from 0 to 1; higher scores are better.", caption)
        self.assertIn(r"$^{*}p<0.05$", caption)
        self.assertNotIn("bootstrap resampling", caption)
        self.assertNotIn("means weight condition", text)
        self.assertEqual(text.count(r"\mathbf{"), 1)
        for forbidden in ("$n=", "Delta", "+0.", "Comparison"):
            self.assertNotIn(forbidden, text)
        coverage["coverage_complete"] = True
        coverage.loc[coverage.variant.eq("patient_retrieval"), "p_value_holm"] = 0.01
        coverage.loc[coverage.variant.eq("patient_retrieval_guidelines"), "p_value_holm"] = 0.05
        boundary = rendering.render_main_results_table(coverage)
        self.assertEqual(boundary.count("0.8234^{*}"), 2)
        self.assertNotIn("0.9234^{", boundary)
        self.assertNotIn("0.6234^{", boundary)
        self.assertNotIn("0.7234^{", boundary)
        self.assertEqual(boundary.count(r"\mathbf{0.9234}"), 2)
        # Plan ties are highlighted independently of the diagnosis maximum.
        tied = coverage.copy()
        tied.loc[tied.metric.eq("plan") & tied.variant.eq("patient_retrieval"), "mean_score"] = 0.92341
        tie_text = rendering.render_main_results_table(tied)
        self.assertEqual(tie_text.count(r"\mathbf{0.9234}"), 3)
        for invalid in (
            coverage.iloc[1:],
            pd.concat([coverage, coverage.iloc[[0]]]),
            coverage.assign(mean_score=np.inf),
        ):
            with self.assertRaises(ValueError):
                rendering.render_main_results_table(invalid)

class AvailableScoreRenderingTests(unittest.TestCase):
    def setUp(self):
        self.averages = pd.DataFrame([
            {'model': 'model_average', 'variant': variant, 'metric': metric,
             'baseline_variant': 'llm_only', 'mean_difference': .08, 'p_value_holm': .02,
             'family_size': 6, 'n_generators_averaged': 4, 'coverage_complete': False}
            for variant in VARIANT_LABELS if variant != 'llm_only' for metric in rendering.METRICS])
        self.main = pd.DataFrame([
            {'model': model, 'variant': variant, 'metric': metric, 'mean_score': .5,
             'mean_ci_low': .4, 'mean_ci_high': .6, 'baseline_delta': .1,
             'p_value_holm': .02, 'coverage_complete': False}
            for model in rendering.MODEL_LABELS for variant in VARIANT_LABELS for metric in rendering.METRICS])
        self.groups = pd.DataFrame([
            {'model': model, 'variant': variant, 'metric': metric, 'group': group,
             'mean_baseline': .4, 'mean_retrieval': .5, 'paired_mean_baseline': .41,
             'paired_mean_retrieval': .49, 'mean_difference': .08, 'p_value_holm': .02,
             'coverage_complete': False}
            for model in rendering.MODEL_LABELS for variant in VARIANT_LABELS if variant != 'llm_only'
            for metric in rendering.METRICS for group in rendering.GROUPS])

    def test_all_seven_variants_and_partial_means_are_displayed(self):
        main = rendering.render_main_results_table(self.main, available_scores=True)
        gains = rendering.render_generator_retrieval_gains_table(self.averages, available_scores=True)
        groups = rendering.render_group_retrieval_gains_table(self.groups, available_scores=True)
        for variant in VARIANT_LABELS:
            label = VARIANT_LABELS[variant]
            self.assertIn(label, main)
            self.assertIn(label, groups)
            if variant != 'llm_only':
                self.assertIn(label, gains)
        self.assertIn(r'0.5000', main)
        self.assertEqual(gains.count(r'$+0.0800^{*}$'), 12)
        self.assertIn(r'0.5000 & $+0.0800^{*}$', groups)
        self.assertEqual(groups.count(r'\begin{table*}'), 4)
        # Raw arm means differ by .1, but matched-pair gains are .08.
        self.assertNotIn('+0.1000', groups)

    def test_empty_outcomes_render_dashes_without_fabricated_zeros_or_stars(self):
        main = self.main.assign(mean_score=np.nan, mean_ci_low=np.nan, mean_ci_high=np.nan,
                                baseline_delta=np.nan, p_value_holm=np.nan)
        averages = self.averages.assign(mean_difference=np.nan, p_value_holm=np.nan)
        groups = self.groups.assign(mean_baseline=np.nan, mean_retrieval=np.nan,
                                    paired_mean_baseline=np.nan, paired_mean_retrieval=np.nan,
                                    mean_difference=np.nan, p_value_holm=np.nan)
        for renderer, data in ((rendering.render_main_results_table, main),
                               (rendering.render_generator_retrieval_gains_table, averages),
                               (rendering.render_group_retrieval_gains_table, groups)):
            with self.subTest(renderer=renderer.__name__):
                text = renderer(data, available_scores=True)
                self.assertIn('--', text)
                self.assertNotIn('nan', text)
                self.assertNotIn('0.0000', text)
                self.assertNotIn('^{*}', text.split(r'\caption{', 1)[0])

    def test_average_renderer_rejects_wrong_weighting_family_and_incomplete_rows(self):
        for data in (self.averages.assign(n_generators_averaged=3),
                     self.averages.assign(family_size=10), self.averages.assign(model='gpt_4o_mini'),
                     self.averages.iloc[1:], pd.concat([self.averages, self.averages.iloc[:1]]),
                     self.averages.loc[self.averages.variant.ne('agentic_both')].assign(family_size=5),
                     self.averages.assign(mean_difference=np.inf), self.averages.assign(p_value_holm=-.1)):
            with self.subTest(rows=len(data)), self.assertRaises(ValueError):
                rendering.render_generator_retrieval_gains_table(data, available_scores=True)

    def test_group_renderer_rejects_inconsistent_paired_or_baseline_means(self):
        bad_baseline = self.groups.copy()
        bad_baseline.loc[0, 'mean_baseline'] = .3
        for data in (bad_baseline, self.groups.assign(paired_mean_retrieval=.5),
                     self.groups.assign(paired_mean_retrieval=1.1), self.groups.iloc[1:],
                     pd.concat([self.groups, self.groups.iloc[:1]])):
            with self.subTest(rows=len(data)), self.assertRaises(ValueError):
                rendering.render_group_retrieval_gains_table(data, available_scores=True)

    def test_rounding_keeps_unrounded_significance_and_removes_negative_zero(self):
        self.averages['p_value_holm'] = .001
        self.averages.loc[self.averages.metric.eq('diagnosis'), 'mean_difference'] = .000049
        self.averages.loc[self.averages.metric.eq('plan'), 'mean_difference'] = -.000049
        text = rendering.render_generator_retrieval_gains_table(self.averages, available_scores=True)
        self.assertEqual(text.count(r'$+0.0000^{**}$ & $+0.0000$'), 6)
        self.assertNotIn('-0.0000', text)

    def test_history_contrasts_with_partial_coverage_remain_available(self):
        data = self.groups.loc[self.groups.variant.eq('patient_retrieval')
                               & self.groups.group.isin(['acute', 'control'])].copy()
        for field, value in {'reference_group': 'history', 'family_size': 8, 'estimate': .03,
                             'p_value_holm': .02, 'test_status': 'tested', 'coverage_complete': False}.items():
            data['history_contrast_' + field] = value
        text = rendering.render_history_gains_over_others_table(data, available_scores=True)
        self.assertEqual(text.count(r'$+0.030^{*}$'), 16)
        with self.assertRaises(ValueError):
            rendering.render_history_gains_over_others_table(data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
