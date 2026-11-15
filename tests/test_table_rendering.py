"""Validate the three retained report-table layouts and score availability."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

# Direct test execution starts in tests/; resolve project modules first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analysis import table_rendering as rendering


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

if __name__ == '__main__':
    unittest.main(verbosity=2)
