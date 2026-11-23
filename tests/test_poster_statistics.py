"""Offline reference and filesystem checks for the focused poster contrasts."""

import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Direct test execution must resolve project modules before importing them.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import poster_statistics as poster


def fixture(count=24):
    rows = []
    for index in range(count):
        patient = f'00000000-0000-0000-0000-{index // 2:012d}'
        case_id = f'case{index}_{patient}'
        for model_index, model in enumerate(poster.MODEL_LABELS):
            diagnosis_baseline = 0.38 + 0.005 * (index % 7) + 0.01 * model_index
            plan_baseline = 0.2 + 0.01 * (index % 6) + 0.012 * model_index
            changes = {
                'llm_only': (0, 0),
                'patient_retrieval': (-0.015 + 0.003 * (index % 5), 0.025 + 0.003 * (index % 5)),
                'guidelines': (0.07 + 0.002 * (index % 4), 0.11 + 0.004 * (index % 7)),
                'patient_retrieval_guidelines': (0.05 + 0.003 * (index % 5), 0.09 + 0.005 * (index % 4)),
                'agentic_patient_retrieval': (-0.005 + 0.002 * (index % 6), 0.018 + 0.005 * (index % 4)),
                'agentic_guidelines': (0.048 + 0.003 * (index % 5), 0.048 + 0.003 * (index % 5)),
                'agentic_both': (0.057 + 0.001 * (index % 7), 0.045 + 0.004 * (index % 6)),
            }
            for variant in poster.VARIANTS:
                diagnosis_change, plan_change = changes[variant]
                rows.append(dict(id=case_id, patient_uuid=patient, model=model, variant=variant,
                                 diagnosis=diagnosis_baseline + diagnosis_change,
                                 plan=plan_baseline + plan_change, plan_eligible=True))
    return pd.DataFrame(rows)


def raw_export(frame):
    raw = frame.drop(columns=['patient_uuid', 'plan_eligible']).rename(
        columns={'diagnosis': 'biolord_score_diagnosis1', 'plan': 'plan1_score'})
    raw['bin'] = 'history'
    raw['condition_label'] = 'essential hypertension'
    raw['plan_label'] = 'Lifestyle counseling'
    raw['plan1_status'] = np.where(raw.plan1_score.notna(), 'scored', 'invalid_response')
    raw['plan1_reason'] = np.where(raw.plan1_score.notna(), '', 'invalid_response')
    raw['diagnosis1_status'] = np.where(raw.biolord_score_diagnosis1.notna(), 'scored', 'unavailable')
    raw['plan1_protocol_version'] = 2
    raw['plan1_rank'] = 1
    return raw


def model_fixture(count=24):
    frame = fixture(count)
    # Different responses across models expose accidental model pooling.
    frame.loc[frame.model.eq('gpt_4o_mini') & frame.variant.eq('patient_retrieval'), 'diagnosis'] += .06
    frame.loc[frame.model.eq('ollama_gemma3_4b') & frame.variant.eq('patient_retrieval'), 'diagnosis'] -= .04
    frame.loc[frame.model.eq('ollama_llama3_2_3b') & frame.variant.eq('agentic_guidelines'), 'plan'] -= .03
    return frame


def select_model(output, model='gpt_4o_mini', metric='diagnosis',
                 question='information', source='patient_retrieval'):
    return output.loc[output.model.eq(model) & output.metric.eq(metric)
                      & output.question.eq(question) & output.source.eq(source)].iloc[0]


class ModelPosterStatisticsTests(unittest.TestCase):
    def test_all_48_tests_are_within_model_and_match_independent_clustered_ols(self):
        frame = model_fixture()
        output = poster.summarize_model_poster_comparisons(frame)
        self.assertEqual(len(output), 48)
        self.assertEqual(output.groupby('model').size().to_dict(), dict.fromkeys(poster.MODEL_LABELS, 12))
        self.assertTrue(output.n_generators_averaged.eq(1).all())
        self.assertTrue(output.family_size.eq(48).all())
        self.assertTrue(output.n_pairs.eq(24).all())
        self.assertTrue(output.n_patients.eq(12).all())
        probabilities = []
        for row in output.itertuples():
            data = frame.loc[frame.model.eq(row.model)]
            pivot = data.pivot(index='id', columns='variant', values=row.metric)
            pair = pivot[[row.baseline_variant, row.comparison_variant]].dropna()
            delta = pair[row.comparison_variant] - pair[row.baseline_variant]
            patients = data.drop_duplicates('id').set_index('id').loc[pair.index, 'patient_uuid']
            reference = sm.OLS(delta.to_numpy(), np.ones((len(delta), 1))).fit(
                cov_type='cluster', cov_kwds={'groups': patients, 'use_correction': True}, use_t=True)
            np.testing.assert_allclose(row.mean_baseline, pair[row.baseline_variant].mean(), rtol=1e-12)
            np.testing.assert_allclose(row.mean_comparison, pair[row.comparison_variant].mean(), rtol=1e-12)
            np.testing.assert_allclose(row.mean_difference, reference.params[0], rtol=1e-12)
            np.testing.assert_allclose(row.standard_error, reference.bse[0], rtol=1e-10)
            np.testing.assert_allclose([row.ci_low, row.ci_high], reference.conf_int()[0], rtol=1e-10)
            np.testing.assert_allclose(row.p_value_two_sided, reference.pvalues[0], rtol=1e-9)
            probabilities.append(reference.pvalues[0])
        adjusted = multipletests(probabilities, method='holm')[1]
        np.testing.assert_allclose(output.p_value_two_sided_holm, adjusted, rtol=1e-9)
        self.assertEqual(output.significant_two_sided_holm_0_01.tolist(), (adjusted < .01).tolist())
        self.assertEqual(output.significant_two_sided_holm_0_05.tolist(), (adjusted < .05).tolist())
        self.assertGreater(select_model(output).mean_difference, 0)
        self.assertLess(select_model(output, model='ollama_gemma3_4b').mean_difference, 0)
        self.assertEqual([c for c in output if 'p_value' in c],
                         ['p_value_two_sided', 'p_value_two_sided_holm'])

    def test_missing_score_excludes_only_its_model_case_comparison_and_metric(self):
        frame = model_fixture()
        case = frame.id.iloc[0]
        slot = frame.id.eq(case) & frame.model.eq('gpt_4o_mini') & frame.variant.eq('patient_retrieval')
        frame.loc[slot, 'diagnosis'] = np.nan
        output = poster.summarize_model_poster_comparisons(frame)
        row = select_model(output)
        self.assertEqual((row.n_eligible, row.n_baseline, row.n_comparison, row.n_pairs), (24, 24, 23, 23))
        self.assertFalse(row.coverage_complete)
        self.assertEqual(select_model(output, question='adaptive').n_pairs, 23)
        self.assertEqual(select_model(output, source='guidelines').n_pairs, 24)
        self.assertEqual(select_model(output, metric='plan').n_pairs, 24)
        self.assertEqual(select_model(output, model='ollama_gemma3_4b').n_pairs, 24)
        expected = frame.loc[~frame.id.eq(case) & frame.model.eq('gpt_4o_mini')].groupby('variant').diagnosis.mean()
        self.assertAlmostEqual(row.mean_baseline, expected.llm_only)
        self.assertAlmostEqual(row.mean_comparison, expected.patient_retrieval)

    def test_zero_and_unavailable_reference_keep_separate_treatment_denominators(self):
        frame = model_fixture()
        cases = frame.id.unique()
        frame.loc[frame.id.eq(cases[0]), 'plan_eligible'] = False
        frame.loc[frame.id.eq(cases[0]), 'plan'] = 1
        frame.loc[frame.id.eq(cases[1]), 'plan'] = 0
        output = poster.summarize_model_poster_comparisons(frame)
        self.assertTrue(output.loc[output.metric.eq('plan'), 'n_pairs'].eq(23).all())
        self.assertTrue(output.loc[output.metric.eq('diagnosis'), 'n_pairs'].eq(24).all())
        for model in poster.MODEL_LABELS:
            expected = frame.loc[frame.model.eq(model) & ~frame.id.eq(cases[0])].groupby('variant').plan.mean()
            row = select_model(output, model=model, metric='plan')
            self.assertAlmostEqual(row.mean_baseline, expected.llm_only)
            self.assertAlmostEqual(row.mean_comparison, expected.patient_retrieval)

    def test_missing_and_degenerate_tests_reserve_48_family_places_without_significance(self):
        frame = model_fixture()
        frame.loc[frame.model.eq('gpt_4o_mini') & frame.variant.eq('patient_retrieval'), 'diagnosis'] = np.nan
        output = poster.summarize_model_poster_comparisons(frame)
        missing = select_model(output)
        self.assertEqual(missing.test_status, 'no_pairs')
        self.assertTrue(pd.isna(missing.p_value_two_sided_holm))
        self.assertFalse(missing.significant_two_sided_holm_0_01)
        self.assertFalse(missing.significant_two_sided_holm_0_05)
        self.assertEqual(missing.stars, '')
        probabilities = output.p_value_two_sided.fillna(1).to_numpy()
        reference = multipletests(probabilities, method='holm')[1]
        observed = output.p_value_two_sided.notna()
        np.testing.assert_allclose(output.loc[observed, 'p_value_two_sided_holm'], reference[observed])
        no_reference = poster.summarize_model_poster_comparisons(model_fixture().assign(plan_eligible=False))
        self.assertTrue(no_reference.loc[no_reference.metric.eq('plan'), 'test_status'].eq('no_eligible').all())
        one_patient = poster.summarize_model_poster_comparisons(model_fixture().assign(patient_uuid='one'))
        self.assertTrue(one_patient.test_status.eq('insufficient_patients').all())
        zero_variance = poster.summarize_model_poster_comparisons(model_fixture().assign(diagnosis=.5, plan=.5))
        self.assertTrue(zero_variance.test_status.eq('zero_variance').all())
        self.assertFalse(zero_variance.significant_two_sided_holm_0_01.any())

    def test_bad_structure_is_rejected_and_input_order_does_not_change_results(self):
        frame = model_fixture()
        for invalid in (frame.iloc[1:], frame.assign(model='wrong'),
                        frame.assign(plan=np.inf), frame.drop(columns='patient_uuid')):
            with self.assertRaises(ValueError):
                poster.summarize_model_poster_comparisons(invalid)
        first = poster.summarize_model_poster_comparisons(frame)
        shuffled = poster.summarize_model_poster_comparisons(frame.sample(frac=1, random_state=14))
        pd.testing.assert_frame_equal(first, shuffled)

    def test_all_invalid_structure_metadata_and_score_values_are_rejected(self):
        frame = model_fixture()
        cases = [
            frame.iloc[:0], frame.drop(columns='patient_uuid'), pd.concat([frame, frame.iloc[:1]]),
            frame.iloc[1:], frame.loc[frame.model.ne('gpt_4o_mini')],
            frame.loc[frame.variant.ne('agentic_both')], frame.assign(model='unknown'),
            frame.assign(variant='unknown'), frame.assign(id=''), frame.assign(patient_uuid=' '),
            frame.assign(plan_eligible='True'), frame.assign(diagnosis=np.inf),
            frame.assign(plan=-.1), frame.assign(diagnosis=1.1), frame.assign(diagnosis='bad'),
        ]
        for column, value in (('patient_uuid', 'different'), ('plan_eligible', False)):
            inconsistent = frame.copy()
            inconsistent.loc[0, column] = value
            cases.append(inconsistent)
        for invalid in cases:
            with self.subTest(rows=len(invalid), columns=invalid.columns.tolist()):
                with self.assertRaises(ValueError):
                    poster.summarize_model_poster_comparisons(invalid)

    def test_missing_both_arms_never_creates_a_zero_or_disrupts_other_models(self):
        frame = model_fixture()
        model_mask = frame.model.eq('ollama_gemma3_4b')
        frame.loc[model_mask & frame.variant.isin(['llm_only', 'guidelines']), 'plan'] = np.nan
        output = poster.summarize_model_poster_comparisons(frame)
        row = select_model(output, model='ollama_gemma3_4b', metric='plan', source='guidelines')
        self.assertEqual((row.n_eligible, row.n_baseline, row.n_comparison, row.n_pairs, row.n_patients),
                         (24, 0, 0, 0, 0))
        self.assertEqual(row.test_status, 'no_pairs')
        self.assertTrue(pd.isna(row.mean_baseline))
        self.assertTrue(pd.isna(row.mean_comparison))
        self.assertTrue(pd.isna(row.mean_difference))
        self.assertTrue(pd.isna(row.p_value_two_sided))
        self.assertEqual(select_model(output, metric='plan', source='guidelines').n_pairs, 24)


class ModelPosterExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'experiment_results.csv'
        raw_export(model_fixture()).to_csv(self.source, index=False)
        self.destination = self.directory / 'analysis' / poster.MODEL_OUTPUT_NAME

    def test_real_csv_export_masks_statuses_records_provenance_and_preserves_other_files(self):
        raw = pd.read_csv(self.source, keep_default_na=False)
        case = raw.id.iloc[0]
        slot = raw.id.eq(case) & raw.model.eq('gpt_4o_mini') & raw.variant.eq('patient_retrieval')
        raw.loc[slot, ['diagnosis1_status', 'plan1_status']] = ['unavailable', 'invalid_response']
        raw.loc[slot, ['biolord_score_diagnosis1', 'plan1_score']] = .99
        raw.to_csv(self.source, index=False)
        original = self.source.read_bytes()
        self.destination.parent.mkdir()
        sentinel = self.destination.with_name('existing_summary.csv')
        sentinel.write_text('previous summary\n')
        output = poster.generate_model_poster_statistics(self.source)
        saved = pd.read_csv(self.destination, keep_default_na=False)
        self.assertEqual(len(saved), 48)
        self.assertTrue(saved.source_csv_sha256.eq(hashlib.sha256(original).hexdigest()).all())
        self.assertTrue(saved.source_csv.eq(str(self.source.resolve())).all())
        self.assertTrue(saved.multiplicity_policy.eq('Holm across all 48 two-sided poster comparisons').all())
        self.assertEqual(select_model(output).n_pairs, 23)
        self.assertEqual(select_model(output, metric='plan').n_pairs, 23)
        self.assertEqual(select_model(output, model='ollama_gemma3_4b').n_pairs, 24)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(sentinel.read_text(), 'previous summary\n')
        first = self.destination.read_bytes()
        poster.generate_model_poster_statistics(self.source)
        self.assertEqual(first, self.destination.read_bytes())
        self.assertEqual({p.name for p in self.destination.parent.iterdir()},
                         {poster.MODEL_OUTPUT_NAME, 'existing_summary.csv'})

    def test_invalid_or_changing_source_preserves_existing_csv_and_custom_output_works(self):
        self.destination.parent.mkdir()
        self.destination.write_text('previous model results\n')
        raw = pd.read_csv(self.source)
        for invalid in (raw.assign(plan1_protocol_version=1), raw.assign(plan1_rank=2),
                        raw.drop(columns='diagnosis1_status'), raw.iloc[1:]):
            invalid.to_csv(self.source, index=False)
            with self.assertRaises(ValueError):
                poster.generate_model_poster_statistics(self.source)
            self.assertEqual(self.destination.read_text(), 'previous model results\n')
        raw.to_csv(self.source, index=False)
        with patch.object(poster, '_sha256', side_effect=['before', 'after']):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                poster.generate_model_poster_statistics(self.source)
        self.assertEqual(self.destination.read_text(), 'previous model results\n')
        with self.assertRaisesRegex(ValueError, 'differ'):
            poster.generate_model_poster_statistics(self.source, self.source)
        custom = self.directory / 'custom' / 'model.csv'
        poster.generate_model_poster_statistics(self.source, custom)
        self.assertEqual(len(pd.read_csv(custom)), 48)

    def test_cli_generates_only_the_model_specific_48_comparison_csv(self):
        custom = self.directory / 'custom' / 'matrix.csv'
        output = poster.main(['--results-csv', str(self.source), '--output-csv', str(custom)])
        self.assertEqual(len(output), 48)
        self.assertTrue(custom.exists())
        self.assertFalse(self.destination.exists())
        self.assertEqual([c for c in output if 'p_value' in c],
                         ['p_value_two_sided', 'p_value_two_sided_holm'])
        self.assertTrue(output.confidence_level.eq(.95).all())
        self.assertTrue(output.interval_policy.eq('Pointwise Student-t CR1 patient-clustered').all())

    def test_source_changed_during_calculation_does_not_create_an_output_directory(self):
        with patch.object(poster, '_sha256', side_effect=['before', 'after']):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                poster.generate_model_poster_statistics(self.source)
        self.assertFalse(self.destination.parent.exists())

    def test_output_cannot_overwrite_raw_data_and_failed_publication_cleans_temporary_csv(self):
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'differ'):
            poster.generate_model_poster_statistics(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)
        self.destination.parent.mkdir()
        self.destination.write_text('old valid output\n')
        with patch.object(poster.os, 'replace', side_effect=OSError('simulated publication failure')):
            with self.assertRaisesRegex(OSError, 'publication failure'):
                poster.generate_model_poster_statistics(self.source)
        self.assertEqual(self.destination.read_text(), 'old valid output\n')
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])
        self.assertEqual(self.source.read_bytes(), original)

    def test_raw_artifact_reference_and_real_zero_have_distinct_eligibility(self):
        raw = pd.read_csv(self.source, keep_default_na=False)
        cases = raw.id.unique()
        raw.loc[raw.id.eq(cases[0]), 'plan_label'] = 'COPD plan (record artifact)'
        raw.loc[raw.id.eq(cases[1]), 'plan1_score'] = 0
        raw.to_csv(self.source, index=False)
        output = poster.generate_model_poster_statistics(self.source)
        plans = output.loc[output.metric.eq('plan')]
        self.assertTrue(plans.n_eligible.eq(23).all())
        self.assertTrue(plans.n_pairs.eq(23).all())
        self.assertTrue(output.loc[output.metric.eq('diagnosis'), 'n_pairs'].eq(24).all())
        for model in poster.MODEL_LABELS:
            expected = raw.loc[raw.model.eq(model) & ~raw.id.eq(cases[0]) & raw.variant.eq('llm_only'),
                               'plan1_score'].mean()
            self.assertAlmostEqual(select_model(output, model=model, metric='plan').mean_baseline, expected)


if __name__ == '__main__':
    unittest.main()
