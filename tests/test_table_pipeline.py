"""Integration tests: experiment export -> statistics CSVs -> report tables."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# Direct test execution starts in tests/; resolve project modules first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analysis import summarize_experiment_results as analysis
from scripts.analysis import render_tables as exporter
from scripts.analysis import compare_agentic_variants as counterparts


MODEL='gpt_4o_mini'
VARIANTS=('llm_only','patient_retrieval','guidelines','patient_retrieval_guidelines')


def case_rows(stem, patient, diagnosis, *, group="history", condition="essential hypertension",
              plans=None, plan_label="Recorded care", statuses=None, reasons=None):
    """Construct actual flat-CSV records without any model or API dependencies."""
    plans = [0.5] * 4 if plans is None else plans
    statuses = ["scored"] * 4 if statuses is None else statuses
    reasons = [""] * 4 if reasons is None else reasons
    return [{
        "id": f"{stem}_{patient}", "model": MODEL, "variant": variant,
        "bin": group, "condition_label": condition, "plan_label": plan_label,
        "biolord_score_diagnosis1": diagnosis[index], "bertf1_diagnosis1": 0.8,
        "plan1_score": plans[index], "plan1_status": statuses[index],
        "plan1_reason": reasons[index],
        # Deliberately different lower-ranked outputs detect accidental pooling.
        "biolord_score_diagnosis2": 0.99, "biolord_score_diagnosis3": 0.01,
        "plan2_score": 0.99, "plan3_score": 0.01,
    } for index, variant in enumerate(VARIANTS)]


class TablePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv = self.root / 'experiment_results.csv'
        self.analysis = self.root / 'analysis'
        self.report = self.root / 'report/figures'
        self.report.parent.mkdir()
        (self.report.parent / 'main.tex').write_text('existing report source')
        (self.root / 'results.jsonl').write_text('protected saved experiment results\n')
        (self.root / 'run.json').write_text('{}\n')

    def rows(self, plan_ready=True):
        rows=[]
        for group,condition in [('history','essential hypertension'),
                                ('acute','acute viral pharyngitis'),
                                ('control','chronic congestive heart failure')]:
            for index in range(8):
                case=case_rows(group,f'00000000-0000-4000-8000-{index:012d}',
                               [0.4+index*0.002,0.5+index*0.004,0.45+index*0.003,0.6+index*0.005],
                               group=group,condition=condition,
                               plans=[0.3+index*0.001,0.5+index*0.002,0.35+index*0.003,0.6+index*0.004],
                               statuses=['scored' if plan_ready else 'pending']*4)
                rows+=case
                rows += [dict(row,model='ollama_gemma3_4b') for row in case]
        return rows

    def summarize(self,plan_ready=True):
        pd.DataFrame(self.rows(plan_ready)).to_csv(self.csv,index=False)
        return analysis.generate_table_statistics(self.csv,bootstrap_samples=64,seed=17)

    def test_three_csvs_provenance_and_protected_experiment_files(self):
        protected={name:(self.root/name).read_bytes() for name in ('run.json','results.jsonl')}
        manifest=self.summarize()
        self.assertEqual({Path(item['path']).name for item in manifest['outputs']},
                         {f'{name}.csv' for name in exporter.TABLE_INPUTS.values()})
        self.assertEqual(manifest['source_csv_sha256'],hashlib.sha256(self.csv.read_bytes()).hexdigest())
        for item in manifest['outputs']:
            self.assertEqual(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest(),item['sha256'])
        self.assertFalse(list(self.analysis.rglob('*.tex')))
        self.assertFalse(list(self.analysis.rglob('*.png')))
        self.assertEqual(protected,{name:(self.root/name).read_bytes() for name in protected})
        generator=pd.read_csv(self.analysis/'generator_retrieval_gains.csv')
        self.assertEqual(set(generator.model),{MODEL,'ollama_gemma3_4b','model_average'})
        self.assertTrue(generator.family_size.eq(6).all())
        group=pd.read_csv(self.analysis/'group_retrieval_gains.csv')
        self.assertTrue(group.family_size.eq(9).all())

    def test_saved_csvs_alone_generate_all_report_copies(self):
        self.summarize()
        self.csv.unlink()
        outputs=exporter.generate_tables(self.analysis,report_dir=self.report)
        self.assertEqual(set(outputs),set(exporter.TABLES))
        self.assertEqual({p.name for p in self.report.iterdir()},{f'{name}.tex' for name in exporter.TABLES})
        for name,paths in outputs.items():
            self.assertEqual(paths['table'].read_bytes(),paths['report'].read_bytes())
        self.assertIn(r'\label{tab:main-results}',outputs['main_results']['table'].read_text())
        self.assertIn(r'\label{tab:group-retrieval-gains}',outputs['group_retrieval_gains']['table'].read_text())
        self.assertIn(r'\label{tab:history-gains-over-others}',outputs['history_gains_over_others']['table'].read_text())
        self.assertFalse((self.analysis/'history_gains_over_others.csv').exists())
        self.assertIn('Average across generators',outputs['generator_retrieval_gains']['table'].read_text())
        self.assertEqual((self.report.parent/'main.tex').read_text(),'existing report source')
        manifest=json.loads((self.analysis/'table_exports_manifest.json').read_text())
        for name,record in manifest['tables'].items():
            self.assertEqual(record['table_sha256'],record['report_sha256'])
            self.assertEqual(record['input_csv_sha256'],hashlib.sha256((self.analysis/f'{exporter.TABLE_INPUTS[name]}.csv').read_bytes()).hexdigest())

    def test_exploratory_checks_are_separate_reproducible_and_ready_for_treatment(self):
        manifest = self.summarize(plan_ready=False)
        policy = manifest['exploratory_two_sided']
        self.assertEqual(policy['alternative'], 'two-sided')
        self.assertEqual(policy['outcomes'], ['diagnosis', 'plan'])
        self.assertIn('Post-hoc', policy['status'])
        self.assertEqual(manifest['test_policy']['alternative'], 'greater')
        for name in policy['tables']:
            saved = pd.read_csv(self.analysis / f'{name}.csv')
            self.assertTrue(saved.loc[saved.metric.eq('plan'), 'p_value_two_sided'].isna().all())
        self.summarize(plan_ready=True)
        for name in policy['tables']:
            saved = pd.read_csv(self.analysis / f'{name}.csv')
            comparisons = saved.loc[saved.variant.ne('llm_only')]
            self.assertTrue(comparisons.p_value_two_sided.notna().all())
            self.assertTrue(comparisons.p_value_two_sided_holm.notna().all())
            extra = [column for column in saved if 'two_sided' in column]
            self.assertEqual(exporter.TABLES[name](saved), exporter.TABLES[name](saved.drop(columns=extra)))
        self.assertNotIn('p_value_two_sided', pd.read_csv(self.analysis / 'group_retrieval_gains.csv'))

    def test_completed_plan_export_populates_all_three_tables(self):
        self.summarize(plan_ready=False)
        exporter.generate_tables(self.analysis,report_dir=self.report)
        before={name:(self.report/f'{name}.tex').read_text() for name in exporter.TABLES}
        diagnosis_before=pd.read_csv(self.analysis/'main_results.csv').query("metric == 'diagnosis'").reset_index(drop=True)
        self.summarize(plan_ready=True)
        exporter.generate_tables(self.analysis,report_dir=self.report)
        diagnosis_after=pd.read_csv(self.analysis/'main_results.csv').query("metric == 'diagnosis'").reset_index(drop=True)
        pd.testing.assert_frame_equal(diagnosis_before,diagnosis_after)
        for name in exporter.TABLES:
            after=(self.report/f'{name}.tex').read_text()
            self.assertNotEqual(before[name],after)
            self.assertIn('--',before[name])
            self.assertNotIn('--',after)
        self.assertIn('**',(self.report/'generator_retrieval_gains.tex').read_text())
        self.assertIn(r'\scriptsize',(self.report/'main_results.tex').read_text())

    def test_history_gain_tests_refresh_from_pending_to_complete_plan_csv(self):
        rows = self.rows(plan_ready=False)
        for row in rows:
            if row['variant'] == 'patient_retrieval':
                patient_index = int(row['id'][-12:])
                change = {'history': 0.004, 'acute': -0.003, 'control': 0.001}[row['bin']] * (patient_index + 1)
                row['biolord_score_diagnosis1'] += change
                row['plan1_score'] += change
        pd.DataFrame(rows).to_csv(self.csv, index=False)
        manifest = analysis.generate_table_statistics(self.csv, bootstrap_samples=32)
        self.assertEqual(manifest['history_gain_contrasts']['family_size'], 4)
        self.assertEqual(manifest['history_gain_contrasts']['alternative'], 'greater')
        before = pd.read_csv(self.analysis / 'group_retrieval_gains.csv')
        contrasts = before.loc[before.history_contrast_reference_group.notna()]
        self.assertTrue(contrasts.loc[contrasts.metric.eq('plan'), 'history_contrast_p_value'].isna().all())
        self.assertTrue(contrasts.loc[contrasts.metric.eq('diagnosis'), 'history_contrast_p_value'].notna().all())
        for row in rows:
            row['plan1_status'] = 'scored'
        pd.DataFrame(rows).to_csv(self.csv, index=False)
        manifest = analysis.generate_table_statistics(self.csv, bootstrap_samples=32)
        after = pd.read_csv(self.analysis / 'group_retrieval_gains.csv')
        contrasts = after.loc[after.history_contrast_reference_group.notna()]
        self.assertTrue(contrasts.history_contrast_p_value.notna().all())
        self.assertTrue(contrasts.history_contrast_p_value_holm.notna().all())
        # CSV inference changes integer columns to floats while Plan cells are blank.
        pd.testing.assert_frame_equal(before.query("metric == 'diagnosis'").reset_index(drop=True),
                                      after.query("metric == 'diagnosis'").reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        without = after.drop(columns=[column for column in after if column.startswith('history_contrast_')])
        self.assertEqual(exporter.TABLES['group_retrieval_gains'](after),
                         exporter.TABLES['group_retrieval_gains'](without))
        pending_tex = exporter.TABLES['history_gains_over_others'](before)
        complete_tex = exporter.TABLES['history_gains_over_others'](after)
        self.assertIn('--', pending_tex)
        self.assertNotIn('--', complete_tex)
        self.assertIn('^{**}', complete_tex)
        # Selecting this table consumes the existing group CSV and copies it verbatim.
        selected = exporter.generate_tables(self.analysis, ['history_gains_over_others'], self.report)
        paths = selected['history_gains_over_others']
        self.assertEqual(paths['table'].read_text(), complete_tex)
        self.assertEqual(paths['table'].read_bytes(), paths['report'].read_bytes())
        record = next(item for item in manifest['outputs'] if Path(item['path']).name == 'group_retrieval_gains.csv')
        self.assertEqual(record['sha256'], hashlib.sha256(Path(record['path']).read_bytes()).hexdigest())

    def test_single_table_refresh_keeps_other_tables_and_manifest_entries(self):
        self.summarize()
        exporter.generate_tables(self.analysis)
        others={name:(self.analysis/'tables'/f'{name}.tex').read_bytes()
                for name in exporter.TABLES if name!='main_results'}
        output=exporter.generate_tables(self.analysis,['main_results'])
        self.assertEqual(set(output),{'main_results'})
        self.assertEqual(others,{name:(self.analysis/'tables'/f'{name}.tex').read_bytes() for name in others})
        self.assertEqual(set(json.loads((self.analysis/'table_exports_manifest.json').read_text())['tables']),set(exporter.TABLES))

    def test_bad_csv_does_not_overwrite_existing_tables(self):
        self.summarize()
        outputs=exporter.generate_tables(self.analysis,report_dir=self.report)
        saved={path:path.read_bytes() for paths in outputs.values() for path in paths.values()}
        path=self.analysis/'group_retrieval_gains.csv'
        pd.read_csv(path).drop(columns='p_value_holm').to_csv(path,index=False)
        with self.assertRaises(ValueError):
            exporter.generate_tables(self.analysis,report_dir=self.report)
        self.assertEqual(saved,{path:path.read_bytes() for path in saved})

    def test_unvalidated_judge_and_unavailable_plan_references_remain_unscored(self):
        (self.root/'run.json').write_text(json.dumps({'plan_evaluation':{'judge_settings':{'reporting':'experimental_unvalidated'}}}))
        manifest=self.summarize()
        self.assertTrue(manifest['unvalidated_plan_scores_excluded'])
        for name in exporter.TABLES:
            frame=pd.read_csv(self.analysis/f'{exporter.TABLE_INPUTS[name]}.csv')
            plan=frame.loc[frame.metric.eq('plan')]
            self.assertTrue(plan.p_value_holm.isna().all())
            if name == 'group_retrieval_gains':
                self.assertTrue(plan.history_contrast_p_value.isna().all())
            if name in ('main_results', 'generator_retrieval_gains'):
                self.assertTrue(plan.p_value_two_sided_holm.isna().all())
        rows=self.rows()
        for row in rows:
            if row['bin']=='history':
                row['plan_label']='(record artifact)'
        (self.root/'run.json').write_text('{}')
        pd.DataFrame(rows).to_csv(self.csv,index=False)
        analysis.generate_table_statistics(self.csv,bootstrap_samples=32)
        group=pd.read_csv(self.analysis/'group_retrieval_gains.csv')
        self.assertTrue(group.loc[group.metric.eq('plan') & group.group.eq('history'),'n_eligible'].eq(0).all())

    def test_input_validation_and_missing_judgments(self):
        rows=self.rows()
        pd.DataFrame(rows+[rows[0]]).to_csv(self.csv,index=False)
        with self.assertRaises(ValueError):
            analysis.load_scored_results(self.csv)
        rows[0]['biolord_score_diagnosis1']=1.1
        pd.DataFrame(rows).to_csv(self.csv,index=False)
        with self.assertRaises(ValueError):
            analysis.load_scored_results(self.csv)
        rows=self.rows()
        rows[0]['plan1_status']='pending'
        pd.DataFrame(rows).to_csv(self.csv,index=False)
        data=analysis.load_scored_results(self.csv)
        self.assertTrue(pd.isna(data.iloc[0]['plan']))

    def test_source_change_aborts_before_publishing(self):
        pd.DataFrame(self.rows()).to_csv(self.csv,index=False)
        with patch.object(analysis,'_sha256',side_effect=['before','after']):
            with self.assertRaisesRegex(RuntimeError,'source CSV changed'):
                analysis.generate_table_statistics(self.csv,bootstrap_samples=32)
        self.assertFalse(self.analysis.exists())

    def test_cli_csv_only_then_selected_table(self):
        pd.DataFrame(self.rows()).to_csv(self.csv,index=False)
        with redirect_stdout(StringIO()):
            analysis.main(['--results-csv',str(self.csv),'--csv-only'])
            exporter.main(['--analysis-dir',str(self.analysis),'--table','main_results','--report-dir',str(self.report)])
        self.assertTrue((self.report/'main_results.tex').exists())
        self.assertFalse((self.report/'group_retrieval_gains.tex').exists())

    def combined_rows(self):
        rows = []
        for base in self.rows():
            if base['model'] != MODEL or base['variant'] != 'llm_only':
                continue
            for m, model in enumerate(analysis.MODEL_LABELS):
                for v, variant in enumerate(analysis.ALL_VARIANT_LABELS):
                    rows.append(dict(base, model=model, variant=variant, plan1_protocol_version=2, plan1_rank=1,
                                     diagnosis1_status='scored', diagnosis1_reason='', generation_status='completed',
                                     generation_error_type='', variant_error_type='',
                                     biolord_score_diagnosis1=base['biolord_score_diagnosis1'] + m * .001 + v * .01))
        return pd.DataFrame(rows)

    def test_combined_requires_all_model_variant_slots_before_writing(self):
        data = self.combined_rows()
        for incomplete in (data.loc[data.model.ne(MODEL)], data.loc[data.variant.ne('agentic_both')], data.iloc[1:]):
            incomplete.to_csv(self.csv, index=False)
            with self.subTest(rows=len(incomplete)), self.assertRaisesRegex(ValueError, 'every record/model/variant'):
                analysis.generate_table_statistics(self.csv, bootstrap_samples=16, combined=True)
            self.assertFalse(self.analysis.exists())

    def test_combined_masks_failed_scores_and_retains_exact_failure_reasons(self):
        data = self.combined_rows()
        ids = data.id.unique()[:4]
        target = data.model.eq(MODEL) & data.variant.eq('agentic_guidelines')
        for patient_id, field, error in ((ids[0], 'generation_error_type', 'ContextWindowExceeded'),
                                         (ids[1], 'variant_error_type', 'AgentValidationError')):
            mask = target & data.id.eq(patient_id)
            data.loc[mask, 'diagnosis1_status'] = 'not_scored'
            data.loc[mask, 'generation_status'] = 'not_processed'
            data.loc[mask, field] = error
        data.loc[target & data.id.eq(ids[2]), 'diagnosis1_status'] = 'not_scored'
        data.loc[target & data.id.eq(ids[3]), ['biolord_score_diagnosis1', 'plan1_score']] = 0.
        failed_plan = target & data.id.eq(ids[0])
        data.loc[failed_plan, 'plan1_status'] = 'invalid_response'
        data.loc[failed_plan, 'plan1_reason'] = 'invalid_response'
        data.to_csv(self.csv, index=False)
        loaded = analysis.load_scored_results(self.csv)
        self.assertTrue(loaded.loc[target & data.id.isin(ids[:3]), 'diagnosis'].isna().all())
        self.assertTrue(loaded.loc[failed_plan, 'plan'].isna().all())
        self.assertEqual(loaded.loc[target & data.id.eq(ids[3]), 'diagnosis'].item(), 0)
        summaries = analysis.generate_table_statistics(self.csv, bootstrap_samples=16, combined=True)
        coverage = summaries['coverage_and_failures']
        diagnosis = coverage.loc[coverage.level.eq('overall') & coverage.model.eq(MODEL)
                                 & coverage.variant.eq('agentic_guidelines') & coverage.metric.eq('diagnosis')]
        self.assertEqual(set(diagnosis.reason) - {''},
                         {'ContextWindowExceeded', 'AgentValidationError', 'score_missing'})
        self.assertEqual(diagnosis.n_status_reason.sum(), data.id.nunique())
        self.assertTrue(diagnosis.n_failed.eq(3).all())
        self.assertFalse(list(self.analysis.glob('*manifest*')))

    def test_combined_source_change_does_not_replace_existing_outputs(self):
        self.combined_rows().to_csv(self.csv, index=False)
        self.analysis.mkdir()
        sentinel = self.analysis / 'main_summary_stats.csv'
        sentinel.write_text('existing statistics\n')
        with patch.object(analysis, '_sha256', side_effect=['before', 'after']):
            with self.assertRaisesRegex(RuntimeError, 'source CSV changed'):
                analysis.generate_table_statistics(self.csv, bootstrap_samples=16, combined=True)
        self.assertEqual(sentinel.read_text(), 'existing statistics\n')
        self.assertEqual(list(self.analysis.iterdir()), [sentinel])

    def combined_pvalue_reference(self):
        data = self.combined_rows()
        cases = pd.factorize(data.id)[0] % 8
        variants = data.variant.map({variant: index for index, variant in enumerate(analysis.ALL_VARIANT_LABELS)})
        data['biolord_score_diagnosis1'] += variants * cases * .001
        data['plan1_score'] += variants * (.01 + cases * .001)
        # Keep untestable comparisons alongside finite p-values in both directions.
        data.loc[data.variant.eq('agentic_both'), 'plan1_status'] = 'invalid_response'
        data.loc[data.variant.eq('agentic_both'), 'plan1_reason'] = 'invalid_response'
        data.to_csv(self.csv, index=False)
        normalized = analysis.load_scored_results(self.csv)
        return {
            'main_summary_stats': analysis.summarize_main_results(normalized, 16, 17, available_scores=True),
            'generator_retrieval_gains': analysis.summarize_generator_retrieval_gains(
                normalized, available_scores=True, average_only=True),
            'group_retrieval_gains': analysis.summarize_group_retrieval_gains(normalized, available_scores=True),
            'coverage_and_failures': analysis.summarize_coverage_and_failures(normalized),
        }

    def test_combined_pvalue_names_change_without_changing_any_values(self):
        original = self.combined_pvalue_reference()
        mapping = {'p_value': 'p_value_one_sided', 'p_value_holm': 'p_value_one_sided_holm',
                   'history_contrast_p_value': 'history_contrast_p_value_one_sided',
                   'history_contrast_p_value_holm': 'history_contrast_p_value_one_sided_holm'}
        sources = {path: path.read_bytes() for path in (self.csv, self.root / 'run.json', self.root / 'results.jsonl')}
        # A combined export must not rewrite the independent two-sided comparison CSV.
        counterparts.generate_agentic_comparisons(self.csv)
        counterpart_path = self.analysis / counterparts.OUTPUT_NAME
        sources[counterpart_path] = counterpart_path.read_bytes()
        summaries = analysis.generate_table_statistics(self.csv, bootstrap_samples=16, seed=17, combined=True)
        for name, frame in original.items():
            with self.subTest(name=name):
                expected = frame.rename(columns=mapping)
                pd.testing.assert_frame_equal(summaries[name], expected, check_exact=True)
                path = self.analysis / f'{name}.csv'
                self.assertEqual(path.read_text(), expected.to_csv(index=False, na_rep=''))
                saved = pd.read_csv(path)
                self.assertFalse(set(mapping) & set(saved))
                for field in saved:
                    if 'p_value' in field:
                        self.assertTrue('one_sided' in field or 'two_sided' in field, field)
                for field in ('p_value_two_sided', 'p_value_two_sided_holm'):
                    if field in frame:
                        pd.testing.assert_series_equal(summaries[name][field], frame[field], check_exact=True)
        self.assertIn('history_contrast_p_value_one_sided_holm', summaries['group_retrieval_gains'])
        self.assertTrue(summaries['main_summary_stats'].p_value_one_sided.notna().any())
        self.assertTrue(summaries['main_summary_stats'].p_value_one_sided.isna().any())
        self.assertEqual(sources, {path: path.read_bytes() for path in sources})

    def test_combined_rendering_is_identical_with_explicit_and_historical_pvalue_names(self):
        original = self.combined_pvalue_reference()
        analysis.generate_table_statistics(self.csv, bootstrap_samples=16, seed=17, combined=True)
        outputs = exporter.generate_tables(self.analysis, combined=True)
        rendered = {name: paths['table'].read_bytes() for name, paths in outputs.items()}
        for name, renderer in exporter.TABLES.items():
            input_name = 'main_summary_stats' if name == 'main_results' else exporter.TABLE_INPUTS[name]
            self.assertEqual(rendered[name], renderer(original[input_name], available_scores=True).encode('utf-8'))
        # Existing saved combined CSVs remain readable before the user regenerates them.
        for name, frame in original.items():
            frame.to_csv(self.analysis / f'{name}.csv', index=False, na_rep='')
        outputs = exporter.generate_tables(self.analysis, combined=True)
        self.assertEqual(rendered, {name: paths['table'].read_bytes() for name, paths in outputs.items()})

    def test_combined_renderer_rejects_ambiguous_pvalue_columns_before_overwriting_tables(self):
        self.combined_pvalue_reference()
        analysis.generate_table_statistics(self.csv, bootstrap_samples=16, seed=17, combined=True)
        outputs = exporter.generate_tables(self.analysis, combined=True)
        protected = {paths['table']: paths['table'].read_bytes() for paths in outputs.values()}
        path = self.analysis / 'group_retrieval_gains.csv'
        original = pd.read_csv(path)
        for old, new in (('p_value', 'p_value_one_sided'), ('p_value_holm', 'p_value_one_sided_holm'),
                         ('history_contrast_p_value', 'history_contrast_p_value_one_sided'),
                         ('history_contrast_p_value_holm', 'history_contrast_p_value_one_sided_holm')):
            ambiguous = original.copy()
            ambiguous[old] = ambiguous[new]
            ambiguous.to_csv(path, index=False)
            with self.subTest(field=old), self.assertRaisesRegex(ValueError, 'Ambiguous'):
                exporter.generate_tables(self.analysis, combined=True)
            self.assertEqual(protected, {destination: destination.read_bytes() for destination in protected})

    def test_default_renderer_never_falls_back_to_legacy_csv_or_updates_report(self):
        # Even when only a legacy filename exists, the default combined path must fail safely.
        self.analysis.mkdir()
        (self.analysis / 'main_results.csv').write_text('old protocol data\n')
        with patch.object(exporter, 'DEFAULT_ANALYSIS', self.analysis):
            with self.assertRaises(FileNotFoundError) as error:
                exporter.main(['--report-dir', str(self.report)])
        self.assertIn('main_summary_stats.csv', str(error.exception))
        self.assertFalse(self.report.exists())
        self.assertFalse((self.analysis / 'tables').exists())
        self.assertFalse((self.analysis / 'figures').exists())

    def test_counterpart_cli_saves_only_its_six_comparisons_and_preserves_sources(self):
        self.combined_rows().to_csv(self.csv, index=False)
        self.analysis.mkdir()
        saved_table = self.analysis / 'main_summary_stats.csv'
        saved_table.write_text('protected existing statistics\n')
        sources = {path: path.read_bytes() for path in (self.csv, self.root / 'run.json', self.root / 'results.jsonl', saved_table)}
        # Exercise the no-argument command without pointing it at production artifacts.
        with patch.object(counterparts, 'ROOT', self.root), patch.object(counterparts, 'RESULTS', self.csv.name):
            with redirect_stdout(StringIO()) as output:
                counterparts.main([])
        destination = self.analysis / 'agentic_vs_nonagentic_significance.csv'
        scores = pd.read_csv(destination)
        self.assertEqual(len(scores), 6)
        self.assertFalse(scores.duplicated(['agentic_variant', 'nonagentic_variant', 'metric']).any())
        self.assertTrue(scores.family_size.eq(6).all())
        self.assertTrue(scores.alternative.eq('two-sided').all())
        self.assertTrue(scores.n_generators_averaged.eq(4).all())
        self.assertEqual({path.name for path in self.analysis.iterdir()}, {destination.name, saved_table.name})
        self.assertIn(str(destination), output.getvalue())
        first = destination.read_bytes()
        with redirect_stdout(StringIO()):
            counterparts.main(['--results-csv', str(self.csv)])
        self.assertEqual(first, destination.read_bytes())
        self.assertEqual(sources, {path: path.read_bytes() for path in sources})

    def test_counterpart_reader_excludes_stale_failed_scores_and_preserves_zero(self):
        data = self.combined_rows()
        ids = data.id.unique()[:3]
        plan_failure = (data.id.eq(ids[0]) & data.model.eq(MODEL)
                        & data.variant.eq('agentic_patient_retrieval'))
        data.loc[plan_failure, 'plan1_status'] = 'invalid_response'  # Stale nonzero score must be masked.
        diagnosis_failure = data.id.eq(ids[1]) & data.model.eq(MODEL) & data.variant.eq('agentic_both')
        data.loc[diagnosis_failure, 'diagnosis1_status'] = 'not_scored'
        data.loc[data.id.eq(ids[2]) & data.variant.eq('agentic_patient_retrieval'), 'plan1_score'] = 0.
        data.to_csv(self.csv, index=False)
        scores = counterparts.generate_agentic_comparisons(self.csv)
        plan = scores.query("agentic_variant == 'agentic_patient_retrieval' and metric == 'plan'").iloc[0]
        self.assertEqual(plan.n_pairs, data.id.nunique() - 1)
        expected = data.loc[data.variant.eq('agentic_patient_retrieval') & data.id.ne(ids[0]), 'plan1_score'].mean()
        self.assertAlmostEqual(plan.mean_agentic, expected)
        diagnosis = scores.query("agentic_variant == 'agentic_both' and metric == 'diagnosis'").iloc[0]
        self.assertEqual(diagnosis.n_pairs, data.id.nunique() - 1)

    def test_counterpart_rejects_wrong_protocol_rank_missing_status_and_incomplete_inputs(self):
        data = self.combined_rows()
        for invalid in (data.assign(plan1_protocol_version=1), data.assign(plan1_rank=2),
                        data.drop(columns='diagnosis1_status'), data.drop(columns='plan1_protocol_version'),
                        data.iloc[1:], data.loc[data.model.ne(MODEL)],
                        data.loc[data.variant.ne('agentic_both')], data.assign(model='unknown')):
            invalid.to_csv(self.csv, index=False)
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                counterparts.generate_agentic_comparisons(self.csv)
            self.assertFalse(self.analysis.exists())

    def test_counterpart_source_change_aborts_without_overwriting_existing_output(self):
        self.combined_rows().to_csv(self.csv, index=False)
        self.analysis.mkdir()
        destination = self.analysis / counterparts.OUTPUT_NAME
        destination.write_text('keep existing counterpart tests\n')
        with patch.object(counterparts, 'sha256_file', side_effect=['before', 'after']):
            with self.assertRaisesRegex(RuntimeError, 'source CSV changed'):
                counterparts.generate_agentic_comparisons(self.csv)
        self.assertEqual(destination.read_text(), 'keep existing counterpart tests\n')
        self.assertEqual(list(self.analysis.iterdir()), [destination])


if __name__=='__main__':
    unittest.main(verbosity=2)
