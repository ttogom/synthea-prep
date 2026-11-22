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


if __name__=='__main__':
    unittest.main(verbosity=2)
