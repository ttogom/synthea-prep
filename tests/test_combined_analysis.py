"""Offline export and end-to-end checks for all seven variants."""

from contextlib import redirect_stdout
import copy
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import yaml

# Direct execution must resolve repository modules before importing them.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import build_analysis_dataset as builder
from scripts.analysis import render_tables, summarize_experiment_results as analysis, table_statistics as stats
from scripts.plan_judge_v2 import protocol, runner


SOURCE_CONFIG = yaml.safe_load((builder.OUTPUT / 'sources.yaml').read_text())
MODELS = SOURCE_CONFIG['models']
VARIANTS = tuple(stats.VARIANT_LABELS)
CONDITIONS = {'history': 'essential hypertension', 'acute': 'acute viral pharyngitis',
              'control': 'chronic congestive heart failure'}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class CombinedExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'combined'
        self.sources = copy.deepcopy(SOURCE_CONFIG)
        self.sources.update(labels='labels.jsonl', patients='patients.jsonl')
        self.sources['runs'][0]['path'] = 'runs/base'
        self.sources['runs'][1]['path'] = 'runs/agentic'
        labels = []
        for group, condition in CONDITIONS.items():
            for i in range(4):
                labels.append({'id': f'{group}_00000000-0000-4000-8000-{i:012d}', 'bin': group,
                               'condition': condition, 'plan': ['(record artifact)'] if group == 'control' and i == 0
                               else ['Action A', 'Action B']})
        self.labels = labels
        write_jsonl(self.root / 'labels.jsonl', labels)
        write_jsonl(self.root / 'patients.jsonl', [{'id': r['id'], 'subjective': 'S', 'objective': 'O'} for r in labels])
        judge = {'backend': 'fake', 'model_id': 'test/judge', 'revision_resolved': 'fixed'}
        for source in self.sources['runs']:
            run_dir = self.root / source['path']
            records = []
            for i, label in enumerate(labels):
                variants = {}
                for variant in source['variants']:
                    slots = {}
                    for m, model in enumerate(MODELS):
                        empty = variant == 'guidelines' and i == 11 and m == 2
                        slots[model] = {'status': 'completed', 'predictions': [
                            {'condition': f'Rank {rank}', 'plan': [] if empty and rank == 1 else
                             [f'Action {rank}; keep this together', ' Action B ', '']}
                            for rank in range(1, 4)], 'evaluation': [
                                {'rank': rank, 'diagnosis_correctness': {'status': 'scored', 'score': 0.2 + i * .02 + m * .01},
                                 'condition': {'status': 'scored', 'metrics': {'BERTScore_F1': .71}},
                                 'plan_correctness': {'status': 'scored', 'score': 999, 'unsupported_extra_count': 99}}
                                for rank in range(1, 4)]}
                    if variant == 'agentic_both' and i == 0:
                        slots[MODELS[0]].update(status='not_processed', predictions=None, evaluation=None)
                    variants[variant] = {'status': 'completed', 'assessment_plans': slots}
                records.append({'id': label['id'], 'variants': variants})
            write_jsonl(run_dir / 'results.jsonl', records)
            (run_dir / 'experiment_results.csv').write_text('OBSOLETE PROTOCOL ONE CSV: NEVER READ\n')
            (run_dir / 'run.json').write_text('{"plan_evaluation": {"protocol": {"version": 1}}}')
            for variant in source['variants']:
                rows = []
                manifest = {'variant': variant, 'run_id': variant, 'protocol': protocol.protocol_identity(),
                            'judge': judge, 'selection': {'rank': 1, 'sample': None,
                                'patients_in_results': len(labels), 'generator_models_expected': MODELS},
                            'inputs': {key: {'sha256': digest(path)} for key, path in (
                                ('labels', self.root / 'labels.jsonl'), ('patients', self.root / 'patients.jsonl'),
                                ('results', run_dir / 'results.jsonl'))}}
                for i, record in enumerate(records):
                    for model in MODELS:
                        slot = record['variants'][variant]['assessment_plans'][model]
                        row = {'run_id': variant, 'patient_id': record['id'], 'generator_model': model, 'rank': 1,
                               'status': 'scored', 'reason': None, 'reasons': [], 'attempt': 1,
                               'parsed': {'reference_items': [{'ref': 1, 'match': 'full', 'generated': [1]},
                                                             {'ref': 2, 'match': 'partial', 'generated': [2]}],
                                          'harmful_actions': []},
                               'credits': [1, .5], 'coverage': .75, 'harmful_count': 0, 'penalized': .75,
                               'source': {'variant_status': 'completed', 'generation_status': slot['status']}}
                        if self.labels[i]['plan'] == ['(record artifact)'] or slot['status'] != 'completed':
                            row.update(status='unavailable', reason='record_artifact_plan' if slot['status'] == 'completed'
                                       else 'generation_not_processed', attempt=0, parsed=None, credits=None,
                                       coverage=None, harmful_count=None, penalized=None)
                        elif variant == 'agentic_patient_retrieval' and i == 4 and model == MODELS[1]:
                            row.update(status='invalid_response', reason='invalid_response', attempt=3,
                                       parsed=None, credits=None, coverage=None, harmful_count=None, penalized=None,
                                       error={'type': 'JudgeResponseError', 'message': 'missing citation'})
                        elif slot['predictions'][0]['plan'] == []:
                            row.update(attempt=0, reason='empty_generated_plan', parsed=None,
                                       credits=[0, 0], coverage=0., penalized=0.)
                        rows.append(row)
                counts = {'plans': len(rows), 'scored': sum(r['status'] == 'scored' for r in rows),
                          'unavailable': sum(r['status'] == 'unavailable' for r in rows),
                          'failed_retryable': 0, 'failed_exhausted': sum(r['status'] == 'invalid_response' for r in rows),
                          'not_attempted': 0, 'duplicate_final_rows': 0}
                manifest['counts'] = counts
                directory = run_dir / 'plan_judging_v2' / variant
                directory.mkdir(parents=True)
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                runner.export_csv(directory, manifest, rows)
        self.originals = {p: digest(p) for p in self.root.rglob('*') if p.is_file()}

    def csv_path(self, variant='llm_only'):
        source = next(s for s in self.sources['runs'] if variant in s['variants'])
        return self.root / source['path'] / 'plan_judging_v2' / variant / 'judgments.csv'

    def test_full_pipeline_and_no_protocol_one_or_provenance(self):
        frame, judgments = builder.build_dataset(self.sources, self.output, self.root)
        self.assertEqual(len(frame), 12 * 4 * 7)
        self.assertEqual(len(judgments), len(frame))
        self.assertEqual(set(frame.variant), set(VARIANTS))
        self.assertFalse(frame.duplicated(['id', 'model', 'variant']).any())
        self.assertFalse(any('unsupported_extra' in name for name in frame))
        self.assertNotIn('plan2_score', frame)
        self.assertNotIn('plan3_score', frame)
        self.assertTrue(frame.plan2_status.eq('not_judged').all())
        self.assertEqual(frame.iloc[0].diagnosis3, 'Rank 3')
        self.assertEqual(json.loads(frame.iloc[0].plan3_actions), ['Action 3; keep this together', ' Action B ', ''])
        self.assertEqual(frame.iloc[0].bertf1_diagnosis2, .71)
        self.assertTrue(frame.plan1_score.replace('', np.nan).dropna().astype(float).le(1).all())
        data_path = self.output / 'experiment_results.csv'
        saved = pd.read_csv(data_path)
        empty = saved.loc[saved.plan1_reason.eq('empty_generated_plan')].iloc[0]
        self.assertEqual(empty.plan1_score, 0)
        self.assertEqual(empty.plan1_status, 'scored')
        failed = saved.loc[saved.plan1_status.eq('invalid_response')].iloc[0]
        self.assertTrue(pd.isna(failed.plan1_score))
        summaries = analysis.generate_table_statistics(data_path, bootstrap_samples=32, combined=True)
        gains = summaries['generator_retrieval_gains']
        self.assertEqual(set(gains.model), {'model_average'})
        self.assertTrue(gains.family_size.eq(6).all())
        self.assertTrue(gains.n_generators_averaged.eq(4).all())
        report = self.root / 'report'
        outputs = render_tables.generate_tables(self.output / 'analysis', report_dir=report, combined=True)
        self.assertEqual(len(outputs), 4)
        self.assertFalse(report.exists())
        self.assertTrue((self.output / 'analysis/tables/main_summary_stats.tex').exists())
        self.assertTrue((self.output / 'analysis/figures').is_dir())
        self.assertFalse(list(self.output.rglob('.gitkeep')))
        coverage = summaries['coverage_and_failures']
        overall = coverage.loc[coverage.level.eq('overall') & coverage.model.eq('all_models')
                               & coverage.metric.eq('plan')]
        for variant, group in overall.groupby('variant'):
            original = judgments.loc[judgments.variant.eq(variant)]
            self.assertEqual(group.n_status_reason.sum(), len(original))
            self.assertEqual(group.iloc[0].n_unavailable, original.status.eq('unavailable').sum())
            self.assertEqual(group.iloc[0].n_failed, original.status.eq('invalid_response').sum())
            self.assertEqual(group.iloc[0].n_scored, original.status.eq('scored').sum())
        self.assertFalse(list(self.output.rglob('*manifest*')))
        self.assertFalse(list(self.output.rglob('main_results.*')))
        for path, value in self.originals.items():
            self.assertEqual(digest(path), value, path)
        # The CSV-only command recognizes a protocol-2 export outside the default directory.
        with redirect_stdout(StringIO()):
            analysis.main(['--results-csv', str(data_path), '--csv-only'])
            render_tables.main(['--analysis-dir', str(self.output / 'analysis'), '--analysis-only'])

    def test_regeneration_is_deterministic(self):
        builder.build_dataset(self.sources, self.output, self.root)
        analysis.generate_table_statistics(self.output / 'experiment_results.csv', bootstrap_samples=32, combined=True)
        before = {p.relative_to(self.output): digest(p) for p in self.output.rglob('*') if p.is_file()}
        builder.build_dataset(self.sources, self.output, self.root)
        analysis.generate_table_statistics(self.output / 'experiment_results.csv', bootstrap_samples=32, combined=True)
        self.assertEqual(before, {p.relative_to(self.output): digest(p) for p in self.output.rglob('*') if p.is_file()})

    def test_rejects_protocol_one_and_bad_csv_before_writing(self):
        path = self.csv_path()
        original = path.read_bytes()
        frame = pd.read_csv(path, keep_default_na=False)
        mutations = [frame.assign(protocol_version=1), pd.concat([frame, frame.iloc[:1]]),
                     frame.iloc[1:], frame.assign(judge_revision='different'), frame.assign(run_id='wrong'),
                     frame.assign(generator_model='unknown')]
        scored = frame.status.eq('scored')
        wrong = frame.copy()
        wrong.loc[scored, 'coverage'] = '.99'
        mutations.append(wrong)
        wrong = frame.copy()
        wrong.loc[~scored, 'coverage'] = '0'
        mutations.append(wrong)
        for column, value in [('matches', '["none", "none"]'), ('credits', '[true, 0.5]'),
                              ('harmful_actions', '[{"generated": 1, "why": "test"}]'),
                              ('generation_status', 'failed_incomplete')]:
            wrong = frame.copy()
            wrong.loc[scored, column] = value
            mutations.append(wrong)
        for mutation in mutations:
            with self.subTest(columns=list(mutation.columns)):
                mutation.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    builder.build_dataset(self.sources, self.output, self.root)
                self.assertFalse(self.output.exists())
        path.write_bytes(original)

    def test_rejects_stale_inputs_and_conflicting_manifests(self):
        path = self.csv_path().with_name('manifest.json')
        original = json.loads(path.read_text())
        for field in ('protocol', 'judge', 'inputs', 'counts'):
            value = copy.deepcopy(original)
            if field == 'protocol':
                value[field]['version'] = 1
            elif field == 'judge':
                value[field]['model_id'] = 'other/judge'
            elif field == 'inputs':
                value[field]['results']['sha256'] = 'stale'
            else:
                value[field]['plans'] += 1
            path.write_text(json.dumps(value))
            with self.subTest(field=field), self.assertRaises(ValueError):
                builder.build_dataset(self.sources, self.output, self.root)
            self.assertFalse(self.output.exists())
        path.write_text(json.dumps(original))

    def test_rejects_duplicate_labels_and_mismatched_cohorts(self):
        path = self.root / 'labels.jsonl'
        with path.open('a') as f:
            f.write(json.dumps(self.labels[0]) + '\n')
        with self.assertRaisesRegex(ValueError, 'duplicate id'):
            builder.build_dataset(self.sources, self.output, self.root)
        write_jsonl(path, self.labels)
        results = self.root / 'runs/agentic/results.jsonl'
        records = [json.loads(s) for s in results.read_text().splitlines()]
        write_jsonl(results, records[1:])
        with self.assertRaisesRegex(ValueError, 'identical patient'):
            builder.build_dataset(self.sources, self.output, self.root)

    def test_failed_validation_preserves_existing_outputs(self):
        builder.build_dataset(self.sources, self.output, self.root)
        before = {p: digest(p) for p in self.output.iterdir() if p.is_file()}
        path = self.csv_path()
        frame = pd.read_csv(path).assign(protocol_version=1)
        frame.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            builder.build_dataset(self.sources, self.output, self.root)
        self.assertEqual(before, {p: digest(p) for p in before})

    def test_builder_cli_reuses_saved_sources_and_does_not_read_old_csv(self):
        sources = copy.deepcopy(self.sources)
        for key in ('labels', 'patients'):
            sources[key] = str(self.root / sources[key])
        for source in sources['runs']:
            source['path'] = str(self.root / source['path'])
        builder.build_dataset(sources, self.output, self.root)
        (self.output / 'sources.yaml').write_text(yaml.safe_dump(sources))
        (self.output / 'README.md').write_text('Keep this reviewed documentation.\n')
        originals = {name: (self.output / name).read_bytes() for name in ('sources.yaml', 'README.md')}
        with redirect_stdout(StringIO()):
            builder.main(['--sources', str(self.output / 'sources.yaml')])
        self.assertEqual(len(pd.read_csv(self.output / 'experiment_results.csv')), 336)
        self.assertEqual(originals, {name: (self.output / name).read_bytes() for name in originals})

    def test_each_diagnosis_metric_keeps_its_own_status_and_literal_none_survives(self):
        source = self.sources['runs'][0]
        directory = self.root / source['path']
        path = directory / 'results.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        slot = records[0]['variants']['llm_only']['assessment_plans'][MODELS[0]]
        slot['predictions'][0]['condition'] = 'None'
        slot['evaluation'][1]['condition'].update(status='error', reason='bert_failed')
        slot['evaluation'][2]['diagnosis_correctness'].update(status='error', reason='biolord_failed')
        write_jsonl(path, records)
        for variant in source['variants']:
            manifest_path = directory / 'plan_judging_v2' / variant / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['inputs']['results']['sha256'] = digest(path)
            manifest_path.write_text(json.dumps(manifest))
        frame, _ = builder.build_dataset(self.sources, self.output, self.root)
        row = frame.iloc[0]
        self.assertTrue(pd.isna(row.bertf1_diagnosis2))
        self.assertEqual(row.bertf1_diagnosis2_status, 'error')
        self.assertEqual(row.bertf1_diagnosis2_reason, 'bert_failed')
        self.assertTrue(pd.notna(row.biolord_score_diagnosis2))
        self.assertTrue(pd.isna(row.biolord_score_diagnosis3))
        self.assertEqual(row.diagnosis3_reason, 'biolord_failed')
        self.assertEqual(analysis.load_scored_results(self.output / 'experiment_results.csv').iloc[0].diagnosis1, 'None')

    def test_combined_summary_rejects_protocol_one_before_writing(self):
        builder.build_dataset(self.sources, self.output, self.root)
        path = self.output / 'experiment_results.csv'
        data = pd.read_csv(path).assign(plan1_protocol_version=1)
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'protocol-2'):
            analysis.generate_table_statistics(path, bootstrap_samples=32, combined=True)
        self.assertFalse((self.output / 'analysis').exists())

    def test_export_requires_all_seven_variants_and_four_models(self):
        for setting in ('models', 'variants'):
            config = copy.deepcopy(self.sources)
            if setting == 'models':
                config['models'].pop()
            else:
                config['runs'][1]['variants'].pop()
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                builder.build_dataset(config, self.output, self.root)
        self.assertFalse(self.output.exists())




if __name__ == '__main__':
    unittest.main(verbosity=2)
