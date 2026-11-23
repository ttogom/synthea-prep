"""Rendering, independent-function, raw-data and publication tests for both matrices."""

import ast
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Direct test execution must resolve repository modules before importing them.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import render_poster_matrices as matrices
from scripts.analysis import poster_statistics as stats
from test_poster_statistics import model_fixture, raw_export


def summary_fixture():
    return stats.summarize_model_poster_comparisons(model_fixture())


class InformationRenderingTests(unittest.TestCase):
    def test_complete_standalone_tex_contains_only_own_cells_and_note_below_chart(self):
        summary = summary_fixture().sample(frac=1, random_state=12)
        document = matrices.render_information_matrix(summary)
        self.assertIn(r'\documentclass[border=2pt]{standalone}', document)
        self.assertEqual(document.count(matrices.NOTE), 1)
        self.assertGreater(document.index(matrices.NOTE), document.index(r'\end{tikzpicture}'))
        self.assertEqual(len(re.findall(r'^% cell ', document, re.M)), 24)
        self.assertNotIn('Adaptive search adds', document)
        self.assertNotIn('Which information helps?', document)
        self.assertNotIn('confidence', document.lower())
        self.assertNotIn('**', document)
        for row in summary.loc[summary.question.eq('information')].itertuples():
            prefix = f'% cell model={row.model} metric={row.metric} source={row.source} '
            trace = next(line for line in document.splitlines() if line.startswith(prefix))
            self.assertIn(f'n_pairs={row.n_pairs}', trace)
            self.assertIn(f'p_value_two_sided_holm={row.p_value_two_sided_holm}', trace)
        for label in matrices.MODELS.values():
            self.assertEqual(document.count('{' + label + '}'), 1)

    def test_display_uses_model_outcome_source_keys_and_full_precision_for_stars(self):
        summary = summary_fixture()
        index = summary.index[summary.model.eq('ollama_gemma3_4b') & summary.question.eq('information')
                              & summary.source.eq('both') & summary.metric.eq('diagnosis')][0]
        summary.loc[index, 'mean_difference'] = -.140229
        summary.loc[index, 'mean_comparison'] = summary.loc[index, 'mean_baseline']-.140229
        summary['p_value_two_sided'] = 1.
        summary.loc[index, 'p_value_two_sided'] = .01/48
        summary['p_value_two_sided_holm'] = multipletests(summary.p_value_two_sided, method='holm')[1]
        document = matrices.render_information_matrix(summary)
        self.assertIn(r'at (8.340,4.200) {-14.0}', document)
        self.assertEqual(document.count(r'\textsuperscript{*}'), 0)
        summary.loc[index, 'p_value_two_sided'] = (.01-1e-7)/48
        summary['p_value_two_sided_holm'] = multipletests(summary.p_value_two_sided, method='holm')[1]
        changed = matrices.render_information_matrix(summary)
        self.assertIn(r'at (8.340,4.200) {-14.0\textsuperscript{*}}', changed)
        self.assertEqual(changed.count(r'\textsuperscript{*}'), 1)

    def test_invalid_cache_or_inference_cannot_produce_a_matrix(self):
        frame = summary_fixture()
        cases = [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), frame.drop(columns='p_value_two_sided')]
        for column, value in [('model', 'unknown'), ('test_status', 'no_pairs'), ('family_size', 24),
                              ('n_generators_averaged', 4), ('mean_comparison', np.inf),
                              ('mean_difference', .5), ('n_patients', 1),
                              ('baseline_variant', 'wrong'), ('p_value_two_sided_holm', 1.)]:
            bad = frame.copy()
            bad.loc[0, column] = value
            cases.append(bad)
        for bad in cases:
            with self.subTest(rows=len(bad)):
                with self.assertRaises(ValueError):
                    matrices.render_information_matrix(bad)


class InformationGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory/'experiment_results.csv'
        raw_export(model_fixture()).to_csv(self.source, index=False)
        self.figures = self.directory/'analysis'/'figures'

    def test_runs_from_raw_csv_without_another_generator_or_cached_artifact(self):
        before = self.source.read_bytes()
        outputs = matrices.generate_information_matrix(self.source, compile_pdf=False)
        self.assertEqual(set(outputs), {'statistics', 'tex'})
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual({p.name for p in self.figures.iterdir()}, {'poster_information_matrix.tex'})
        self.assertEqual(len(pd.read_csv(outputs['statistics'])), 48)
        tex = outputs['tex'].read_text()
        self.assertIn('source_csv_sha256=', tex)
        code = Path(matrices.__file__).read_text()
        self.assertNotIn('import render_adaptive_matrix', code)
        self.assertNotIn('poster_matrix_layout import', code)
        self.assertNotIn('render_poster_options import', code)
        first = outputs['tex'].read_bytes()
        matrices.generate_information_matrix(self.source, compile_pdf=False)
        self.assertEqual(first, outputs['tex'].read_bytes())

    def test_compile_failure_preserves_old_pair_and_tex_only_removes_stale_pdf(self):
        self.figures.mkdir(parents=True)
        old_tex = self.figures/'poster_information_matrix.tex'
        old_pdf = old_tex.with_suffix('.pdf')
        old_tex.write_bytes(b'old TeX')
        old_pdf.write_bytes(b'old PDF')
        with patch.object(matrices, '_compile', side_effect=RuntimeError('compiler failed')):
            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                matrices.generate_information_matrix(self.source)
        self.assertEqual(old_tex.read_bytes(), b'old TeX')
        self.assertEqual(old_pdf.read_bytes(), b'old PDF')
        self.assertEqual(len(list(self.figures.iterdir())), 2)
        matrices.generate_information_matrix(self.source, compile_pdf=False)
        self.assertFalse(old_pdf.exists())

    def test_invalid_raw_export_preserves_existing_tex(self):
        self.figures.mkdir(parents=True)
        output = self.figures/'poster_information_matrix.tex'
        output.write_text('previous')
        bad = pd.read_csv(self.source).iloc[1:]
        bad.to_csv(self.source, index=False)
        with self.assertRaises(ValueError):
            matrices.generate_information_matrix(self.source)
        self.assertEqual(output.read_text(), 'previous')

    def test_cli_custom_destination_is_independent(self):
        destination = self.directory/'custom'
        outputs = matrices.main(['--results-csv', str(self.source), '--output-dir', str(destination), '--skip-compile', '--matrix', 'information'])['information']
        self.assertEqual(outputs['tex'].parent, destination.resolve())
        self.assertEqual({path.name for path in destination.iterdir()}, {'poster_information_matrix.tex'})

    def test_source_change_during_generation_aborts_before_publishing(self):
        original = matrices.generate_model_poster_statistics

        def changed_source(*args):
            result = original(*args)
            self.source.write_text(self.source.read_text()+'\n')
            return result

        with patch.object(matrices, 'generate_model_poster_statistics', side_effect=changed_source):
            with self.assertRaisesRegex(RuntimeError, 'source CSV changed'):
                matrices.generate_information_matrix(self.source, compile_pdf=False)
        self.assertFalse(self.figures.exists())

    @unittest.skipUnless(shutil.which('pdflatex') and shutil.which('pdftotext'), 'LaTeX/Poppler unavailable')
    def test_real_pdf_contains_all_visible_cells_and_its_own_note(self):
        outputs = matrices.generate_information_matrix(self.source)
        self.assertTrue(outputs['pdf'].read_bytes().startswith(b'%PDF-'))
        text = subprocess.run(['pdftotext', '-layout', str(outputs['pdf']), '-'], check=True,
                              capture_output=True, text=True).stdout
        self.assertEqual(text.count('\f'), 1)
        self.assertEqual(text.count('Holm-adjusted'), 1)
        self.assertGreater(text.index('Holm-adjusted'), text.index('Change vs notes alone'))
        self.assertEqual(len(re.findall(r'[+-]\d+\.\d', text)), 24)
        self.assertNotIn('Adaptive search adds', text)



def statistics_fixture():
    rows = []
    for model_index, model in enumerate(matrices.MODELS):
        for question in ('information', 'adaptive'):
            for source_index, (source, (_, fixed, agentic)) in enumerate(matrices.SOURCES.items()):
                for metric in matrices.METRICS:
                    effect = (model_index+1)*.01+source_index*.001
                    if metric == 'plan':
                        effect *= -1
                    rows.append(dict(model=model, question=question, source=source, metric=metric,
                                     baseline_variant='llm_only' if question == 'information' else fixed,
                                     comparison_variant=fixed if question == 'information' else agentic,
                                     mean_baseline=.4, mean_comparison=.4+effect, mean_difference=effect,
                                     n_pairs=100, n_patients=80, family_size=48,
                                     n_generators_averaged=1, test_status='tested',
                                     p_value_two_sided=.000001*(len(rows)+1)))
    frame = pd.DataFrame(rows)
    frame['p_value_two_sided_holm'] = multipletests(frame.p_value_two_sided, method='holm')[1]
    return frame


def raw_fixture(count=24):
    rows = []
    for index in range(count):
        patient = f'00000000-0000-0000-0000-{index//2:012d}'
        case = f'hypertension_{index}_{patient}'
        for model_index, model in enumerate(matrices.MODELS):
            base_d = .42+.006*(index%5)+.01*model_index
            base_p = .24+.004*(index%4)+.008*model_index
            fixed = {
                'patient_retrieval': (.025+.002*(index%3)+model_index*.002,
                                      .022+.001*(index%5)+model_index*.002),
                'guidelines': (.045+.003*(index%4)+model_index*.002,
                               .07+.003*(index%6)+model_index*.002),
                'patient_retrieval_guidelines': (.035+.004*(index%5)+model_index*.002,
                                                .06+.002*(index%3)+model_index*.002),
            }
            changes = {'llm_only': (0, 0), **fixed}
            changes['agentic_patient_retrieval'] = (fixed['patient_retrieval'][0]+.008+.001*(index%4),
                                                    fixed['patient_retrieval'][1]-.006+.001*(index%5))
            changes['agentic_guidelines'] = (fixed['guidelines'][0]-.012+.001*(index%3),
                                             fixed['guidelines'][1]-.035+.001*(index%4))
            changes['agentic_both'] = (fixed['patient_retrieval_guidelines'][0]+.007+.001*(index%6),
                                      fixed['patient_retrieval_guidelines'][1]-.021+.002*(index%3))
            for variant, (delta_d, delta_p) in changes.items():
                rows.append(dict(id=case, model=model, variant=variant,
                                 bin='history', condition_label='essential hypertension',
                                 plan_label='Lifestyle counseling',
                                 biolord_score_diagnosis1=base_d+delta_d, plan1_score=base_p+delta_p,
                                 diagnosis1_status='scored', plan1_status='scored', plan1_reason='',
                                 plan1_protocol_version=2, plan1_rank=1))
    return pd.DataFrame(rows)


class AdaptiveRenderingTests(unittest.TestCase):
    def test_full_document_has_only_own_24_cells_and_note_below_chart(self):
        frame = statistics_fixture().sample(frac=1, random_state=42)
        document = matrices.render_adaptive_matrix(frame)
        self.assertIn(r'\documentclass[border=2pt]{standalone}', document)
        self.assertIn(r'\useasboundingbox (0,0) rectangle (16,9);', document)
        self.assertEqual(document.count(matrices.NOTE), 1)
        self.assertGreater(document.index(matrices.NOTE), document.index(r'\end{tikzpicture}'))
        cells = [dict(item.split('=', 1) for item in line.removeprefix('% cell ').split())
                 for line in document.splitlines() if line.startswith('% cell ')]
        self.assertEqual(len(cells), 24)
        for model in matrices.MODELS.values():
            self.assertEqual(document.count('{' + model + '}'), 1)
        self.assertEqual(document.count(r'\textsuperscript{*}'), 24)
        self.assertNotIn('**', document)
        self.assertNotIn('confidence', document.lower())
        self.assertNotIn('Does adaptive searching add value?', document)
        self.assertNotIn('Which information helps?', document)
        for cell in cells:
            source = frame.loc[frame.model.eq(cell['model']) & frame.question.eq('adaptive')
                               & frame.metric.eq(cell['metric']) & frame.source.eq(cell['source'])].iloc[0]
            for field in ('baseline_variant', 'comparison_variant', 'n_pairs', 'n_patients',
                          'mean_baseline', 'mean_comparison', 'mean_difference',
                          'p_value_two_sided', 'p_value_two_sided_holm'):
                actual = cell[field] if isinstance(source[field], str) else float(cell[field])
                self.assertEqual(actual, source[field])

    def test_model_source_outcome_mapping_and_unrounded_values_are_retained(self):
        frame = statistics_fixture()
        select = frame.model.eq('ollama_gemma3_4b') & frame.question.eq('adaptive') & frame.metric.eq('diagnosis') & frame.source.eq('both')
        frame.loc[select, ['mean_difference', 'mean_comparison']] = [-.140229, .259771]
        result = matrices.render_adaptive_matrix(frame.sample(frac=1, random_state=8))
        self.assertIn(r'at (8.340,4.200) {-14.0\textsuperscript{*}}', result)
        self.assertIn('mean_difference=-0.140229', result)
        self.assertIn('matrixnegative!36.05', result)

    def test_full_precision_p_threshold_and_nonsignificant_values_remain_visible(self):
        frame = statistics_fixture()
        frame['p_value_two_sided'] = 1e-10
        indices = frame.loc[frame.question.eq('adaptive')].index[:2]
        frame.loc[indices, 'p_value_two_sided'] = [.005, .02]
        frame['p_value_two_sided_holm'] = multipletests(frame.p_value_two_sided, method='holm')[1]
        self.assertEqual(frame.loc[indices[0], 'p_value_two_sided_holm'], .01)
        document = matrices.render_adaptive_matrix(frame)
        self.assertEqual(document.count(r'\textsuperscript{*}'), 22)
        self.assertIn('{+1.0}', document)
        self.assertIn('{-1.0}', document)
        frame.loc[indices[0], 'p_value_two_sided'] = .0049999999
        frame['p_value_two_sided_holm'] = multipletests(frame.p_value_two_sided, method='holm')[1]
        self.assertEqual(matrices.render_adaptive_matrix(frame).count(r'\textsuperscript{*}'), 23)
        frame.loc[indices[0], ['mean_difference', 'mean_comparison']] = [-.000001, .399999]
        self.assertIn(r'{+0.0\textsuperscript{*}}', matrices.render_adaptive_matrix(frame))

    def test_invalid_schema_counts_variants_scores_or_holm_are_rejected(self):
        frame = statistics_fixture()
        bad = [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), frame.drop(columns='n_pairs')]
        for column, value in [('family_size', 12), ('n_generators_averaged', 4),
                              ('test_status', 'no_pairs'), ('mean_comparison', .9),
                              ('mean_difference', np.inf), ('n_pairs', 2), ('n_pairs', 100.5),
                              ('n_patients', 1), ('baseline_variant', 'llm_only'),
                              ('p_value_two_sided_holm', .5), ('p_value_two_sided', np.nan)]:
            changed = frame.copy()
            # Select an adaptive row so llm_only is an incorrect baseline.
            index = changed.index[changed.question.eq('adaptive')][0]
            changed[column] = changed[column].astype(object)
            changed.loc[index, column] = value
            bad.append(changed)
        for candidate in bad:
            with self.subTest(rows=len(candidate)):
                with self.assertRaises(ValueError):
                    matrices.render_adaptive_matrix(candidate)

    def test_generator_imports_no_other_figure_implementation(self):
        source = Path(matrices.__file__).read_text()
        tree = ast.parse(source)
        imported = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        for forbidden in ('render_information_matrix', 'poster_matrix_layout', 'poster_bar_layout',
                          'render_poster_options', 'render_poster_figures'):
            self.assertFalse(any(forbidden in module for module in imported))


class AdaptiveGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory/'experiment_results.csv'
        raw_fixture().to_csv(self.source, index=False)
        self.destination = self.directory/'analysis'/'figures'

    def test_raw_csv_cli_runs_alone_and_preserves_unrelated_outputs(self):
        self.destination.mkdir(parents=True)
        unrelated = self.destination/'poster_information_matrix.tex'
        unrelated.write_text('independent information matrix')
        original = self.source.read_bytes()
        outputs = matrices.main(['--results-csv', str(self.source), '--skip-compile', '--matrix', 'adaptive'])['adaptive']
        self.assertEqual(outputs['tex'].name, 'poster_adaptive_matrix.tex')
        self.assertEqual(set(outputs), {'statistics', 'tex'})
        self.assertEqual(len(pd.read_csv(outputs['statistics'])), 48)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(unrelated.read_text(), 'independent information matrix')
        self.assertEqual({path.name for path in self.destination.iterdir()},
                         {'poster_information_matrix.tex', 'poster_adaptive_matrix.tex'})
        document = outputs['tex'].read_text()
        self.assertIn(str(self.source.resolve()), document)
        self.assertIn(hashlib.sha256(original).hexdigest(), document)
        self.assertEqual(document.count(matrices.NOTE), 1)

    def test_compile_failure_preserves_old_pair_and_tex_only_removes_stale_pdf(self):
        self.destination.mkdir(parents=True)
        tex = self.destination/f'{matrices.OUTPUT_STEMS["adaptive"]}.tex'
        pdf = tex.with_suffix('.pdf')
        tex.write_text('old valid TeX')
        pdf.write_bytes(b'old valid PDF')
        with patch.object(matrices, '_compile', side_effect=RuntimeError('compiler failed')):
            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                matrices.generate_adaptive_matrix(self.source)
        self.assertEqual(tex.read_text(), 'old valid TeX')
        self.assertEqual(pdf.read_bytes(), b'old valid PDF')
        matrices.generate_adaptive_matrix(self.source, compile_pdf=False)
        self.assertFalse(pdf.exists())
        self.assertEqual({p.name for p in self.destination.iterdir()}, {'poster_adaptive_matrix.tex'})

    def test_changed_source_during_compilation_cannot_publish_new_pair(self):
        def compiler(tex):
            self.source.write_text(self.source.read_text()+'\n')
            pdf = tex.with_suffix('.pdf')
            pdf.write_bytes(b'new PDF')
            return pdf

        with patch.object(matrices, '_compile', side_effect=compiler):
            with self.assertRaisesRegex(RuntimeError, 'source CSV changed'):
                matrices.generate_adaptive_matrix(self.source)
        self.assertFalse((self.destination/f'{matrices.OUTPUT_STEMS["adaptive"]}.tex').exists())
        self.assertFalse((self.destination/f'{matrices.OUTPUT_STEMS["adaptive"]}.pdf').exists())

    @unittest.skipUnless(shutil.which('pdflatex') and shutil.which('pdftotext'), 'LaTeX/Poppler unavailable')
    def test_real_compilation_contains_own_note_once_and_no_other_figure(self):
        output = self.directory/'custom figures'
        outputs = matrices.generate_adaptive_matrix(self.source, output)
        self.assertTrue(outputs['pdf'].read_bytes().startswith(b'%PDF-'))
        text = subprocess.run(['pdftotext', str(outputs['pdf']), '-'], check=True,
                              capture_output=True, text=True).stdout
        self.assertEqual(text.count('\f'), 1)
        self.assertEqual(text.count('Holm-adjusted'), 1)
        self.assertGreater(text.index('Holm-adjusted'), text.index('Ministral'))
        self.assertIn('48 comparisons', text)
        self.assertIn('Adaptive search adds few gains and several losses.', text)
        self.assertNotIn('Evidence choice depends', text)
        self.assertEqual({path.suffix for path in output.iterdir()}, {'.tex', '.pdf'})



class CombinedMatrixGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory/'experiment_results.csv'
        raw_fixture().to_csv(self.source, index=False)
        self.destination = self.directory/'figures'

    def test_default_cli_generates_both_and_computes_statistics_once(self):
        with patch.object(matrices, 'generate_model_poster_statistics',
                          wraps=matrices.generate_model_poster_statistics) as calculate:
            outputs = matrices.main(['--results-csv', str(self.source), '--output-dir',
                                     str(self.destination), '--skip-compile'])
        self.assertEqual(calculate.call_count, 1)
        self.assertEqual(set(outputs), {'information', 'adaptive'})
        self.assertEqual({p.name for p in self.destination.iterdir()},
                         {'poster_information_matrix.tex', 'poster_adaptive_matrix.tex'})
        for output in outputs.values():
            self.assertEqual(output['tex'].read_text().count(matrices.NOTE), 1)

    def test_single_matrix_functions_do_not_call_the_other_rendering_function(self):
        with patch.object(matrices, 'render_adaptive_matrix', side_effect=AssertionError('wrong figure')):
            matrices.generate_information_matrix(self.source, self.destination, compile_pdf=False)
        with patch.object(matrices, 'render_information_matrix', side_effect=AssertionError('wrong figure')):
            matrices.generate_adaptive_matrix(self.source, self.destination, compile_pdf=False)

    def test_second_compilation_failure_preserves_both_previous_pairs(self):
        self.destination.mkdir()
        before = {}
        for stem in matrices.OUTPUT_STEMS.values():
            for suffix in ('.tex', '.pdf'):
                path = self.destination/(stem+suffix)
                path.write_bytes(b'previous '+path.name.encode())
                before[path] = path.read_bytes()

        def compile_document(tex):
            if tex.stem == matrices.OUTPUT_STEMS['adaptive']:
                raise RuntimeError('second compilation failed')
            pdf = tex.with_suffix('.pdf')
            pdf.write_bytes(b'new PDF')
            return pdf

        with patch.object(matrices, '_compile', side_effect=compile_document):
            with self.assertRaisesRegex(RuntimeError, 'second compilation failed'):
                matrices.generate_poster_matrices(self.source, self.destination)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(len(list(self.destination.iterdir())), 4)

    @unittest.skipUnless(shutil.which('pdflatex') and shutil.which('pdftotext'), 'LaTeX/Poppler unavailable')
    def test_both_public_generator_compiles_two_distinct_one_page_pdfs(self):
        outputs = matrices.generate_poster_matrices(self.source, self.destination)
        for question, output in outputs.items():
            text = subprocess.run(['pdftotext', str(output['pdf']), '-'], check=True,
                                  capture_output=True, text=True).stdout
            self.assertEqual(text.count('\f'), 1)
            self.assertEqual(text.count('Holm-adjusted'), 1)
            self.assertIn('Evidence choice' if question == 'information' else 'Adaptive search adds', text)
        self.assertEqual({path.suffix for path in self.destination.iterdir()}, {'.tex', '.pdf'})


if __name__ == '__main__':
    unittest.main()
