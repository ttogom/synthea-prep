"""Render both poster matrices with separate functions in one shared script.

Each matrix keeps its own drawing function and standalone TeX/p-value note.
The common pipeline computes the 48-test family once and stages every selected
figure before publishing. Either matrix can also be generated on its own.
"""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

# Direct execution must resolve repository modules before importing them.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.poster_statistics import (
    MODEL_FAMILY_SIZE,
    MODEL_OUTPUT_NAME,
    generate_model_poster_statistics,
)
from scripts.analysis.summarize_experiment_results import RESULTS


MODELS = {
    'gpt_4o_mini': 'GPT-4o mini',
    'ollama_gemma3_4b': 'Gemma 3 4B',
    'ollama_llama3_2_3b': 'Llama 3.2 3B',
    'ollama_ministral_3_3b': 'Ministral 3 3B',
}
SOURCES = {
    'patient_retrieval': ('Patients', 'patient_retrieval', 'agentic_patient_retrieval'),
    'guidelines': ('Guidelines', 'guidelines', 'agentic_guidelines'),
    'both': ('Both', 'patient_retrieval_guidelines', 'agentic_both'),
}
METRICS = {'diagnosis': 'Diagnosis', 'plan': 'Plan coverage'}
SIGNIFICANCE_LEVEL = 0.01
OUTPUT_STEMS = {'information': 'poster_information_matrix', 'adaptive': 'poster_adaptive_matrix'}
NOTE = r'* Holm-adjusted $p<.01$ (48 comparisons).'


def _validate_statistics(statistics):
    required = {
        'model', 'question', 'source', 'metric', 'baseline_variant', 'comparison_variant',
        'mean_baseline', 'mean_comparison', 'mean_difference', 'n_pairs', 'n_patients',
        'p_value_two_sided', 'p_value_two_sided_holm', 'family_size',
        'n_generators_averaged', 'test_status',
    }
    if not isinstance(statistics, pd.DataFrame) or required - set(statistics):
        raise ValueError('Require the complete model-specific poster statistics schema')
    frame = statistics.copy()
    keys = ['model', 'question', 'source', 'metric']
    expected = {(model, question, source, metric) for model in MODELS
                for question in ('information', 'adaptive') for source in SOURCES for metric in METRICS}
    if (len(frame) != MODEL_FAMILY_SIZE or frame.duplicated(keys).any()
            or set(frame[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError('Require all 48 distinct within-model comparisons')
    if (not frame.family_size.eq(MODEL_FAMILY_SIZE).all()
            or not frame.n_generators_averaged.eq(1).all()
            or not frame.test_status.eq('tested').all()):
        raise ValueError('Require testable within-model comparisons in the joint 48-test family')
    numeric = ['mean_baseline', 'mean_comparison', 'mean_difference', 'n_pairs', 'n_patients',
               'p_value_two_sided', 'p_value_two_sided_holm']
    try:
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError) as error:
        raise ValueError('Plotted statistics must be numeric') from error
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Plotted statistics must be finite')
    if (not frame[['mean_baseline', 'mean_comparison']].apply(lambda x: x.between(0, 1)).all().all()
            or not frame[['p_value_two_sided', 'p_value_two_sided_holm']].apply(
                lambda x: x.between(0, 1)).all().all()
            or not frame.n_pairs.ge(frame.n_patients).all() or not frame.n_patients.ge(2).all()
            or not (frame[['n_pairs', 'n_patients']] % 1).eq(0).all().all()):
        raise ValueError('Invalid paired scores, p-values or sample counts')
    if not np.allclose(frame.mean_comparison-frame.mean_baseline, frame.mean_difference,
                       rtol=1e-10, atol=1e-12):
        raise ValueError('Paired arm means disagree with their differences')
    probabilities = frame.p_value_two_sided.to_numpy(dtype=float)
    order = np.argsort(probabilities, kind='stable')
    adjusted = np.empty(len(probabilities))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        probabilities[order] * np.arange(len(probabilities), 0, -1)))
    if not np.allclose(adjusted, frame.p_value_two_sided_holm, rtol=1e-10, atol=1e-150):
        raise ValueError('Adjusted p-values disagree with Holm correction across all 48 comparisons')
    for row in frame.itertuples(index=False):
        _, fixed, adaptive = SOURCES[row.source]
        expected_variants = ('llm_only', fixed) if row.question == 'information' else (fixed, adaptive)
        if (row.baseline_variant, row.comparison_variant) != expected_variants:
            raise ValueError('Saved comparison variants do not match their labels')
    for column in ('source_csv', 'source_csv_sha256'):
        if column in frame and (frame[column].isna().any() or frame[column].nunique() != 1):
            raise ValueError('Require one consistent raw source provenance')
    if 'source_csv_sha256' in frame and not re.fullmatch('[0-9a-f]{64}', frame.source_csv_sha256.iloc[0]):
        raise ValueError('Invalid raw source SHA-256')
    return frame


def _appearance(effect, adjusted_p):
    points = 100 * effect
    value = 0.0 if abs(points) < 0.05 else points
    label = f'{value:+.1f}'
    if adjusted_p >= SIGNIFICANCE_LEVEL:
        return 'matrixink!4', 'matrixmuted', label
    color = 'matrixpositive' if points >= 0 else 'matrixnegative'
    intensity = min(48, 8 + 2*abs(points))
    return f'{color}!{intensity:.2f}', 'matrixink', label + r'\textsuperscript{*}'


def render_information_matrix(statistics):
    """Return the complete standalone TeX, including this figure's own p-value note."""
    frame = _validate_statistics(statistics)
    rows = frame.loc[frame.question.eq('information')].set_index(['model', 'metric', 'source'])
    lines = [
        '% Generated by scripts/analysis/render_poster_matrices.py (information).',
        r'\documentclass[border=2pt]{standalone}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[scaled=0.95]{helvet}',
        r'\renewcommand{\familydefault}{\sfdefault}',
        r'\usepackage{tikz}',
        r'\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi',
        r'\ifdefined\pdftrailerid\pdftrailerid{}\fi',
        r'\begin{document}',
        r'\begin{minipage}{16cm}',
        r'\begin{tikzpicture}[x=1cm,y=1cm]',
        r'\definecolor{matrixink}{HTML}{172B40}',
        r'\definecolor{matrixmuted}{HTML}{6B7280}',
        r'\definecolor{matrixpositive}{HTML}{3E79B8}',
        r'\definecolor{matrixnegative}{HTML}{D98042}',
        r'\useasboundingbox (0,0) rectangle (16,9);',
        r'\fill[white] (0,0) rectangle (16,9);',
        r'\node[anchor=west,text=matrixink,font=\fontsize{15}{18}\selectfont\bfseries] at (0.2,8.25) {Evidence choice depends on the model.};',
        r'\node[anchor=west,text=matrixmuted,font=\fontsize{10.5}{12}\selectfont] at (0.2,6.36) {Language model};',
    ]
    for column in ('source_csv', 'source_csv_sha256'):
        if column in frame:
            if frame[column].nunique(dropna=False) != 1 or frame[column].isna().any():
                raise ValueError('Require one consistent raw-source provenance value')
            lines.insert(1, f'% {column}={frame[column].iloc[0]}')
    starts = {'diagnosis': 3.35, 'plan': 9.52}
    width, height, step = 1.94, 1.12, 2.01
    for metric, start in starts.items():
        center = start + (2*step+width)/2
        lines.append(rf'\node[text=matrixink,font=\fontsize{{12}}{{14}}\selectfont\bfseries] at ({center:.3f},7.0) {{{METRICS[metric]}}};')
        for index, (label, _, _) in enumerate(SOURCES.values()):
            x = start + index*step + width/2
            lines.append(rf'\node[text=matrixink,font=\fontsize{{10.5}}{{12}}\selectfont] at ({x:.3f},6.36) {{{label}}};')
    for index, (model, label) in enumerate(MODELS.items()):
        y = 5.45-index*1.25
        lines.append(rf'\node[anchor=west,text=matrixink,font=\fontsize{{11.5}}{{14}}\selectfont] at (0.2,{y:.3f}) {{{label}}};')
        for metric, start in starts.items():
            for column, source in enumerate(SOURCES):
                row = rows.loc[(model, metric, source)]
                points = 100*row.mean_difference
                value = 0.0 if abs(points) < .05 else points
                significant = row.p_value_two_sided_holm < .01
                text = f'{value:+.1f}' + (r'\textsuperscript{*}' if significant else '')
                color = 'matrixpositive' if points >= 0 else 'matrixnegative'
                fill = f'{color}!{min(48, 8+2*abs(points)):.2f}' if significant else 'matrixink!4'
                ink = 'matrixink' if significant else 'matrixmuted'
                x = start+column*step
                trace = ' '.join(f'{name}={row[name]}' for name in (
                    'baseline_variant', 'comparison_variant', 'n_pairs', 'n_patients',
                    'mean_baseline', 'mean_comparison', 'mean_difference',
                    'p_value_two_sided', 'p_value_two_sided_holm'))
                lines.extend([
                    f'% cell model={model} metric={metric} source={source} {trace}',
                    rf'\fill[{fill},rounded corners=1.2pt] ({x:.3f},{y-height/2:.3f}) rectangle ({x+width:.3f},{y+height/2:.3f});',
                    rf'\node[text={ink},font=\fontsize{{13}}{{15}}\selectfont\bfseries] at ({x+width/2:.3f},{y:.3f}) {{{text}}};',
                ])
    lines.extend([
        r'\node[text=matrixmuted,font=\fontsize{10.5}{12}\selectfont] at (8,0.55) {Change vs notes alone (score points)};',
        r'\end{tikzpicture}',
        r'\par\centering\vspace{1mm}',
        r'{\fontsize{8}{10}\selectfont\color{black!65}' + NOTE + r'\par}',
        r'\end{minipage}',
        r'\end{document}',
    ])
    return '\n'.join(lines)+'\n'


def render_adaptive_matrix(statistics):
    """Return the complete standalone adaptive matrix TeX, including its note.

    Every displayed cell has a machine-readable trace comment retaining both
    variants, matched counts, unrounded means/effect, raw and adjusted p-values.
    Significance uses the full-precision adjusted probability, never rounding.
    """
    frame = _validate_statistics(statistics)
    rows = frame.loc[frame.question.eq('adaptive')].set_index(['model', 'metric', 'source'])
    lines = [
        '% Generated by scripts/analysis/render_poster_matrices.py (adaptive).',
    ]
    for column in ('source_csv', 'source_csv_sha256'):
        if column in frame:
            value = str(frame[column].iloc[0]).replace('\r', ' ').replace('\n', ' ')
            lines.append(f'% {column}={value}')
    lines.extend([
        r'\documentclass[border=2pt]{standalone}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage[scaled=0.95]{helvet}',
        r'\renewcommand{\familydefault}{\sfdefault}',
        r'\usepackage{tikz}',
        r'\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi',
        r'\ifdefined\pdftrailerid\pdftrailerid{}\fi',
        r'\setlength{\parindent}{0pt}',
        r'\begin{document}',
        r'\begin{minipage}{16cm}',
        r'\begin{tikzpicture}[x=1cm,y=1cm]',
        r'\definecolor{matrixink}{HTML}{172B40}',
        r'\definecolor{matrixmuted}{HTML}{6B7280}',
        r'\definecolor{matrixpositive}{HTML}{3E79B8}',
        r'\definecolor{matrixnegative}{HTML}{D98042}',
        r'\useasboundingbox (0,0) rectangle (16,9);',
        r'\fill[white] (0,0) rectangle (16,9);',
        r'\node[anchor=west,text=matrixink,font=\fontsize{15}{18}\selectfont\bfseries] at (0.2,8.25) {Adaptive search adds few gains and several losses.};',
        r'\node[anchor=west,text=matrixmuted,font=\fontsize{10.5}{12}\selectfont] at (0.2,6.36) {Language model};',
    ])
    starts = {'diagnosis': 3.35, 'plan': 9.52}
    cell_width, cell_height, step = 1.94, 1.12, 2.01
    for metric, start in starts.items():
        center = start + (2*step + cell_width)/2
        lines.append(rf'\node[text=matrixink,font=\fontsize{{12}}{{14}}\selectfont\bfseries] at ({center:.3f},7.0) {{{METRICS[metric]}}};')
        for column, (source, (label, _, _)) in enumerate(SOURCES.items()):
            x = start + column*step + cell_width/2
            lines.append(rf'\node[text=matrixink,font=\fontsize{{10.5}}{{12}}\selectfont] at ({x:.3f},6.36) {{{label}}};')
    for index, (model, model_label) in enumerate(MODELS.items()):
        y = 5.45-index*1.25
        lines.append(rf'\node[anchor=west,text=matrixink,font=\fontsize{{11.5}}{{14}}\selectfont] at (0.2,{y:.3f}) {{{model_label}}};')
        for metric, start in starts.items():
            for column, source in enumerate(SOURCES):
                row = rows.loc[(model, metric, source)]
                fill, text_color, label = _appearance(row.mean_difference, row.p_value_two_sided_holm)
                trace = dict(model=model, metric=metric, source=source,
                             baseline_variant=row.baseline_variant, comparison_variant=row.comparison_variant,
                             n_pairs=int(row.n_pairs), n_patients=int(row.n_patients),
                             mean_baseline=float(row.mean_baseline), mean_comparison=float(row.mean_comparison),
                             mean_difference=float(row.mean_difference),
                             p_value_two_sided=float(row.p_value_two_sided),
                             p_value_two_sided_holm=float(row.p_value_two_sided_holm),
                             display_value=f'{0.0 if abs(100*row.mean_difference) < .05 else 100*row.mean_difference:+.1f}',
                             display_significant=bool(row.p_value_two_sided_holm < SIGNIFICANCE_LEVEL))
                x = start+column*step
                lines.extend([
                    '% cell ' + ' '.join(f'{key}={value}' for key, value in trace.items()),
                    rf'\fill[{fill},rounded corners=1.2pt] ({x:.3f},{y-cell_height/2:.3f}) rectangle ({x+cell_width:.3f},{y+cell_height/2:.3f});',
                    rf'\node[text={text_color},font=\fontsize{{13}}{{15}}\selectfont\bfseries] at ({x+cell_width/2:.3f},{y:.3f}) {{{label}}};',
                ])
    lines.extend([
        r'\node[text=matrixmuted,font=\fontsize{10.5}{12}\selectfont] at (8,0.55) {Adaptive minus fixed search (score points)};',
        r'\end{tikzpicture}',
        r'\par\centering\vspace{1mm}',
        r'{\fontsize{8}{10}\selectfont\color{black!65}' + NOTE + r'\par}',
        r'\end{minipage}',
        r'\end{document}',
    ])
    return '\n'.join(lines)+'\n'


def _compile(tex_path):
    executable = shutil.which('pdflatex')
    if executable is None:
        raise RuntimeError('pdflatex is required to compile the poster matrix')
    with TemporaryDirectory(prefix='poster-matrix-tex-') as directory:
        completed = subprocess.run(
            [executable, '-no-shell-escape', '-halt-on-error', '-interaction=nonstopmode',
             f'-output-directory={directory}', str(tex_path)],
            cwd=tex_path.parent, capture_output=True, text=True, timeout=120, check=False,
        )
        pdf = Path(directory)/tex_path.with_suffix('.pdf').name
        if completed.returncode or not pdf.is_file():
            raise RuntimeError(f'Poster matrix LaTeX compilation failed:\n{completed.stdout[-5000:]}\n{completed.stderr[-1000:]}')
        if 'Overfull \\hbox' in completed.stdout or 'Overfull \\vbox' in completed.stdout:
            raise RuntimeError('Poster matrix LaTeX reported overflowing content')
        destination = tex_path.with_suffix('.pdf')
        shutil.copyfile(pdf, destination)
    return destination


def _generate_matrices(results_path, output_dir, questions, compile_pdf):
    source = Path(results_path).expanduser().resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    statistics_path = source.parent/'analysis'/MODEL_OUTPUT_NAME
    generate_model_poster_statistics(source, statistics_path)
    frame = pd.read_csv(statistics_path, keep_default_na=False, na_values=[''])
    if (not frame.source_csv_sha256.eq(source_hash).all()
            or hashlib.sha256(source.read_bytes()).hexdigest() != source_hash):
        raise RuntimeError('The source CSV changed during poster-matrix generation')
    renderers = {'information': render_information_matrix, 'adaptive': render_adaptive_matrix}
    documents = {question: renderers[question](frame) for question in questions}
    destination = (Path(output_dir).expanduser().resolve() if output_dir is not None
                   else statistics_path.parent/'figures')
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with TemporaryDirectory(prefix='.poster-matrices-', dir=destination) as directory:
        prepared = {}
        for question, document in documents.items():
            tex = Path(directory)/(OUTPUT_STEMS[question]+'.tex')
            tex.write_text(document, encoding='utf-8')
            prepared[question] = {'tex': tex}
            if compile_pdf:
                prepared[question]['pdf'] = _compile(tex)
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError('The source CSV changed before publishing the poster matrices')
        for question, artifacts in prepared.items():
            outputs[question] = {'statistics': statistics_path}
            for kind, path in artifacts.items():
                target = destination/path.name
                os.replace(path, target)
                outputs[question][kind] = target
            if not compile_pdf:
                outputs[question]['tex'].with_suffix('.pdf').unlink(missing_ok=True)
    return outputs


def generate_information_matrix(results_path=ROOT/RESULTS, output_dir=None, *, compile_pdf=True):
    """Generate only the information matrix, including its own TeX p-value note."""
    return _generate_matrices(results_path, output_dir, ('information',), compile_pdf)['information']


def generate_adaptive_matrix(results_path=ROOT/RESULTS, output_dir=None, *, compile_pdf=True):
    """Generate only the adaptive matrix, including its own TeX p-value note."""
    return _generate_matrices(results_path, output_dir, ('adaptive',), compile_pdf)['adaptive']


def generate_poster_matrices(results_path=ROOT/RESULTS, output_dir=None, *, compile_pdf=True):
    """Generate both matrices, calculating shared statistics once and staging both."""
    return _generate_matrices(results_path, output_dir, ('information', 'adaptive'), compile_pdf)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-csv', type=Path, default=ROOT/RESULTS)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--matrix', choices=('all', 'information', 'adaptive'), default='all')
    parser.add_argument('--skip-compile', action='store_true')
    args = parser.parse_args(argv)
    questions = ('information', 'adaptive') if args.matrix == 'all' else (args.matrix,)
    outputs = _generate_matrices(args.results_csv, args.output_dir, questions, not args.skip_compile)
    for artifacts in outputs.values():
        for path in artifacts.values():
            print(f'Saved {path}')
    return outputs


if __name__ == '__main__':
    main()
