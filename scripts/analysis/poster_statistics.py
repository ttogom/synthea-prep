"""Paired model-specific statistics for the two selected poster matrices.

All 48 within-model comparisons form one two-sided Holm family. Each model
keeps its own observed paired cohort, and inference clusters repeated condition
examples by patient. No model averaging is performed.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# Direct script execution must resolve repository modules before importing them.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.summarize_experiment_results import (
    MODEL_LABELS,
    RESULTS,
    load_scored_results,
)


MODEL_OUTPUT_NAME = 'poster_model_comparisons.csv'
SOURCES = (
    ('patient_retrieval', 'patient_retrieval', 'agentic_patient_retrieval'),
    ('guidelines', 'guidelines', 'agentic_guidelines'),
    ('both', 'patient_retrieval_guidelines', 'agentic_both'),
)
VARIANTS = ('llm_only',) + tuple(variant for _, fixed, adaptive in SOURCES
                               for variant in (fixed, adaptive))
MODEL_FAMILY_SIZE = 48


def _validate_results(results):
    required = {'id', 'patient_uuid', 'model', 'variant', 'diagnosis', 'plan', 'plan_eligible'}
    if not isinstance(results, pd.DataFrame) or required - set(results):
        raise ValueError('Pass the normalized frame returned by load_scored_results')
    frame = results.copy()
    if frame.empty:
        raise ValueError('Require a nonempty experiment')
    for column in ('id', 'patient_uuid', 'model', 'variant'):
        if frame[column].isna().any() or not frame[column].map(
                lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'{column} must contain nonempty strings')
    if frame.duplicated(['id', 'model', 'variant']).any():
        raise ValueError('Require unique condition-example/model/variant slots')
    if set(frame.model) != set(MODEL_LABELS) or set(frame.variant) != set(VARIANTS):
        raise ValueError('Require the fixed four generators and all seven variants')
    if not frame.groupby('id').size().eq(len(MODEL_LABELS) * len(VARIANTS)).all():
        raise ValueError('Require every condition-example/model/variant slot')
    if frame.plan_eligible.isna().any() or not frame.plan_eligible.map(
            lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError('plan_eligible must contain booleans')
    if frame.groupby('id')[['patient_uuid', 'plan_eligible']].nunique().gt(1).any().any():
        raise ValueError('Each condition example must have consistent patient and eligibility metadata')
    for metric in ('diagnosis', 'plan'):
        try:
            frame[metric] = pd.to_numeric(frame[metric], errors='raise')
        except (TypeError, ValueError) as error:
            raise ValueError(f'{metric} scores must be numeric') from error
        observed = frame[metric].dropna()
        if not np.isfinite(observed).all() or not observed.between(0, 1).all():
            raise ValueError(f'{metric} scores must be finite within [0, 1] or missing')
    # Eligibility is a reference property; unavailable references never enter plan comparisons.
    frame.loc[~frame.plan_eligible, 'plan'] = np.nan
    return frame


def _holm_adjust(values):
    probabilities = np.asarray(values, dtype=float)
    order = np.argsort(np.where(np.isnan(probabilities), 1.0, probabilities), kind='stable')
    sorted_values = np.where(np.isnan(probabilities[order]), 1.0, probabilities[order])
    adjusted = np.empty(len(probabilities))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(
        sorted_values * np.arange(len(probabilities), 0, -1)))
    adjusted[np.isnan(probabilities)] = np.nan
    return adjusted


def _paired_inference(differences, patients):
    codes, unique = pd.factorize(patients, sort=True)
    clusters = len(unique)
    inference = dict(standard_error=np.nan, ci_low=np.nan, ci_high=np.nan,
                     test_statistic=np.nan, test_df=clusters - 1 if clusters else np.nan,
                     p_value_two_sided=np.nan, test_status='no_pairs')
    if not len(differences):
        return inference
    if clusters < 2:
        inference['test_status'] = 'insufficient_patients'
        return inference
    mean = float(differences.mean())
    cluster_sums = np.bincount(codes, weights=differences - mean)
    variance = clusters / (clusters - 1) * np.dot(cluster_sums, cluster_sums) / len(differences) ** 2
    if variance <= np.finfo(float).eps ** 2:
        inference['test_status'] = 'zero_variance'
        return inference
    standard_error = float(np.sqrt(variance))
    statistic = mean / standard_error
    radius = student_t.ppf(0.975, clusters - 1) * standard_error
    inference.update(standard_error=standard_error, ci_low=mean - radius, ci_high=mean + radius,
                     test_statistic=statistic, p_value_two_sided=float(
                         2 * student_t.sf(abs(statistic), clusters - 1)), test_status='tested')
    return inference


def summarize_model_poster_comparisons(results):
    """Return all 48 within-model contrasts as one two-sided Holm family.

    Each language model keeps its own observed paired cohort. A missing score
    excludes that model's example from that comparison, not another model's
    example. Repeated condition examples are clustered by patient UUID. The
    pointwise confidence intervals remain available for auditing even when the
    poster deliberately omits them; significance is never inferred from the
    displayed rounded effect.
    """
    frame = _validate_results(results)
    metadata = frame.drop_duplicates('id').set_index('id').sort_index()
    rows = []
    for model in MODEL_LABELS:
        model_frame = frame.loc[frame.model.eq(model)]
        for metric in ('diagnosis', 'plan'):
            eligible = metadata if metric == 'diagnosis' else metadata.loc[metadata.plan_eligible]
            wide = model_frame.pivot(index='id', columns='variant', values=metric).reindex(
                index=eligible.index, columns=VARIANTS)
            for question in ('information', 'adaptive'):
                for source, fixed, adaptive in SOURCES:
                    baseline, comparison = ('llm_only', fixed) if question == 'information' else (fixed, adaptive)
                    paired = wide[[baseline, comparison]].dropna()
                    delta = (paired[comparison] - paired[baseline]).to_numpy(dtype=float)
                    patients = eligible.loc[paired.index, 'patient_uuid']
                    n_eligible = len(eligible)
                    inference = _paired_inference(delta, patients)
                    if not n_eligible:
                        inference['test_status'] = 'no_eligible'
                    rows.append(dict(
                        model=model, question=question, source=source, metric=metric,
                        baseline_variant=baseline, comparison_variant=comparison,
                        n_eligible=n_eligible, n_baseline=int(wide[baseline].notna().sum()),
                        n_comparison=int(wide[comparison].notna().sum()), n_pairs=len(paired),
                        n_patients=int(patients.nunique()), coverage_complete=bool(n_eligible and len(paired) == n_eligible),
                        mean_baseline=float(paired[baseline].mean()),
                        mean_comparison=float(paired[comparison].mean()),
                        mean_difference=float(delta.mean()) if len(delta) else np.nan,
                        n_generators_averaged=1, family_size=MODEL_FAMILY_SIZE,
                        **inference,
                    ))
    output = pd.DataFrame(rows)
    output['p_value_two_sided_holm'] = _holm_adjust(output.p_value_two_sided)
    output['stars'] = output.p_value_two_sided_holm.map(
        lambda value: '**' if value < 0.01 else '*' if value < 0.05 else '')
    for alpha, suffix in ((0.01, '0_01'), (0.05, '0_05')):
        output[f'significant_two_sided_holm_{suffix}'] = output.p_value_two_sided_holm.lt(alpha)
    return output


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_model_poster_statistics(results_path=ROOT / RESULTS, output_path=None):
    """Validate the raw export and atomically save the 48-row matrix statistics.

    Source fingerprints and effective statistical settings accompany the paired
    estimates. Raw results, judgments, and existing analysis summaries are never
    regenerated by this entry point.
    """
    results_path = Path(results_path).expanduser().resolve()
    output_path = (Path(output_path).expanduser().resolve() if output_path is not None
                   else results_path.parent / 'analysis' / MODEL_OUTPUT_NAME)
    if output_path == results_path:
        raise ValueError('The statistics output must differ from its raw input')
    source_hash = _sha256(results_path)
    results = load_scored_results(results_path)
    required = {'plan1_protocol_version', 'plan1_rank', 'diagnosis1_status'}
    if (required - set(results) or not results.plan1_protocol_version.eq(2).all()
            or not results.plan1_rank.eq(1).all()):
        raise ValueError('Require protocol-2 rank-1 judgments and diagnosis statuses')
    summary = summarize_model_poster_comparisons(results)
    if source_hash != _sha256(results_path):
        raise RuntimeError('The source CSV changed during poster analysis')
    summary['source_csv'] = str(results_path)
    summary['source_csv_sha256'] = source_hash
    summary['confidence_level'] = 0.95
    summary['interval_policy'] = 'Pointwise Student-t CR1 patient-clustered'
    summary['multiplicity_policy'] = f'Holm across all {MODEL_FAMILY_SIZE} two-sided poster comparisons'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', dir=output_path.parent, suffix='.csv', delete=False) as stream:
            temporary = Path(stream.name)
            summary.to_csv(stream, index=False, na_rep='')
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-csv', type=Path, default=ROOT / RESULTS)
    parser.add_argument('--output-csv', type=Path)
    args = parser.parse_args(argv)
    summary = generate_model_poster_statistics(args.results_csv, args.output_csv)
    destination = args.output_csv or args.results_csv.parent / 'analysis' / MODEL_OUTPUT_NAME
    print(f'Saved {len(summary)} paired poster comparisons to {destination}')
    return summary


if __name__ == '__main__':
    main()
