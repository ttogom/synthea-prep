"""Rebuild the seven-variant CSVs from saved predictions and protocol-2 judgments."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

# Direct execution must resolve the repository before importing its judge schema.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plan_judge_v2.inputs import load_by_id, sha256_file
from scripts.plan_judge_v2.protocol import MATCH_CREDITS, protocol_identity
from scripts.plan_judge_v2.runner import CSV_COLUMNS
from scripts.analysis.table_statistics import VARIANT_LABELS


OUTPUT = ROOT / 'experiments/results_all_variants'
SOURCE_KEYS = {'labels', 'patients', 'models', 'protocol_version', 'rank', 'runs'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def build_dataset(sources, output_dir=OUTPUT, root=ROOT):
    """Validate all sources before replacing the two derived CSVs."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    _require(isinstance(sources, dict) and set(sources) == SOURCE_KEYS, 'Unknown or missing source settings')
    _require(sources['protocol_version'] == 2 and sources['rank'] == 1, 'Require protocol 2, rank 1')
    models = sources['models']
    _require(len(models) == 4 and len(models) == len(set(models)), 'Require four distinct generator models')
    selected = [variant for run in sources['runs'] for variant in run['variants']]
    _require(len(selected) == len(VARIANT_LABELS) and set(selected) == set(VARIANT_LABELS),
             'Require all seven variants, exactly once')
    labels_path, patients_path = (root / sources[key] for key in ('labels', 'patients'))
    labels = load_by_id(labels_path, ('bin', 'condition', 'plan'))
    hashes = {'labels': sha256_file(labels_path), 'patients': sha256_file(patients_path)}
    records, judgments, seen_variants = [], [], set()
    cohort, judge_identity = None, None
    for source in sources['runs']:
        _require(set(source) == {'path', 'variants'} and source['variants'], 'Invalid run selection')
        variants = source['variants']
        _require(len(set(variants)) == len(variants) and not seen_variants.intersection(variants),
                 'Each variant must have exactly one source run')
        seen_variants.update(variants)
        run_dir = root / source['path']
        _require(output_dir != run_dir and output_dir not in run_dir.parents,
                 'Output directory must not contain a source run')
        seen_ids, digest = set(), hashlib.sha256()
        with (run_dir / 'results.jsonl').open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                patient_id = record['id']
                _require(patient_id not in seen_ids and patient_id in labels,
                         'Duplicate patient ID or missing label')
                seen_ids.add(patient_id)
                label = labels[patient_id]
                for variant in variants:
                    body = record['variants'].get(variant, {})
                    slots = body.get('assessment_plans', {})
                    _require(not set(slots) - set(models), 'Unexpected generator model')
                    for model in models:
                        slot = slots.get(model, {})
                        row = {'id': patient_id, 'model': model, 'variant': variant,
                               'source_run': source['path'], 'bin': label['bin'],
                               'condition_label': label['condition'],
                               'plan_label': '; '.join(label['plan'] or []),
                               'plan_label_actions': json.dumps(label['plan'] or [], ensure_ascii=False),
                               'variant_status': body.get('status'),
                               'variant_error_type': (body.get('error') or {}).get('type'),
                               'generation_status': slot.get('status'),
                               'generation_error_type': (slot.get('error') or {}).get('type')}
                        predictions = slot.get('predictions') or []
                        evaluations = {entry['rank']: entry for entry in (slot.get('evaluation') or [])}
                        for rank in range(1, 4):
                            prediction = predictions[rank - 1] if len(predictions) >= rank else {}
                            evaluation = evaluations.get(rank, {})
                            biolord = evaluation.get('diagnosis_correctness') or {}
                            bert = evaluation.get('condition') or {}
                            # Deliberately never read evaluation.plan_correctness (protocol 1).
                            row[f'diagnosis{rank}'] = prediction.get('condition') or ''
                            row[f'biolord_score_diagnosis{rank}'] = (
                                biolord.get('score') if biolord.get('status') == 'scored' else None)
                            row[f'bertf1_diagnosis{rank}'] = (
                                bert.get('metrics', {}).get('BERTScore_F1') if bert.get('status') == 'scored' else None)
                            row[f'diagnosis{rank}_status'] = biolord.get('status') or 'not_scored'
                            row[f'diagnosis{rank}_reason'] = biolord.get('reason')
                            row[f'bertf1_diagnosis{rank}_status'] = bert.get('status') or 'not_scored'
                            row[f'bertf1_diagnosis{rank}_reason'] = bert.get('reason')
                            actions = prediction.get('plan') or []
                            row[f'plan{rank}'] = '; '.join(actions)
                            row[f'plan{rank}_actions'] = json.dumps(actions, ensure_ascii=False)
                            if rank > 1:
                                row[f'plan{rank}_status'] = 'not_judged'
                        records.append(row)
        _require(seen_ids and (cohort is None or cohort == seen_ids), 'Runs must cover identical patient IDs')
        cohort = seen_ids
        for variant in variants:
            directory = run_dir / 'plan_judging_v2' / variant
            manifest = json.loads((directory / 'manifest.json').read_text())
            _require(manifest['protocol'] == protocol_identity(), 'Judging protocol mismatch')
            _require(manifest['variant'] == variant and manifest['selection']['rank'] == 1
                     and manifest['selection']['sample'] is None
                     and manifest['selection']['patients_in_results'] == len(seen_ids)
                     and set(manifest['selection']['generator_models_expected']) == set(models),
                     'Judging selection mismatch')
            _require(manifest['inputs']['results']['sha256'] == digest.hexdigest()
                     and all(manifest['inputs'][key]['sha256'] == value for key, value in hashes.items()),
                     'Judging inputs do not match source files')
            _require(judge_identity is None or judge_identity == manifest['judge'], 'Judge settings differ')
            judge_identity = manifest['judge']
            frame = pd.read_csv(directory / 'judgments.csv', keep_default_na=False)
            _require(list(frame.columns) == CSV_COLUMNS, 'Unexpected judging CSV columns')
            _require(frame.variant.eq(variant).all() and frame.protocol_version.eq(2).all()
                     and frame['rank'].eq(1).all() and frame.run_id.eq(manifest['run_id']).all(),
                     'Judging CSV identity/protocol mismatch')
            _require(frame.judge_model.eq(judge_identity['model_id']).all()
                     and frame.judge_revision.eq(judge_identity['revision_resolved']).all(),
                     'Judging CSV model mismatch')
            keys = set(zip(frame.patient_id, frame.generator_model))
            expected = {(patient, model) for patient in seen_ids for model in models}
            _require(keys == expected and len(frame) == len(expected), 'Duplicate or missing judging keys')
            counts = manifest['counts']
            _require(len(frame) == counts['plans'] and frame.status.eq('scored').sum() == counts['scored']
                     and frame.status.eq('unavailable').sum() == counts['unavailable']
                     and frame.status.isin(['error', 'invalid_response']).sum()
                     == counts['failed_retryable'] + counts['failed_exhausted']
                     and not counts['not_attempted'] and not counts['duplicate_final_rows'],
                     'Judging CSV counts differ from manifest or contain unfinished work')
            _require(frame.status.isin(['scored', 'unavailable', 'error', 'invalid_response']).all(),
                     'Unknown judging status')
            for field in ('coverage', 'penalized', 'harmful_count'):
                numeric = pd.to_numeric(frame[field], errors='raise')
                scored = frame.status.eq('scored')
                _require(numeric[~scored].isna().all() and np.isfinite(numeric[scored]).all(),
                         'Scores must exist only for scored judgments')
                _require((numeric[scored].ge(0) & (numeric[scored].le(1) if field != 'harmful_count'
                                                else numeric[scored].mod(1).eq(0))).all(), 'Invalid score')
            for row in frame.loc[frame.status.eq('scored')].to_dict('records'):
                credits = json.loads(row['credits'])
                _require(len(credits) == len(labels[row['patient_id']]['plan']) and credits
                         and all(type(c) in (int, float) and c in (0, 0.5, 1) for c in credits),
                         'Invalid reference credits')
                matches = json.loads(row['matches'] or '[]')
                if row['attempt']:
                    _require(all(match in MATCH_CREDITS for match in matches)
                             and [MATCH_CREDITS[match] for match in matches] == credits,
                             'Match labels differ from reference credits')
                else:
                    _require(row['reason'] == 'empty_generated_plan' and not matches and not any(credits),
                             'Invalid automatic judgment')
                harmful = json.loads(row['harmful_actions'] or '[]')
                numbers = [item['generated'] for item in harmful]
                _require(len(harmful) == float(row['harmful_count']) and len(set(numbers)) == len(numbers),
                         'Harmful action count differs from distinct actions')
                _require(np.isclose(float(row['coverage']), sum(credits) / len(credits), rtol=0, atol=1e-12)
                         and np.isclose(float(row['penalized']), max(0, (sum(credits) -
                             0.5 * float(row['harmful_count'])) / len(credits)), rtol=0, atol=1e-12),
                         'Judgment score does not match credits')
            judgments.append(frame)
    _require(seen_variants, 'No variants selected')
    patient_df = pd.DataFrame(records)
    judge_df = pd.concat(judgments, ignore_index=True)
    renamed = {column: 'plan1_' + column for column in judge_df if column not in
               ('patient_id', 'generator_model', 'variant')}
    renamed.update(patient_id='id', generator_model='model', coverage='plan1_score',
                   credits='plan1_reference_item_credits', run_id='plan1_judge_run_id')
    patient_df = patient_df.merge(judge_df.rename(columns=renamed), on=['id', 'model', 'variant'],
                                  how='left', validate='one_to_one')
    for field in ('generation_status', 'variant_status'):
        _require(patient_df[field].fillna('').eq(patient_df['plan1_' + field].fillna('')).all(),
                 'Judging generation status differs from source run')
    _require(len(patient_df) == len(judge_df) == len(cohort) * len(models) * len(seen_variants),
             'Combined denominator mismatch')
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(output_dir / 'experiment_results.csv', index=False, na_rep='')
    judge_df.to_csv(output_dir / 'plan_judgments_v2.csv', index=False, na_rep='')
    return patient_df, judge_df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, help='Source YAML; outputs go beside this file')
    args = parser.parse_args(argv)
    path = args.sources or OUTPUT / 'sources.yaml'
    sources = yaml.safe_load(path.read_text())
    frame, judgments = build_dataset(sources, path.parent)
    print(f'Saved {len(frame)} experiment rows and {len(judgments)} judgments to {path.parent}')


if __name__ == '__main__':
    main()
