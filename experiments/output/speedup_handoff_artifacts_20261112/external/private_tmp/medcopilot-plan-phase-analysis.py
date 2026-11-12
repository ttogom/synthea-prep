from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import os
import sys
import tempfile

ROOT = Path('/Users/rasikaramanan/Documents/usc/by_semester/f26/csci566/project/synthea-prep')
sys.path.insert(0, str(ROOT))
from scripts.med_copilot_judges import plan_entry, plan_action_messages, plan_extras_messages, request_key

RUN = ROOT / 'experiments/output/2026-11-11_232045PST__pilot_speedup_so__speedup'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def iso(value):
    return datetime.fromisoformat(value)


metadata = json.loads((RUN / 'run.json').read_text())
if not metadata['plan_evaluation'].get('finished_at'):
    print(json.dumps({'status': 'waiting_for_finished_at', 'started_at': metadata['plan_evaluation']['started_at'],
                      'cache_rows': sum(1 for line in (RUN / 'judge_responses.jsonl').open() if line.strip())}))
    raise SystemExit(0)

selection = metadata['pilot_selection']
patients = {row['id']: row for row in read_rows(ROOT / metadata['input']['file'])}
labels = {row['id']: row for row in read_rows(ROOT / metadata['evaluation']['labels_path'])}
records = read_rows(RUN / 'results.jsonl')
cache_bytes = (RUN / 'judge_responses.jsonl').read_bytes()
cache = [json.loads(line) for line in cache_bytes.decode('utf-8').split('\n') if line.strip()]
judge = metadata['experiment']['plan_judge']
expected = defaultdict(set)
owners = defaultdict(set)
logical_calls = Counter()
plan_entries = 0
plan_statuses = Counter()
plan_reasons = Counter()
plan_errors = []
for record in records:
    patient_id = record['id']
    patient, label = patients[patient_id], labels[patient_id]
    context = label['condition'].replace('_', ' '), patient['subjective'], patient['objective']
    for variant_id, variant in record['variants'].items():
        for model_id, output in variant['assessment_plans'].items():
            for evaluation in output.get('evaluation') or []:
                correctness = evaluation.get('plan_correctness') or {}
                plan_statuses[correctness.get('status', 'not_recorded')] += 1
                if correctness.get('reason'):
                    plan_reasons[correctness['reason']] += 1
                if correctness.get('status') in ('error', 'interrupted', 'pending'):
                    plan_errors.append({'patient_id': patient_id, 'stratum': selection['patient_strata'][patient_id],
                                        'variant': variant_id, 'model': model_id, 'rank': evaluation.get('rank'),
                                        'status': correctness.get('status'), 'reason': correctness.get('reason'),
                                        'error': correctness.get('error'),
                                        'failed_judges': [{key: item.get(key) for key in
                                                          ('request_key', 'call', 'status', 'error')}
                                                         for item in correctness.get('judges', [])
                                                         if item.get('status') != 'scored']})
            if output['status'] != 'completed':
                continue
            predictions = output.get('predictions')
            for index in range(3):
                generated = predictions[index].get('plan') if predictions is not None else None
                entry = plan_entry(index + 1, label['plan'], generated)
                if entry['status'] != 'pending':
                    continue
                plan_entries += 1
                requests = [('plan_action', plan_action_messages(*context, action, generated))
                            for action in entry['reference_actions']]
                requests.append(('plan_extras', plan_extras_messages(*context, entry['reference_actions'], generated)))
                for task, messages in requests:
                    key = request_key(judge, messages)
                    expected[patient_id].add(key)
                    owners[key].add(patient_id)
                    logical_calls[task] += 1

assert all(len(values) == 1 for values in owners.values()), 'Request keys shared across patients'
owner = {key: next(iter(values)) for key, values in owners.items()}
all_expected = set(owner)
assert len(all_expected) == 8006 and sum(logical_calls.values()) == 8015
key_hashes_match = all(row['key'] == request_key(row['judge_model'], row['messages']) for row in cache)
unexpected = sorted({row['key'] for row in cache} - all_expected)
stages = [*metadata.get('plan_evaluation_history', []), metadata['plan_evaluation']]
log = (RUN / 'ollama-judge.log').read_text(errors='replace')
markers = list(re.finditer(r'^.*msg="server config".*$', log, re.MULTILINE))
log_sections = [log[marker.start():markers[index + 1].start() if index + 1 < len(markers) else len(log)]
                for index, marker in enumerate(markers)]
stage_rows = []
stage_reports = []
for index, stage in enumerate(stages):
    start, finish = iso(stage['started_at']), iso(stage['finished_at'])
    responses = [row for row in cache if start <= iso(row['recorded_at']) <= finish]
    stage_rows.append(responses)
    wall = (finish - start).total_seconds()
    valid = [row for row in responses if row['valid'] is True]
    native = log_sections[index] if index < len(log_sections) else ''
    positive_truncation = [line for line in native.split('\n') if re.search(
        r'truncated\s*=\s*[1-9]|truncating input|context shift', line, re.IGNORECASE)]
    stage_reports.append({
        'stage_index': index, 'status': stage['status'], 'started_at': stage['started_at'],
        'finished_at': stage['finished_at'], 'concurrency': stage.get('concurrency'),
        'context_length': stage.get('context_length', 8192), 'wall_seconds': wall,
        'calls_reported': stage.get('calls'), 'cache_hits_reported': stage.get('cache_hits'),
        'response_rows': len(responses), 'valid_response_rows': len(valid),
        'valid_unique_keys': len({row['key'] for row in valid}),
        'invalid_response_rows': len(responses) - len(valid),
        'calls_per_second': stage.get('calls', 0) / wall if wall else None,
        'valid_responses_per_second': len(valid) / wall if wall else None,
        'max_reported_prompt_tokens': max(((row.get('model_response') or {}).get('token_usage') or {}).get('prompt_tokens') or 0
                                          for row in responses),
        'max_reported_completion_tokens': max(((row.get('model_response') or {}).get('token_usage') or {}).get('completion_tokens') or 0
                                              for row in responses),
        'native_slot_context_counts': dict(Counter(re.findall(r'n_ctx_slot\s*=\s*(\d+)', native))),
        'native_truncation_or_context_shift_detected': bool(positive_truncation),
        'native_matching_lines': positive_truncation[:10],
        'counts': stage.get('counts'),
    })

c4, c8 = stage_rows[0], stage_rows[-1]
valid4 = {row['key'] for row in c4 if row['valid'] is True}
keys4 = {row['key'] for row in c4}
valid8 = {row['key'] for row in c8 if row['valid'] is True}
keys8 = {row['key'] for row in c8}
valid_all = {row['key'] for row in cache if row['valid'] is True}
allocation = defaultdict(float)
previous = iso(stages[-1]['started_at'])
for row in c8:
    current = iso(row['recorded_at'])
    allocation[owner[row['key']]] += max(0.0, (current - previous).total_seconds())
    previous = current
categories = defaultdict(Counter)
patients_summary = []
pure_attempted = []
for record in records:
    patient_id = record['id']
    keys = expected[patient_id]
    stratum = selection['patient_strata'][patient_id]
    if not keys:
        category = 'no_scalar_judging'
    elif not keys & valid4:
        if keys <= valid8:
            category = 'C8_only_valid_complete'
        elif keys <= keys8:
            category = 'C8_only_fully_attempted_with_invalid_reply'
        else:
            category = 'C8_only_incomplete'
        if keys <= keys8:
            pure_attempted.append(patient_id)
    elif keys <= valid4:
        category = 'C4_valid_complete'
    elif keys <= valid_all:
        category = 'C4_C8_mixed_valid_complete'
    elif keys <= keys4 | keys8:
        category = 'C4_C8_mixed_fully_attempted_with_invalid_reply'
    else:
        category = 'C4_C8_mixed_incomplete'
    categories[category][stratum] += 1
    patients_summary.append({'patient_id': patient_id, 'stratum': stratum, 'category': category,
                             'expected_unique_keys': len(keys), 'valid_C4_keys': len(keys & valid4),
                             'attempted_C4_keys': len(keys & keys4), 'valid_C8_keys': len(keys & valid8),
                             'C8_allocated_seconds': allocation[patient_id]})

strata = {}
for stratum, values in selection['strata'].items():
    members = [patient_id for patient_id in pure_attempted if selection['patient_strata'][patient_id] == stratum]
    seconds = [allocation[patient_id] for patient_id in members]
    strata[stratum] = {'population_count': values['population_count'], 'population_weight': values['weight'],
                       'C8_only_fully_attempted_patients': len(members),
                       'C8_only_valid_complete_patients': sum(expected[patient_id] <= valid8 for patient_id in members),
                       'allocated_seconds': sum(seconds), 'mean_allocated_seconds_per_patient': mean(seconds) if seconds else None}
projected_seconds = (sum(values['population_count'] * values['mean_allocated_seconds_per_patient'] for values in strata.values())
                     if all(values['mean_allocated_seconds_per_patient'] is not None for values in strata.values()) else None)
generation = json.loads((RUN / 'runtime_projection_first_pass.json').read_text())
baseline = generation['pipeline']['wall_time']['hours']
c4_measurement = json.loads((RUN / 'plan_measurement_c4/measurement.json').read_text())
prefix_size = c4_measurement['cache_prefix_bytes']
prefix_preserved = hashlib.sha256(cache_bytes[:prefix_size]).hexdigest() == c4_measurement['cache_prefix_sha256']
report = {
    'method': 'read-only-plan-phase-analysis-v1', 'run_directory': str(RUN.relative_to(ROOT)),
    'analyzed_at': datetime.now().astimezone().isoformat(),
    'final_plan_status': metadata['plan_evaluation']['status'],
    'final_plan_counts': metadata['plan_evaluation'].get('counts'),
    'canonical_plan_status_counts': dict(plan_statuses), 'canonical_plan_reason_counts': dict(plan_reasons),
    'canonical_unfinished_plan_entries': plan_errors,
    'judgeable_plan_entries': plan_entries, 'logical_scalar_calls': dict(logical_calls),
    'logical_scalar_total': sum(logical_calls.values()), 'expected_unique_request_keys': len(all_expected),
    'valid_unique_cached_request_keys': len(valid_all), 'expected_keys_without_valid_reply': sorted(all_expected - valid_all),
    'expected_keys_never_attempted': sorted(all_expected - {row['key'] for row in cache}),
    'cache_row_count': len(cache), 'all_cache_key_hashes_match_current_messages': key_hashes_match,
    'unexpected_cache_keys': unexpected, 'C4_cache_prefix_preserved': prefix_preserved,
    'C4_valid_keys_recalled_under_C8': sorted(valid4 & keys8),
    'C4_valid_keys_not_recalled_under_C8': len(valid4 - keys8),
    'stages': stage_reports, 'patient_categories_by_stratum': {key: dict(value) for key, value in categories.items()},
    'C8_patient_interval_allocation': patients_summary, 'C8_stratum_allocation': strata,
    'C8_only_projected_full_judge_seconds': projected_seconds,
    'C8_only_projected_full_judge_hours': projected_seconds / 3600 if projected_seconds is not None else None,
    'generation_baseline': {'file': 'runtime_projection_first_pass.json', 'pipeline_wall_time_hours': baseline,
                            'note': 'Original first-pass projection retained; resumed generation metadata was not projected.'},
    'generation_plus_C8_judge_hours': baseline + projected_seconds / 3600 if projected_seconds is not None else None,
    'forecast_limitations': [
        'Approximate timestamp allocation: each interval between durable replies is assigned to the patient owning its ending reply.',
        'Timestamps have one-second precision; concurrent patient boundaries can transfer wall time between patients.',
        'C8-only fully attempted patients have no valid C4 prefix key. Invalid replies still count as work, avoiding exclusion bias.',
        'The C8 subset is a nonrandom input-order suffix after C4; stratum weighting does not eliminate condition/action-count mix bias.',
        'C4 and C8 processed different work slices; their observed reply rates do not establish causal concurrency speedup.',
        'This projects initial attempted judgement work; invalid replies may still require additional retries or review.',
        'Native log absence of truncation is observed evidence, not a model-independent proof for unseen full-cohort prompts.',
    ],
    'source_sha256': {name: hashlib.sha256((RUN / name).read_bytes()).hexdigest() for name in
                      ['run.json', 'results.jsonl', 'judge_responses.jsonl', 'runtime_projection_first_pass.json']},
}
destination = RUN / 'plan_phase_analysis.json'
with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=RUN, prefix='.plan-phase-analysis-', delete=False) as handle:
    temporary = Path(handle.name)
    json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
print(json.dumps({'file': str(destination), 'status': report['final_plan_status'],
                  'counts': report['final_plan_counts'], 'stages': stage_reports,
                  'C8_only_projected_full_judge_hours': report['C8_only_projected_full_judge_hours'],
                  'generation_plus_C8_judge_hours': report['generation_plus_C8_judge_hours'],
                  'C4_valid_keys_recalled_under_C8': len(report['C4_valid_keys_recalled_under_C8']),
                  'C4_cache_prefix_preserved': prefix_preserved,
                  'C8_stratum_allocation': strata}, indent=2))
