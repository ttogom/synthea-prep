"""Matched local Plan-judge throughput: 32 identical saved requests at C4 and C8.

Prepare/inspect: python /tmp/medcopilot_speedup_judge_benchmark.py RUN_DIRECTORY --inspect
Run only after Plan judging stops: omit --inspect. Existing run artifacts stay untouched.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from urllib.parse import urlsplit

# The ad hoc script lives in /tmp; project imports require the explicit checkout.
ROOT = Path('/Users/rasikaramanan/Documents/usc/by_semester/f26/csci566/project/synthea-prep')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'med_copilot/upstream'))

from scripts.med_copilot_config import read_yaml_config
from scripts.med_copilot_execution import bounded_map
from scripts.med_copilot_helpers import model_response_metadata
from scripts.med_copilot_judges import (
    parse_action_judgement, parse_extras_judgement, plan_action_messages,
    plan_entry, plan_extras_messages, request_key,
)
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.med_copilot_ollama import ManagedOllama
from scripts.med_copilot_run_file import _fsync_directory, pacific_now
from scripts.med_copilot_run_lock import exclusive_run_lock


STRATA = ('short', 'typical', 'long', 'complex')
TASKS = ('plan_action', 'plan_extras')
CONTEXT = 4096


def rows(path):
    # splitlines() would also split Unicode line separators inside clinical text.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_requests(run_dir, metadata):
    judge = metadata['experiment']['plan_judge']
    patients = {row['id']: row for row in rows(ROOT / metadata['input']['file'])}
    labels = {row['id']: row for row in rows(ROOT / metadata['evaluation']['labels_path'])}
    strata = metadata['pilot_selection']['patient_strata']
    expected = {}
    for row in rows(run_dir / 'results.jsonl'):
        patient, label = patients[row['id']], labels[row['id']]
        context = (label['condition'].replace('_', ' '), patient['subjective'], patient['objective'])
        for variant in row['variants'].values():
            for output in variant['assessment_plans'].values():
                if output['status'] != 'completed':
                    continue
                for rank, prediction in enumerate(output.get('predictions') or [], 1):
                    generated = prediction.get('plan')
                    if plan_entry(rank, label['plan'], generated)['status'] != 'pending':
                        continue
                    requests = [('plan_action', plan_action_messages(*context, action, generated))
                                for action in label['plan']]
                    requests.append(('plan_extras', plan_extras_messages(*context, label['plan'], generated)))
                    for task, messages in requests:
                        key = request_key(judge, messages)
                        request = {'key': key, 'patient_id': patient['id'], 'stratum': strata[patient['id']],
                                   'task': task, 'messages': messages, 'generated_count': len(generated)}
                        if key in expected and expected[key] != request:
                            raise ValueError('Ambiguous saved request identity')
                        expected[key] = request
    return expected


def parse(request, content):
    if request['task'] == 'plan_action':
        return parse_action_judgement(content)
    return parse_extras_judgement(content, request['generated_count'])


def select_requests(run_dir, metadata):
    expected = expected_requests(run_dir, metadata)
    cached = {}
    judge = metadata['experiment']['plan_judge']
    for entry in rows(run_dir / 'judge_responses.jsonl'):
        key = entry.get('key')
        if entry.get('valid') is not True or key not in expected or entry.get('judge_model') != judge:
            continue
        if request_key(judge, entry['messages']) != key:
            raise ValueError('Saved cache key does not match its messages')
        request = dict(expected[key])
        parse(request, entry['content'])
        request['prompt_bytes'] = sum(len(message['content'].encode('utf-8')) for message in request['messages'])
        cached[key] = request
    groups = {(stratum, task): sorted(
        [request for request in cached.values() if (request['stratum'], request['task']) == (stratum, task)],
        key=lambda request: (request['prompt_bytes'], request['key'])) for stratum in STRATA for task in TASKS}
    if any(len(group) < 4 for group in groups.values()):
        raise ValueError('Need four cached valid calls for every stratum/action-extras group')
    selected = []
    # Interleave strata/tasks so both conditions see the same varied request order.
    for quartile in range(4):
        for stratum in STRATA:
            for task in TASKS:
                group = groups[(stratum, task)]
                index = min(len(group) - 1, int((quartile + 0.5) * len(group) / 4))
                selected.append({**group[index], 'length_quartile': quartile + 1})
    if len({request['key'] for request in selected}) != 32:
        raise ValueError('Matched sample must contain 32 unique requests')
    chosen = {request['key'] for request in selected}
    warmup = next(request for key, request in sorted(cached.items()) if key not in chosen)
    return selected, warmup


def call(models, judge, request):
    began = perf_counter()
    try:
        response = models.judge(judge, request['messages'])
        choice = response.choices[0]
        content = choice.message.content
        error, parsed = None, None
        try:
            if getattr(choice.message, 'refusal', None) or getattr(choice, 'finish_reason', None) not in (None, 'stop'):
                raise ValueError('Judge reply refused or unfinished')
            parsed = parse(request, content)
        except (ValueError, TypeError, KeyError) as exc:
            error = models.error_metadata(exc)
        return {'key': request['key'], 'patient_id': request['patient_id'], 'stratum': request['stratum'],
                'task': request['task'], 'prompt_bytes': request['prompt_bytes'],
                'seconds': perf_counter() - began, 'valid': error is None, 'parsed': parsed, 'error': error,
                'raw_response': response.model_dump(mode='json'), 'provider': model_response_metadata(response)}
    except Exception as exc:
        return {'key': request['key'], 'patient_id': request['patient_id'], 'stratum': request['stratum'],
                'task': request['task'], 'prompt_bytes': request['prompt_bytes'],
                'seconds': perf_counter() - began, 'valid': False, 'error': models.error_metadata(exc)}


def publish(path, report):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.plan_concurrency_benchmark.', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic publication that refuses to replace even a concurrently created file.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--inspect', action='store_true', help='Select/describe inputs without any inference or writes')
    args = parser.parse_args()
    run_dir = args.run_directory.resolve()
    metadata = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    selected, warmup = select_requests(run_dir, metadata)
    if args.inspect:
        print(json.dumps({'requests': len(selected), 'context_length': CONTEXT,
            'stratum_counts': dict(Counter(request['stratum'] for request in selected)),
            'task_counts': dict(Counter(request['task'] for request in selected)),
            'prompt_bytes_range': [min(request['prompt_bytes'] for request in selected),
                                   max(request['prompt_bytes'] for request in selected)],
            'condition_order': [4, 8], 'warmup_calls_excluded': 2}, indent=2))
        return
    output = run_dir / 'plan_concurrency_benchmark.json'
    logs = {slots: run_dir / f'plan_concurrency_benchmark.c{slots}.log' for slots in (4, 8)}
    with exclusive_run_lock(run_dir):
        if output.exists() or any(path.exists() for path in logs.values()):
            raise FileExistsError('Refusing to overwrite benchmark output or logs')
        metadata = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if metadata.get('plan_evaluation', {}).get('status') not in ('completed', 'failed_incomplete', 'interrupted'):
            raise ValueError('Wait for the active Plan stage to finish or stop')
        judge = metadata['experiment']['plan_judge']
        registry_path = ROOT / metadata['assessment_plan_models']['config_file']
        registry, _ = read_yaml_config(registry_path)
        entry = registry['models'][judge]
        address = urlsplit(entry.get('api_base', ''))
        settings = metadata['experiment']['execution']
        owned = urlsplit('http://' + settings['ollama_host'])
        if (entry.get('api_key_env') or address.hostname not in (owned.hostname, 'localhost')
                or address.port != owned.port or address.scheme != 'http' or address.path.rstrip('/') != '/v1'
                or entry.get('max_tokens') != 64):
            raise ValueError('Benchmark requires the unauthenticated owned local judge with max_tokens=64')
        models = AssessmentPlanModels(registry_path, [judge], temperature=0,
                                      max_retries=metadata['experiment']['max_retries'])
        protected = {name: fingerprint(run_dir / name) for name in ('run.json', 'results.jsonl', 'judge_responses.jsonl')}
        report = {'method': 'matched-plan-concurrency-v1', 'started_at': pacific_now().isoformat(),
                  'request_count_per_condition': 32, 'condition_order': [4, 8], 'context_length': CONTEXT,
                  'judge_alias': judge, 'temperature': 0, 'max_tokens': 64,
                  'model_settings': models.metadata(ROOT), 'sample': selected,
                  'protected_artifact_sha256': protected, 'conditions': []}
        try:
            for slots in (4, 8):
                with ManagedOllama({**settings, 'manage_ollama': True}, logs[slots],
                                   parallel=slots, context=CONTEXT, max_loaded_models=1) as server:
                    condition = {'parallel_slots': slots, 'client_concurrency': slots,
                                 'server': server.metadata(), 'log_file': logs[slots].name}
                    condition['warmup_excluded'] = call(models, judge, warmup)
                    condition['ps_after_warmup'] = server.ps_snapshot()
                    began = perf_counter()
                    responses = [result for _, result, _ in bounded_map(
                        lambda request: call(models, judge, request), selected, workers=slots)]
                    condition.update(elapsed_seconds=perf_counter() - began,
                        responses=responses, valid_count=sum(response['valid'] for response in responses))
                    condition['valid_replies_per_second'] = condition['valid_count'] / condition['elapsed_seconds']
                    condition['ps_after_measured_calls'] = server.ps_snapshot()
                    report['conditions'].append(condition)
                    server.unload(entry['model'])
        finally:
            report['finished_at'] = pacific_now().isoformat()
            report['protected_artifacts_unchanged'] = all(fingerprint(run_dir / name) == sha for name, sha in protected.items())
            publish(output, report)
        if not report['protected_artifacts_unchanged']:
            raise RuntimeError('Protected run artifacts changed while benchmark held its lock')
        print(f'Benchmark saved: {output}')


if __name__ == '__main__':
    main()
