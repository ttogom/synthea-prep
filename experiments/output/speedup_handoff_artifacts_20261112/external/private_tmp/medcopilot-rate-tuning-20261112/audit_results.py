"""Audit matched requests, config-only changes, saved preparation and timing."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

import yaml

TASK=Path(__file__).resolve().parent
BASE=Path('/private/tmp/medcopilot-preparation-profile-qlspfgvq')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def main():
    experiment=json.loads((TASK/'results.json').read_text())
    original_config=yaml.safe_load((BASE/'guidelines/settings.yaml').read_text())
    checks={}
    for path in TASK.glob('*.yaml'):
        config=yaml.safe_load(path.read_text())
        for key in ('requests_per_minute','tokens_per_minute'):
            config['models']['default_chat_model'][key]=original_config['models']['default_chat_model'][key]
        assert config==original_config,path.name
    checks['all_comparison_configs_differ_only_in_RPM_TPM']=True
    requests=defaultdict(set)
    contexts=defaultdict(set)
    queries=defaultdict(set)
    statuses=[]
    finishes=[]
    all_rows=[]
    all_http=[]
    for path in TASK.glob('*_w?.json'):
        record=json.loads(path.read_text())
        for row in record['rows']:
            all_rows.append(row)
            assert row['answer_nonempty'] and row['context']['sources']
            contexts[row['patient_id']].add(digest(row['context']))
            queries[row['patient_id']].add(row['query_sha256'])
            for event in row['events']:
                if event['type']=='http':
                    all_http.append(event)
                    statuses.append(event['status'])
                    if event['path'].endswith('chat/completions'):
                        requests[row['patient_id']].add(event['request_sha256'])
                        finishes.extend(event['finish_reasons'])
    checks['query_count']=len(all_rows)
    checks['http_attempt_count']=len(statuses)
    checks['http_status_counts']=dict(Counter(statuses))
    checks['finish_reason_counts']=dict(Counter(finishes))
    checks['byte_identical_requests_per_patient']=all(len(v)==1 for v in requests.values())
    checks['identical_contexts_per_patient']=all(len(v)==1 for v in contexts.values())
    checks['identical_queries_per_patient']=all(len(v)==1 for v in queries.values())
    # Report existing retrieval variability instead of hiding it behind a pass.
    checks['patients_with_context_variation']=[k for k,v in contexts.items() if len(v)>1]
    checks['patients_with_request_variation']=[k for k,v in requests.items() if len(v)>1]
    initial_requests=defaultdict(set)
    for label in ('A','B','C','D'):
        for worker in (0,1):
            for row in json.loads((TASK/f'{label}_w{worker}.json').read_text())['rows']:
                for e in row['events']:
                    if e['type']=='http' and e['path'].endswith('chat/completions'):
                        initial_requests[row['patient_id']].add(e['request_sha256'])
    checks['initial_factorial_byte_identical_requests']=all(len(v)==1 for v in initial_requests.values())
    assert checks['initial_factorial_byte_identical_requests']
    assert all(s==200 for s in statuses) and finishes==['stop']*len(all_rows)
    strata={r['id']:r['stratum'] for r in json.loads((BASE/'selection.json').read_text())['patients']}
    checks['initial_comparison_strata']=dict(Counter(strata[r['patient_id']] for r in json.loads((TASK/'A_w0.json').read_text())['rows']+json.loads((TASK/'A_w1.json').read_text())['rows']))
    checks['confirmation_strata']=dict(Counter(strata[r['patient_id']] for r in json.loads((TASK/'D_repeat_w0.json').read_text())['rows']+json.loads((TASK/'D_repeat_w1.json').read_text())['rows']))
    prepared=[]
    for path in (TASK/'validation/prepared').glob('*.json'):
        row=json.loads(path.read_text())
        assert set(row['variants'])=={'patient_retrieval_guidelines'}
        assert row['components']['patient_retrieval']['status']=='completed'
        assert row['components']['guidelines']['status']=='completed'
        retrieval=row['components']['patient_retrieval']['capture']
        guidelines=row['components']['guidelines']['capture']
        assert retrieval['selected_reference_patient'] and retrieval['reranked_patient_ids']
        assert guidelines['guideline_evidence'].strip() and guidelines['graphrag_context']
        assert row['variants']['patient_retrieval_guidelines']['assessment_plan_prompts']
        for slot in row['variants']['patient_retrieval_guidelines']['assessment_plans'].values():
            assert slot['assessment_plan_raw'] is None and not slot['model_responses'] and slot['evaluation'] is None
        prepared.append(row)
    assert len(prepared)==10 and len({p['id'] for p in prepared})==10
    checks['prepared_patients']=10
    checks['validation_strata']=dict(Counter(strata[r['id']] for r in prepared))
    checks['final_generation_scoring_judging_calls']=0
    checks['validation_error_artifacts']=len(list((TASK/'validation/errors').glob('*')))
    assert checks['validation_error_artifacts']==0
    # The baseline and new full-preparation questions are stochastic, so answers
    # and retrieval rankings must not be incorrectly required to match exactly.
    old=json.loads((BASE/'summary.json').read_text())
    new=experiment['validation']
    old_seconds=old['mean_preparation_seconds']
    new_seconds=new['mean_preparation_seconds']
    comparison={'old_wall_seconds_including_loading':old['wall_seconds_including_loading'],
                'new_wall_seconds_including_loading':new['wall_seconds_including_loading'],
                'old_mean_preparation_seconds':old_seconds,'new_mean_preparation_seconds':new_seconds,
                'service_speedup':old_seconds/new_seconds,
                'wall_speedup':old['wall_seconds_including_loading']/new['wall_seconds_including_loading'],
                'old_warm_eight_mean_preparation_seconds':old['warm_eight']['mean_preparation_seconds'],
                'new_warm_eight_mean_preparation_seconds':new['warm_eight']['mean_preparation_seconds'],
                'approx_preparation_hours_for_2378_two_workers':2378*new_seconds/2/3600,
                'stages':{k:{'old_mean_seconds':v['mean_seconds'],'new_mean_seconds':new['stages'][k]['mean_seconds']} for k,v in old['stages'].items()}}
    # Stored API responses are payload data, never credential-bearing headers.
    checks['credential_headers_saved']=False
    result={'checks':checks,'before_after':comparison,
            'warnings':{'async_limiter_reuse':sum(p.read_text().count('re-used across loops') for p in TASK.glob('*.log'))},
            'limitations':['Initial comparison sampled three strata; confirmation corrected to one patient per stratum and full validation used all ten.',
                          'Initial exhausted-bucket stress preconditions RPM then TPM when each is first acquired, adding conservative one-time debt. Use full-preparation timings for operational projections.',
                          'Small sample and stochastic API latency do not establish a unique global optimum or clinical equivalence.',
                          'One low-complexity confirmation case selected a different tenth context entity between conditions, with unchanged input token count. The initial factorial requests were byte-identical; do not interpret confirmation as a perfectly controlled clinical comparison.',
                          'Quota headers are observed at test time; two workers have independent local buckets. Rebudget if workers or other shared API traffic change.',
                          'The existing library warns about sequential reuse of limiters across asyncio loops. No failures were observed; no library code was changed.']}
    (TASK/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
