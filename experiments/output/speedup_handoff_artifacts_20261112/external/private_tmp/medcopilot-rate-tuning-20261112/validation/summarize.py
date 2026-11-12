from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

TASK=Path(__file__).resolve().parent
status=json.loads((TASK/'status.json').read_text())
manifest=json.loads((TASK/'selection.json').read_text())
profiles=[json.loads(p.read_text()) for p in (TASK/'profiles').glob('*.json')]
assert status['status']=='completed' and len(profiles)==10
assert all(p['error_type'] is None for p in profiles)
first={w['completed'][0]['patient_id'] for w in status['workers']}
warm=[p for p in profiles if p['patient_id'] not in first]
by_id={p['patient_id']:p for p in profiles}
strata={p['id']:p['stratum'] for p in manifest['patients']}
total=sum(p['total_seconds'] for p in profiles)
stages={name:{'mean_seconds':mean(p['stages'][name] for p in profiles),
              'median_seconds':median(p['stages'][name] for p in profiles),
              'min_seconds':min(p['stages'][name] for p in profiles),
              'max_seconds':max(p['stages'][name] for p in profiles),
              'share_percent':100*sum(p['stages'][name] for p in profiles)/total}
        for name in profiles[0]['stages']}
http=Counter()
request_captures=Counter()
for path in (TASK/'prepared').glob('*.json'):
    result=json.loads(path.read_text())
    assert len(result['variants'])==1 and 'patient_retrieval_guidelines' in result['variants']
    for comp in result['components'].values():
        assert comp['status']=='completed'
        for key, values in comp['capture']['model_responses'].items():
            request_captures[key]+=len(values)
    for variant in result['variants'].values():
        assert variant['assessment_plan_prompts']
        for slot in variant['assessment_plans'].values():
            assert slot['status']=='not_processed' and not slot['model_responses'] and slot['evaluation'] is None
for profile in profiles:
    for event in profile['events']:
        if event['name']=='http':
            http[(event['stage'],event['path'],event['status'])]+=1
wait=sum(p['diagnostics'].get('limiter_rpm',0) for p in profiles)
guide=sum(p['stages']['guideline_retrieval'] for p in profiles)
diagnostics={name:mean(p['diagnostics'].get(name,0) for p in profiles)
             for name in ('attribution','patient_search','patient_rerank','limiter_total','limiter_rpm','limiter_tpm',
                          'graphrag_setup:load_config','graphrag_setup:_resolve_output_files','checkpoint_ack')}
report={'n':10,'sampling':manifest['method'],'stratum_counts':dict(Counter(strata.values())),
        'wall_seconds_including_loading':status['wall_seconds_including_loading'],
        'initialization_seconds_by_worker':[w['initialization_seconds'] for w in status['workers']],
        'approx_processing_wall_seconds_excluding_initialization':status['wall_seconds_including_loading']-max(w['initialization_seconds'] for w in status['workers']),
        'mean_preparation_seconds':total/10,'stages':stages,'nested_diagnostics_mean_seconds':diagnostics,
        'rpm_wait_share_of_all_preparation_percent':100*wait/total,
        'rpm_wait_share_of_guideline_retrieval_percent':100*wait/guide,
        'warm_eight':{'mean_preparation_seconds':mean(p['total_seconds'] for p in warm),
                      'mean_rpm_wait_seconds':mean(p['diagnostics'].get('limiter_rpm',0) for p in warm)},
        'http_attempt_counts':[{'stage':s,'path':p,'http_status':code,'count':count} for (s,p,code),count in http.items()],
        'captured_model_response_counts':dict(request_captures),
        'by_complexity':{name:{'n':sum(strata[p['patient_id']]==name for p in profiles),
                                'mean_preparation_seconds':mean(p['total_seconds'] for p in profiles if strata[p['patient_id']]==name),
                                'mean_rpm_wait_seconds':mean(p['diagnostics'].get('limiter_rpm',0) for p in profiles if strata[p['patient_id']]==name)}
                         for name in sorted(set(strata.values()))},
        'verification':{'prepared_patients':10,'final_generation_calls':0,'diagnosis_scoring_calls':0,'plan_judging_calls':0,'errors':0},
        'limitations':['Two workers and two CPU threads per worker match the main pilot; other parent-session work can still affect CPU/network timing.',
                      'First request in each process incurs no steady-state RPM wait, making the overall ten-case average optimistic for a large cohort.',
                      'HTTP streaming timings cover headers only; the remainder includes streamed generation and uninstrumented local work.',
                      'Substep timings are observational, not a test of removing the limiter.']}
(TASK/'summary.json').write_text(json.dumps(report,indent=2))
lines=['# Ten-patient preparation-only profile','',
       'Variant: patient_retrieval_guidelines. Two isolated CPU workers, two Torch threads each. No generation or scoring.','',
       '| Preparation step | Mean seconds/patient | Share |','|---|---:|---:|']
for name,value in stages.items():
    lines.append(f'| {name} | {value["mean_seconds"]:.3f} | {value["share_percent"]:.2f}% |')
lines+=['',f'Total wall time including loading: {report["wall_seconds_including_loading"]:.3f} seconds.',
        f'Worker initialization times: {report["initialization_seconds_by_worker"]}.',
        f'Mean local RPM wait: {diagnostics["limiter_rpm"]:.3f} seconds/patient.',
        f'Warm eight: {report["warm_eight"]}.','', '## Limitations','']
lines += ['- '+v for v in report['limitations']]
(TASK/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,indent=2))
