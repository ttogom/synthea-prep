"""Compare eight matched cases against saved two-worker service measurements."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import yaml

TASK=Path(__file__).resolve().parent
BASE=Path('/private/tmp/medcopilot-rate-tuning-20261112/validation')


def read(path):
    return json.loads(path.read_text())


def main():
    status=read(TASK/'status.json')
    assert status['status']=='completed' and len(status['workers'])==4
    selected=read(TASK/'selection.json')
    identifiers={p['id'] for p in selected['patients']}
    strata={p['id']:p['stratum'] for p in selected['patients']}
    assert len(identifiers)==8 and Counter(strata.values())=={'low':2,'moderate_low':2,'moderate_high':2,'high':2}
    old_profiles={p['patient_id']:p for p in (read(f) for f in (BASE/'profiles').glob('*.json')) if p['patient_id'] in identifiers}
    new_profiles={p['patient_id']:p for p in (read(f) for f in (TASK/'profiles').glob('*.json'))}
    assert set(new_profiles)==set(old_profiles)==identifiers
    assert all(p['error_type'] is None for p in new_profiles.values())
    old_status=read(BASE/'status.json')
    baseline_worker_service={str(w['worker']):sum(p['seconds'] for p in w['completed'] if p['patient_id'] in identifiers) for w in old_status['workers']}
    old_processing=max(baseline_worker_service.values())
    old_cold_wall=max(w['initialization_seconds']+baseline_worker_service[str(w['worker'])] for w in old_status['workers'])
    new_processing=max(w['processing_wall_seconds'] for w in status['workers'])
    old_mean=mean(p['total_seconds'] for p in old_profiles.values())
    new_mean=mean(p['total_seconds'] for p in new_profiles.values())
    old_service=sum(p['total_seconds'] for p in old_profiles.values())
    new_service=sum(p['total_seconds'] for p in new_profiles.values())
    per_case=[{'patient_id':pid,'stratum':strata[pid],'two_worker_seconds':old_profiles[pid]['total_seconds'],
               'four_worker_seconds':new_profiles[pid]['total_seconds'],'ratio_new_over_old':new_profiles[pid]['total_seconds']/old_profiles[pid]['total_seconds']} for pid in identifiers]
    stages={name:{'two_worker_mean_seconds':mean(p['stages'][name] for p in old_profiles.values()),
                  'four_worker_mean_seconds':mean(p['stages'][name] for p in new_profiles.values())} for name in next(iter(old_profiles.values()))['stages']}
    http=Counter()
    for p in new_profiles.values():
        for e in p['events']:
            if e['name']=='http':
                http[(e['stage'],e['path'],e['status'])]+=1
    assert all(status==200 for stage,path,status in http)
    new_prepared=[read(f) for f in (TASK/'prepared').glob('*.json')]
    assert len(new_prepared)==8 and {p['id'] for p in new_prepared}==identifiers
    api_captures=Counter()
    for p in new_prepared:
        assert set(p['variants'])=={'patient_retrieval_guidelines'}
        for component in p['components'].values():
            assert component['status']=='completed'
            for k,v in component['capture']['model_responses'].items():
                api_captures[k]+=len(v)
        patient_capture=p['components']['patient_retrieval']['capture']
        guideline_capture=p['components']['guidelines']['capture']
        assert patient_capture['selected_reference_patient'] and patient_capture['reranked_patient_ids']
        assert guideline_capture['guideline_evidence'].strip() and guideline_capture['graphrag_context']
        variant=p['variants']['patient_retrieval_guidelines']
        assert variant['assessment_plan_prompts']
        for slot in variant['assessment_plans'].values():
            assert slot.get('assessment_plan_raw') is None and not slot['model_responses'] and slot['evaluation'] is None
    guide=yaml.safe_load((TASK/'guidelines/settings.yaml').read_text())
    assert guide==yaml.safe_load((BASE/'guidelines/settings.yaml').read_text())
    old_inputs={p['id']:p for p in map(json.loads,(BASE/'patients.jsonl').read_text().splitlines())}
    new_inputs={p['id']:p for p in map(json.loads,(TASK/'patients.jsonl').read_text().splitlines())}
    assert all(old_inputs[k]==new_inputs[k] for k in identifiers)
    profiles_by_new_pid=Counter(p['worker_pid'] for p in new_profiles.values())
    assert len(profiles_by_new_pid)==4 and set(profiles_by_new_pid.values())=={2}
    assert not list((TASK/'errors').glob('*'))
    resource=read(TASK/'resources.json')
    peak_rss_kib=max(sum(s['workers'].values()) for s in resource)
    result={'patients':8,'strata':dict(Counter(strata.values())),'workers':4,
            'settings':{'requests_per_minute_per_worker':12,'tokens_per_minute_per_worker':200000},
            'two_worker_baseline':{'matched_service_sum_seconds':old_service,'matched_mean_patient_seconds':old_mean,
                                   'reconstructed_processing_wall_seconds':old_processing,'reconstructed_total_wall_including_startup_seconds':old_cold_wall,
                                   'worker_service_sums_seconds':baseline_worker_service,
                                   'initialization_seconds':[w['initialization_seconds'] for w in old_status['workers']]},
            'four_worker_run':{'matched_service_sum_seconds':new_service,'matched_mean_patient_seconds':new_mean,
                               'measured_processing_wall_seconds':new_processing,'measured_total_wall_including_startup_seconds':status['wall_seconds_including_loading'],
                               'initialization_seconds':[w['initialization_seconds'] for w in status['workers']]},
            'speedup':{'processing_wall':old_processing/new_processing,'startup_inclusive_wall':old_cold_wall/status['wall_seconds_including_loading'],
                       'concurrency_normalized_service_throughput':(old_service/2)/(new_service/4),
                       'mean_patient_latency_new_over_old':new_mean/old_mean},
            'stages':stages,'per_case':per_case,
            'diagnostics_mean_seconds':{key:mean(p['diagnostics'].get(key,0) for p in new_profiles.values()) for key in ['limiter_rpm','limiter_tpm','attribution']},
            'http_attempts':[{'stage':s,'path':p,'status':status,'count':n} for (s,p,status),n in http.items()],
            'captured_model_response_counts':dict(api_captures),
            'resources':{'peak_sum_worker_rss_gib':peak_rss_kib/1024**2,'swap_at_first_sample':resource[0]['swap'],'swap_at_last_sample':resource[-1]['swap']},
            'integrity':{'unique_patients':8,'workers_each_completed_two':True,'completed_evidence_components':16,'all_inputs_match_baseline':True,
                         'guidelines_settings_unchanged':True,'prompts_and_ranking_metadata_present':True,'final_generation_scoring_judging_calls':0,'error_artifacts':0},
            'limitations':['Baseline eight-case wall time is reconstructed from the existing two-worker per-patient durations and original assignments, excluding the two unselected cases; no new baseline run was performed.',
                           'Four-worker measured processing wall includes tiny protocol and shutdown overhead; baseline reconstruction sums recorded service durations.',
                           'Only two cases per new worker; fresh limiter buckets cannot establish long-run throttling behavior.',
                           'API-generated keywords/questions/answers are stochastic. Different screening retry counts and provider latency can affect timing.',
                           'This is preparation-only. Concurrent final generation/scoring interference was not reproduced; existing parent work was left untouched.',
                           'Two-worker and four-worker runs occurred at different times, so throughput ratios are estimates rather than a controlled simultaneous comparison.']}
    (TASK/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='per_case'},indent=2))
    lines=['# Eight-patient four-worker preparation comparison','',
           'Eight matched patients, two per complexity group, drawn without reference to their timing. Four CPU workers with two CPU threads each. Same copied pipeline and numeric GraphRAG settings: 12 RPM / 200,000 TPM per worker. No branch or repository changes.','',
           '| Measure | Existing two-worker baseline | New four-worker run |','|---|---:|---:|',
           f'| Mean preparation duration per patient | {old_mean:.2f} s | {new_mean:.2f} s |',
           f'| Processing wall, excluding initialization | {old_processing:.2f} s (reconstructed) | {new_processing:.2f} s (measured) |',
           f'| Total wall including initialization | {old_cold_wall:.2f} s (reconstructed) | {status["wall_seconds_including_loading"]:.2f} s (measured) |','',
           f'Processing-wall speedup: {old_processing/new_processing:.3f}x. Concurrency-normalized service-throughput speedup: {(old_service/2)/(new_service/4):.3f}x. Startup-inclusive speedup: {old_cold_wall/status["wall_seconds_including_loading"]:.3f}x.','',
           '| Preparation step | Two-worker seconds/patient | Four-worker seconds/patient |','|---|---:|---:|']
    for name,v in stages.items():
        lines.append(f'| {name} | {v["two_worker_mean_seconds"]:.3f} | {v["four_worker_mean_seconds"]:.3f} |')
    lines+=['',f'Peak sampled sum of worker RSS: {peak_rss_kib/1024**2:.2f} GiB (excludes other applications).',
            '', 'All eight distinct cases completed both evidence components, saved prompts/ranking metadata, and ran no final generation/scoring/judging. All recorded HTTP attempts returned 200.','',
            '## Limitations','']+['- '+v for v in result['limitations']]
    (TASK/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
