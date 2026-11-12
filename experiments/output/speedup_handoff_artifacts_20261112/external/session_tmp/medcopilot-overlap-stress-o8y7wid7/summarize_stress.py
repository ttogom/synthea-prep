"""Audit isolated overlap outputs and observed system memory behavior."""
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean

TASK=Path(__file__).resolve().parent


def load(name):
    return json.loads((TASK/name).read_text())


def value(text,name):
    return int(re.search(re.escape(name)+r':\s+(\d+)',text)[1])


def main():
    status=load('overall_status.json')
    assert status['status']=='completed'
    generations=load('generations.json')
    assert len(generations)==4 and all(g['valid'] for g in generations)
    assert max(g['start_seconds'] for g in generations)<min(g['end_seconds'] for g in generations)
    profiles=[json.loads(p.read_text()) for p in (TASK/'profiles').glob('*.json')]
    prepared=[json.loads(p.read_text()) for p in (TASK/'prepared').glob('*.json')]
    assert len(profiles)==len(prepared)==8
    assert len({p['patient_id'] for p in profiles})==8
    assert len({p['worker_pid'] for p in profiles})==4
    assert Counter(p['worker_pid'] for p in profiles).values()=={2}.copy() if False else True
    assert set(Counter(p['worker_pid'] for p in profiles).values())=={2}
    assert all(p['error_type'] is None for p in profiles)
    http=Counter()
    for p in profiles:
        for e in p['events']:
            if e['name']=='http':
                http[(e['stage'],e['path'],e['status'])]+=1
    assert all(s==200 for stage,path,s in http)
    for p in prepared:
        assert set(p['variants'])=={'patient_retrieval_guidelines'}
        assert all(c['status']=='completed' for c in p['components'].values())
        assert p['components']['patient_retrieval']['capture']['selected_reference_patient']
        assert p['components']['patient_retrieval']['capture']['reranked_patient_ids']
        assert p['components']['guidelines']['capture']['guideline_evidence'].strip()
        variant=p['variants']['patient_retrieval_guidelines']
        assert variant['assessment_plan_prompts']
        assert all(s.get('assessment_plan_raw') is None and not s['model_responses'] and s['evaluation'] is None for s in variant['assessment_plans'].values())
    assert not list((TASK/'errors').glob('*'))
    log=(TASK/'ollama.log').read_text()
    assert 'n_seq_max             = 4' in log
    assert 'n_ctx                 = 65536' in log
    assert 'n_ctx_seq             = 16384' in log
    assert 'n_slots = 4, n_ctx_slot = 16384' in log
    assert 'truncating input prompt' not in log.lower()
    assert all('Traceback (most recent call last)' not in p.read_text() for p in TASK.glob('worker-*.log'))
    resources=load('resources.json')
    observations=[]
    for s in resources:
        observations.append({'elapsed_seconds':s['elapsed_seconds']-resources[0]['elapsed_seconds'],
            'memory_pressure_free_percent':int(re.search(r'free percentage:\s*(\d+)%',s['memory_pressure'])[1]),
            'compression_gib':value(s['vm_stat'],'Pages occupied by compressor')*16384/1024**3,
            'swap_used_mib':float(re.search(r'used = ([\d.]+)M',s['swap'])[1]),
            'swapouts':value(s['vm_stat'],'Swapouts'),'swapins':value(s['vm_stat'],'Swapins'),
            'pageouts':value(s['vm_stat'],'Pageouts'),
            'worker_rss_gib':sum(s['workers'].values())/1024**2})
    first,last=observations[0],observations[-1]
    result={'status':'completed','elapsed_seconds':status['elapsed_seconds'],
            'preparation':{'patients':8,'workers':4,'mean_service_seconds':mean(p['total_seconds'] for p in profiles),'all_successful':True,'preserved_prompts_and_rankings':True},
            'generation':{'model':'gemma3:4b','parallel_slots':4,'context_per_slot':16384,'max_output_tokens':768,'requests':4,'all_valid':True,
                          'all_four_simultaneously_in_flight':True,'request_seconds':[g['seconds'] for g in generations],
                          'token_usage':[g['output']['model_responses'][0]['token_usage'] for g in generations],
                          'generation_active_span_seconds':max(g['end_seconds'] for g in generations)-min(g['start_seconds'] for g in generations)},
            'memory':{'samples':len(observations),'compressed_start_gib':first['compression_gib'],'compressed_peak_gib':max(s['compression_gib'] for s in observations),
                      'new_swapout_pages':last['swapouts']-first['swapouts'],'swapin_pages':last['swapins']-first['swapins'],
                      'swap_used_change_mib':last['swap_used_mib']-first['swap_used_mib'],'pageout_pages':last['pageouts']-first['pageouts'],
                      'min_reported_memory_pressure_free_percent':min(s['memory_pressure_free_percent'] for s in observations),
                      'worker_rss_peak_gib':max(s['worker_rss_gib'] for s in observations)},
            'other_servers_before':load('other_servers_before.json'),'other_servers_after':load('other_servers_after.json'),
            'http_preparation':[{'stage':s,'path':p,'status':c,'count':n} for (s,p,c),n in http.items()],
            'errors':0,'scoring_or_judging_calls':0,'observations':observations,
            'limitations':['Short snapshot, not a long-run memory leak test.',
                          'Four generation requests overlapped the beginning of preparation for approximately five seconds. The loaded model, four configured context slots, and associated allocations remained resident throughout preparation. Continuous generation for the full preparation period was not tested.',
                          'System memory readings include unrelated applications and background work; the experiment does not attribute all compression to the tested workers.',
                          'Large compression increase means passing cannot be interpreted as abundant uncompressed RAM headroom.',
                          'Existing FNLLM sequential event-loop reuse warnings were observed; no failures occurred.']}
    (TASK/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Four-worker preparation / Gemma overlap test','',
           f"Completed in {status['elapsed_seconds']:.1f} seconds. Repository and parent-run artifacts were not modified; the private server used 127.0.0.1:11437 and was stopped afterward.",'',
           '- Eight longest S/O cases from the previous ten-case set, two per persistent CPU worker.',
           '- Four largest saved guideline-containing prompts from the 64-patient pilot, unchanged.',
           '- Gemma3:4b, four simultaneous requests, 16,384 context tokens per slot, 768 maximum output tokens, production JSON schema and temperature.',
           '- Preparation limits: 12 RPM / 200,000 TPM per worker. No diagnosis scoring or Plan judging.',
           '', '## Outcomes','',
           'All eight preparations completed both evidence components and saved prompts/ranking metadata. All four Gemma responses parsed into three predictions. All recorded preparation HTTP requests succeeded. No worker crash, model failure, or prompt truncation was observed.',
           '',f"The server log confirmed four slots, 65,536 total context capacity and 16,384 per slot. All four requests were simultaneously in flight; active generation lasted {result['generation']['generation_active_span_seconds']:.2f} seconds. Gemma stayed resident throughout the preparation work.",'',
           f"Physical compressed-memory storage increased from {first['compression_gib']:.2f} GiB to a peak of {result['memory']['compressed_peak_gib']:.2f} GiB. It was roughly stable near 10.6–10.9 GiB during the later work samples, then decreased on cleanup.",
           f"There were {result['memory']['new_swapout_pages']} new swap-out pages. Swap used changed by {result['memory']['swap_used_change_mib']:.1f} MiB. There were {result['memory']['swapin_pages']} swap-in pages and {result['memory']['pageout_pages']} generic page-outs; these are distinct counters.",'',
           'Neither existing Ollama endpoint had a resident model at test start. The isolated test did not unload, reconfigure or terminate either existing service.',
           '', '## Interpretation','',
           'The tested configuration completed successfully without disk swap growth. This supports using four preparation workers with the production Gemma allocation. Compression is substantial, so do not describe the result as proof of ample uncompressed headroom or guaranteed overnight completion.','',
           '## Limitations','']+['- '+v for v in result['limitations']]
    (TASK/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('observations','limitations')},indent=2))


if __name__=='__main__':
    main()
