"""Bounded two-worker limiter comparison plus isolated ten-case validation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import yaml

TASK=Path(__file__).resolve().parent
BASE=Path('/private/tmp/medcopilot-preparation-profile-qlspfgvq')
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
START=perf_counter()


def atomic(name,data):
    (TASK/name).write_text(json.dumps(data,indent=2,default=str))


def child(label,rpm,tpm,indices):
    with (TASK/(label+'.log')).open('w') as log:
        proc=subprocess.run([sys.executable,'-B',str(TASK/'query_worker.py'),label,str(rpm),str(tpm),indices,'--prefill'],stdout=log,stderr=subprocess.STDOUT,env=ENV,timeout=100)
    if proc.returncode:
        raise RuntimeError(f'{label} failed; see temporary log')
    return json.loads((TASK/(label+'.json')).read_text())


def arm(label,rpm,tpm,reverse=False):
    start=perf_counter()
    indices=['0,5','2,8'] if not reverse else ['5,0','8,2']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(child,f'{label}_w{i}',rpm,tpm,idx) for i,idx in enumerate(indices)]
        workers=[f.result() for f in futures]
    rows=[row for w in workers for row in w['rows']]
    # Exclude Python import costs; each child measures from before its first query.
    service_wall=max(w['wall_seconds'] for w in workers)
    events=[e for r in rows for e in r['events']]
    record={'label':label,'rpm':rpm,'tpm':tpm,'wall_seconds_including_imports':perf_counter()-start,'service_wall_seconds':service_wall,
            'queries':len(rows),'valid_answers':sum(r['answer_nonempty'] for r in rows),
            'rpm_wait_mean':sum(e['seconds'] for e in events if e['type']=='rpm_wait')/len(rows),
            'tpm_wait_mean':sum(e['seconds'] for e in events if e['type']=='tpm_wait')/len(rows),
            'http_statuses':[e['status'] for e in events if e['type']=='http'],
            'finish_reasons':[f for e in events if e['type']=='http' for f in e['finish_reasons']],
            'workers':workers}
    record['valid_per_minute']=60*record['valid_answers']/service_wall
    record['accepted']=record['valid_answers']==4 and all(s==200 for s in record['http_statuses']) and len(record['http_statuses'])==8 and record['finish_reasons']==['stop']*4
    atomic(label+'.summary.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='workers'}),flush=True)
    return record


def validation(selected):
    dest=TASK/'validation'
    dest.mkdir()
    shutil.copytree(BASE/'snapshot/scripts',dest/'snapshot/scripts')
    (dest/'snapshot/med_copilot').symlink_to(BASE/'snapshot/med_copilot',target_is_directory=True)
    for name in ('patients.jsonl','selection.json','condition_plan_models.yaml','source_hashes.json'):
        shutil.copy2(BASE/name,dest/name)
    (dest/'fresh').symlink_to(BASE/'fresh',target_is_directory=True)
    config=yaml.safe_load((BASE/'config.yaml').read_text())
    (dest/'guidelines').mkdir()
    for entry in (BASE/'guidelines').iterdir():
        if entry.name!='settings.yaml':
            (dest/'guidelines'/entry.name).symlink_to(entry,target_is_directory=entry.is_dir())
    guide=yaml.safe_load((BASE/'guidelines/settings.yaml').read_text())
    guide['models']['default_chat_model'].update(requests_per_minute=selected['rpm'],tokens_per_minute=selected['tpm'])
    (dest/'guidelines/settings.yaml').write_text(yaml.safe_dump(guide,sort_keys=False))
    config.update(guidelines=str(dest/'guidelines'),patient_input=str(dest/'patients.jsonl'),experiment_output=str(dest/'unused-experiment-output'))
    (dest/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for name in ('prep_profile.py','run_profile.py','summarize.py'):
        (dest/name).write_text((BASE/name).read_text())
    worker=dest/'snapshot/scripts/med_copilot_worker.py'
    worker.write_text(worker.read_text().replace(str(BASE),str(dest)))
    start=perf_counter()
    with (dest/'run.log').open('w') as log:
        proc=subprocess.run([sys.executable,'-B',str(dest/'run_profile.py')],stdout=log,stderr=subprocess.STDOUT,env=ENV,timeout=150)
    if proc.returncode:
        raise RuntimeError('Ten-case validation failed; see validation/run.log')
    subprocess.run([sys.executable,'-B',str(dest/'summarize.py')],stdout=(dest/'summary.log').open('w'),stderr=subprocess.STDOUT,env=ENV,check=True,timeout=10)
    summary=json.loads((dest/'summary.json').read_text())
    summary['settings']={'rpm':selected['rpm'],'tpm':selected['tpm']}
    summary['wall_including_validation_controller']=perf_counter()-start
    return summary


def main():
    probe=json.loads((TASK/'quota_probe.json').read_text())
    headers=next(e['rate_headers'] for e in probe['rows'][0]['events'] if e['type']=='http' and e['path'].endswith('chat/completions'))
    quota_rpm=int(headers['x-ratelimit-limit-requests'])
    quota_tpm=int(headers['x-ratelimit-limit-tokens'])
    high_rpm=min(12,quota_rpm//4)
    high_tpm=min(200000,quota_tpm//4)
    low_rpm=max(1,high_rpm//2)
    low_tpm=high_tpm//2
    if high_tpm<30000 or low_rpm==high_rpm:
        raise RuntimeError('Actual quota cannot support the planned bounded factorial safely')
    design={'quota_rpm':quota_rpm,'quota_tpm':quota_tpm,'per_worker_rpm':[low_rpm,high_rpm],'per_worker_tpm':[low_tpm,high_tpm],
            'workers':2,'reserved_capacity_fraction':0.5,'indices':[0,2,5,8],'scope':'Four same saved queries, 2x2 factorial, exhausted initial local buckets; then reverse-order repeat and ten-case full preparation validation'}
    atomic('design.json',design)
    print(json.dumps({'design':design}),flush=True)
    records=[]
    # Alternate TPM while balancing RPM; comparison repeat uses reverse query order.
    for label,rpm,tpm in [('A',low_rpm,low_tpm),('D',high_rpm,high_tpm),('B',high_rpm,low_tpm),('C',low_rpm,high_tpm)]:
        if perf_counter()-START>290:
            raise RuntimeError('Comparison exceeded allotted budget')
        records.append(arm(label,rpm,tpm))
        atomic('progress.json',{'status':'comparison','arms_completed':len(records),'elapsed_seconds':perf_counter()-START})
    eligible=sorted([r for r in records if r['accepted']],key=lambda r:-r['valid_per_minute'])
    if len(eligible)<2:
        raise RuntimeError('Fewer than two settings passed integrity/error checks')
    repeats=[arm(r['label']+'_repeat',r['rpm'],r['tpm'],True) for r in eligible[:2]]
    candidates=[]
    for original,repeat in zip(eligible[:2],repeats):
        if repeat['accepted']:
            seconds=original['service_wall_seconds']+repeat['service_wall_seconds']
            candidates.append({'label':original['label'],'rpm':original['rpm'],'tpm':original['tpm'],'pooled_valid_per_minute':480/seconds})
    if not candidates:
        raise RuntimeError('No repeated condition passed')
    best=max(r['pooled_valid_per_minute'] for r in candidates)
    # Within five percent choose lower token allowance then lower request allowance.
    selected=min([r for r in candidates if r['pooled_valid_per_minute']>=.95*best],key=lambda r:(r['tpm'],r['rpm']))
    atomic('selected.json',selected)
    print(json.dumps({'selected':selected}),flush=True)
    result=validation(selected)
    atomic('results.json',{'status':'completed','design':design,'comparisons':[{k:v for k,v in r.items() if k!='workers'} for r in records+repeats],
                         'selected':selected,'validation':result,'elapsed_seconds':perf_counter()-START})
    print(json.dumps({'completed':True,'elapsed_seconds':perf_counter()-START,'validation_wall':result['wall_seconds_including_loading'],'preparation_seconds_mean':result['mean_preparation_seconds']}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        atomic('failure.json',{'type':type(exc).__name__,'message':str(exc),'elapsed_seconds':perf_counter()-START})
        raise
