"""Run ten preparation jobs in two isolated temporary workers; never generate/score."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from threading import Barrier, Event, Lock, Thread

TASK = Path(__file__).resolve().parent
SNAPSHOT = TASK/'snapshot'
PATIENTS = [json.loads(x) for x in (TASK/'patients.jsonl').read_text().splitlines() if x.strip()]
BARRIER=Barrier(4)
PROCS={}
PROCS_LOCK=Lock()
STOP=Event()
RESOURCES=[]
ENV = {**os.environ, 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
       'LITELLM_LOCAL_MODEL_COST_MAP':'True','MPLCONFIGDIR':str(TASK/'matplotlib'),
       'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
       'PYTHONDONTWRITEBYTECODE':'1'}


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as handle:
        json.dump(value,handle,ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def run_worker(number):
    start = perf_counter()
    ready = None
    completed = []
    log = (TASK/f'worker-{number}.log').open('w')
    proc = subprocess.Popen([sys.executable,str(SNAPSHOT/'scripts/med_copilot_worker.py'),
                             str(TASK/'config.yaml'),'--device','cpu','--threads','2'],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                            text=True,bufsize=1,cwd=SNAPSHOT,env=ENV)
    with PROCS_LOCK:
        PROCS[number]=proc
    def send(value):
        proc.stdin.write(json.dumps(value)+'\n')
        proc.stdin.flush()
    def receive():
        line=proc.stdout.readline()
        if not line:
            raise RuntimeError(f'Worker {number} ended; inspect its temporary log')
        return json.loads(line)
    try:
        event=receive()
        if event.get('type')!='ready':
            raise RuntimeError(f'Worker {number} initialization failed: {event.get("error")}')
        ready=perf_counter()-start
        print(json.dumps({'worker':number,'loading_and_initialization_seconds':ready}),flush=True)
        BARRIER.wait(timeout=90)
        processing_start=perf_counter()
        for patient in PATIENTS[number::4]:
            # Import-free equivalent fresh result construction is delegated to
            # the snapshot helper via a prebuilt, validated template on disk.
            identifier=hashlib.sha256(patient['id'].encode()).hexdigest()[:24]
            result=json.loads((TASK/'fresh'/f'{identifier}.json').read_text())
            send({'type':'prepare','patient':patient,'result':result})
            while True:
                event=receive()
                kind=event.get('type')
                if kind=='checkpoint':
                    atomic(TASK/'checkpoints'/f'{identifier}.json',event['result'])
                    send({'type':'ack'})
                elif kind=='error':
                    atomic(TASK/'errors'/f'{identifier}.json',event['error'])
                    send({'type':'ack'})
                elif kind=='result':
                    result=event['result']
                    if any(x['status']!='completed' for x in result['components'].values()):
                        raise RuntimeError(f'Worker {number} preparation component failed; see temporary artifacts')
                    # Safety check: no final generations or evaluations are permitted.
                    for variant in result['variants'].values():
                        for slot in variant['assessment_plans'].values():
                            if slot.get('assessment_plan_raw') is not None or slot.get('model_responses') or slot.get('evaluation') is not None:
                                raise RuntimeError('Unexpected final-generation/scoring work')
                    atomic(TASK/'prepared'/f'{identifier}.json',result)
                    completed.append({'patient_id':patient['id'],'seconds':event['duration_seconds'],'finished_seconds_since_processing_start':perf_counter()-processing_start})
                    print(json.dumps({'worker':number,'completed_jobs_in_worker':len(completed),
                                      'seconds':event['duration_seconds']}),flush=True)
                    break
                else:
                    raise RuntimeError(f'Unexpected worker message: {kind}')
        send({'type':'shutdown'})
        proc.wait(timeout=20)
        if proc.returncode:
            raise RuntimeError(f'Worker {number} exit {proc.returncode}')
        return {'worker':number,'initialization_seconds':ready,'processing_wall_seconds':perf_counter()-processing_start,'completed':completed}
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=20)
        proc.stdin.close()
        proc.stdout.close()
        log.close()


def main():
    for name in ('checkpoints','prepared','errors'):
        (TASK/name).mkdir(exist_ok=True)
    start=perf_counter()
    atomic(TASK/'status.json',{'status':'running','workers':4,'patients':8})
    try:
        monitor=Thread(target=monitor_resources,daemon=True)
        monitor.start()
        with ThreadPoolExecutor(max_workers=4) as pool:
            workers=list(pool.map(run_worker,(0,1,2,3)))
        STOP.set()
        monitor.join(timeout=6)
        atomic(TASK/'resources.json',RESOURCES)
        result={'status':'completed','wall_seconds_including_loading':perf_counter()-start,'workers':workers}
        atomic(TASK/'status.json',result)
        print(json.dumps({'status':'completed','wall_seconds_including_loading':result['wall_seconds_including_loading']}),flush=True)
    except BaseException as exc:
        atomic(TASK/'status.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc)})
        raise


def monitor_resources():
    while not STOP.is_set():
        with PROCS_LOCK:
            active={n:p.pid for n,p in PROCS.items() if p.poll() is None}
        sample={'elapsed_seconds':perf_counter(),'workers':{}}
        for n,pid in active.items():
            output=subprocess.run(['ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,timeout=3).stdout.strip()
            if output:
                sample['workers'][str(n)]=int(output)
        sample['swap']=subprocess.run(['sysctl','-n','vm.swapusage'],capture_output=True,text=True,timeout=3).stdout.strip()
        RESOURCES.append(sample)
        STOP.wait(3)


if __name__=='__main__':
    main()
