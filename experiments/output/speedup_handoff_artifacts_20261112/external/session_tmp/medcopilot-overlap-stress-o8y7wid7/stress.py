"""Run ten preparation jobs in two isolated temporary workers; never generate/score."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from urllib.request import Request, urlopen
import re
from threading import Barrier, Event, Lock, Thread

TASK = Path(__file__).resolve().parent
SNAPSHOT = TASK/'snapshot'
# Use the frozen snapshot, not source files under active parent-session edits.
sys.path[:0]=[str(SNAPSHOT),str(SNAPSHOT/'med_copilot/upstream')]
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP','True')
from scripts.med_copilot_ollama import ManagedOllama
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.med_copilot_helpers import new_model_result, parse_condition_plan
PATIENTS = [json.loads(x) for x in (TASK/'patients.jsonl').read_text().splitlines() if x.strip()]
BARRIER=Barrier(5)
HOLD=Event()
GENERATION=[]
SERVER=None
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
        HOLD.wait(timeout=150)
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


def prep_main():
    for name in ('checkpoints','prepared','errors'):
        (TASK/name).mkdir(exist_ok=True)
    start=perf_counter()
    atomic(TASK/'status.json',{'status':'running','workers':4,'patients':8})
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            generation=pool.submit(run_generations)
            workers=list(pool.map(run_worker,(0,1,2,3)))
            generation.result()
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
        sample['vm_stat']=subprocess.run(['vm_stat'],capture_output=True,text=True,timeout=3).stdout
        sample['memory_pressure']=subprocess.run(['memory_pressure','-Q'],capture_output=True,text=True,timeout=3).stdout
        try:
            sample['own_ollama']=SERVER.ps_snapshot() if SERVER is not None and SERVER._process is not None else None
        except Exception as exc:
            sample['own_ollama_error']=type(exc).__name__
        for n,pid in active.items():
            output=subprocess.run(['ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,timeout=3).stdout.strip()
            if output:
                sample['workers'][str(n)]=int(output)
        sample['swap']=subprocess.run(['sysctl','-n','vm.swapusage'],capture_output=True,text=True,timeout=3).stdout.strip()
        RESOURCES.append(sample)
        STOP.wait(2)



def read_other_servers():
    result={}
    for port in (11434,11435):
        try:
            with urlopen(f'http://127.0.0.1:{port}/api/ps',timeout=2) as response:
                result[str(port)]=json.load(response)
        except Exception as exc:
            result[str(port)]={'unavailable':type(exc).__name__}
    return result


def generate_one(models,prompt):
    started=perf_counter()
    output=new_model_result()
    models.generate_one('ollama_gemma3_4b',prompt['messages'],output)
    parsed=parse_condition_plan(output.get('assessment_plan_raw'))
    predictions=parsed.get('predictions')
    valid=output['status']=='completed' and isinstance(predictions,list) and len(predictions)==3
    record={k:v for k,v in prompt.items() if k!='messages'}
    record.update(start_seconds=started,end_seconds=perf_counter(),seconds=perf_counter()-started,valid=valid,output=output)
    return record


def run_generations():
    try:
        models=AssessmentPlanModels(TASK/'condition_plan_models.yaml',['ollama_gemma3_4b'],temperature=1,max_retries=3)
        BARRIER.wait(timeout=90)
        atomic(TASK/'generation_status.json',{'status':'running','start_seconds':perf_counter()})
        prompts=json.loads((TASK/'generation_prompts.json').read_text())
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(generate_one,models,p) for p in prompts]
            for future in futures:
                record=future.result()
                GENERATION.append(record)
                atomic(TASK/'generations.json',GENERATION)
                print(json.dumps({'generation_completed':len(GENERATION),'seconds':record['seconds'],'valid':record['valid']}),flush=True)
        atomic(TASK/'generation_status.json',{'status':'completed','end_seconds':perf_counter(),'valid':sum(r['valid'] for r in GENERATION)})
        if not all(r['valid'] for r in GENERATION):
            raise RuntimeError('A generation did not return three valid predictions')
    finally:
        HOLD.set()


def main():
    global SERVER
    started=perf_counter()
    atomic(TASK/'other_servers_before.json',read_other_servers())
    print(json.dumps({'other_servers_before':read_other_servers()}),flush=True)
    monitor=Thread(target=monitor_resources,daemon=True)
    monitor.start()
    settings={'manage_ollama':True,'ollama_host':'127.0.0.1:11437','ollama_parallel_slots':4,'ollama_context_length':16384}
    try:
        with ManagedOllama(settings,TASK/'ollama.log',parallel=4,context=16384,max_loaded_models=1) as server:
            SERVER=server
            atomic(TASK/'server_metadata.json',server.metadata())
            # Empty native request loads the model and production-sized KV slots
            # without generating an extra answer. All four timed requests use
            # unchanged saved experiment prompts and production output settings.
            loaded=server._request('POST','/api/generate',{'model':'gemma3:4b','stream':False,'keep_alive':-1},timeout=90)
            atomic(TASK/'model_load.json',{'response':loaded,'ps':server.ps_snapshot()})
            print(json.dumps({'model_loaded':True,'ps':server.ps_snapshot()}),flush=True)
            prep_main()
            atomic(TASK/'owned_server_final_ps.json',server.ps_snapshot())
        atomic(TASK/'overall_status.json',{'status':'completed','elapsed_seconds':perf_counter()-started,'preparation_cases':8,'generation_requests':4,'valid_generations':sum(r['valid'] for r in GENERATION)})
    except BaseException as exc:
        HOLD.set()
        BARRIER.abort()
        with PROCS_LOCK:
            for proc in PROCS.values():
                if proc.poll() is None:
                    proc.terminate()
        atomic(TASK/'overall_status.json',{'status':'failed','elapsed_seconds':perf_counter()-started,'error_type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        STOP.set()
        monitor.join(timeout=10)
        atomic(TASK/'resources.json',RESOURCES)
        atomic(TASK/'other_servers_after.json',read_other_servers())


if __name__=='__main__':
    main()
