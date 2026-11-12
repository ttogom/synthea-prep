"""Isolated GraphRAG timing process; no source repository writes."""
import argparse
from contextlib import redirect_stdout
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

BASE=Path('/private/tmp/medcopilot-preparation-profile-qlspfgvq')
TASK=Path(__file__).resolve().parent
# Snapshot imports and credentials precede API client construction.
sys.path[:0]=[str(BASE/'snapshot'),str(BASE/'snapshot/med_copilot/upstream')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LITELLM_LOCAL_MODEL_COST_MAP='True',MPLCONFIGDIR=str(TASK/'matplotlib'),PYTHONDONTWRITEBYTECODE='1')
from dotenv import load_dotenv
load_dotenv('/Users/rasikaramanan/Documents/usc/by_semester/f26/csci566/project/synthea-prep/.env',override=False)
import httpx
import tiktoken
import yaml
from fnllm.limiting.rpm import RPMLimiter
from fnllm.limiting.tpm import TPMLimiter
from graphrag.cli import query as cli
from scripts.med_copilot_worker_stream import RedactingStream

parser=argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('rpm',type=int)
parser.add_argument('tpm',type=int)
parser.add_argument('indices')
parser.add_argument('--prefill',action='store_true')
args=parser.parse_args()
sys.stderr=RedactingStream(sys.stderr,[os.environ.get(k) for k in ('OPENAI_API_KEY','GRAPHRAG_API_KEY','HF_TOKEN')])
EVENTS=[]
ORIGINAL_SEND=httpx.AsyncClient.send
ORIGINAL_RPM=RPMLimiter.acquire
ORIGINAL_TPM=TPMLimiter.acquire
ENC=tiktoken.encoding_for_model('gpt-4o-mini')


def acquire_wrapper(original,name):
    @wraps(original)
    async def acquire(self,manifest):
        if args.prefill and not getattr(self,'_tuning_prefilled',False):
            await self._limiter.acquire(self._limiter.max_rate)
            self._tuning_prefilled=True
        start=perf_counter()
        try:
            return await original(self,manifest)
        finally:
            EVENTS.append({'type':name,'seconds':perf_counter()-start,'input_tokens':manifest.request_tokens,'post_tokens':manifest.post_request_tokens})
    return acquire


async def send(self,request,*a,**kw):
    start=perf_counter()
    body=json.loads(request.content)
    response=await ORIGINAL_SEND(self,request,*a,**kw)
    event={'type':'http','path':request.url.path,'status':response.status_code,'headers_seconds':perf_counter()-start,
           'rate_headers':{k:v for k,v in response.headers.items() if k.startswith('x-ratelimit-')},
           'request_sha256':hashlib.sha256(request.content).hexdigest(),'model':body.get('model'),
           'max_tokens':body.get('max_tokens'),'max_completion_tokens':body.get('max_completion_tokens'),
           'input_tokens_text_estimate':len(ENC.encode(json.dumps(body.get('messages',body.get('input'))))),
           'finish_reasons':[],'usage':None}
    EVENTS.append(event)
    chunks=[]
    original=response.aiter_raw
    async def raw(*x,**y):
        try:
            async for chunk in original(*x,**y):
                chunks.append(chunk)
                yield chunk
        finally:
            event['seconds']=perf_counter()-start
            for line in b''.join(chunks).decode('utf-8',errors='replace').splitlines():
                if line.startswith('data: ') and line[6:]!='[DONE]':
                    try:
                        data=json.loads(line[6:])
                        for choice in data.get('choices',[]):
                            if choice.get('finish_reason'):
                                event['finish_reasons'].append(choice['finish_reason'])
                        if data.get('usage'):
                            event['usage']=data['usage']
                    except ValueError:
                        pass
    if kw.get('stream'):
        response.aiter_raw=raw
    else:
        event['seconds']=perf_counter()-start
    return response


httpx.AsyncClient.send=send
RPMLimiter.acquire=acquire_wrapper(ORIGINAL_RPM,'rpm_wait')
TPMLimiter.acquire=acquire_wrapper(ORIGINAL_TPM,'tpm_wait')
config=yaml.safe_load((BASE/'guidelines/settings.yaml').read_text())
config['models']['default_chat_model'].update(requests_per_minute=args.rpm,tokens_per_minute=args.tpm)
configpath=TASK/(args.label+'.yaml')
configpath.write_text(yaml.safe_dump(config,sort_keys=False))
patients=[json.loads(s) for s in (BASE/'patients.jsonl').read_text().splitlines()]
rows=[]
start_all=perf_counter()
# The saved JSONL is shuffled relative to selection.json. Correct confirmation
# to cover all four strata: low=7, moderate_low=2, moderate_high=0, high=8.
indices=list(map(int,args.indices.split(',')))
if '_repeat_' in args.label:
    indices=[7 if i==5 else i for i in indices]
for index in indices:
    patient=patients[index]
    identifier=hashlib.sha256(patient['id'].encode()).hexdigest()[:24]
    saved=json.loads((BASE/'prepared'/f'{identifier}.json').read_text())
    question=saved['components']['guidelines']['capture']['guideline_query']
    EVENTS.clear()
    start=perf_counter()
    with redirect_stdout(sys.stderr):
        answer,context=cli.run_local_search(config_filepath=configpath,data_dir=BASE/'guidelines/output',root_dir=BASE/'guidelines',community_level=2,response_type='Multiple Paragraphs',streaming=False,query=question,verbose=False)
    row={'patient_id':patient['id'],'index':index,'seconds':perf_counter()-start,'answer':answer,
         'query_sha256':hashlib.sha256(question.encode()).hexdigest(),'events':list(EVENTS),
         'context':{k:v.to_dict(orient='records') if hasattr(v,'to_dict') else v for k,v in context.items()},
         'answer_nonempty':isinstance(answer,str) and bool(answer.strip())}
    rows.append(row)
    (TASK/(args.label+'.json')).write_text(json.dumps({'label':args.label,'rpm':args.rpm,'tpm':args.tpm,'prefilled':args.prefill,'wall_seconds':perf_counter()-start_all,'rows':rows},indent=2,default=str))
    print(json.dumps({'label':args.label,'index':index,'seconds':row['seconds'],'ok':row['answer_nonempty']}),flush=True)
sys.stderr.finish()
