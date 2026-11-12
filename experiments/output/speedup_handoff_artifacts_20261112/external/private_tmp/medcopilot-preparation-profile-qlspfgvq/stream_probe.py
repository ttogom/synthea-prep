"""One diagnostic GraphRAG replay of an already sampled patient's saved query."""
from contextlib import redirect_stdout
from functools import wraps
import json
import os
from pathlib import Path
import sys
from time import perf_counter

TASK=Path(__file__).resolve().parent
ROOT=TASK/'snapshot'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'med_copilot/upstream'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LITELLM_LOCAL_MODEL_COST_MAP='True',
                  MPLCONFIGDIR=str(TASK/'matplotlib'),PYTHONDONTWRITEBYTECODE='1')
from dotenv import load_dotenv
load_dotenv('/Users/rasikaramanan/Documents/usc/by_semester/f26/csci566/project/synthea-prep/.env',override=False)
import httpx
from fnllm.limiting.base import LimitContext
from graphrag.cli import query as cli
from graphrag.api import query as api
from scripts.med_copilot_worker_stream import RedactingStream

sys.stderr=RedactingStream(sys.stderr,[os.environ.get(k) for k in ('OPENAI_API_KEY','GRAPHRAG_API_KEY','HF_TOKEN')])
EVENTS=[]
ASEND=httpx.AsyncClient.send
LIMIT_ENTER=LimitContext.__aenter__


def timed(name, fn):
    @wraps(fn)
    def call(*args, **kwargs):
        start=perf_counter()
        try:
            return fn(*args,**kwargs)
        finally:
            EVENTS.append({'step':name,'seconds':perf_counter()-start})
    return call


async def limiter(self):
    start=perf_counter()
    try:
        return await LIMIT_ENTER(self)
    finally:
        EVENTS.append({'step':'limiter','seconds':perf_counter()-start})


async def send(self, request, *args, **kwargs):
    start=perf_counter()
    response=await ASEND(self,request,*args,**kwargs)
    header_seconds=perf_counter()-start
    original=response.aiter_raw
    async def raw(*a,**kw):
        try:
            async for chunk in original(*a,**kw):
                yield chunk
        finally:
            EVENTS.append({'step':'http_full','path':request.url.path,'seconds':perf_counter()-start,
                           'headers_seconds':header_seconds,'stream':bool(kwargs.get('stream')),
                           'status':response.status_code})
    if kwargs.get('stream'):
        response.aiter_raw=raw
    else:
        EVENTS.append({'step':'http_full','path':request.url.path,'seconds':perf_counter()-start,
                       'headers_seconds':header_seconds,'stream':False,'status':response.status_code})
    return response


httpx.AsyncClient.send=send
LimitContext.__aenter__=limiter
cli.load_config=timed('load_config',cli.load_config)
cli._resolve_output_files=timed('load_tables',cli._resolve_output_files)
for name in list(vars(api)):
    if name.startswith('read_indexer_') and callable(getattr(api,name)):
        setattr(api,name,timed(name,getattr(api,name)))
if hasattr(api,'get_local_search_engine'):
    api.get_local_search_engine=timed('build_search_engine',api.get_local_search_engine)

row=json.loads(next((TASK/'prepared').glob('*.json')).read_text())
question=row['components']['guidelines']['capture']['guideline_query']
start=perf_counter()
with redirect_stdout(sys.stderr):
    answer, context=cli.run_local_search(config_filepath=TASK/'guidelines/settings.yaml',
        data_dir=TASK/'guidelines/output',root_dir=TASK/'guidelines',community_level=2,
        response_type='Multiple Paragraphs',streaming=False,query=question,verbose=False)
result={'scope':'One additional guideline-only replay from the ten sampled patients; excluded from ten-case averages',
        'total_seconds':perf_counter()-start,'events':EVENTS}
(TASK/'stream_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
sys.stderr.finish()
