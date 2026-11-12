"""Temporary timing wrappers; no prompts, model settings or return values changed."""
from collections import defaultdict
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import httpx
from fnllm.limiting.base import LimitContext
from fnllm.limiting.rpm import RPMLimiter
from fnllm.limiting.tpm import TPMLimiter

TASK = Path(__file__).resolve().parent
EVENTS = []
STACK = []
CURRENT = None


def record(name, start, **details):
    if CURRENT is not None:
        EVENTS.append({'name':name, 'seconds':perf_counter()-start,
                       'stage':STACK[-1] if STACK else None, **details})


def timed(name, original):
    @wraps(original)
    def call(*args, **kwargs):
        start = perf_counter()
        STACK.append(name)
        try:
            return original(*args, **kwargs)
        finally:
            record(name, start)
            STACK.pop()
    return call


def timed_async(name, original):
    @wraps(original)
    async def call(*args, **kwargs):
        start = perf_counter()
        try:
            return await original(*args, **kwargs)
        finally:
            record(name, start)
    return call


def install(worker):
    preparation = sys.modules['scripts.med_copilot_preparation']
    helpers = sys.modules['scripts.med_copilot_helpers']
    retrieval = sys.modules['scripts.med_copilot_retrieval']
    helpers.JSONKeywordExtractor.extract = timed('keywords', helpers.JSONKeywordExtractor.extract)
    retrieval.HybridRetriever.search = timed('patient_search', retrieval.HybridRetriever.search)
    retrieval.CrossEncoderReranker.rerank = timed('patient_rerank', retrieval.CrossEncoderReranker.rerank)
    retrieval.HybridRetriever.compute_case_phrase_contributions = timed(
        'attribution', retrieval.HybridRetriever.compute_case_phrase_contributions)
    preparation.prepare_guideline_query = timed('guideline_questions', preparation.prepare_guideline_query)
    preparation.prepare_variant_messages = timed('prompt_assembly', preparation.prepare_variant_messages)
    worker['run_local_search'] = timed('graphrag', worker['run_local_search'])
    worker['_acknowledged_event'] = timed('checkpoint_ack', worker['_acknowledged_event'])
    # These are nested diagnostics; do not add them to the five top-level stages.
    LimitContext.__aenter__ = timed_async('limiter_total', LimitContext.__aenter__)
    RPMLimiter.acquire = timed_async('limiter_rpm', RPMLimiter.acquire)
    TPMLimiter.acquire = timed_async('limiter_tpm', TPMLimiter.acquire)
    query = sys.modules['graphrag.cli.query']
    for name in ('load_config', '_resolve_output_files'):
        if hasattr(query, name):
            setattr(query, name, timed('graphrag_setup:'+name, getattr(query, name)))
    original_send, original_asend = httpx.Client.send, httpx.AsyncClient.send

    def request_info(request):
        return {'host':request.url.host, 'path':request.url.path}

    @wraps(original_send)
    def send(self, request, *args, **kwargs):
        start = perf_counter()
        response = None
        try:
            response = original_send(self, request, *args, **kwargs)
            return response
        finally:
            record('http', start, **request_info(request), stream=bool(kwargs.get('stream')),
                   status=None if response is None else response.status_code)

    @wraps(original_asend)
    async def asend(self, request, *args, **kwargs):
        start = perf_counter()
        response = None
        try:
            response = await original_asend(self, request, *args, **kwargs)
            return response
        finally:
            record('http', start, **request_info(request), stream=bool(kwargs.get('stream')),
                   status=None if response is None else response.status_code)

    httpx.Client.send, httpx.AsyncClient.send = send, asend
    original_prepare = preparation.EvidencePreparer.prepare

    @wraps(original_prepare)
    def prepare(self, patient, result):
        global CURRENT
        CURRENT = patient['id']
        EVENTS.clear()
        STACK.clear()
        start = perf_counter()
        error = None
        try:
            return original_prepare(self, patient, result)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            seconds = perf_counter()-start
            totals = defaultdict(float)
            for event in EVENTS:
                totals[event['name']] += event['seconds']
            stages = {
                'keyword_extraction':totals['keywords'],
                'patient_retrieval_reranking_attribution':totals['patient_search']-totals['keywords']+totals['patient_rerank'],
                'guideline_question_generation':totals['guideline_questions'],
                'guideline_retrieval':totals['graphrag'],
            }
            stages['prompt_assembly_saving_other_overhead'] = seconds-sum(stages.values())
            output = {'patient_id':CURRENT, 'worker_pid':os.getpid(), 'total_seconds':seconds,
                      'stages':stages,'diagnostics':dict(totals),'events':list(EVENTS),'error_type':error}
            name = hashlib.sha256(CURRENT.encode()).hexdigest()[:24]+'.json'
            destination = TASK/'profiles'/name
            destination.parent.mkdir(exist_ok=True)
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(output,indent=2))
            temporary.replace(destination)
            CURRENT = None
    preparation.EvidencePreparer.prepare = prepare
