"""Scoped observation of the configured GraphRAG FNLLM OpenAI/Azure providers.

FNLLM hides SDK streaming chunks and embedding metadata. Observe those values
before its adapters discard them, including raw models returned from its cache.
No model or client instance is mutated; ContextVars also cross GraphRAG's
run_coroutine_threadsafe boundary. Hooks are restored even on interruption.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from unittest.mock import patch

from fnllm.openai.llm.openai_embeddings_llm import OpenAIEmbeddingsLLMImpl
from fnllm.openai.llm.openai_streaming_chat_llm import OpenAIStreamingChatLLMImpl
from fnllm.openai.llm.openai_text_chat_llm import OpenAITextChatLLMImpl

from scripts.med_copilot_helpers import model_response_metadata
from scripts.med_copilot_run_file import RunMetadataError, checkpoint


_sink = ContextVar("med_copilot_response_sink", default=None)
_seen = ContextVar("med_copilot_response_seen", default=None)
_checkpoint_errors = ContextVar("med_copilot_response_checkpoint_errors", default=None)
_lock = RLock()
_users = 0
_hooks = None


def _save_usage():
    try:
        checkpoint()
    except Exception as exc:
        error = exc if isinstance(exc, RunMetadataError) else RunMetadataError("Could not save GraphRAG usage")
        errors = _checkpoint_errors.get()
        if errors is not None:
            errors.append(error)
        if error is exc:
            raise
        raise error from exc


def _record(response, sink, stage, *, cache_hit=False):
    seen = _seen.get()
    if seen is not None:
        if any(response is previous for previous in seen):
            return
        seen.append(response)
    metadata = model_response_metadata(response)
    if cache_hit:
        metadata["cache_hit"] = True
    sink[stage].append(metadata)
    _save_usage()


class _Proxy:
    def __init__(self, target, **overrides):
        self._target = target
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._target, name)


class _RawResponse(_Proxy):
    def __init__(self, target, sink, stage):
        super().__init__(target)
        self._sink, self._stage = sink, stage

    def parse(self):
        response = self._target.parse()
        _record(response, self._sink, self._stage)
        return response


class _Stream(_Proxy):
    def __init__(self, target, sink, stage):
        super().__init__(target)
        self._sink, self._stage = sink, stage
        self._entry = None

    async def __aiter__(self):
        async for chunk in self._target:
            metadata = model_response_metadata(chunk)
            changed = False
            if self._entry is None:
                self._entry = metadata
                self._sink[self._stage].append(self._entry)
                changed = True
            else:
                for key in ("reported_model", "response_id", "request_id"):
                    if metadata.get(key) is not None and self._entry.get(key) != metadata[key]:
                        self._entry[key] = metadata[key]
                        changed = True
                if metadata["token_usage"] is not None:
                    if self._entry["token_usage"] is None:
                        self._entry["token_usage"] = metadata["token_usage"]
                        changed = True
                    else:
                        for key, count in metadata["token_usage"].items():
                            if count is not None and self._entry["token_usage"].get(key) != count:
                                self._entry["token_usage"][key] = count
                                changed = True
            if changed:
                _save_usage()
            # Pass the original chunk through unchanged, including usage events.
            yield chunk


class _Resource(_Proxy):
    def __init__(self, target, sink, stage, raw=False):
        super().__init__(target)
        self._sink, self._stage, self._raw = sink, stage, raw

    @property
    def with_raw_response(self):
        return _Resource(self._target.with_raw_response, self._sink, self._stage, raw=True)

    async def create(self, *args, **kwargs):
        if kwargs.get("stream"):
            kwargs["stream_options"] = {**(kwargs.get("stream_options") or {}),
                                        "include_usage": True}
        response = await self._target.create(*args, **kwargs)
        if self._raw:
            return _RawResponse(response, self._sink, self._stage)
        if kwargs.get("stream"):
            return _Stream(response, self._sink, self._stage)
        _record(response, self._sink, self._stage)
        return response


def _execute_hook(original, stage):
    @wraps(original)
    async def execute(self, *args, **kwargs):
        sink = _sink.get()
        if sink is None:
            return await original(self, *args, **kwargs)
        client = self._client
        if stage == "guideline_query_embedding":
            client = _Proxy(client, embeddings=_Resource(client.embeddings, sink, stage))
        else:
            client = _Proxy(client, chat=_Proxy(
                client.chat, completions=_Resource(client.chat.completions, sink, stage)))
        return await original(_Proxy(self, _client=client), *args, **kwargs)
    return execute


def _call_hook(original, stage):
    @wraps(original)
    async def call(self, *args, **kwargs):
        sink = _sink.get()
        if sink is None:
            return await original(self, *args, **kwargs)
        token = _seen.set([])
        try:
            response = await original(self, *args, **kwargs)
            raw = getattr(response.output, "raw_model", None)
            if raw is not None:
                _record(raw, sink, stage, cache_hit=getattr(response, "cache_hit", False) is True)
            return response
        finally:
            _seen.reset(token)
    return call


@contextmanager
def capture_graphrag_responses(model_responses):
    """Append provider metadata to this patient's lists during GraphRAG calls.

    Scoped class hooks cover both newly created and ModelManager-cached FNLLM
    instances. Unrelated contexts pass through unchanged. Stream chunks update
    one response entry rather than adding entries or summing repeated usage.
    """
    global _users, _hooks
    with _lock:
        if _users == 0:
            _hooks = ExitStack()
            try:
                for cls, stage in (
                    (OpenAIEmbeddingsLLMImpl, "guideline_query_embedding"),
                    (OpenAIStreamingChatLLMImpl, "guideline_evidence"),
                    (OpenAITextChatLLMImpl, "guideline_evidence"),
                ):
                    _hooks.enter_context(patch.object(
                        cls, "_execute_llm", _execute_hook(cls._execute_llm, stage)))
                    _hooks.enter_context(patch.object(
                        cls, "__call__", _call_hook(cls.__call__, stage)))
            except BaseException:
                _hooks.close()
                raise
        _users += 1
    token = _sink.set(model_responses)
    errors = []
    errors_token = _checkpoint_errors.set(errors)
    try:
        yield
    finally:
        _checkpoint_errors.reset(errors_token)
        _sink.reset(token)
        with _lock:
            _users -= 1
            if _users == 0:
                _hooks.close()
                _hooks = None
        # GraphRAG may swallow or replace provider exceptions. A failed durable
        # usage write must remain fatal after restoring all scoped hooks.
        if errors:
            raise errors[0]
