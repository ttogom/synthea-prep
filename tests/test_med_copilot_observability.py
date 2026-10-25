"""Offline integration tests through real FNLLM adapters and fake SDK clients."""

import ast
import asyncio
from contextlib import contextmanager
import logging
from pathlib import Path
import sys
import tempfile
import time
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

# Direct test execution puts tests/ on sys.path; project imports need the root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from httpx import Headers
from fnllm.base.services.cached import Cached
from fnllm.caching.file import FileCache
from fnllm.events.base import LLMEvents
from fnllm.openai.llm.openai_embeddings_llm import OpenAIEmbeddingsLLMImpl
from fnllm.openai.llm.openai_streaming_chat_llm import OpenAIStreamingChatLLMImpl
from fnllm.openai.llm.openai_text_chat_llm import OpenAITextChatLLMImpl
from fnllm.openai.services.openai_embeddings_cache_adapter import OpenAIEmbeddingsCacheAdapter
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from scripts.med_copilot_helpers import new_patient_result
from scripts.med_copilot_observability import capture_graphrag_responses



def upstream_provider_types():
    """Execute the vendored wrappers and singleton manager, without their setup imports."""
    files = [(ROOT / "med_copilot/upstream/graphrag/language_model/providers/fnllm/models.py",
              {"OpenAIChatFNLLM", "OpenAIEmbeddingFNLLM"}),
             (ROOT / "med_copilot/upstream/graphrag/language_model/manager.py", {"ModelManager"})]
    namespace = {}
    for path, names in files:
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(source, str(path), "exec"), namespace)
    return namespace


class FakeStream:
    def __init__(self, chunks, error=None):
        self.chunks, self.error = chunks, error
        self.close = AsyncMock()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class ProviderHarness:
    """Use real cached/shared FNLLM models; replace only their SDK clients."""

    def __init__(self):
        self.usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19)
        self.chunks = [self.chunk("evidence "), self.chunk("café"),
                       self.chunk(None, self.usage), self.chunk(None, self.usage)]
        self.stream_error = None
        self.chat_create = AsyncMock(side_effect=self.new_stream)
        self.embedding_response = CreateEmbeddingResponse.model_construct(
            model="reported-embedding-snapshot", object="list",
            data=[Embedding(index=0, object="embedding", embedding=[0.1, 0.2])],
            usage=SimpleNamespace(prompt_tokens=11, total_tokens=11))
        self.embedding_raw = SimpleNamespace(parse=Mock(side_effect=lambda: self.embedding_response), headers=Headers({}))
        self.embedding_create = AsyncMock(return_value=self.embedding_raw)
        self.text_response = ChatCompletion.model_construct(
            id="text-provider-id", model="reported-text-snapshot", created=0, object="chat.completion",
            choices=[], usage=None)
        self.text_create = AsyncMock(return_value=SimpleNamespace(
            parse=Mock(side_effect=lambda: self.text_response), headers=Headers({})))
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=self.chat_create,
                with_raw_response=SimpleNamespace(create=self.text_create))),
            embeddings=SimpleNamespace(with_raw_response=SimpleNamespace(create=self.embedding_create)))
        self.chat = OpenAIStreamingChatLLMImpl(self.client, "requested-chat", model_parameters={"temperature": 0.2})
        self.embedding = OpenAIEmbeddingsLLMImpl(self.client, "requested-embedding")
        self.text = OpenAITextChatLLMImpl(self.client, "requested-text")
        wrappers = upstream_provider_types()
        chat_provider = object.__new__(wrappers["OpenAIChatFNLLM"])
        chat_provider.model = self.chat
        embedding_provider = object.__new__(wrappers["OpenAIEmbeddingFNLLM"])
        embedding_provider.model = self.embedding
        wrappers["ModelFactory"] = SimpleNamespace(
            create_chat_model=lambda *args, **kwargs: chat_provider,
            create_embedding_model=lambda *args, **kwargs: embedding_provider)
        self.manager = wrappers["ModelManager"]()

    @staticmethod
    def chunk(content, usage=None):
        return ChatCompletionChunk.model_construct(
            id="stream-provider-id", model="reported-chat-snapshot", created=0,
            object="chat.completion.chunk", usage=usage,
            choices=([SimpleNamespace(delta=SimpleNamespace(content=content))]
                     if content is not None else []))

    async def new_stream(self, **kwargs):
        await asyncio.sleep(0)  # Let concurrent capture scopes overlap.
        return FakeStream(self.chunks, self.stream_error)

    async def query(self):
        embedder = self.manager.get_or_create_embedding_model("embedding", "openai_embedding")
        chat = self.manager.get_or_create_chat_model("chat", "openai_chat")
        embeddings = await embedder.aembed_batch(["guideline questions"])
        answer = "".join([text async for text in chat.achat_stream("guideline prompt", history=[])])
        return answer, embeddings


@contextmanager
def background_loop():
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


class GraphRAGCaptureTests(unittest.TestCase):
    def setUp(self):
        self.harness = ProviderHarness()
        self.result = new_patient_result("patient")
        self.metadata = self.result["model_responses"]

    def test_real_fnllm_stream_and_embedding_keep_provider_metadata_and_behavior(self):
        original_execute = OpenAIStreamingChatLLMImpl._execute_llm
        with capture_graphrag_responses(self.metadata):
            answer, embeddings = asyncio.run(self.harness.query())
        self.assertIs(OpenAIStreamingChatLLMImpl._execute_llm, original_execute)
        self.assertEqual(answer, "evidence café")
        self.assertEqual(embeddings, [[0.1, 0.2]])
        self.assertEqual(self.metadata["guideline_evidence"], [{
            "reported_model": "reported-chat-snapshot", "response_id": "stream-provider-id",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}])
        self.assertEqual(self.metadata["guideline_query_embedding"], [{
            "reported_model": "reported-embedding-snapshot", "response_id": None,
            "token_usage": {"prompt_tokens": 11, "completion_tokens": None, "total_tokens": 11}}])
        self.assertEqual(self.harness.chat_create.await_count, 1)
        self.assertEqual(self.harness.embedding_create.await_count, 1)
        kwargs = self.harness.chat_create.call_args.kwargs
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(kwargs["model"], "requested-chat")
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertTrue(kwargs["stream"])
        self.assertEqual(self.harness.embedding_create.call_args.kwargs,
                         {"input": ["guideline questions"], "model": "requested-embedding"})

    def test_scopes_do_not_leak_across_shared_instances_or_outside_scope(self):
        with capture_graphrag_responses(self.metadata):
            asyncio.run(self.harness.query())
        other = new_patient_result("second")["model_responses"]
        with capture_graphrag_responses(other):
            asyncio.run(self.harness.query())
            asyncio.run(self.harness.query())
        asyncio.run(self.harness.query())
        self.assertEqual(len(self.metadata["guideline_evidence"]), 1)
        self.assertEqual(len(other["guideline_evidence"]), 2)
        self.assertEqual(len(other["guideline_query_embedding"]), 2)
        self.assertNotIn("stream_options", self.harness.chat_create.call_args.kwargs)

    def test_partial_stream_exception_and_empty_stream_preserve_only_obtained_responses(self):
        self.harness.stream_error = RuntimeError("stream broke")
        with capture_graphrag_responses(self.metadata):
            with self.assertRaisesRegex(RuntimeError, "stream broke"):
                asyncio.run(self.harness.query())
        self.assertEqual(len(self.metadata["guideline_evidence"]), 1)
        self.harness.stream_error = None
        self.harness.chunks = []
        other = new_patient_result("second")["model_responses"]
        with capture_graphrag_responses(other):
            asyncio.run(self.harness.query())
        self.assertEqual(other["guideline_evidence"], [])

    def test_usage_and_ids_missing_are_not_replaced_by_configuration(self):
        self.harness.chunks = [self.harness.chunk("answer")]
        self.harness.chunks[0].id = None
        self.harness.chunks[0].model = None
        self.harness.embedding_response.usage = None
        with capture_graphrag_responses(self.metadata):
            asyncio.run(self.harness.query())
        self.assertEqual(self.metadata["guideline_evidence"], [
            {"reported_model": None, "response_id": None, "token_usage": None}])
        self.assertIsNone(self.metadata["guideline_query_embedding"][0]["token_usage"])

    def test_rejected_text_completion_is_recorded_before_fnllm_choice_validation(self):
        with capture_graphrag_responses(self.metadata):
            with self.assertRaises(Exception):
                asyncio.run(self.harness.text("prompt"))
        self.assertEqual(self.metadata["guideline_evidence"], [{
            "reported_model": "reported-text-snapshot", "response_id": "text-provider-id", "token_usage": None}])
        self.assertEqual(self.harness.text_create.await_count, 1)

    def test_context_propagates_through_graphrag_background_loop(self):
        with background_loop() as loop, capture_graphrag_responses(self.metadata):
            answer, _ = asyncio.run_coroutine_threadsafe(self.harness.query(), loop).result(timeout=5)
        self.assertEqual(answer, "evidence café")
        self.assertEqual(len(self.metadata["guideline_query_embedding"]), 1)
        self.assertEqual(len(self.metadata["guideline_evidence"]), 1)

    def test_nested_and_concurrent_contexts_remain_isolated(self):
        other = new_patient_result("second")["model_responses"]
        async def invoke(sink):
            with capture_graphrag_responses(sink):
                await self.harness.query()
        async def concurrent():
            await asyncio.gather(invoke(self.metadata), invoke(other))
        asyncio.run(concurrent())
        self.assertEqual(len(self.metadata["guideline_evidence"]), 1)
        self.assertEqual(len(other["guideline_evidence"]), 1)
        with capture_graphrag_responses(self.metadata):
            with capture_graphrag_responses(other):
                asyncio.run(self.harness.query())
            asyncio.run(self.harness.query())
        self.assertEqual(len(self.metadata["guideline_evidence"]), 2)
        self.assertEqual(len(other["guideline_evidence"]), 2)

    def test_instrumentation_imports_remain_at_top(self):
        tree = ast.parse((ROOT / "scripts/med_copilot_observability.py").read_text())
        self.assertEqual(sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
                         sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body))

    def test_cached_embedding_raw_metadata_is_captured_without_a_second_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = FileCache(directory)
            events = LLMEvents()
            cached = Cached(cache=cache, events=events, cache_adapter=OpenAIEmbeddingsCacheAdapter(
                cache, "requested-embedding"))
            # Real SDK usage serializes and validates through FNLLM's cache adapter.
            self.harness.embedding_response.usage = EmbeddingUsage(prompt_tokens=11, total_tokens=11)
            model = OpenAIEmbeddingsLLMImpl(self.harness.client, "requested-embedding", cached=cached)
            with capture_graphrag_responses(self.metadata):
                first = asyncio.run(model(["same question"]))
            other = new_patient_result("second")["model_responses"]
            with capture_graphrag_responses(other):
                second = asyncio.run(model(["same question"]))
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(self.harness.embedding_create.await_count, 1)
            self.assertEqual(len(self.metadata["guideline_query_embedding"]), 1)
            self.assertEqual(other["guideline_query_embedding"], self.metadata["guideline_query_embedding"])

    def test_successful_nonstreaming_fnllm_response_is_recorded_once(self):
        self.harness.text_response.choices = [SimpleNamespace(message=SimpleNamespace(content="answer"))]
        # Valid SDK response objects are required by FNLLM's output model.
        self.harness.text_response = ChatCompletion(
            id="text-id", model="reported-text", created=0, object="chat.completion",
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
            usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10})
        with capture_graphrag_responses(self.metadata):
            response = asyncio.run(self.harness.text("prompt"))
        self.assertEqual(response.output.content, "answer")
        self.assertEqual(self.metadata["guideline_evidence"], [{
            "reported_model": "reported-text", "response_id": "text-id",
            "token_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}])
        self.harness.text_create.assert_awaited_once()

    def test_partial_usage_chunks_merge_without_summing_or_erasing_counts(self):
        self.harness.chunks = [self.harness.chunk("answer"),
                               self.harness.chunk(None, SimpleNamespace(prompt_tokens=4, completion_tokens=2)),
                               self.harness.chunk(None, SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6))]
        with capture_graphrag_responses(self.metadata):
            asyncio.run(self.harness.query())
        self.assertEqual(self.metadata["guideline_evidence"][0]["token_usage"],
                         {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        self.assertEqual(len(self.metadata["guideline_evidence"]), 1)

    def test_partial_usage_survives_fnllm_rejecting_missing_counts(self):
        self.harness.chunks = [self.harness.chunk("answer"), self.harness.chunk(
            None, SimpleNamespace(prompt_tokens=4, completion_tokens=None))]
        with capture_graphrag_responses(self.metadata):
            with self.assertRaises(ValueError):
                asyncio.run(self.harness.query())
        self.assertEqual(self.metadata["guideline_evidence"][0]["token_usage"],
                         {"prompt_tokens": 4, "completion_tokens": None, "total_tokens": None})

    def test_api_exception_and_interruption_restore_hooks_and_leave_obtained_metadata(self):
        original = OpenAIEmbeddingsLLMImpl._execute_llm
        self.harness.embedding_create.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            with capture_graphrag_responses(self.metadata):
                asyncio.run(self.harness.query())
        self.assertEqual(self.metadata["guideline_query_embedding"], [])
        self.assertIs(OpenAIEmbeddingsLLMImpl._execute_llm, original)
        self.harness.embedding_create.side_effect = None
        self.harness.stream_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            with capture_graphrag_responses(self.metadata):
                asyncio.run(self.harness.query())
        self.assertEqual(len(self.metadata["guideline_evidence"]), 1)
        self.assertIs(OpenAIEmbeddingsLLMImpl._execute_llm, original)

    def test_vendored_local_search_streams_with_sync_embedding_on_background_loop(self):
        wrappers = upstream_provider_types()
        chat = object.__new__(wrappers["OpenAIChatFNLLM"])
        chat.model = self.harness.chat
        embedding = object.__new__(wrappers["OpenAIEmbeddingFNLLM"])
        embedding.model = self.harness.embedding
        utils_path = ROOT / "med_copilot/upstream/graphrag/language_model/providers/fnllm/utils.py"
        sync_function = next(node for node in ast.parse(utils_path.read_text()).body
                             if isinstance(node, ast.FunctionDef) and node.name == "run_coroutine_sync")
        search_path = ROOT / "med_copilot/upstream/graphrag/query/structured_search/local_search/search.py"
        local_class = next(node for node in ast.parse(search_path.read_text()).body
                           if isinstance(node, ast.ClassDef) and node.name == "LocalSearch")
        stream_function = next(node for node in local_class.body
                               if isinstance(node, ast.AsyncFunctionDef) and node.name == "stream_search")
        search_namespace = {"time": time, "logger": logging.getLogger(__name__)}
        exec(compile("from __future__ import annotations\n" + ast.unparse(stream_function),
                     str(search_path), "exec"), search_namespace)
        vectors = []
        def build_context(**kwargs):
            vectors.append(embedding.embed(kwargs["query"]))
            return SimpleNamespace(context_chunks="guideline context", context_records={"sources": []})
        callback = Mock()
        engine = SimpleNamespace(
            context_builder=SimpleNamespace(build_context=build_context), context_builder_params={},
            system_prompt="Context: {context_data}; response: {response_type}", response_type="paragraphs",
            callbacks=[callback], model=chat, model_params={"temperature": .2})
        async def consume():
            return "".join([text async for text in search_namespace["stream_search"](engine, "query")])
        with background_loop() as loop, capture_graphrag_responses(self.metadata):
            wrappers.update(asyncio=asyncio, _loop=loop, _thr=SimpleNamespace(is_alive=lambda: True))
            exec(compile("from __future__ import annotations\n" + ast.unparse(sync_function),
                         str(utils_path), "exec"), wrappers)
            answer = asyncio.run(consume())
        self.assertEqual(answer, "evidence café")
        self.assertEqual(vectors, [[.1, .2]])
        callback.on_context.assert_called_once_with({"sources": []})
        self.assertEqual(len(self.metadata["guideline_query_embedding"]), 1)
        self.assertEqual(len(self.metadata["guideline_evidence"]), 1)
        self.harness.embedding_create.assert_awaited_once()
        self.harness.chat_create.assert_awaited_once()
        request = self.harness.chat_create.call_args.kwargs
        self.assertEqual(request["temperature"], .2)
        self.assertEqual(request["stream_options"], {"include_usage": True})


if __name__ == "__main__":
    unittest.main()
