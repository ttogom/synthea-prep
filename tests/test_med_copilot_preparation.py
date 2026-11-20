"""Offline evidence-worker regressions; no models, APIs, or downloads are used."""

import builtins
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

# Direct test execution needs the repository and vendored imports before loading
# the preparation adapter. Its model dependencies are injected below.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from graphrag.config.load_config import load_config
from graphrag.language_model.factory import ModelFactory
from scripts.collate_med_copilot_results import load_report_data
from scripts.med_copilot_helpers import GUIDELINE_TABLES, new_patient_result
from scripts.experiment_runtime.preparation import EvidencePreparer, validate_preparation_result
from scripts.med_copilot_run_file import RunMetadataError
from scripts.experiment_runtime.worker_stream import RedactingStream
from scripts.experiment_runtime.execution import JobStore
from scripts.experiment_runtime.provenance import write_provenance
from test_med_copilot_retrieval import load_retrieval_module


class FakeClient:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if kwargs.get("tools"):
            assert kwargs["parallel_tool_calls"] is False
            names = {tool["function"]["name"] for tool in kwargs["tools"]}
            returned = {message["tool_call_id"] for message in kwargs["messages"]
                        if message["role"] == "tool"}
            observed = {call["function"]["name"] for message in kwargs["messages"]
                        if message["role"] == "assistant" for call in message.get("tool_calls", [])
                        if call["id"] in returned}
            for name, argument in (("search_similar_patients", {"query": "cough and oxygen saturation"}),
                                   ("search_guidelines", {"question": "What findings should be checked?"})):
                if name in names and name not in observed and (
                        kwargs["tool_choice"] == "required" or
                        kwargs["tool_choice"].get("function", {}).get("name") == name):
                    break
            else:
                name, argument = "finish", {}
                if "search_similar_patients" in observed:
                    argument["patient_ref"] = "P1"
                if "search_guidelines" in observed:
                    argument["guideline_refs"] = ["G1"]
            calls = [SimpleNamespace(id=f"call-{len(self.requests)}", function=SimpleNamespace(
                name=name, arguments=json.dumps(argument)))]
            return SimpleNamespace(model="fake-agent", id=f"agent-{len(self.requests)}", usage=None,
                choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
                    content=None, refusal=None, tool_calls=calls))])
        content = ('{"symptoms": ["cough"], "diagnoses": [], "medications": []}'
                   if kwargs.get("temperature") == 0 else '{"Key Questions": ["What findings should be checked?"]}')
        return SimpleNamespace(model="fake", id="reply", usage=None, choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=content, refusal=None))])


class FakeRetriever:
    def __init__(self, df, *, encoder, **kwargs):
        self.df = df
        self.alpha = kwargs["alpha"]
        self.embedder = SimpleNamespace(model=encoder, emb_path="embeddings.npy",
                                        index_path="index.faiss", metadata_path="matching.json")
        for name in ("embeddings.npy", "index.faiss", "matching.json"):
            Path(name).write_text("artifact")
        self.keyword_extractor = SimpleNamespace(client=CLIENT, model="fake-keywords")
        self.requests = []

    def search(self, query, topk):
        self.requests.append((query, topk))
        self.keyword_extractor.extract(query)
        values = [{"idx": 0, "embedding_score": 0.8, "hybrid_score": 0.7}]
        self.ranked_candidates = values
        if getattr(self, "on_ranking", None):
            self.on_ranking(values)
        return values


class FakeReranker:
    def __init__(self, model_name):
        self.model = SimpleNamespace(device="cpu")
        self.requests = []

    def rerank(self, query, candidates, topk, format_top_k):
        self.requests.append((query, topk, format_top_k))
        candidates[0]["rerank_score"] = 1.0
        if getattr(self, "on_ranking", None):
            self.on_ranking(candidates)
        return ["Reference SOAP example"]


def agent_message(*calls, content=None):
    return SimpleNamespace(content=content, tool_calls=[
        SimpleNamespace(id=f"call-{index}", function=SimpleNamespace(
            name=name, arguments=json.dumps(arguments)))
        for index, (name, arguments) in enumerate(calls)
    ])


class ScriptedAgentClient(FakeClient):
    def __init__(self, replies):
        super().__init__()
        self.replies = iter(replies)

    def create(self, **kwargs):
        if not kwargs.get("tools"):
            return super().create(**kwargs)
        self.requests.append(deepcopy(kwargs))
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=next(self.replies))])


class MeteredAgentClient(ScriptedAgentClient):
    def __init__(self, replies, prefix="attempt"):
        super().__init__(replies)
        self.prefix = prefix

    def create(self, **kwargs):
        response = super().create(**kwargs)
        response.model = "gpt-4o-mini-2024-07-18"
        response.id = f"{self.prefix}-{len(self.requests)}"
        response.usage = SimpleNamespace(prompt_tokens=101, completion_tokens=7, total_tokens=108,
                                        prompt_tokens_details=SimpleNamespace(cached_tokens=32))
        return response


CLIENT = None


class GuidelineRateConfigurationTests(unittest.TestCase):
    def test_production_chat_limits_are_numeric_and_embedding_limits_remain_disabled(self):
        settings = yaml.safe_load((ROOT / "med_copilot/data/guidelines/settings.yaml").read_text())
        chat = settings["models"]["default_chat_model"]
        embedding = settings["models"]["default_embedding_model"]
        self.assertIs(type(chat["requests_per_minute"]), int)
        self.assertIs(type(chat["tokens_per_minute"]), int)
        self.assertEqual((chat["requests_per_minute"], chat["tokens_per_minute"]), (12, 200000))
        self.assertIsNone(embedding["requests_per_minute"])
        self.assertIsNone(embedding["tokens_per_minute"])
        self.assertEqual(settings["local_search"]["chat_model_id"], "default_chat_model")
        self.assertEqual(settings["local_search"]["embedding_model_id"], "default_embedding_model")

    def test_real_graphrag_loader_and_fnllm_provider_receive_numeric_limits_without_calls(self):
        guidelines = ROOT / "med_copilot/data/guidelines"
        with patch("graphrag.config.load_config._load_dotenv") as dotenv, \
                patch.dict(os.environ, {"GRAPHRAG_API_KEY": "offline-guideline-key"}), \
                patch("graphrag.language_model.providers.fnllm.models.create_openai_client") as client, \
                patch("graphrag.language_model.providers.fnllm.models.create_openai_chat_llm") as chat_llm, \
                patch("graphrag.language_model.providers.fnllm.models.create_openai_embeddings_llm") as embedding_llm:
            config = load_config(guidelines)
            dotenv.assert_called_once_with(guidelines / "settings.yaml")
            chat = config.models[config.local_search.chat_model_id]
            embedding = config.models[config.local_search.embedding_model_id]
            ModelFactory.create_chat_model(chat.type, name="offline-guideline-chat", config=chat)
            ModelFactory.create_embedding_model(embedding.type, name="offline-guideline-embedding", config=embedding)
            effective_chat = chat_llm.call_args.args[0]
            effective_embedding = embedding_llm.call_args.args[0]
            self.assertEqual((effective_chat.requests_per_minute, effective_chat.tokens_per_minute), (12, 200000))
            self.assertIs(type(effective_chat.requests_per_minute), int)
            self.assertIs(type(effective_chat.tokens_per_minute), int)
            self.assertIsNone(effective_embedding.requests_per_minute)
            self.assertIsNone(effective_embedding.tokens_per_minute)
            self.assertEqual(effective_chat.api_key, "offline-guideline-key")
            self.assertIs(client.call_args_list[0].args[0], effective_chat)
            self.assertIs(client.call_args_list[1].args[0], effective_embedding)
            self.assertEqual(client.call_count, 2)
            chat_llm.return_value.assert_not_called()
            embedding_llm.return_value.assert_not_called()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        global CLIENT
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "experiment.yaml"
        self.patient = {"id": "patient-a", "subjective": "Cough café {literal}", "objective": "O₂ 98%"}
        self.corpus = self.root / "corpus.json"
        self.corpus.write_text(json.dumps([{"id": "reference-a", "subjective": "Reference S",
                                             "objective": "Reference O", "condition": "DX", "plan": ["P"]}]))
        self.guidelines = self.root / "guidelines"
        (self.guidelines / "output").mkdir(parents=True)
        (self.guidelines / "settings.yaml").write_text("vector_store: {}\n")
        for name in GUIDELINE_TABLES:
            (self.guidelines / "output" / f"{name}.parquet").write_bytes(b"table")
        self.data = {"selected_models": ["model-a", "model-b"], "selected_variants": [
            "llm_only", "patient_retrieval", "guidelines", "patient_retrieval_guidelines"],
            "temperature": 1, "max_retries": 3, "max_patients": None,
            "patient_corpus": str(self.corpus), "guidelines": str(self.guidelines),
            "patient_retrieval_artifacts_dir": str(self.root / "artifacts")}
        self.client = CLIENT = FakeClient()
        self.snapshots = []
        self.errors = []
        self.searches = []
        self.env = patch.dict(os.environ, OPENAI_API_KEY="private-openai", GRAPHRAG_API_KEY="private-graph")
        self.env.start()
        self.addCleanup(self.env.stop)
        self.graph_capture = patch("scripts.experiment_runtime.preparation.capture_graphrag_responses",
                                   side_effect=lambda responses: nullcontext())
        self.graph_capture.start()
        self.addCleanup(self.graph_capture.stop)

    def graph_search(self, **kwargs):
        self.searches.append(kwargs)
        return "Guideline evidence", {"sources": [{"id": "source"}]}

    def preparer(self, **kwargs):
        self.path.write_text(yaml.safe_dump(self.data))
        options = dict(device="cpu", encoder=SimpleNamespace(device="cpu"),
                       checkpoint=self.snapshots.append, error_callback=self.errors.append,
                       retriever_factory=FakeRetriever, reranker_factory=FakeReranker,
                       guideline_search=self.graph_search, client_factory=lambda **kw: self.client,
                       root=self.root)
        options.update(kwargs)
        return EvidencePreparer(self.path, **options)

    def result(self, preparer):
        experiment = preparer.experiment
        return new_patient_result(self.patient["id"], model_ids=experiment.selected_models,
                                  variant_ids=experiment.selected_variants,
                                  component_plan=experiment.component_plan,
                                  variant_components=experiment.variant_components)

    def test_all_variants_share_evidence_and_render_without_generating(self):
        preparer = self.preparer()
        result = self.result(preparer)
        current_dir = Path.cwd()
        self.assertIs(preparer.prepare(self.patient, result), result)
        self.assertEqual(Path.cwd(), current_dir)
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(len(self.searches), 1)
        self.assertTrue(all(component["status"] == "completed" for component in result["components"].values()))
        for variant, output in result["variants"].items():
            prompt = output["assessment_plan_prompts"]["user"]
            self.assertIn(self.patient["subjective"], prompt)
            self.assertIn(self.patient["objective"], prompt)
            self.assertEqual("Reference SOAP example" in prompt, "patient_retrieval" in variant)
            self.assertEqual("Guideline evidence" in prompt, "guidelines" in variant)
            self.assertTrue(all(slot["status"] == "not_processed" for slot in output["assessment_plans"].values()))
        self.assertEqual(preparer.reranker.requests[0][1:], (5, 1))
        self.assertEqual(result["components"]["patient_retrieval"]["capture"]["selected_reference_patient"],
                         {"id": "reference-a"})

    def test_selected_agentic_variants_start_and_prepare_only_the_selected_set_offline(self):
        variants = ["agentic_patient_retrieval", "agentic_guidelines", "agentic_both"]
        self.data["selected_variants"] = variants
        preparer = self.preparer()
        result = preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(list(result["variants"]), variants)
        self.assertTrue(all(component["status"] == "completed" for component in result["components"].values()),
                        {name: component["error"] for name, component in result["components"].items()})
        self.assertTrue(all(result["variants"][name]["status"] == "failed_incomplete" for name in variants),
                        {name: result["variants"][name]["status"] for name in variants})
        self.assertTrue(all(result["variants"][name]["assessment_plan_prompts"] is not None
                            for name in variants))
        self.assertEqual(set(result["components"]), {"agent_patients", "agent_guidelines", "agent_both"})
        self.assertEqual(result["components"]["agent_patients"]["outputs"]["example"], "Reference SOAP example")
        self.assertEqual(result["components"]["agent_guidelines"]["outputs"]["Key_info"], "Guideline evidence")
        self.assertEqual(result["components"]["agent_both"]["outputs"], {
            "example": "Reference SOAP example", "Key_info": "Guideline evidence"})
        self.assertTrue(all(component["capture"]["agent"]["finished"]
                            for component in result["components"].values()))
        agent_requests = [request for request in self.client.requests if request.get("tools")]
        self.assertEqual(len(agent_requests), 7)  # Two turns for each single-source variant, three for both.
        self.assertTrue(all(request["parallel_tool_calls"] is False for request in agent_requests))

    def test_agent_sees_three_full_reranked_cases_and_selects_the_third_for_both_patient_variants(self):
        records = [{"id": f"reference-{i}", "subjective": f"Case {i}: " + str(i) * 4000,
                    "objective": "cough", "condition": f"Condition {i}", "plan": [f"Plan {i}"]}
                   for i in range(3)]
        self.corpus.write_text(json.dumps(records))
        self.data["selected_variants"] = ["agentic_patient_retrieval", "agentic_both"]
        self.client = ScriptedAgentClient([
            agent_message(("search_similar_patients", {"query": "cough"})),
            agent_message(("finish", {"patient_ref": "P3"})),
            agent_message(("search_similar_patients", {"query": "cough"})),
            agent_message(("search_guidelines", {"question": "What findings?"})),
            agent_message(("finish", {"patient_ref": "P3", "guideline_refs": ["G1"]})),
        ])
        adapter = load_retrieval_module()

        def retriever_factory(*args, **kwargs):
            retriever = FakeRetriever(*args, **kwargs)

            def search(query, topk):
                candidates = [dict(record, idx=i, dominant_terms=[]) for i, record in enumerate(records)]
                if getattr(retriever, "on_ranking", None):
                    retriever.on_ranking(candidates)
                return candidates

            retriever.search = search
            return retriever

        def reranker_factory(model_name):
            reranker = adapter.CrossEncoderReranker()
            reranker.model.predict = lambda pairs: [0.1, 0.9, 0.5]
            return reranker

        preparer = self.preparer(retriever_factory=retriever_factory, reranker_factory=reranker_factory)
        result = preparer.prepare(self.patient, self.result(preparer))
        for name in ("agent_patients", "agent_both"):
            component = result["components"][name]
            self.assertEqual(component["status"], "completed", component["error"])
            self.assertEqual(component["capture"]["selected_reference_patient"], {"id": "reference-0"})
            self.assertEqual(list(component["capture"]["agent"]["patient_results"]), ["P1", "P2", "P3"])
            self.assertIn(records[0]["subjective"], component["outputs"]["example"])
            self.assertNotIn(records[1]["subjective"], component["outputs"]["example"])
        for request in (self.client.requests[1], self.client.requests[4]):
            reply = next(message["content"] for message in request["messages"]
                         if message["role"] == "tool" and "[P1]" in message["content"])
            for i in range(1, 4):
                self.assertIn(f"[P{i}]", reply)
            for record in records:
                self.assertIn(record["subjective"], reply)
                self.assertIn(record["condition"], reply)
                self.assertIn(record["plan"][0], reply)
        for output in result["variants"].values():
            self.assertIn(records[0]["subjective"], output["assessment_plan_prompts"]["user"])
            self.assertNotIn(records[1]["subjective"], output["assessment_plan_prompts"]["user"])

    def test_unfinished_agent_blocks_final_prompts_and_retains_partial_capture(self):
        self.data["selected_variants"] = ["llm_only", "agentic_patient_retrieval"]
        self.client = ScriptedAgentClient([
            agent_message(("search_similar_patients", {"query": "cough"})),
            *[agent_message(content="done") for _ in range(10)],
        ])
        preparer = self.preparer()
        result = preparer.prepare(self.patient, self.result(preparer))
        component = result["components"]["agent_patients"]
        self.assertEqual(component["status"], "failed_incomplete")
        self.assertEqual(component["error"]["type"], "RetrievalAgentError")
        self.assertEqual(component["outputs"], {})
        agent = component["capture"]["agent"]
        self.assertFalse(agent["finished"])
        self.assertEqual(agent["stop_reason"], "validation_error")
        self.assertEqual(agent["searches"], 1)
        self.assertEqual(len(agent["trace"]), 1)
        self.assertEqual(agent["patient_results"], {"P1": "Reference SOAP example"})
        self.assertIsNone(result["variants"]["agentic_patient_retrieval"]["assessment_plan_prompts"])
        self.assertEqual(result["variants"]["agentic_patient_retrieval"]["error"]["type"], "EvidenceUnavailable")
        self.assertIsNotNone(result["variants"]["llm_only"]["assessment_plan_prompts"])
        self.assertTrue(self.errors)
        self.assertEqual(self.snapshots[-1], result)

    def test_agent_sees_the_complete_guideline_answer_used_in_the_final_prompt(self):
        self.data["selected_variants"] = ["agentic_guidelines"]
        answer = "First recommendation. " + "More supporting evidence. " * 300 + "Last recommendation."
        self.client = ScriptedAgentClient([
            agent_message(("search_guidelines", {"question": "What findings?"})),
            agent_message(("finish", {"guideline_refs": ["G1"]})),
        ])
        preparer = self.preparer(guideline_search=lambda **kwargs: (answer, {}))
        result = preparer.prepare(self.patient, self.result(preparer))
        component = result["components"]["agent_guidelines"]
        self.assertEqual(component["status"], "completed")
        self.assertEqual(self.client.requests[1]["messages"][-1]["content"],
                         f"[G1] evidence_bytes={len(answer.encode('utf-8'))}\n{answer}")
        self.assertEqual(component["outputs"]["Key_info"], answer)
        self.assertEqual(component["capture"]["agent"]["guideline_results"], {"G1": answer})
        self.assertIn(answer, result["variants"]["agentic_guidelines"]["assessment_plan_prompts"]["user"])
        self.assertEqual(self.snapshots[-1], result)

    def test_finish_rejects_unknown_reference_ids_instead_of_silently_substituting_empty_evidence(self):
        for variant, component_id, selection in [
                ("agentic_patient_retrieval", "agent_patients", {"patient_ref": "P99"}),
                ("agentic_guidelines", "agent_guidelines", {"guideline_refs": ["G99"]}),
                ("agentic_both", "agent_both", {"patient_ref": "P99", "guideline_refs": ["G99"]})]:
            with self.subTest(variant=variant):
                self.data["selected_variants"] = [variant]
                searches = []
                if "patient_ref" in selection:
                    searches.append(agent_message(("search_similar_patients", {"query": "cough"})))
                if "guideline_refs" in selection:
                    searches.append(agent_message(("search_guidelines", {"question": "What findings?"})))
                self.client = ScriptedAgentClient([*searches, agent_message(("finish", selection)),
                                                  agent_message(("finish", selection))])
                preparer = self.preparer()
                result = preparer.prepare(self.patient, self.result(preparer))
                component = result["components"][component_id]
                self.assertEqual(component["status"], "failed_incomplete")
                self.assertIn("Unknown", component["error"]["message"])
                self.assertEqual(component["outputs"], {})
                self.assertIsNone(result["variants"][variant]["assessment_plan_prompts"])

    def test_explicit_empty_finish_is_rejected_and_never_publishes_fallback_evidence(self):
        self.data["selected_variants"] = ["agentic_both"]
        self.client = ScriptedAgentClient([agent_message(("finish", {})), agent_message(("finish", {}))])
        preparer = self.preparer()
        result = preparer.prepare(self.patient, self.result(preparer))
        component = result["components"]["agent_both"]
        self.assertEqual(component["status"], "failed_incomplete")
        self.assertFalse(component["capture"]["agent"]["finished"])
        self.assertEqual(component["outputs"], {})
        self.assertIsNone(result["variants"]["agentic_both"]["assessment_plan_prompts"])
        self.assertEqual([e["action"] for e in component["capture"]["agent_diagnostics"]], ["retry", "exit"])

    def test_agent_usage_is_durable_before_tools_and_survives_store_reopen_and_retry(self):
        self.data["selected_variants"] = ["agentic_patient_retrieval"]
        self.client = MeteredAgentClient([agent_message(("search_similar_patients", {"query": "cough"}))])
        path = self.root / "jobs.sqlite3"
        store = JobStore(path)
        self.addCleanup(store.close)
        key = ("preparation", self.patient["id"], None, None)
        interrupted = []

        def save(result):
            store.save(key, result)
            responses = result["components"]["agent_patients"]["capture"]["model_responses"]
            if responses.get("agent_control") and not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt()

        preparer = self.preparer(checkpoint=save)
        with self.assertRaises(KeyboardInterrupt):
            preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(preparer.retriever.requests, [], "No retrieval before the response's usage is saved")
        store.close()
        store = JobStore(path)
        self.addCleanup(store.close)
        recovered = store.load(key)
        calls = recovered["components"]["agent_patients"]["capture"]["model_responses"]["agent_control"]
        self.assertEqual(calls, [{"reported_model": "gpt-4o-mini-2024-07-18", "response_id": "attempt-1",
            "token_usage": {"prompt_tokens": 101, "completion_tokens": 7, "total_tokens": 108,
                            "cached_tokens": 32}}])
        self.client = MeteredAgentClient([
            agent_message(("search_similar_patients", {"query": "cough"})),
            agent_message(("finish", {"patient_ref": "P1"})),
        ], prefix="retry")
        resumed = self.preparer(checkpoint=lambda result: store.save(key, result))
        completed = resumed.prepare(self.patient, recovered)
        calls = completed["components"]["agent_patients"]["capture"]["model_responses"]["agent_control"]
        self.assertEqual([call["response_id"] for call in calls], ["attempt-1", "retry-1", "retry-2"])
        self.assertEqual(store.load(key), completed)
        run_dir = self.root / "run"
        run_dir.mkdir()
        paths = write_provenance(run_dir, metadata={"status": "completed_with_errors"}, records=[completed])
        measured = json.loads(paths["usage"].read_text())["measured_usage"]["by_stage"]["agent_control"]
        self.assertEqual(measured["responses"], 3)
        self.assertEqual(measured["reported_tokens"], {"prompt_tokens": 303, "completion_tokens": 21,
                                                     "total_tokens": 324, "cached_tokens": 96})
        ledger = [json.loads(line) for line in (run_dir / "provenance/api-usage.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["contexts"][0]["variants"] == ["agentic_patient_retrieval"] for row in ledger))

    def test_unknown_reference_is_corrected_in_conversation_for_each_variant(self):
        for variant, component_id, patients, guidelines in [
                ("agentic_patient_retrieval", "agent_patients", True, False),
                ("agentic_guidelines", "agent_guidelines", False, True),
                ("agentic_both", "agent_both", True, True)]:
            with self.subTest(variant=variant):
                self.data["selected_variants"] = [variant]
                script, bad, good = [], {}, {}
                if patients:
                    script.append(agent_message(("search_similar_patients", {"query": "cough"})))
                    bad["patient_ref"], good["patient_ref"] = "P99", "P1"
                if guidelines:
                    script.append(agent_message(("search_guidelines", {"question": "What findings?"})))
                    bad["guideline_refs"], good["guideline_refs"] = ["G99"], ["G1"]
                self.client = MeteredAgentClient([*script, agent_message(("finish", bad)),
                                                  agent_message(("finish", good))])
                preparer = self.preparer()
                result = preparer.prepare(self.patient, self.result(preparer))
                component = result["components"][component_id]
                self.assertEqual(component["status"], "completed")
                self.assertIsNone(component["error"])
                self.assertEqual(self.errors, [])
                events = component["capture"]["agent_diagnostics"]
                self.assertEqual([e["action"] for e in events], ["retry", "recovered"])
                self.assertIn("Unknown", self.client.requests[-1]["messages"][-1]["content"])
                first = events[0]
                self.assertEqual(first["patient_id"], self.patient["id"])
                self.assertEqual(first["component"], component_id)
                self.assertEqual(first["variants"], [variant])
                self.assertEqual(first["stage"], "agentic_retrieval")
                self.assertEqual(first["substage"], "finish_validation")
                self.assertEqual(first["turn"], len(script) + 1)
                self.assertEqual(first["searches"], len(script))
                self.assertEqual(first["calls"][0]["tool"], "finish")
                self.assertEqual(first["response_id"], f"attempt-{len(script) + 1}")
                self.assertTrue(first["timestamp"])
                self.assertTrue(any(s["components"].get(component_id, {}).get("capture", {}).get("agent_diagnostics")
                                    for s in self.snapshots))

    def test_both_requires_patient_and_guidelines_and_retries_omitted_source(self):
        for invalid in [{"patient_ref": "P1"}, {"guideline_refs": ["G1"]}]:
            with self.subTest(invalid=invalid):
                self.data["selected_variants"] = ["agentic_both"]
                self.client = ScriptedAgentClient([
                    agent_message(("search_similar_patients", {"query": "cough"})),
                    agent_message(("search_guidelines", {"question": "What findings?"})),
                    agent_message(("finish", invalid)),
                    agent_message(("finish", {"patient_ref": "P1", "guideline_refs": ["G1"]})),
                ])
                preparer = self.preparer()
                result = preparer.prepare(self.patient, self.result(preparer))
                self.assertEqual(result["components"]["agent_both"]["status"], "completed")
                self.assertEqual(result["components"]["agent_both"]["outputs"], {
                    "example": "Reference SOAP example", "Key_info": "Guideline evidence"})
                self.assertIn("Error:", self.client.requests[-1]["messages"][-1]["content"])

    def test_both_selects_earlier_patient_and_combines_complete_earlier_and_later_guidelines(self):
        self.data["selected_variants"] = ["agentic_both"]
        self.client = ScriptedAgentClient([
            agent_message(("search_similar_patients", {"query": "first patient search"})),
            agent_message(("search_guidelines", {"question": "First findings?"})),
            agent_message(("search_similar_patients", {"query": "second patient search"})),
            agent_message(("search_guidelines", {"question": "Later findings?"})),
            agent_message(("finish", {"patient_ref": "P1", "guideline_refs": ["G1", "G2"]})),
        ])
        answers = ["Earlier answer " * 100, "Later answer " * 100]
        replies = iter(answers)
        preparer = self.preparer(guideline_search=lambda **kwargs: (next(replies), {}))
        patient_answers = ["Closest earlier patient", "Less similar later patient"]
        with patch.object(preparer.reranker, "rerank", side_effect=[[p] for p in patient_answers]):
            result = preparer.prepare(self.patient, self.result(preparer))
        component = result["components"]["agent_both"]
        self.assertEqual(component["status"], "completed")
        self.assertEqual(component["outputs"], {"example": patient_answers[0], "Key_info": "\n\n".join(answers)})
        self.assertEqual(component["capture"]["agent"]["patient_results"], dict(zip(["P1", "P2"], patient_answers)))
        self.assertEqual(component["capture"]["agent"]["guideline_results"], dict(zip(["G1", "G2"], answers)))
        final = result["variants"]["agentic_both"]["assessment_plan_prompts"]["user"]
        self.assertIn(patient_answers[0], final)
        self.assertNotIn(patient_answers[1], final)
        for answer in answers:
            self.assertIn(answer, final)
            self.assertTrue(any(m["role"] == "tool" and answer in m["content"]
                                for m in self.client.requests[-1]["messages"]))

    def test_guideline_budget_counts_utf8_and_joins_then_recovers_using_any_prior_answer(self):
        for variant, logical in [("agentic_guidelines", "agent_guidelines"), ("agentic_both", "agent_both")]:
            for budget, kept in [(12, ["G1", "G2"]), (11, ["G1"]), (4, ["G2"])]:
                with self.subTest(variant=variant, budget=budget):
                    self.data.update(selected_variants=[variant], variant_settings={
                        variant: {logical: {"guideline_byte_budget": budget}}})
                    selection = {"patient_ref": "P1"} if variant == "agentic_both" else {}
                    script = ([agent_message(("search_similar_patients", {"query": "cough"}))] if selection else [])
                    script += [agent_message(("search_guidelines", {"question": "First findings?"})),
                               agent_message(("search_guidelines", {"question": "Later findings?"})),
                               agent_message(("finish", {**selection, "guideline_refs": ["G1", "G2"]}))]
                    if budget < 12:
                        script.append(agent_message(("finish", {**selection, "guideline_refs": kept})))
                    self.client = MeteredAgentClient(script)
                    answers = {"G1": "ééé", "G2": "🙂"}  # 6 + 2 separator + 4 = 12 bytes, only 6 characters.
                    replies = iter(answers.values())
                    preparer = self.preparer(guideline_search=lambda **kw: (next(replies), {}))
                    result = preparer.prepare(self.patient, self.result(preparer))
                    instance = preparer.experiment.variant_components[variant][logical]
                    component = result["components"][instance]
                    self.assertEqual(component["status"], "completed")
                    self.assertEqual(component["settings"]["guideline_byte_budget"], budget)
                    joined = "\n\n".join(answers[ref] for ref in kept)
                    self.assertEqual(component["outputs"]["Key_info"], joined)
                    self.assertEqual(component["capture"]["agent"]["guideline_results"], answers)
                    final = result["variants"][variant]["assessment_plan_prompts"]["user"]
                    self.assertIn(joined, final)
                    self.assertNotIn("evidence_bytes=", final)
                    tool_replies = [m["content"] for m in self.client.requests[-1]["messages"] if m["role"] == "tool"]
                    self.assertIn("[G1] evidence_bytes=6\nééé", tool_replies)
                    self.assertIn("[G2] evidence_bytes=4\n🙂", tool_replies)
                    events = component["capture"].get("agent_diagnostics", [])
                    self.assertEqual([e["action"] for e in events], [] if budget == 12 else ["retry", "recovered"])
                    if events:
                        self.assertIn(f"12 UTF-8 bytes; allowed {budget}; excess {12-budget}", events[0]["message"])
                    self.assertEqual(len(component["capture"]["model_responses"]["agent_control"]), len(script))
                    self.assertEqual(len(self.client.requests), len(script))

    def test_default_variant_budgets_allow_more_guidelines_without_truncation(self):
        for variant, logical, budget in [("agentic_guidelines", "agent_guidelines", 8637),
                                           ("agentic_both", "agent_both", 4117)]:
            with self.subTest(variant=variant):
                self.data.update(selected_variants=[variant, "llm_only"], variant_settings={})
                selection = {"guideline_refs": ["G1"]}
                script = []
                if variant == "agentic_both":
                    selection["patient_ref"] = "P1"
                    script.append(agent_message(("search_similar_patients", {"query": "cough"})))
                script += [agent_message(("search_guidelines", {"question": "Findings?"})),
                           agent_message(("finish", selection))]
                if budget < 5000:
                    script.append(agent_message(("finish", selection)))
                self.client = MeteredAgentClient(script)
                answer = "x" * 5000
                preparer = self.preparer(guideline_search=lambda **kw: (answer, {}))
                result = preparer.prepare(self.patient, self.result(preparer))
                component = result["components"][logical]
                self.assertEqual(component["settings"]["guideline_byte_budget"], budget)
                self.assertEqual(component["capture"]["agent"]["guideline_results"], {"G1": answer})
                if budget > 5000:
                    self.assertEqual(component["outputs"]["Key_info"], answer)
                    self.assertIn(answer, result["variants"][variant]["assessment_plan_prompts"]["user"])
                else:
                    self.assertEqual(component["status"], "failed_incomplete")
                    self.assertEqual(component["outputs"], {})
                    self.assertIsNone(result["variants"][variant]["assessment_plan_prompts"])
                    events = component["capture"]["agent_diagnostics"]
                    self.assertEqual([e["action"] for e in events], ["retry", "exit"])
                    self.assertIn("excess 883", events[-1]["message"])
                    self.assertEqual(events[-1]["patient_id"], self.patient["id"])
                    self.assertEqual(events[-1]["variants"], [variant])
                    self.assertEqual(events[-1]["substage"], "finish_validation")
                    self.assertEqual(events[-1]["calls"][0]["tool"], "finish")
                    self.assertEqual(len(component["capture"]["model_responses"]["agent_control"]), len(script))
                    self.assertTrue(any(s["components"].get(logical, {}).get("capture", {}).get("agent_diagnostics")
                                        for s in self.snapshots))
                self.assertIsNotNone(result["variants"]["llm_only"]["assessment_plan_prompts"])
                preparer.client = FakeClient()
                later = {**self.patient, "id": "later-patient"}
                self.assertIsNotNone(preparer.prepare(later, self.result_for_patient(preparer, later))["variants"]["llm_only"]["assessment_plan_prompts"])

    def result_for_patient(self, preparer, patient):
        result = self.result(preparer)
        result["id"] = patient["id"]
        return result

    def test_budget_error_shares_the_existing_correction_allowance(self):
        for first, second in [(["G99"], ["G1"]), (["G1"], ["G99"])]:
            with self.subTest(first=first):
                self.data.update(selected_variants=["agentic_guidelines"], variant_settings={
                    "agentic_guidelines": {"agent_guidelines": {"guideline_byte_budget": 1}}})
                self.client = MeteredAgentClient([
                    agent_message(("search_guidelines", {"question": "Findings?"})),
                    agent_message(("finish", {"guideline_refs": first})),
                    agent_message(("finish", {"guideline_refs": second})),
                ])
                preparer = self.preparer()
                result = preparer.prepare(self.patient, self.result(preparer))
                instance = preparer.experiment.variant_components["agentic_guidelines"]["agent_guidelines"]
                component = result["components"][instance]
                self.assertEqual(component["status"], "failed_incomplete")
                self.assertEqual(len(self.client.requests), 3)
                events = component["capture"]["agent_diagnostics"]
                self.assertEqual([e["action"] for e in events], ["retry", "exit"])
                self.assertTrue(any("allowed 1" in e["message"] for e in events))
                self.assertTrue(any("Unknown" in e["message"] for e in events))

    def test_budget_correction_can_retrieve_a_shorter_answer_within_search_limit(self):
        self.data.update(selected_variants=["agentic_guidelines"], variant_settings={
            "agentic_guidelines": {"agent_guidelines": {"guideline_byte_budget": 3}}})
        self.client = ScriptedAgentClient([
            agent_message(("search_guidelines", {"question": "First findings?"})),
            agent_message(("finish", {"guideline_refs": ["G1"]})),
            agent_message(("search_guidelines", {"question": "Different findings?"})),
            agent_message(("finish", {"guideline_refs": ["G2"]})),
        ])
        replies = iter(["too long", "yes"])
        preparer = self.preparer(guideline_search=lambda **kw: (next(replies), {}))
        result = preparer.prepare(self.patient, self.result(preparer))
        instance = preparer.experiment.variant_components["agentic_guidelines"]["agent_guidelines"]
        component = result["components"][instance]
        self.assertEqual(component["status"], "completed")
        self.assertEqual(component["outputs"]["Key_info"], "yes")
        self.assertEqual(component["capture"]["agent"]["searches"], 2)
        self.assertEqual([e["action"] for e in component["capture"]["agent_diagnostics"]], ["retry", "recovered"])

    def test_blank_selected_evidence_can_be_replaced_by_another_search(self):
        for variant, component_id, tool, argument, field, blank_ref, valid_ref in [
                ("agentic_patient_retrieval", "agent_patients", "search_similar_patients", "query", "patient_ref", "P1", "P2"),
                ("agentic_guidelines", "agent_guidelines", "search_guidelines", "question", "guideline_refs", ["G1"], ["G2"])]:
            with self.subTest(variant=variant):
                self.data["selected_variants"] = [variant]
                self.client = ScriptedAgentClient([
                    agent_message((tool, {argument: "First findings?"})),
                    agent_message(("finish", {field: blank_ref})),
                    agent_message((tool, {argument: "Other findings?"})),
                    agent_message(("finish", {field: valid_ref})),
                ])
                replies = iter(["  ", "Complete nonempty evidence"])
                preparer = self.preparer(guideline_search=lambda **kwargs: (next(replies), {}))
                with patch.object(preparer.reranker, "rerank", side_effect=lambda *a, **kw: [next(replies)]) \
                        if preparer.reranker is not None else nullcontext():
                    result = preparer.prepare(self.patient, self.result(preparer))
                component = result["components"][component_id]
                self.assertEqual(component["status"], "completed")
                self.assertIn("blank evidence", self.client.requests[2]["messages"][-1]["content"])
                self.assertIn("Complete nonempty evidence", component["outputs"].values())

    def test_empty_searches_exhaust_budget_fail_locally_and_next_patient_still_runs(self):
        for patients in [True, False]:
            variant, component_id, tool, argument = (
                ("agentic_patient_retrieval", "agent_patients", "search_similar_patients", "query") if patients else
                ("agentic_guidelines", "agent_guidelines", "search_guidelines", "question"))
            with self.subTest(variant=variant):
                self.data["selected_variants"] = ["llm_only", variant]
                self.client = MeteredAgentClient([
                    *[agent_message((tool, {argument: f"findings {i}?"})) for i in range(4)],
                    agent_message((tool, {argument: "cough?"})),
                    agent_message(("finish", {"patient_ref": "P1"} if patients else {"guideline_refs": ["G1"]})),
                ])
                preparer = self.preparer()
                with patch.object(preparer.retriever, "search", return_value=[]) if patients else \
                        patch.object(preparer, "guideline_search", return_value=(" ", {})):
                    result = preparer.prepare(self.patient, self.result(preparer))
                component = result["components"][component_id]
                self.assertEqual(component["status"], "failed_incomplete")
                self.assertEqual(component["outputs"], {})
                self.assertEqual(len(self.client.requests), 4, "No useless correction without available evidence or searches")
                self.assertEqual(component["capture"]["agent"]["searches"], 4)
                event = component["capture"]["agent_diagnostics"][-1]
                self.assertEqual(event["substage"], "evidence_availability")
                self.assertEqual(event["action"], "exit")
                self.assertIsNone(result["variants"][variant]["assessment_plan_prompts"])
                self.assertIsNotNone(result["variants"]["llm_only"]["assessment_plan_prompts"])
                next_patient = {**self.patient, "id": "patient-b"}
                next_result = self.result(preparer)
                next_result["id"] = "patient-b"
                completed = preparer.prepare(next_patient, next_result)
                self.assertEqual(completed["components"][component_id]["status"], "completed")

    def test_diagnostics_are_additive_durable_exportable_and_optional_on_resume(self):
        self.data["selected_variants"] = ["agentic_patient_retrieval"]
        self.client = MeteredAgentClient([
            agent_message(("finish", {})),
            agent_message(("search_similar_patients", {"query": "cough"})),
            agent_message(("finish", {"patient_ref": "P1"})),
        ])
        preparer = self.preparer()
        result = preparer.prepare(self.patient, self.result(preparer))
        capture = result["components"]["agent_patients"]["capture"]
        self.assertEqual(set(capture["agent"]), {"patient_ref", "guideline_refs", "searches", "finished", "stop_reason",
            "trace", "usage", "patient_results", "patient_reference_ids", "guideline_results", "screened_questions"})
        self.assertEqual(set(capture["agent"]["trace"][0]), {"search", "tool", "input", "output"})
        self.assertEqual(capture["agent"]["usage"], [{"prompt_tokens": 101, "completion_tokens": 7}] * 3)
        before = deepcopy(result)
        store = JobStore(self.root / "diagnostics.sqlite3")
        key = ("preparation", self.patient["id"], None, None)
        store.save(key, result)
        store.close()
        store = JobStore(self.root / "diagnostics.sqlite3")
        self.addCleanup(store.close)
        restored = store.load(key)
        self.assertEqual(restored, before)
        validate_preparation_result(preparer.experiment, restored)
        self.assertEqual(preparer.prepare(self.patient, restored), before)
        legacy = deepcopy(restored)
        del legacy["components"]["agent_patients"]["capture"]["agent_diagnostics"]
        validate_preparation_result(preparer.experiment, legacy)
        self.assertEqual(preparer.prepare(self.patient, legacy), legacy)
        self.assertEqual(len(self.client.requests), 3, "Completed evidence is reused with or without new diagnostics")
        run_dir = self.root / "diagnostic-run"
        run_dir.mkdir()
        results_path = run_dir / "results.jsonl"
        results_path.write_text(json.dumps(restored) + "\n")
        saved_bytes = results_path.read_bytes()
        paths = write_provenance(run_dir, metadata={"status": "completed_with_errors"}, records=[restored])
        self.assertEqual(results_path.read_bytes(), saved_bytes)
        usage = json.loads(paths["usage"].read_text())["measured_usage"]["by_stage"]["agent_control"]
        self.assertEqual(usage["responses"], 3)
        self.assertEqual(usage["reported_tokens"]["total_tokens"], 324)
        self.assertEqual(restored, before)
        labels = self.root / "labels.jsonl"
        labels.write_text(json.dumps({"id": self.patient["id"], "condition": "DX", "plan": ["P"]}) + "\n")
        (run_dir / "run.json").write_text(json.dumps({"schema_version": 4,
            "experiment": {"selected_variants": ["agentic_patient_retrieval"]},
            "evaluation": {"labels_path": str(labels), "selected_patient_ids": [self.patient["id"]]}}))
        augmented_report = load_report_data(run_dir, root=self.root)
        results_path.write_text(json.dumps(legacy) + "\n")
        self.assertEqual(load_report_data(run_dir, root=self.root), augmented_report)

    def test_diagnostic_checkpoint_failure_is_fatal_and_arguments_are_redacted(self):
        self.data["selected_variants"] = ["agentic_both"]
        self.client = ScriptedAgentClient([agent_message(("private-openai", {"value": "private-graph"}))])

        def cannot_save(result):
            events = result["components"]["agent_both"]["capture"].get("agent_diagnostics")
            if events:
                text = json.dumps(events)
                self.assertNotIn("private-openai", text)
                self.assertNotIn("private-graph", text)
                raise OSError("disk full")

        preparer = self.preparer(checkpoint=cannot_save)
        with self.assertRaises(RunMetadataError):
            preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(preparer.retriever.requests, [])

    def test_agent_usage_is_kept_when_a_tool_fails_and_a_later_variant_can_continue(self):
        self.data["selected_variants"] = ["agentic_guidelines", "agentic_patient_retrieval"]
        self.client = MeteredAgentClient([
            agent_message(("search_similar_patients", {"query": "cough"})),
            agent_message(("finish", {"patient_ref": "P1"})),
            agent_message(("search_guidelines", {"question": "What findings?"})),
        ])

        def graph_failure(**kwargs):
            raise RuntimeError("Graph failed after the agent response")

        preparer = self.preparer(guideline_search=graph_failure)
        result = preparer.prepare(self.patient, self.result(preparer))
        component = result["components"]["agent_guidelines"]
        self.assertEqual(component["status"], "failed_incomplete")
        self.assertEqual(component["capture"]["model_responses"]["agent_control"][0]["response_id"], "attempt-3")
        self.assertEqual(component["capture"]["agent"]["stop_reason"], "tool_error")
        diagnostic = component["capture"]["agent_diagnostics"][-1]
        self.assertEqual((diagnostic["substage"], diagnostic["error_type"], diagnostic["action"]),
                         ("tool_execution", "RuntimeError", "exit"))
        self.assertEqual(diagnostic["calls"][0]["tool"], "search_guidelines")
        self.assertEqual(result["components"]["agent_patients"]["status"], "completed")
        self.assertTrue(any(snapshot["components"]["agent_guidelines"]["capture"]["model_responses"].get(
            "agent_control") for snapshot in self.snapshots))

    def test_agent_usage_checkpoint_failure_is_fatal_before_retrieval(self):
        self.data["selected_variants"] = ["agentic_patient_retrieval"]
        self.client = MeteredAgentClient([agent_message(("search_similar_patients", {"query": "cough"}))])

        def cannot_save(result):
            if result["components"]["agent_patients"]["capture"]["model_responses"].get("agent_control"):
                raise OSError("disk full")

        preparer = self.preparer(checkpoint=cannot_save)
        with self.assertRaises(RunMetadataError):
            preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(preparer.retriever.requests, [])
        self.assertEqual(len(self.client.requests), 1)

    def test_agent_question_rewrite_precedes_retrieval_and_preserves_exact_query_and_usage(self):
        self.patient.update(subjective="Breathlessness", objective="Oxygen saturation 89%; BP 172/104")
        bad = "Which hypertension treatment fits?"
        question = "For breathlessness, oxygen saturation 89% and BP 172/104, which investigations are recommended?"
        for variant, logical, patients in [("agentic_guidelines", "agent_guidelines", False),
                                           ("agentic_both", "agent_both", True)]:
            with self.subTest(variant=variant):
                self.data["selected_variants"] = [variant]
                self.searches.clear()
                self.snapshots.clear()
                prefix = [agent_message(("search_similar_patients", {"query": "breathlessness"}))] if patients else []
                selection = {"guideline_refs": ["G1"], **({"patient_ref": "P1"} if patients else {})}
                self.client = MeteredAgentClient([*prefix,
                    agent_message(("search_guidelines", {"question": bad})),
                    agent_message(("search_guidelines", {"question": "  " + question + "  "})),
                    agent_message(("finish", selection))])

                def graph(**kwargs):
                    saved = self.snapshots[-1]["components"][logical]["capture"]
                    self.assertEqual(saved["agent_diagnostics"][0]["action"], "retry")
                    self.assertEqual(len(saved["model_responses"]["agent_control"]), len(prefix) + 2)
                    return self.graph_search(**kwargs)

                preparer = self.preparer(guideline_search=graph)
                result = preparer.prepare(self.patient, self.result(preparer))
                component = result["components"][logical]
                capture, agent = component["capture"], component["capture"]["agent"]
                self.assertEqual(component["status"], "completed")
                self.assertEqual([q["query"] for q in self.searches], [question])
                self.assertEqual(agent["screened_questions"], [{"question": bad, "terms": ["hypertension"]}])
                self.assertEqual(agent["searches"], len(prefix) + 1)
                self.assertEqual(agent["guideline_results"], {"G1": "Guideline evidence"})
                self.assertEqual(agent["trace"][-1]["input"], question)
                self.assertEqual(set(agent["trace"][-1]), {"search", "tool", "input", "output"})
                events = capture["agent_diagnostics"]
                self.assertEqual([e["action"] for e in events], ["retry", "recovered"])
                self.assertEqual((events[0]["code"], events[0]["substage"], events[0]["searches"]),
                                 ("invalid_guideline_question", "query_validation", len(prefix)))
                self.assertEqual(json.loads(events[0]["calls"][0]["arguments"]), {"question": bad})
                self.assertEqual(events[0]["response_id"], f"attempt-{len(prefix) + 1}")
                self.assertEqual((events[0]["patient_id"], events[0]["variants"]), (self.patient["id"], [variant]))
                retry = self.client.requests[len(prefix) + 1]
                self.assertIn("hypertension", retry["messages"][-1]["content"])
                self.assertIn("recorded findings and measurement values", retry["messages"][-1]["content"])
                self.assertEqual(retry["tool_choice"]["function"]["name"], "search_guidelines")
                self.assertEqual(component["outputs"]["Key_info"], "Guideline evidence")
                self.assertNotIn("evidence_bytes", result["variants"][variant]["assessment_plan_prompts"]["user"])
                self.assertIsNone(component["error"])
                run_dir = self.root / variant
                run_dir.mkdir()
                paths = write_provenance(run_dir, metadata={"status": "completed"}, records=[result])
                measured = json.loads(paths["usage"].read_text())["measured_usage"]["by_stage"]["agent_control"]
                self.assertEqual(measured["responses"], len(prefix) + 3)
                self.assertEqual(measured["reported_tokens"]["total_tokens"], 108 * (len(prefix) + 3))

    def test_repeated_flagged_questions_fail_locally_and_later_patient_continues(self):
        bad = "Which hypertension treatment fits?"
        for variant, logical, patients in [("agentic_guidelines", "agent_guidelines", False),
                                           ("agentic_both", "agent_both", True)]:
            with self.subTest(variant=variant):
                self.data["selected_variants"] = ["llm_only", variant]
                self.searches.clear()
                prefix = [agent_message(("search_similar_patients", {"query": "cough"}))] if patients else []
                self.client = MeteredAgentClient([*prefix, agent_message(("search_guidelines", {"question": bad})),
                    agent_message(("search_guidelines", {"question": "Which HYPERTENSION stage fits?"}))])
                preparer = self.preparer()
                result = preparer.prepare(self.patient, self.result(preparer))
                component = result["components"][logical]
                self.assertEqual(component["status"], "failed_incomplete")
                self.assertEqual(component["outputs"], {})
                self.assertEqual(self.searches, [])
                self.assertEqual(len(self.client.requests), len(prefix) + 2)
                self.assertEqual(component["capture"]["agent"]["searches"], len(prefix))
                self.assertEqual(component["capture"]["agent"]["stop_reason"], "validation_error")
                self.assertEqual([e["action"] for e in component["capture"]["agent_diagnostics"]], ["retry", "exit"])
                self.assertEqual(self.errors[-1]["stage"], "agentic_retrieval")
                self.assertIsNone(result["variants"][variant]["assessment_plan_prompts"])
                self.assertIsNotNone(result["variants"]["llm_only"]["assessment_plan_prompts"])
                preparer.client = FakeClient()
                later = {**self.patient, "id": "later-patient"}
                completed = preparer.prepare(later, self.result_for_patient(preparer, later))
                self.assertEqual(completed["components"][logical]["status"], "completed")

    def test_agent_question_screen_respects_existing_setting_and_recorded_diagnoses(self):
        question = "Which hypertension treatment fits?"
        for allow, subjective in [(True, "Cough"), (False, "Cough; history of hypertension")]:
            with self.subTest(allow=allow):
                self.data.update(selected_variants=["agentic_guidelines"], variant_settings={
                    "agentic_guidelines": {"agent_guidelines": {"allow_diagnosis_terms": allow}}})
                self.patient["subjective"] = subjective
                self.searches.clear()
                self.client = ScriptedAgentClient([
                    agent_message(("search_guidelines", {"question": question})),
                    agent_message(("finish", {"guideline_refs": ["G1"]})),
                ])
                preparer = self.preparer()
                result = preparer.prepare(self.patient, self.result(preparer))
                instance = preparer.experiment.variant_components["agentic_guidelines"]["agent_guidelines"]
                component = result["components"][instance]
                self.assertEqual(component["status"], "completed")
                self.assertEqual([q["query"] for q in self.searches], [question])
                self.assertEqual(component["capture"]["agent"]["screened_questions"], [])
                self.assertNotIn("agent_diagnostics", component["capture"])

    def test_agent_question_errors_share_correction_with_finish_byte_budget_errors(self):
        bad = agent_message(("search_guidelines", {"question": "Which hypertension treatment fits?"}))
        good = agent_message(("search_guidelines", {"question": "For cough and O2 98%, which tests?"}))
        finish = agent_message(("finish", {"guideline_refs": ["G1"]}))
        for script, codes in [([bad, good, finish], ["invalid_guideline_question", "invalid_guideline_question", "invalid_finish"]),
                              ([good, finish, bad], ["invalid_finish", "invalid_guideline_question"])]:
            with self.subTest(codes=codes):
                self.data.update(selected_variants=["agentic_guidelines"], variant_settings={
                    "agentic_guidelines": {"agent_guidelines": {"guideline_byte_budget": 1}}})
                self.searches.clear()
                self.client = MeteredAgentClient(script)
                preparer = self.preparer()
                result = preparer.prepare(self.patient, self.result(preparer))
                instance = preparer.experiment.variant_components["agentic_guidelines"]["agent_guidelines"]
                component = result["components"][instance]
                self.assertEqual(component["status"], "failed_incomplete")
                self.assertEqual(component["outputs"], {})
                self.assertEqual(len(self.searches), 1)
                self.assertEqual(len(self.client.requests), 3)
                self.assertEqual(component["capture"]["agent"]["searches"], 1)
                events = component["capture"]["agent_diagnostics"]
                self.assertEqual([e["code"] for e in events], codes)
                self.assertEqual(sum(e["action"] == "retry" for e in events), 1)
                self.assertEqual(events[-1]["action"], "exit")

    def test_rejected_question_leaves_all_four_searches_and_earlier_answers_available(self):
        self.data["selected_variants"] = ["agentic_guidelines"]
        questions = [f"For cough and O2 98%, what {topic} is recommended?"
                     for topic in ["investigation", "monitoring", "follow-up", "education"]]
        self.client = ScriptedAgentClient([
            agent_message(("search_guidelines", {"question": "Which hypertension treatment fits?"})),
            *[agent_message(("search_guidelines", {"question": q})) for q in questions],
            agent_message(("finish", {"guideline_refs": ["G1", "G4"]})),
        ])
        answers = [f"Complete answer {i}" for i in range(4)]
        replies = iter(answers)

        def graph(**kwargs):
            self.searches.append(kwargs)
            return next(replies), {}

        preparer = self.preparer(guideline_search=graph)
        result = preparer.prepare(self.patient, self.result(preparer))
        component = result["components"]["agent_guidelines"]
        self.assertEqual(component["status"], "completed")
        self.assertEqual([q["query"] for q in self.searches], questions)
        self.assertEqual(component["capture"]["agent"]["searches"], 4)
        self.assertEqual(len(self.client.requests), 6)
        self.assertEqual(component["capture"]["agent"]["guideline_results"],
                         {f"G{i + 1}": a for i, a in enumerate(answers)})
        self.assertEqual(component["outputs"]["Key_info"], "\n\n".join([answers[0], answers[3]]))
        self.assertEqual(self.client.requests[-1]["tool_choice"]["function"]["name"], "finish")

    def test_all_seven_variants_preserve_fixed_screening_and_agent_patient_behavior(self):
        self.data["selected_variants"] += ["agentic_patient_retrieval", "agentic_guidelines", "agentic_both"]
        bad = "Which hypertension treatment fits?"
        good = "For cough and oxygen saturation 98%, which investigations are recommended?"
        self.client = ScriptedAgentClient([
            agent_message(("search_similar_patients", {"query": "cough"})),
            agent_message(("finish", {"patient_ref": "P1"})),
            agent_message(("search_guidelines", {"question": bad})),
            agent_message(("search_guidelines", {"question": good})),
            agent_message(("finish", {"guideline_refs": ["G1"]})),
            agent_message(("search_similar_patients", {"query": "cough"})),
            agent_message(("search_guidelines", {"question": bad})),
            agent_message(("search_guidelines", {"question": good})),
            agent_message(("finish", {"patient_ref": "P1", "guideline_refs": ["G1"]})),
        ])
        original = self.client.chat.completions.create
        raw = json.dumps({"Key Questions": [bad]})

        def create(**kwargs):
            if kwargs.get("response_format") == {"type": "json_object"}:
                self.client.requests.append(deepcopy(kwargs))
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="stop",
                    message=SimpleNamespace(content=raw, refusal=None))])
            return original(**kwargs)

        self.client.chat.completions.create = create
        preparer = self.preparer()
        result = preparer.prepare(self.patient, self.result(preparer))
        fixed_query = json.dumps({"Key Questions": ["Which these findings treatment fits?"]})
        self.assertEqual([q["query"] for q in self.searches], [fixed_query, good, good])
        fixed = result["components"]["guidelines"]["capture"]
        self.assertEqual(fixed["guideline_query"], fixed_query)
        self.assertEqual(fixed["guideline_question_screen"], {"first_reply": raw, "first_terms": ["hypertension"],
                         "retry_reply": raw, "retry_terms": ["hypertension"]})
        fixed_requests = [q for q in self.client.requests if q.get("response_format")]
        self.assertEqual(len(fixed_requests), 2)
        self.assertEqual(fixed_requests[0], fixed_requests[1])
        self.assertIn("Keep recorded measurements", fixed_requests[0]["messages"][1]["content"])
        self.assertTrue(all(c["status"] == "completed" for c in result["components"].values()))
        self.assertEqual(len(preparer.retriever.requests), 3)
        self.assertNotIn("agent_diagnostics", result["components"]["agent_patients"]["capture"])
        for variant in self.data["selected_variants"]:
            self.assertIsNotNone(result["variants"][variant]["assessment_plan_prompts"])

    def test_question_diagnostic_survives_interruption_and_store_reopen_before_any_retrieval(self):
        self.data["selected_variants"] = ["agentic_guidelines"]
        self.client = MeteredAgentClient([
            agent_message(("search_guidelines", {"question": "Which hypertension treatment fits?"}))])
        path, interrupted = self.root / "query.sqlite3", []
        key = ("preparation", self.patient["id"], None, None)
        store = JobStore(path)
        self.addCleanup(store.close)

        def save(result):
            store.save(key, result)
            if result["components"]["agent_guidelines"]["capture"].get("agent_diagnostics") and not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt()

        preparer = self.preparer(checkpoint=save)
        with self.assertRaises(KeyboardInterrupt):
            preparer.prepare(self.patient, self.result(preparer))
        store.close()
        store = JobStore(path)
        self.addCleanup(store.close)
        saved = store.load(key)
        capture = saved["components"]["agent_guidelines"]["capture"]
        self.assertEqual(self.searches, [])
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(capture["agent_diagnostics"][0]["substage"], "query_validation")
        self.assertEqual(capture["agent"]["screened_questions"][0]["terms"], ["hypertension"])
        self.assertEqual(capture["model_responses"]["agent_control"][0]["token_usage"]["total_tokens"], 108)
        validate_preparation_result(preparer.experiment, saved)

    def test_question_diagnostic_persistence_failure_is_fatal_before_retrieval(self):
        self.data["selected_variants"] = ["agentic_guidelines"]
        self.client = MeteredAgentClient([
            agent_message(("search_guidelines", {"question": "Which hypertension treatment fits?"}))])

        def save(result):
            if result["components"]["agent_guidelines"]["capture"].get("agent_diagnostics"):
                raise OSError("disk full")

        preparer = self.preparer(checkpoint=save)
        with self.assertRaises(RunMetadataError):
            preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(self.searches, [])
        self.assertEqual(len(self.client.requests), 1)

    def test_completed_preparation_and_exact_prompts_are_reused(self):
        preparer = self.preparer()
        result = preparer.prepare(self.patient, self.result(preparer))
        before = deepcopy(result)
        preparer.prepare(self.patient, result)
        self.assertEqual(result, before)
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(len(self.searches), 1)

    def test_persistent_resources_keep_each_patients_capture_and_prompts_isolated(self):
        preparer = self.preparer()
        first = preparer.prepare(self.patient, self.result(preparer))
        saved_first = deepcopy(first)
        second_patient = {"id": "patient-b", "subjective": "Different S", "objective": "Different O"}
        second = self.result(preparer)
        second["id"] = second_patient["id"]
        preparer.prepare(second_patient, second)
        self.assertEqual(first, saved_first)
        self.assertEqual(len(preparer.retriever.requests), 2)
        self.assertEqual(len(preparer.reranker.requests), 2)
        self.assertIn("Different S", second["variants"]["llm_only"]["assessment_plan_prompts"]["user"])
        self.assertNotIn("Different S", first["variants"]["llm_only"]["assessment_plan_prompts"]["user"])
        self.assertIsNone(getattr(preparer.retriever.keyword_extractor, "capture_result", None))

    def test_one_evidence_failure_blocks_only_its_consumers(self):
        preparer = self.preparer()
        preparer.retriever.search = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Retrieval failed"))
        result = preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(result["components"]["patient_retrieval"]["status"], "failed_incomplete")
        self.assertEqual(result["components"]["guidelines"]["status"], "completed")
        for variant in ("llm_only", "guidelines"):
            self.assertIsNotNone(result["variants"][variant]["assessment_plan_prompts"])
        for variant in ("patient_retrieval", "patient_retrieval_guidelines"):
            self.assertIsNone(result["variants"][variant]["assessment_plan_prompts"])
        self.assertEqual(len(self.errors), 1)
        self.assertEqual((self.errors[0]["patient_id"], self.errors[0]["stage"]),
                         (self.patient["id"], "patient_retrieval"))

    def test_screened_query_survives_graph_failure_and_is_not_paid_again(self):
        requests = []
        def fail_once(**kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                raise RuntimeError("Graph unavailable private-graph")
            return "Recovered evidence", {}
        preparer = self.preparer(guideline_search=fail_once)
        result = preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(result["components"]["guidelines"]["status"], "failed_incomplete")
        query = result["components"]["guidelines"]["capture"]["guideline_query"]
        self.assertTrue(any(snapshot["components"]["guidelines"]["capture"].get("guideline_query") == query
                            for snapshot in self.snapshots))
        preparer.prepare(self.patient, result)
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["query"], requests[1]["query"])
        self.assertEqual(result["components"]["guidelines"]["status"], "completed")
        self.assertNotIn("private-graph", json.dumps(self.errors))

    def test_keyword_checkpoint_is_detached_and_reused_after_reranker_crash(self):
        preparer = self.preparer()
        normal = preparer.reranker.rerank
        preparer.reranker.rerank = lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt())
        result = self.result(preparer)
        with self.assertRaises(KeyboardInterrupt):
            preparer.prepare(self.patient, result)
        saved = next(snapshot for snapshot in self.snapshots
                     if "_keyword_query" in snapshot["components"]["patient_retrieval"]["capture"])
        result["components"]["patient_retrieval"]["capture"]["retrieval_keywords"].append("later mutation")
        self.assertNotIn("later mutation", saved["components"]["patient_retrieval"]["capture"]["retrieval_keywords"])
        preparer.reranker.rerank = normal
        preparer.prepare(self.patient, saved)
        keyword_requests = [request for request in self.client.requests if request.get("temperature") == 0]
        self.assertEqual(len(keyword_requests), 1)

    def test_failed_parent_ack_propagates_instead_of_becoming_an_evidence_failure(self):
        preparer = self.preparer(checkpoint=lambda result: (_ for _ in ()).throw(OSError("disk full")))
        with self.assertRaises(RunMetadataError):
            preparer.prepare(self.patient, self.result(preparer))

    def test_llm_only_requires_no_encoders_clients_or_reference_files(self):
        self.data["selected_variants"] = ["llm_only"]
        self.corpus.unlink()
        preparer = self.preparer(encoder=None, retriever_factory=None, reranker_factory=None,
                                 guideline_search=None, client_factory=lambda **kw: self.fail("Unexpected client"))
        result = preparer.prepare(self.patient, self.result(preparer))
        self.assertEqual(result["components"], {})
        self.assertIsNotNone(result["variants"]["llm_only"]["assessment_plan_prompts"])
        self.assertEqual(preparer.metadata()["reference_data"], {"patient_retrieval": None, "guidelines": None})

    def test_metadata_records_reference_fingerprints_and_effective_devices(self):
        metadata = self.preparer().metadata()
        self.assertEqual(metadata["devices"], {"requested": "cpu", "patient_encoder": "cpu", "reranker": "cpu"})
        self.assertIsNotNone(metadata["reference_data"]["patient_retrieval"]["sha256"])
        self.assertIsNotNone(metadata["reference_data"]["guidelines"]["sha256"])

    def test_invalid_identity_and_missing_runtime_are_rejected(self):
        preparer = self.preparer()
        with self.assertRaisesRegex(ValueError, "belong"):
            preparer.prepare({**self.patient, "id": "wrong"}, self.result(preparer))
        with self.assertRaisesRegex(ValueError, "preloaded encoder"):
            self.preparer(encoder=None)
        with self.assertRaisesRegex(ValueError, "local-search"):
            self.preparer(guideline_search=None)

    def test_malformed_recovery_matrix_is_rejected_before_paid_work(self):
        preparer = self.preparer()
        cases = []
        value = self.result(preparer)
        value["components"].pop("patient_retrieval")
        cases.append(value)
        value = self.result(preparer)
        value["components"]["patient_retrieval"]["settings"]["alpha"] = 0.99
        cases.append(value)
        value = self.result(preparer)
        value["components"]["guidelines"]["capture"]["model_responses"] = None
        cases.append(value)
        value = self.result(preparer)
        value["components"]["guidelines"]["status"] = "completed"
        cases.append(value)
        value = self.result(preparer)
        value["variants"]["llm_only"]["assessment_plans"].pop("model-b")
        cases.append(value)
        value = self.result(preparer)
        value["variants"]["llm_only"]["assessment_plan_prompts"] = {"user": "Missing system"}
        cases.append(value)
        value = self.result(preparer)
        value["variants"]["llm_only"]["assessment_plans"]["model-a"]["_generation_reply_complete"] = 1
        cases.append(value)
        for value in cases:
            with self.subTest(state=value), self.assertRaises(ValueError):
                preparer.prepare(self.patient, value)
        self.assertEqual(self.client.requests, [])
        self.assertEqual(preparer.retriever.requests, [])
        self.assertEqual(self.searches, [])

    def test_received_generation_reply_is_accepted_and_preserved_during_preparation(self):
        preparer = self.preparer()
        result = self.result(preparer)
        slot = result["variants"]["llm_only"]["assessment_plans"]["model-a"]
        slot.update(status="failed_incomplete", _generation_reply_complete=True,
                    assessment_plan_raw='{"predictions": []}')
        saved = deepcopy(slot)
        preparer.prepare(self.patient, result)
        self.assertEqual(slot, saved)

    def test_public_validation_accepts_fresh_and_partial_results_without_mutating_them(self):
        preparer = self.preparer()
        for status in (None, "not_processed", "failed_incomplete", "completed"):
            result = self.result(preparer)
            result["status"] = status
            slot = result["variants"]["llm_only"]["assessment_plans"]["model-a"]
            slot.update(status="failed_incomplete", _generation_reply_complete=True,
                        assessment_plan_raw="received before parsing")
            before = deepcopy(result)
            self.assertIsNone(validate_preparation_result(preparer.experiment, result))
            self.assertEqual(result, before)
        self.assertEqual(self.client.requests, [])

    def test_public_validation_rejects_prediction_and_rank_corruption_before_work(self):
        preparer = self.preparer()
        invalid = []
        for predictions in ({}, [], [{"condition": "DX", "plan": []}] * 2,
                            [{"condition": "DX", "plan": [1]}] * 3,
                            [{"condition": 1, "plan": []}] * 3,
                            [{"condition": "DX", "plan": [], "unknown": "value"}] * 3):
            result = self.result(preparer)
            result["variants"]["llm_only"]["assessment_plans"]["model-a"]["predictions"] = predictions
            invalid.append(result)
        for evaluation in ([], [{"rank": True}, {"rank": 2}, {"rank": 3}],
                           [{"rank": 1}, {"rank": 3}, {"rank": 2}], [{"rank": 1}] * 3):
            result = self.result(preparer)
            result["variants"]["llm_only"]["assessment_plans"]["model-a"]["evaluation"] = evaluation
            invalid.append(result)
        for result in (None, {}, {"id": ""}, *invalid):
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate_preparation_result(preparer.experiment, result)
        self.assertEqual(self.client.requests, [])


class WorkerProtocolTests(unittest.TestCase):
    """Run the actual worker entry point with model and preparation doubles."""

    def execute(self, commands, *, variants=None, bootstrap_failure=False):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "experiment.yaml"
            config.write_text(yaml.safe_dump({
                "selected_models": ["generator"], "selected_variants": variants or ["patient_retrieval"],
                "temperature": 1, "max_retries": 3, "max_patients": None,
            }))
            events = []
            modules = {name: ModuleType(name) for name in (
                "torch", "sentence_transformers", "scripts.med_copilot_retrieval",
                "graphrag.cli.query", "scripts.experiment_runtime.preparation")}
            modules["torch"].set_num_threads = lambda n: events.append(("threads", n))
            def encoder(name, **kwargs):
                events.append(("encoder", kwargs))
                if bootstrap_failure:
                    raise RuntimeError("private-openai failed")
                return SimpleNamespace(device=kwargs["device"])
            modules["sentence_transformers"].SentenceTransformer = encoder
            modules["sentence_transformers"].CrossEncoder = lambda name, **kwargs: SimpleNamespace(device=kwargs["device"])
            modules["scripts.med_copilot_retrieval"].HybridRetriever = object
            modules["scripts.med_copilot_retrieval"].CrossEncoderReranker = type("Reranker", (), {})
            modules["graphrag.cli.query"].run_local_search = object()
            class Preparer:
                def __init__(self, path, **kwargs):
                    self.checkpoint = kwargs["checkpoint"]
                    self.error = kwargs["error_callback"]
                    self.errors = []
                    self.reranker = kwargs["reranker_factory"]("reranker")
                    events.append(("reranker_device", str(self.reranker.model.device)))

                def metadata(self):
                    return {"devices": {"requested": "cpu"}}

                def prepare(self, patient, result):
                    print("library diagnostic")
                    self.checkpoint(result)
                    entry = {"message": "private-openai was redacted"}
                    self.error(entry)
                    self.errors.append(entry)
                    return {**result, "prepared": True}
            modules["scripts.experiment_runtime.preparation"].EvidencePreparer = Preparer
            original_import = builtins.__import__
            def tracked_import(name, *args, **kwargs):
                if name == "scripts.med_copilot_retrieval":
                    events.append(("retrieval_import", None))
                return original_import(name, *args, **kwargs)
            output, errors = io.StringIO(), io.StringIO()
            source = io.StringIO("".join(json.dumps(command) + "\n" for command in commands))
            with patch.dict(sys.modules, modules), \
                    patch.dict(os.environ, OPENAI_API_KEY="private-openai"), \
                    patch.object(sys, "argv", ["worker.py", str(config)]), \
                    patch.object(sys, "stdin", source), patch("builtins.__import__", side_effect=tracked_import), \
                    redirect_stdout(output), redirect_stderr(errors), self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(ROOT / "scripts/experiment_runtime/worker.py"), run_name="__main__")
            return stopped.exception.code, [json.loads(line) for line in output.getvalue().splitlines()], errors.getvalue(), events

    def test_bootstrap_order_and_acknowledged_protocol_with_clean_stdout(self):
        code, output, diagnostics, events = self.execute([
            {"type": "prepare", "patient": {"id": "a"}, "result": {"id": "a"}},
            {"type": "ack"}, {"type": "ack"}, {"type": "shutdown"},
        ])
        self.assertEqual(code, 0)
        self.assertEqual([event["type"] for event in output], ["ready", "checkpoint", "error", "result"])
        self.assertTrue(output[-1]["result"]["prepared"])
        self.assertIn("library diagnostic", diagnostics)
        self.assertLess(next(i for i, e in enumerate(events) if e[0] == "encoder"),
                        next(i for i, e in enumerate(events) if e[0] == "retrieval_import"))
        self.assertIn(("reranker_device", "cpu"), events)
        self.assertNotIn("private-openai", json.dumps(output))

    def test_parent_disappearance_stops_before_publishing_success(self):
        code, output, _, _ = self.execute([
            {"type": "prepare", "patient": {"id": "a"}, "result": {"id": "a"}},
        ])
        self.assertEqual(code, 1)
        self.assertEqual([event["type"] for event in output], ["ready", "checkpoint", "fatal"])
        self.assertEqual(output[-1]["error"]["type"], "RunMetadataError")

    def test_bootstrap_failure_is_a_redacted_fatal_protocol_event(self):
        code, output, _, _ = self.execute([], bootstrap_failure=True)
        self.assertEqual(code, 1)
        self.assertEqual([event["type"] for event in output], ["fatal"])
        self.assertNotIn("private-openai", json.dumps(output))

    def test_idle_worker_exits_on_eof(self):
        code, output, _, _ = self.execute([], variants=["llm_only"])
        self.assertEqual(code, 0)
        self.assertEqual([event["type"] for event in output], ["ready"])


class WorkerStreamTests(unittest.TestCase):
    def test_split_secret_is_redacted_even_across_flushes(self):
        target = io.StringIO()
        stream = RedactingStream(target, ["private-key", "private", None, ""])
        stream.write("failure: priv")
        stream.flush()
        self.assertEqual(target.getvalue(), "")
        stream.write("ate-key\nnext line")
        self.assertEqual(target.getvalue(), "failure: [REDACTED]\n")
        stream.finish()
        self.assertEqual(target.getvalue(), "failure: [REDACTED]\nnext line")

    def test_unterminated_diagnostic_is_redacted_at_shutdown(self):
        target = io.StringIO()
        stream = RedactingStream(target, ["credential"])
        self.assertEqual(stream.write("credential"), 10)
        stream.finish()
        stream.finish()
        self.assertEqual(target.getvalue(), "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
