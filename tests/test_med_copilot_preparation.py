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
from scripts.med_copilot_helpers import GUIDELINE_TABLES, new_patient_result
from scripts.experiment_runtime.preparation import EvidencePreparer, validate_preparation_result
from scripts.med_copilot_run_file import RunMetadataError
from scripts.experiment_runtime.worker_stream import RedactingStream


class FakeClient:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
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
