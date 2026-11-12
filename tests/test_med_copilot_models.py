"""Offline tests using real upstream config/factory/completion wrappers, mocked transport."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from contextvars import copy_context
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Barrier, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

# Resolve the vendored package and its local model catalogue before model imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.factory import ModelFactory
import graphrag.language_model.providers.litellm.chat_model as upstream_chat
from litellm import RateLimitError
from med_copilot.prompts.diagnosis_plan import CONDITION_PLAN_RESPONSE_FORMAT
from scripts.med_copilot_config import ExperimentConfig
from scripts.med_copilot_helpers import new_patient_result
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.med_copilot_run_file import RunMetadataError, error_scope


def response(marker="condition", **overrides):
    content = json.dumps({"predictions": [
        {"condition": marker + str(rank), "plan": [" action café "]} for rank in range(3)
    ]}, ensure_ascii=False)
    values = dict(model="provider-model", id="provider-id",
                  usage=SimpleNamespace(prompt_tokens=0, completion_tokens=4, total_tokens=4),
                  choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                      content=content, refusal=None))])
    return SimpleNamespace(**{**values, **overrides})


class AssessmentPlanModelsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "models.yaml"
        self.run_settings = {"temperature": 1, "max_retries": 3}
        self.data = {"models": {
            "mini": {"type": "chat", "model_provider": "openai", "model": "gpt-4o-mini",
                     "api_key_env": "TEST_MODEL_KEY", "retry_strategy": "none"},
            "llama": {"type": "chat", "model_provider": "groq", "model": "llama-3.3-70b-versatile",
                      "api_key_env": "TEST_LLAMA_KEY", "output_mode": "json_object", "retry_strategy": "none"},
        }}
        self.write()
        env = patch.dict(os.environ, {"TEST_MODEL_KEY": "secret-mini", "TEST_LLAMA_KEY": "secret-llama"})
        env.start()
        self.addCleanup(env.stop)
        self.messages = [{"role": "system", "content": "clinical system"},
                         {"role": "user", "content": "exact evidence café\n"}]
        self.transport = Mock(return_value=response())
        transport = patch.object(upstream_chat, "completion", self.transport)
        transport.start()
        self.addCleanup(transport.stop)
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def write(self):
        self.path.write_text(yaml.safe_dump(self.data, sort_keys=False), encoding="utf-8")

    def load_models(self, selected=None):
        return AssessmentPlanModels(self.path, ["mini"] if selected is None else selected,
                                    **self.run_settings)

    def generate(self, selected=None):
        models = self.load_models(selected)
        result = new_patient_result("p", model_ids=models.model_ids)
        models.generate(self.messages, result)
        return models, result

    def test_generate_one_routes_only_requested_alias_and_returns_its_owned_slot(self):
        models = self.load_models(["mini", "llama"])
        result = new_patient_result("p", model_ids=models.model_ids)
        patient_status = result["status"]
        sibling = deepcopy(result["assessment_plans"]["mini"])
        output = result["assessment_plans"]["llama"]
        self.assertIs(models.generate_one("llama", self.messages, output), output)
        self.transport.assert_called_once()
        self.assertEqual(self.transport.call_args.kwargs["model"], "groq/llama-3.3-70b-versatile")
        self.assertEqual(output["status"], "completed")
        self.assertEqual(result["assessment_plans"]["mini"], sibling)
        self.assertEqual(result["status"], patient_status)  # The coordinator owns patient status.

    def test_generate_one_rejects_unselected_alias_before_mutating_or_checkpointing(self):
        models = self.load_models()
        output = new_patient_result("p", model_ids=models.model_ids)["assessment_plans"]["mini"]
        saved = deepcopy(output)
        with patch("scripts.med_copilot_models.checkpoint") as checkpoint, \
                self.assertRaisesRegex(ValueError, "not selected"):
            models.generate_one("missing", self.messages, output)
        self.transport.assert_not_called()
        checkpoint.assert_not_called()
        self.assertEqual(output, saved)

    def test_generate_one_persists_received_reply_before_parser_runs(self):
        models = self.load_models()
        output = new_patient_result("p", model_ids=models.model_ids)["assessment_plans"]["mini"]
        snapshots = []
        owner = SimpleNamespace(checkpoint=lambda: snapshots.append(deepcopy(output)))
        parse = models.generate_one.__func__.__globals__["parse_condition_plan"]

        def checked_parse(content):
            self.assertEqual(len(snapshots), 1)
            self.assertTrue(snapshots[0]["_generation_reply_complete"])
            self.assertEqual(snapshots[0]["assessment_plan_raw"], content)
            self.assertEqual(len(snapshots[0]["model_responses"]), 1)
            self.assertIsNone(snapshots[0]["predictions"])
            return parse(content)

        with error_scope(owner, patient_id="p"), \
                patch("scripts.med_copilot_models.parse_condition_plan", side_effect=checked_parse):
            models.generate_one("mini", self.messages, output)
        self.assertEqual([snapshot["status"] for snapshot in snapshots], ["failed_incomplete", "completed"])
        self.assertNotIn("_generation_reply_complete", snapshots[-1])

    def test_generate_one_durable_reply_checkpoint_failure_stops_before_parse(self):
        models = self.load_models()
        output = new_patient_result("p", model_ids=models.model_ids)["assessment_plans"]["mini"]
        with patch("scripts.med_copilot_models.checkpoint", side_effect=[RunMetadataError("disk full"), None]), \
                patch("scripts.med_copilot_models.parse_condition_plan") as parse, \
                self.assertRaisesRegex(RunMetadataError, "disk full"):
            models.generate_one("mini", self.messages, output)
        parse.assert_not_called()
        self.assertTrue(output["_generation_reply_complete"])
        self.assertEqual(output["status"], "failed_incomplete")
        models.generate_one("mini", self.messages, output)
        self.transport.assert_called_once()
        self.assertEqual(output["status"], "completed")

    def test_generate_one_overlaps_same_alias_requests_without_crossing_job_state(self):
        models = self.load_models()
        barrier = Barrier(2)
        snapshots = {name: [] for name in ("FIRST", "SECOND")}
        lock = Lock()
        outputs = {name: new_patient_result(name, model_ids=models.model_ids)["assessment_plans"]["mini"]
                   for name in snapshots}

        def answer(**request):
            marker = request["messages"][0]["content"]
            barrier.wait(timeout=5)
            return response(marker)

        def save(name):
            with lock:
                snapshots[name].append(deepcopy(outputs[name]))

        self.transport.side_effect = answer
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for name in snapshots:
                owner = SimpleNamespace(checkpoint=lambda name=name: save(name))
                with error_scope(owner, patient_id=name):
                    futures.append(executor.submit(copy_context().run, models.generate_one, "mini",
                                                   [{"role": "user", "content": name}], outputs[name]))
            for future in futures:
                future.result(timeout=10)
        self.assertEqual(self.transport.call_count, 2)
        for name, output in outputs.items():
            self.assertEqual(output["predictions"][0]["condition"], name + "0")
            self.assertEqual(len(output["model_responses"]), 1)
            self.assertEqual(len(snapshots[name]), 2)
            self.assertTrue(snapshots[name][0]["_generation_reply_complete"])
            self.assertEqual(snapshots[name][-1], output)
        self.assertEqual(models.metadata(self.root)["selected_models"], ["mini"])

    def test_model_is_local_recognizes_loopback_endpoints_only(self):
        for endpoint, expected in ((None, False), ("http://localhost:11434/v1", True),
                                   ("http://127.0.0.1:8000/v1", True), ("http://127.0.0.2:8000/v1", True),
                                   ("http://[::1]:11434/v1", True), ("https://api.example.com/v1", False),
                                   ("http://192.168.1.2:11434/v1", False)):
            with self.subTest(endpoint=endpoint):
                if endpoint is None:
                    self.data["models"]["mini"].pop("api_base", None)
                else:
                    self.data["models"]["mini"]["api_base"] = endpoint
                self.write()
                self.assertIs(self.load_models().model_is_local("mini"), expected)
        with self.assertRaisesRegex(ValueError, "not selected"):
            self.load_models().model_is_local("missing")
        self.transport.assert_not_called()

    def test_resume_reuses_received_reply_after_interruption_during_local_parsing(self):
        models = self.load_models(["mini", "llama"])
        result = new_patient_result("p", model_ids=models.model_ids)
        for _ in range(2):
            with patch("scripts.med_copilot_models.parse_condition_plan", side_effect=KeyboardInterrupt()), \
                    self.assertRaises(KeyboardInterrupt):
                models.generate(self.messages, result)
            result = json.loads(json.dumps(result))
            self.assertEqual(self.transport.call_count, 1)
        saved = json.loads(json.dumps(result))  # Exercise the persisted representation.
        raw = saved["assessment_plans"]["mini"]["assessment_plan_raw"]
        self.assertIsInstance(raw, str)
        self.assertEqual(self.transport.call_count, 1)
        self.load_models(["mini", "llama"]).generate(self.messages, saved)
        self.assertEqual(self.transport.call_count, 2)  # Only the unstarted sibling needs a request.
        first = saved["assessment_plans"]["mini"]
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["assessment_plan_raw"], raw)
        self.assertIsNotNone(first["predictions"])
        self.assertEqual(len(first["model_responses"]), 1)

    def test_resume_retries_parsing_failure_without_repeating_completed_api_call(self):
        models = self.load_models()
        result = new_patient_result("p", model_ids=models.model_ids)
        with patch("scripts.med_copilot_models.parse_condition_plan", side_effect=RuntimeError("parser failed")):
            models.generate(self.messages, result)
        self.assertEqual(result["assessment_plans"]["mini"]["status"], "failed_incomplete")
        models.generate(self.messages, result)
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(result["assessment_plans"]["mini"]["status"], "completed")

    def test_resume_requests_replacement_for_truncated_or_refused_replies(self):
        for finish, refusal in (("length", None), ("stop", "refused")):
            with self.subTest(finish=finish, refusal=refusal):
                self.transport.reset_mock()
                reply = response("rejected")
                reply.choices[0].finish_reason = finish
                reply.choices[0].message.refusal = refusal
                self.transport.return_value = reply
                models, result = self.generate()
                self.assertEqual(result["assessment_plans"]["mini"]["status"], "failed_incomplete")
                self.transport.return_value = response("healthy")
                models.generate(self.messages, result)
                self.assertEqual(self.transport.call_count, 2)
                self.assertEqual(result["assessment_plans"]["mini"]["predictions"][0]["condition"], "healthy0")

    def test_reuses_real_upstream_config_factory_and_preserves_provider_response(self):
        with patch.object(ModelFactory, "create_chat_model", wraps=ModelFactory.create_chat_model) as factory:
            models, result = self.generate()
        factory.assert_called_once()
        config = factory.call_args.kwargs["config"]
        self.assertIsInstance(config, LanguageModelConfig)
        self.assertEqual(config.api_key, "secret-mini")
        request = self.transport.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-4o-mini")
        self.assertEqual(request["messages"], self.messages)
        self.assertEqual(request["response_format"], CONDITION_PLAN_RESPONSE_FORMAT)
        self.assertIs(request["drop_params"], False)
        self.assertIs(request["stream"], False)
        output = result["assessment_plans"]["mini"]
        self.assertEqual(output["model_responses"], [{"reported_model": "provider-model",
            "response_id": "provider-id", "token_usage": {
                "prompt_tokens": 0, "completion_tokens": 4, "total_tokens": 4}}])
        self.assertEqual(output["assessment_plan_raw"], self.transport.return_value.choices[0].message.content)
        self.assertEqual(output["predictions"][0]["plan"], [" action café "])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(models.model_ids, ("mini",))

    def test_generation_leaves_final_console_output_to_the_runner(self):
        output = io.StringIO()
        with redirect_stdout(output):
            _, result = self.generate(["mini", "llama"])
        self.assertEqual(output.getvalue(), "")
        self.assertTrue(all(item["assessment_plan_raw"] is not None
                            for item in result["assessment_plans"].values()))

    def test_resume_skips_completed_sibling_and_checkpoints_only_new_generation(self):
        models = self.load_models(["mini", "llama"])
        result = new_patient_result("p", model_ids=models.model_ids)
        completed = result["assessment_plans"]["mini"]
        completed.update(status="completed", assessment_plan_raw="saved reply",
                         predictions=[{"condition": "saved", "plan": ["saved action"]}],
                         evaluation=[{"saved": "score"}])
        saved = deepcopy(completed)
        snapshots = []
        with patch("scripts.med_copilot_models.checkpoint", create=True,
                   side_effect=lambda: snapshots.append(deepcopy(result))):
            models.generate(self.messages, result)
        self.transport.assert_called_once()
        self.assertEqual(self.transport.call_args.kwargs["model"], "groq/llama-3.3-70b-versatile")
        self.assertEqual(completed, saved)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(snapshots), 2)
        pending = snapshots[0]["assessment_plans"]["llama"]
        self.assertEqual(pending["status"], "failed_incomplete")
        self.assertTrue(pending["_generation_reply_complete"])
        self.assertIsNone(pending["predictions"])
        self.assertEqual(snapshots[1]["assessment_plans"]["llama"]["status"], "completed")

    def test_resume_preserves_completed_but_invalid_generation_without_requests(self):
        models = self.load_models()
        result = new_patient_result("p", model_ids=models.model_ids)
        result["assessment_plans"]["mini"].update(status="completed", assessment_plan_raw="invalid JSON")
        saved = deepcopy(result["assessment_plans"]["mini"])
        with patch("scripts.med_copilot_models.checkpoint", create=True) as checkpoint:
            models.generate(self.messages, result)
        self.transport.assert_not_called()
        checkpoint.assert_not_called()
        self.assertEqual(result["assessment_plans"]["mini"], saved)
        self.assertEqual(result["status"], "completed")

    def test_resume_retries_interrupted_slot_without_repeating_earlier_model(self):
        models = self.load_models(["mini", "llama"])
        result = new_patient_result("p", model_ids=models.model_ids)
        self.transport.side_effect = [response("first"), KeyboardInterrupt()]
        snapshots = []
        with patch("scripts.med_copilot_models.checkpoint", create=True,
                   side_effect=lambda: snapshots.append(deepcopy(result))):
            with self.assertRaises(KeyboardInterrupt):
                models.generate(self.messages, result)
        first = deepcopy(result["assessment_plans"]["mini"])
        self.assertEqual([snapshot["assessment_plans"]["llama"]["status"]
                          for snapshot in snapshots], ["not_processed", "not_processed", "failed_incomplete"])
        self.assertEqual(result["assessment_plans"]["llama"]["error"]["type"], "KeyboardInterrupt")
        self.transport.reset_mock()
        self.transport.side_effect = None
        self.transport.return_value = response("retried")
        with patch("scripts.med_copilot_models.checkpoint", create=True) as checkpoint:
            models.generate(self.messages, result)
        self.transport.assert_called_once()
        self.assertEqual(self.transport.call_args.kwargs["model"], "groq/llama-3.3-70b-versatile")
        self.assertEqual(checkpoint.call_count, 2)
        self.assertEqual(result["assessment_plans"]["mini"], first)
        self.assertIsNone(result["assessment_plans"]["llama"]["error"])
        self.assertEqual(result["assessment_plans"]["llama"]["predictions"][0]["condition"], "retried0")
        self.assertEqual(result["status"], "completed")

    def test_failed_calls_checkpoint_failure_and_successful_sibling(self):
        self.transport.side_effect = [RuntimeError("temporary failure"), response("good")]
        snapshots = []
        models = self.load_models(["mini", "llama"])
        result = new_patient_result("p", model_ids=models.model_ids)
        with patch("scripts.med_copilot_models.checkpoint", create=True,
                   side_effect=lambda: snapshots.append(deepcopy(result))):
            models.generate(self.messages, result)
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(snapshots[0]["assessment_plans"]["mini"]["error"]["type"], "RuntimeError")
        self.assertTrue(snapshots[1]["assessment_plans"]["llama"]["_generation_reply_complete"])
        self.assertEqual(snapshots[2]["assessment_plans"]["llama"]["status"], "completed")

    def test_checkpoint_write_failure_stops_before_requesting_another_model(self):
        models = self.load_models(["mini", "llama"])
        result = new_patient_result("p", model_ids=models.model_ids)
        with patch("scripts.med_copilot_models.checkpoint", create=True,
                   side_effect=OSError("checkpoint is unwritable")):
            with self.assertRaisesRegex(OSError, "checkpoint is unwritable"):
                models.generate(self.messages, result)
        self.transport.assert_called_once()
        self.assertEqual(result["assessment_plans"]["mini"]["status"], "failed_incomplete")
        self.assertTrue(result["assessment_plans"]["mini"]["_generation_reply_complete"])
        self.assertIsNone(result["assessment_plans"]["mini"]["predictions"])
        self.assertEqual(result["assessment_plans"]["llama"]["status"], "not_processed")

    def test_retry_discards_failed_attempt_payload_and_scores_but_keeps_response_history(self):
        models = self.load_models()
        for reply in (response("new"), RuntimeError("retry failed before response")):
            with self.subTest(reply=type(reply).__name__):
                result = new_patient_result("p", model_ids=models.model_ids)
                output = result["assessment_plans"]["mini"]
                output.update(status="failed_incomplete", error={"type": "OldFailure"},
                              predictions=[{"condition": "old", "plan": ["old action"]}],
                              assessment_plan_raw="old partial reply", evaluation=[{"old": "score"}],
                              model_responses=[{"response_id": "old-response"}])
                self.transport.side_effect = reply if isinstance(reply, Exception) else None
                self.transport.return_value = reply
                models.generate(self.messages, result)
                self.assertIsNone(output["evaluation"])
                self.assertEqual(output["model_responses"][0], {"response_id": "old-response"})
                if isinstance(reply, Exception):
                    self.assertIsNone(output["predictions"])
                    self.assertIsNone(output["assessment_plan_raw"])
                    self.assertEqual(len(output["model_responses"]), 1)
                else:
                    self.assertEqual(output["predictions"][0]["condition"], "new0")
                    self.assertEqual(len(output["model_responses"]), 2)

    def test_shipped_registry_defaults_and_all_hosted_local_examples_use_upstream(self):
        experiment = ExperimentConfig(ROOT / "experiments/configs/experiment.yaml")
        path = experiment.model_config
        settings = {"temperature": experiment.temperature, "max_retries": experiment.max_retries}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-openai", "GROQ_API_KEY": "offline-groq"}):
            self.assertEqual(AssessmentPlanModels(path, experiment.selected_models, **settings).model_ids,
                             ("gpt_4o_mini", "ollama_gemma3_4b", "ollama_llama3_2_3b", "ollama_ministral_3_3b"))
            models = AssessmentPlanModels(path, ["gpt_4o_mini", "llama_3_3_70b", "llama_3_3_70b_local"],
                                          **settings)
            result = new_patient_result("p", model_ids=models.model_ids)
            models.generate(self.messages, result)
        self.assertEqual(result["status"], "completed")
        requests = [call.kwargs for call in self.transport.call_args_list]
        self.assertEqual([request["model"] for request in requests], ["openai/gpt-4o-mini",
                         "groq/llama-3.3-70b-versatile", "openai/meta-llama/Llama-3.3-70B-Instruct"])
        self.assertEqual(requests[-1]["api_base"], "http://localhost:8000/v1")
        self.assertEqual(requests[-1]["api_key"], "local-no-key")
        self.assertTrue(all(request["temperature"] == experiment.temperature for request in requests))

    def test_shared_experiment_values_reach_every_upstream_client_request_and_metadata(self):
        self.run_settings.update(temperature=0.35, max_retries=5)
        with patch.object(ModelFactory, "create_chat_model", wraps=ModelFactory.create_chat_model) as factory:
            models, result = self.generate(["mini", "llama"])
        self.assertEqual(result["status"], "completed")
        for call in factory.call_args_list:
            self.assertEqual(call.kwargs["config"].temperature, 0.35)
            self.assertEqual(call.kwargs["config"].max_retries, 5)
        self.assertTrue(all(call.kwargs["temperature"] == 0.35 for call in self.transport.call_args_list))
        for settings in models.metadata(self.root)["models"].values():
            self.assertEqual(settings["config"]["temperature"], 0.35)
            self.assertEqual(settings["config"]["max_retries"], 5)

    def test_downloaded_ollama_models_use_existing_upstream_transport_and_strict_schema_without_keys(self):
        experiment = ExperimentConfig(ROOT / "experiments/configs/experiment.yaml")
        aliases = ["ollama_healthcare", "ollama_llama3", "ollama_gemma2_2b"]
        tags = ["Jayasimma/healthcare:latest", "llama3:latest", "gemma2:2b"]
        with patch.dict(os.environ, {}, clear=True):
            models = AssessmentPlanModels(experiment.model_config, aliases,
                temperature=experiment.temperature, max_retries=experiment.max_retries)
            result = new_patient_result("p", model_ids=models.model_ids)
            models.generate(self.messages, result)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(list(result["assessment_plans"]), aliases)
        self.assertEqual(self.transport.call_count, 3)
        for request, tag in zip(self.transport.call_args_list, tags):
            self.assertEqual(request.kwargs["model"], "openai/" + tag)
            self.assertEqual(request.kwargs["api_base"], "http://localhost:11434/v1")
            self.assertEqual(request.kwargs["api_key"], "local-no-key")
            self.assertEqual(request.kwargs["response_format"], CONDITION_PLAN_RESPONSE_FORMAT)
            self.assertEqual(request.kwargs["messages"], self.messages)
            self.assertIs(request.kwargs["drop_params"], False)
        metadata = models.metadata(ROOT)
        for alias, tag in zip(aliases, tags):
            self.assertIsNone(metadata["models"][alias]["api_key_env"])
            self.assertEqual(metadata["models"][alias]["config"]["model"], tag)
            self.assertNotIn("api_key", metadata["models"][alias]["config"])
            self.assertEqual(result["assessment_plans"][alias]["status"], "completed")
            self.assertEqual(len(result["assessment_plans"][alias]["predictions"]), 3)

    def test_run_settings_cannot_be_reintroduced_in_registry_or_per_model(self):
        original = deepcopy(self.data)
        for key, value in (("selected_models", ["mini"]), ("temperature", 1), ("max_retries", 3)):
            for target in ("root", "model"):
                with self.subTest(key=key, target=target):
                    self.data = deepcopy(original)
                    destination = self.data if target == "root" else self.data["models"]["mini"]
                    destination[key] = value
                    self.write()
                    with patch.object(ModelFactory, "create_chat_model") as factory, self.assertRaises(ValueError):
                        self.load_models()
                    factory.assert_not_called()

    def test_multiple_providers_modes_order_and_independent_mutable_messages(self):
        self.transport.side_effect = [response("llama"), response("mini")]
        models, result = self.generate(["llama", "mini"])
        first, second = [call.kwargs for call in self.transport.call_args_list]
        self.assertEqual([first["model"], second["model"]],
                         ["groq/llama-3.3-70b-versatile", "openai/gpt-4o-mini"])
        self.assertEqual(first["response_format"], {"type": "json_object"})
        self.assertEqual(second["response_format"], CONDITION_PLAN_RESPONSE_FORMAT)
        self.assertEqual(first["messages"], second["messages"])
        first["messages"][0]["content"] = "mutated"
        self.assertEqual(second["messages"], self.messages)
        self.assertEqual(list(result["assessment_plans"]), ["llama", "mini"])
        self.assertEqual(result["assessment_plans"]["llama"]["predictions"][0]["condition"], "llama0")
        self.assertEqual(result["assessment_plans"]["mini"]["predictions"][0]["condition"], "mini0")
        self.assertEqual(models.metadata(self.root)["selected_models"], ["llama", "mini"])

    def test_all_native_provider_routes_and_local_no_key_endpoint(self):
        for provider in ("anthropic", "gemini", "openai"):
            with self.subTest(provider=provider):
                self.data["models"]["mini"]["model_provider"] = provider
                self.write()
                self.generate()
                self.assertEqual(self.transport.call_args.kwargs["model"], provider + "/gpt-4o-mini")
        entry = self.data["models"]["mini"]
        entry.pop("api_key_env")
        entry.update(api_base="http://localhost:8000/v1", model="meta-llama/Llama-3.3-70B-Instruct",
                     output_mode="prompt")
        self.write()
        self.generate()
        request = self.transport.call_args.kwargs
        self.assertEqual(request["api_base"], "http://localhost:8000/v1")
        self.assertEqual(request["api_key"], "local-no-key")
        self.assertNotIn("response_format", request)

    def test_only_selected_credentials_are_required_and_clients_reused_across_patients(self):
        with patch.dict(os.environ, {"TEST_LLAMA_KEY": ""}), \
                patch.object(ModelFactory, "create_chat_model", wraps=ModelFactory.create_chat_model) as factory:
            models, first = self.generate()
            second = new_patient_result("other", model_ids=models.model_ids)
            models.generate(self.messages, second)
            factory.assert_called_once()
        first["assessment_plans"]["mini"]["model_responses"].clear()
        self.assertEqual(len(second["assessment_plans"]["mini"]["model_responses"]), 1)

    def test_metadata_is_effective_sanitized_snapshot_with_exact_config_fingerprint(self):
        models = self.load_models()
        metadata = models.metadata(self.root)
        self.assertEqual(metadata["config_file"], "models.yaml")
        self.assertEqual(metadata["config_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertNotIn("secret-mini", json.dumps(metadata))
        self.assertNotIn("api_key", metadata["models"]["mini"]["config"])
        self.assertEqual(metadata["models"]["mini"]["config"]["temperature"], 1)
        metadata["models"]["mini"]["config"]["temperature"] = 99
        self.assertEqual(models.metadata(self.root)["models"]["mini"]["config"]["temperature"], 1)
        self.assertEqual(models.metadata(self.root / "other")["config_file"], str(self.path.resolve()))

    def test_api_failure_retains_later_model_success_and_redacts_all_selected_secrets(self):
        self.transport.side_effect = [RuntimeError("secret-mini secret-llama"), response("good")]
        _, result = self.generate(["mini", "llama"])
        bad, good = result["assessment_plans"].values()
        self.assertEqual(bad["status"], "failed_incomplete")
        self.assertEqual(bad["model_responses"], [])
        self.assertEqual(bad["error"], {"type": "RuntimeError", "message": "[REDACTED] [REDACTED]"})
        self.assertIsNone(bad["evaluation"])
        self.assertEqual(good["status"], "completed")
        self.assertEqual(result["status"], "failed_incomplete")

    def test_refused_truncated_and_missing_choices_preserve_available_response_capture(self):
        for finish, refusal, choices in (("length", None, True), ("stop", "refused", True),
                                         ("stop", None, False)):
            with self.subTest(finish=finish, choices=choices, refusal=refusal):
                reply = response()
                reply.choices[0].finish_reason = finish
                reply.choices[0].message.refusal = refusal
                raw = reply.choices[0].message.content
                if not choices:
                    reply.choices = []
                self.transport.return_value = reply
                _, result = self.generate()
                output = result["assessment_plans"]["mini"]
                self.assertEqual(output["status"], "failed_incomplete")
                self.assertEqual(len(output["model_responses"]), 1)
                self.assertEqual(output["assessment_plan_raw"], raw if choices else None)
                self.assertIsNone(output["predictions"])

    def test_invalid_json_keeps_exact_raw_and_existing_field_salvage_semantics(self):
        for raw in ('  {bad json café\n', None, 12, '{"predictions":[]}'):
            with self.subTest(raw=raw):
                self.transport.return_value = response()
                self.transport.return_value.choices[0].message.content = raw
                _, result = self.generate()
                output = result["assessment_plans"]["mini"]
                self.assertEqual(output["assessment_plan_raw"], raw)
                self.assertIsNone(output["predictions"])
                self.assertEqual(output["status"], "completed")
        raw = json.dumps({"predictions": [{"condition": "A", "plan": [1]},
                    {"condition": "B", "plan": []}, {"condition": "C", "plan": ["P"]}]})
        self.transport.return_value.choices[0].message.content = raw
        _, result = self.generate()
        self.assertEqual(result["assessment_plans"]["mini"]["predictions"][0],
                         {"condition": "A", "plan": None})

    def test_interruption_keeps_completed_model_and_marks_later_models_unattempted(self):
        for exception in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(exception=type(exception).__name__):
                self.data["models"]["third"] = deepcopy(self.data["models"]["mini"])
                self.write()
                self.transport.side_effect = [response(), exception]
                models = self.load_models(["mini", "llama", "third"])
                result = new_patient_result("p", model_ids=models.model_ids)
                with self.assertRaises(type(exception)):
                    models.generate(self.messages, result)
                first, middle, last = result["assessment_plans"].values()
                self.assertEqual(first["status"], "completed")
                self.assertIsNotNone(first["predictions"])
                self.assertEqual(middle["error"]["type"], type(exception).__name__)
                self.assertEqual(middle["status"], "failed_incomplete")
                self.assertEqual(last["status"], "not_processed")

    def test_upstream_retry_wrapper_is_used_without_reimplementing_retries(self):
        entry = self.data["models"]["mini"]
        entry.update(retry_strategy="exponential_backoff", max_retry_wait=0.001)
        self.run_settings["max_retries"] = 2
        self.write()
        self.transport.side_effect = [RateLimitError("retry", llm_provider="openai", model="gpt-4o-mini"), response()]
        _, result = self.generate()
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual(result["status"], "completed")

    def test_shared_credential_preflight_collects_selection_and_key_errors_before_factory(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(ModelFactory, "create_chat_model") as factory, \
                self.assertRaises(ValueError) as error:
            AssessmentPlanModels(self.path, ["typo", "mini", "llama"],
                                 **self.run_settings, require_shared_credentials=True)
        factory.assert_not_called()
        self.transport.assert_not_called()
        for expected in ("Unknown model alias: typo", "TEST_MODEL_KEY", "TEST_LLAMA_KEY",
                         "OPENAI_API_KEY", "GRAPHRAG_API_KEY", "experiment.yaml",
                         "condition_plan_models.yaml", ".env in the repository root"):
            self.assertIn(expected, str(error.exception))

    def test_shared_key_preflight_requires_only_the_requested_evidence_sources(self):
        with patch.dict(os.environ, {"TEST_MODEL_KEY": "mini", "OPENAI_API_KEY": "shared"}, clear=True):
            models = AssessmentPlanModels(self.path, ["mini"], **self.run_settings,
                                          required_shared_keys=["OPENAI_API_KEY"])
            self.assertEqual(models.model_ids, ("mini",))
            with patch.object(ModelFactory, "create_chat_model") as factory, self.assertRaisesRegex(ValueError, "GRAPHRAG_API_KEY"):
                AssessmentPlanModels(self.path, ["mini"], **self.run_settings,
                                     required_shared_keys=["OPENAI_API_KEY", "GRAPHRAG_API_KEY"])
            factory.assert_not_called()
        with patch.dict(os.environ, {"TEST_MODEL_KEY": "mini"}, clear=True):
            self.assertEqual(AssessmentPlanModels(self.path, ["mini"], **self.run_settings,
                                                  required_shared_keys=[]).model_ids, ("mini",))
        self.transport.assert_not_called()

    def test_public_error_metadata_redacts_final_and_shared_credentials_without_mutating_error(self):
        models = self.load_models(["mini", "llama"])
        exc = RuntimeError("secret-mini secret-llama shared-openai shared-graph café\nsecond line")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "shared-openai", "GRAPHRAG_API_KEY": "shared-graph"}):
            self.assertEqual(models.error_metadata(exc), {
                "type": "RuntimeError", "message": "[REDACTED] [REDACTED] [REDACTED] [REDACTED] café\nsecond line"})
        self.assertIn("secret-mini", str(exc))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(models.error_metadata(ValueError("normal failure")),
                             {"type": "ValueError", "message": "normal failure"})

    def test_error_metadata_redacts_overlapping_keys_in_full_before_shorter_prefixes(self):
        models = self.load_models()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-mini-extended", "GRAPHRAG_API_KEY": "secret-mini-extended-graph"}):
            exc = RuntimeError("secret-mini-extended-graph secret-mini-extended secret-mini")
            self.assertEqual(models.error_metadata(exc)["message"], "[REDACTED] [REDACTED] [REDACTED]")

    def test_preflight_rejects_bad_selection_without_initializing_clients(self):
        for selection in ([], ["missing"], ["mini", "mini"], [1], "mini"):
            with self.subTest(selection=selection), patch.object(ModelFactory, "create_chat_model") as factory:
                with self.assertRaises(ValueError):
                    self.load_models(selection)
                factory.assert_not_called()

    def test_preflight_rejects_missing_credentials_literal_keys_and_unknown_settings(self):
        original = deepcopy(self.data)
        for changes in ({"api_key_env": "MISSING_MODEL_TOKEN"}, {"api_key_env": ""},
                        {"api_key": "literal-secret"}, {"temperature_typo": 1},
                        {"output_mode": "automatic"}, {"type": "openai_chat"}, {"n": 2}, {"n": True},
                        {"temperature": float("nan")}, {"temperature": float("inf")},
                        {"request_timeout": "invalid"}, {"model_provider": ""},
                        {"api_base": "https://user:password@example.com/v1"},
                        {"api_base": "https://example.com/v1?api_key=secret"}):
            with self.subTest(changes=changes), patch.object(ModelFactory, "create_chat_model") as factory:
                self.data = deepcopy(original)
                self.data["models"]["mini"].update(changes)
                self.write()
                with self.assertRaises(ValueError) as error:
                    self.load_models()
                self.assertNotIn("secret-mini", str(error.exception))
                factory.assert_not_called()

    def test_validates_all_selected_models_before_constructing_any(self):
        self.data["models"]["llama"]["type"] = "invalid"
        self.write()
        with patch.object(ModelFactory, "create_chat_model") as factory, self.assertRaises(ValueError):
            self.load_models(["mini", "llama"])
        factory.assert_not_called()

    def test_uncredentialed_hosted_model_and_initialization_failure_have_safe_errors(self):
        self.data["models"]["mini"].pop("api_key_env")
        self.write()
        with self.assertRaisesRegex(ValueError, "api_key_env"):
            self.load_models()
        self.data["models"]["mini"]["api_key_env"] = "TEST_MODEL_KEY"
        self.write()
        with patch.object(ModelFactory, "create_chat_model", side_effect=RuntimeError("secret-mini")):
            with self.assertRaisesRegex(ValueError, "initialization failed") as error:
                self.load_models()
        self.assertNotIn("secret-mini", str(error.exception))

    def test_missing_malformed_duplicate_or_wrong_shape_configuration(self):
        for contents in ("[", "null", "[]", "models: {}", "selected_models: []\nmodels: {}",
                         "selected_models: [a]\nmodels: {a: {}, a: {}}", "!!python/object:builtins.object {}"):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                with self.assertRaises(ValueError):
                    self.load_models()
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load_models()

    def test_imports_stay_in_header(self):
        tree = ast.parse((ROOT / "scripts/med_copilot_models.py").read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertTrue(all(node in tree.body for node in imports))
        first_class = min(node.lineno for node in tree.body if isinstance(node, ast.ClassDef))
        self.assertTrue(all(node.lineno < first_class for node in imports))


if __name__ == "__main__":
    unittest.main()
