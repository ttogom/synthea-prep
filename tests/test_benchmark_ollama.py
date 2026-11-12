"""Offline calibration tests; no Ollama server or model transport is used."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

# Path setup precedes benchmark's upstream completion-wrapper imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from scripts import benchmark_ollama as benchmark


class FakeModels:
    def __init__(self, path, aliases, *, temperature, max_retries):
        self.aliases = aliases
        self.options = {"temperature": temperature, "max_retries": max_retries}
        self.calls = []
        self.lock = Lock()
        self.active = {}
        self.max_active = {}
        self.global_active = self.max_global_active = 0
        self.observations = []
        self.invalid = False
        self.interrupt_at = None

    def metadata(self, root):
        return {"selected_models": self.aliases, "models": {
            alias: {"output_mode": "json_schema", "config": {"model": alias + ":tag", "max_tokens": 768}}
            for alias in self.aliases}}

    def generate_one(self, alias, messages, output):
        with self.lock:
            self.calls.append((alias, deepcopy(messages)))
            number = len(self.calls)
            self.active[alias] = self.active.get(alias, 0) + 1
            self.global_active += 1
            self.max_active[alias] = max(self.max_active.get(alias, 0), self.active[alias])
            self.max_global_active = max(self.max_global_active, self.global_active)
            self.observations.append((len(FakeServer.instances), dict(self.active), self.global_active))
        try:
            if number == self.interrupt_at:
                raise KeyboardInterrupt()
            time.sleep(0.001)
            raw = "invalid" if self.invalid else json.dumps({"predictions": [
                {"condition": f"condition {rank}", "plan": ["action"]} for rank in range(3)]})
            output.update(status="completed", assessment_plan_raw=raw,
                          predictions=benchmark.parse_condition_plan(raw)["predictions"],
                          model_responses=[{"reported_model": alias, "response_id": str(number),
                                            "token_usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}])
            return output
        finally:
            with self.lock:
                self.active[alias] -= 1
                self.global_active -= 1

    def error_metadata(self, exc):
        return {"type": type(exc).__name__, "message": str(exc)}


class FakeServer:
    instances = []

    def __init__(self, settings, log, *, parallel, max_loaded_models):
        self.settings = settings
        self.parallel = parallel
        self.max_loaded_models = max_loaded_models
        self.unloaded = []
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.closed = True

    def metadata(self):
        return {"base_url": "http://127.0.0.1:11435", "parallel_slots": self.parallel,
                "max_loaded_models": self.max_loaded_models, "context_length": 16384}

    def ps_snapshot(self):
        return {"models": [{"name": "native-model", "size_vram": 123, "context_length": 16384}]}

    def unload(self, tag):
        self.unloaded.append(tag)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "saved-run"
        self.run.mkdir()
        self.rows = [{"id": f"patient-{index}", "variants": {
            variant: {"assessment_plan_prompts": {"system": "same system café",
                       "user": variant + "\n" + "x" * (10 + index * 30)}}
            for variant in benchmark.VARIANTS}} for index in range(5)]
        self.source = self.run / "results.jsonl"
        self.write_rows()
        self.aliases = ["local_gemma", "local_llama", "local_ministral"]
        self.registry = {"models": {alias: {"type": "chat", "model_provider": "openai",
            "model": alias + ":tag", "api_base": "http://127.0.0.1:11435/v1", "max_tokens": 768,
            "output_mode": "json_schema"} for alias in self.aliases}}
        self.registry["models"]["hosted"] = {"type": "chat", "model": "gpt-4o-mini",
                                                "api_key_env": "MISSING_HOSTED_KEY"}
        self.config = self.root / "models.yaml"
        self.write_registry()
        self.experiment = SimpleNamespace(
            model_config=self.config, selected_models=["hosted", *self.aliases],
            temperature=1, max_retries=3,
            execution={"manage_ollama": False, "ollama_host": "127.0.0.1:11435",
                       "ollama_context_length": 16384},
            metadata=lambda root: {"temperature": 1, "execution": {"ollama_context_length": 16384}},
        )
        self.models = FakeModels(self.config, self.aliases, temperature=1, max_retries=3)
        self.factory = Mock(return_value=self.models)
        FakeServer.instances = []
        self.output = self.root / "benchmark.json"

    def write_rows(self):
        self.source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8")

    def write_registry(self):
        self.config.write_text(yaml.safe_dump(self.registry), encoding="utf-8")

    def run_benchmark(self, count=4):
        return benchmark.run_benchmark(self.experiment, benchmark.select_prompts(self.run, count), self.output,
                                       models_factory=self.factory, server_factory=FakeServer)

    def test_selects_every_variant_at_short_median_long_quantiles_without_mutation(self):
        before = self.source.read_bytes()
        prompts = benchmark.select_prompts(self.run)
        self.assertEqual(len(prompts), 12)
        self.assertEqual([prompt["variant"] for prompt in prompts], list(benchmark.VARIANTS) * 3)
        self.assertEqual([prompt["patient_id"] for prompt in prompts],
                         [f"patient-{index}" for index in (0, 2, 4) for _ in range(4)])
        prompt = prompts[0]
        self.assertEqual(prompt["messages"], [{"role": role, "content": self.rows[0]["variants"]["llm_only"][
            "assessment_plan_prompts"][role]} for role in ("system", "user")])
        self.assertEqual(len(prompt["prompt_sha256"]), 64)
        self.assertEqual(prompt["prompt_bytes"], sum(len(message["content"].encode()) for message in prompt["messages"]))
        prompt["messages"][0]["content"] = "detached"
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(benchmark.select_prompts(self.run, 4)[0]["patient_id"], "patient-2")

    def test_rejects_invalid_count_inadequate_variants_and_malformed_prompts(self):
        for count in (0, 6, True, 24):
            with self.subTest(count=count), self.assertRaises(ValueError):
                benchmark.select_prompts(self.run, count)
        self.rows[0]["variants"]["llm_only"]["assessment_plan_prompts"] = {"system": "bad"}
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "prompts"):
            benchmark.select_prompts(self.run)
        self.rows[0] = {"id": "legacy", "assessment_plan_prompts": {"system": "old", "user": "old"}}
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "native variants"):
            benchmark.select_prompts(self.run)

    def test_unicode_line_separators_remain_inside_saved_prompt_content(self):
        self.rows[2]["variants"]["llm_only"]["assessment_plan_prompts"]["user"] = "x" * 65 + "\u2028\u2029café"
        self.write_rows()
        self.assertIn("\u2028", self.source.read_text())
        prompts = benchmark.select_prompts(self.run, 12)
        self.assertTrue(any("\u2028\u2029café" in prompt["messages"][1]["content"] for prompt in prompts))

    def test_prompt_artifact_rejects_duplicate_ids_keys_and_nonfinite_values(self):
        self.rows.append(deepcopy(self.rows[0]))
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "unique patient IDs"):
            benchmark.select_prompts(self.run)
        for text in ('{"id":"first","id":"second"}', '{"id":NaN}', '{"id":1e999}', "\n"):
            with self.subTest(text=text):
                self.source.write_text(text)
                with self.assertRaises(ValueError):
                    benchmark.select_prompts(self.run)

    def test_all_scheduling_conditions_keep_production_calls_and_record_valid_outputs(self):
        source = self.source.read_bytes()
        report = self.run_benchmark()
        self.factory.assert_called_once_with(self.config, self.aliases, temperature=1, max_retries=3)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(json.loads(self.output.read_text()), report)
        self.assertEqual(self.source.read_bytes(), source)
        self.assertEqual([(server.parallel, server.max_loaded_models) for server in FakeServer.instances],
                         [(1, 1), (2, 1), (4, 1), (1, 3)])
        self.assertTrue(all(server.settings["manage_ollama"] and server.closed for server in FakeServer.instances))
        self.assertTrue(all(server.unloaded == [alias + ":tag" for alias in self.aliases]
                            for server in FakeServer.instances))
        self.assertEqual(len(self.models.calls), 4 * 3 * (1 + 4))
        for condition in report["conditions"]:
            self.assertEqual(len(condition["cold_requests"]), 3)
            self.assertEqual(condition["summary"]["requests"], 12)
            self.assertEqual(condition["summary"]["valid_requests"], 12)
            self.assertGreater(condition["summary"]["p95_seconds"], 0)
            self.assertEqual(condition["status"], "completed")
            for record in condition["requests"]:
                self.assertEqual(record["valid_prediction_ranks"], 3)
                self.assertEqual(record["output"]["model_responses"][0]["token_usage"]["total_tokens"], 12)
                expected = self.rows[2]["variants"][record["variant"]]["assessment_plan_prompts"]
                self.assertIn((record["model"], [{"role": role, "content": expected[role]}
                                               for role in ("system", "user")]), self.models.calls)
        self.assertIn("size_vram", report["conditions"][-1]["warm_ps"]["models"][0])

    def test_mixed_challenger_has_one_request_per_model_and_at_most_three_total(self):
        self.run_benchmark()
        observations = [(active, total) for condition, active, total in self.models.observations if condition == 4]
        self.assertTrue(observations)
        self.assertTrue(all(all(count <= 1 for count in active.values()) and total <= 3
                            for active, total in observations))

    def test_completed_status_with_unparseable_reply_is_not_counted_as_valid(self):
        self.models.invalid = True
        report = self.run_benchmark()
        for condition in report["conditions"]:
            self.assertEqual(condition["summary"]["valid_requests"], 0)
            self.assertTrue(all(record["output"]["status"] == "completed" and not record["valid_predictions"]
                                and record["valid_prediction_ranks"] == 0 for record in condition["requests"]))
        partial = json.dumps({"predictions": [{"condition": "x", "plan": ["ok"]},
                                                {"condition": None, "plan": []},
                                                {"condition": "y", "plan": []}]})
        self.assertEqual(benchmark._valid_ranks(partial), 2)

    def test_interrupt_preserves_received_cold_result_and_closes_owned_server(self):
        self.models.interrupt_at = 2
        with self.assertRaises(KeyboardInterrupt):
            self.run_benchmark()
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved["status"], "interrupted")
        self.assertEqual(saved["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(len(saved["conditions"][0]["cold_requests"]), 1)
        self.assertTrue(FakeServer.instances[0].closed)

    def test_failed_start_preserves_report_and_does_not_run_inference(self):
        server = Mock(side_effect=RuntimeError("server unavailable"))
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            benchmark.run_benchmark(self.experiment, benchmark.select_prompts(self.run, 4), self.output,
                                    models_factory=self.factory, server_factory=server)
        self.assertEqual(json.loads(self.output.read_text())["status"], "failed")
        self.assertEqual(self.models.calls, [])

    def test_refuses_overwrite_wrong_local_port_and_missing_local_alias_before_client_creation(self):
        self.output.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_benchmark()
        self.assertEqual(self.output.read_text(), "keep")
        self.output.unlink()
        self.registry["models"][self.aliases[0]]["api_base"] = "http://localhost:11434/v1"
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "owned"):
            self.run_benchmark()
        self.factory.assert_not_called()
        del self.registry["models"][self.aliases[0]]
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "Missing selected model"):
            self.run_benchmark()
        self.factory.assert_not_called()

    def test_atomic_save_failure_preserves_old_json_and_cleans_temporary_file(self):
        self.output.write_text('{"old": true}\n')
        with patch.object(benchmark.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                benchmark._write(self.output, {"new": True})
        self.assertEqual(json.loads(self.output.read_text()), {"old": True})
        self.assertEqual(list(self.root.glob(".ollama-benchmark-*")), [])

    def test_main_uses_requested_saved_run_and_prints_saved_report_location(self):
        report = {"status": "completed", "conditions": [{"summary": {"requests": 12}}]}
        with patch.object(benchmark, "ExperimentConfig", return_value=self.experiment) as config, \
                patch.object(benchmark, "run_benchmark", return_value=report) as run, \
                redirect_stdout(io.StringIO()) as stdout:
            code = benchmark.main(["--config", str(self.config), "--prompts-run", str(self.run),
                                   "--output", str(self.output), "--requests", "4"])
        self.assertEqual(code, 0)
        config.assert_called_once_with(self.config)
        self.assertEqual(len(run.call_args.args[1]), 4)
        self.assertEqual(json.loads(stdout.getvalue())["output"], str(self.output.resolve()))


if __name__ == "__main__":
    unittest.main()
