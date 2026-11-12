"""Offline pipeline regressions for the concurrent experiment coordinator.

The scheduler, durable store, generation adapter, scoring, and canonical exports
are real. Evidence subprocesses, model transport, and encoder runtimes are fake.
"""

from copy import deepcopy
import hashlib
from io import StringIO
import json
import os
import ast
from pathlib import Path
from queue import Queue
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nltk
import numpy as np
import yaml

# Direct execution starts in tests/; import the project and vendored adapters first.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
with patch.object(nltk, "download", return_value=True):
    from scripts import med_copilot_scheduler as scheduler_module
    from scripts.med_copilot_execution import write_results
    from scripts.med_copilot_config import EXECUTION_DEFAULTS, PATH_SETTINGS, ExperimentConfig
    from scripts.med_copilot_helpers import GUIDELINE_TABLES, new_patient_result, reference_data_metadata
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from scripts.med_copilot_metrics import METRIC_NAMES, MetricScore
    from scripts.med_copilot_models import AssessmentPlanModels
    from scripts.med_copilot_run_file import RunFile, RunMetadataError, pacific_now
    from scripts.med_copilot_scoring import PatientScorer


VARIANTS = ("llm_only", "patient_retrieval", "guidelines", "patient_retrieval_guidelines")
ALIASES = ("hosted", "local-a", "local-b", "local-c")


class FakeExperiment:
    def __init__(self, root):
        self.config_path = root / "experiment.yaml"
        self.patient_input = root / "patients.jsonl"
        self.patient_labels = root / "labels.jsonl"
        self.selected_models = list(ALIASES)
        self.selected_variants = list(VARIANTS)
        self.component_plan = {"evidence": {"component": "patient_retrieval", "settings": {}}}
        self.variant_components = {variant: {"patient_retrieval": "evidence"} for variant in VARIANTS}
        self.plan_judge = "local-c"
        self.execution = {**EXECUTION_DEFAULTS, "overlap_scoring": False,
                          "preparation_workers": 2, "hosted_concurrency": 2,
                          "local_concurrency": 2, "scoring_batch_patients": 2,
                          "encoder_device": "cpu", "cpu_threads": 2}
        self.config_path.write_text("offline experiment\n")

    def metadata(self, root):
        return {"config_file": str(self.config_path),
                "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
                "execution": deepcopy(self.execution), "selected_models": list(ALIASES),
                "selected_variants": list(VARIANTS), "plan_judge": self.plan_judge}


class BatchMetrics:
    def __init__(self):
        self.batches = []

    def score_pairs(self, pairs):
        self.batches.append([pair.key for pair in pairs])
        return iter(MetricScore(pair.key, {name: 0.75 for name in METRIC_NAMES}) for pair in pairs)


class FakeServer:
    def __init__(self):
        self.unloaded = []

    def unload(self, tag):
        self.unloaded.append(tag)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        return None

    def metadata(self):
        return {"managed": False, "offline_test": True}


class PipelineFixture:
    def __init__(self, root):
        self.root, self.run_dir = root, root / "run"
        self.run_dir.mkdir()
        self.experiment = FakeExperiment(root)
        self.patients = [{"id": name, "subjective": "S " + name, "objective": "O " + name}
                         for name in ("SHORT", "TYPICAL", "LONG")]
        self.labels = [{"id": item["id"], "condition": "reference_condition",
                        "plan": ["Reference action"], "has_presenting_evidence": True}
                       for item in self.patients]
        self.experiment.patient_input.write_text("".join(json.dumps(item) + "\n" for item in self.patients))
        self.experiment.patient_labels.write_text("".join(json.dumps(item) + "\n" for item in self.labels))
        self.registry = root / "models.yaml"
        self.registry.write_text(yaml.safe_dump({"models": {
            alias: {"type": "chat", "model_provider": "openai", "model": alias,
                    "output_mode": "json_schema", "retry_strategy": "none",
                    **({"api_key_env": "OFFLINE_MODEL_KEY"} if alias == "hosted" else
                       {"api_base": "http://127.0.0.1:11435/v1"})}
            for alias in ALIASES}}))
        self.metrics = BatchMetrics()
        self.calls, self.events, self.prepared, self.paid_preparation = [], [], [], []
        self.active = dict.fromkeys(ALIASES, 0)
        self.maximum = dict.fromkeys(ALIASES, 0)
        self.lock = Lock()
        self.failure = None
        self.closed_workers = 0
        self.reference_data = {"patient_retrieval": None, "guidelines": None}
        self.server = FakeServer()
        self.models = None
        self.scorer = PatientScorer(self.experiment.patient_labels, self.patients,
                                    metric_evaluator=self.metrics)
        self.run_file = None

    def model_factory(self, kind, name, config):
        def complete(**request):
            prompt = request["messages"][-1]["content"]
            with self.lock:
                self.calls.append((name, deepcopy(request["messages"])))
                self.active[name] += 1
                self.maximum[name] = max(self.maximum[name], self.active[name])
                self.events.append(("start", name, prompt))
            try:
                time.sleep(0.004)
                if self.failure is not None:
                    outcome = self.failure(name, prompt)
                    if isinstance(outcome, BaseException):
                        raise outcome
                content = json.dumps({"predictions": [
                    {"condition": name + ":" + prompt + ":" + str(rank), "plan": [name + " action"]}
                    for rank in (1, 2, 3)]})
                return SimpleNamespace(model=name, id=name + ":" + prompt,
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
                    choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content, refusal=None))])
            finally:
                with self.lock:
                    self.events.append(("end", name, prompt))
                    self.active[name] -= 1
        return SimpleNamespace(completion=complete)

    def create(self):
        with patch.dict(os.environ, {"OFFLINE_MODEL_KEY": "offline-secret", "OPENAI_API_KEY": "offline-openai",
                                     "GRAPHRAG_API_KEY": "offline-graph"}), \
                patch("scripts.med_copilot_models.ModelFactory.create_chat_model", side_effect=self.model_factory):
            self.models = AssessmentPlanModels(self.registry, list(ALIASES), temperature=1, max_retries=1)
        metadata = {"run_name": "run", "status": "running", "errors": [],
                    "experiment": self.experiment.metadata(self.root),
                    "assessment_plan_models": self.models.metadata(self.root),
                    "evaluation": self.scorer.metadata(self.root)}
        self.run_file = RunFile(self.run_dir, metadata)
        self.run_file.update()
        self.run_file.redact = self.models.redact
        return self.scheduler()

    def scheduler(self):
        return scheduler_module.ExperimentScheduler(self.experiment, self.patients, self.models,
            self.scorer, self.run_file, root=self.root, server=self.server)

    def preparer_factory(self):
        fixture = self
        class Preparer:
            def __init__(self, config_path, run_dir, number, settings, store, run_file, root, stop=None):
                self.store = store
                self.stop = stop
                self.closed = False
                self.metadata = {"reference_data": deepcopy(fixture.reference_data), "devices": {"requested": "cpu"}}

            def prepare(self, item):
                patient, result = item
                fixture.prepared.append(patient["id"])
                first = next(iter(result["components"].values()))
                if first["capture"].get("keyword_extraction_raw") is None:
                    fixture.paid_preparation.append(patient["id"])
                    first["capture"]["keyword_extraction_raw"] = "paid keyword reply " + patient["id"]
                    self.store.save(("preparation", patient["id"], None, None), result)
                for component in result["components"].values():
                    component["status"] = "completed"
                    key = "Key_info" if component["component"] == "guidelines" else "example"
                    component["outputs"] = {key: "evidence " + patient["id"]}
                for variant, output in result["variants"].items():
                    if output.get("assessment_plan_prompts") is None:
                        output["assessment_plan_prompts"] = {"system": "System " + variant,
                                                               "user": patient["id"] + "/" + variant}
                self.store.save(("preparation", patient["id"], None, None), result, "completed")
                return {"result": result}

            def close(self):
                if not self.closed:
                    fixture.closed_workers += 1
                    self.closed = True
        return Preparer

    def records(self):
        return [json.loads(line) for line in (self.run_dir / "results.jsonl").read_text().splitlines()]


class SchedulerPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PipelineFixture(Path(self.temporary.name))
        self.scheduler = self.fixture.create()
        self.addCleanup(self.scheduler.close)
        self.preparation = patch.object(scheduler_module, "PreparationWorker", self.fixture.preparer_factory())
        self.preparation.start()
        self.addCleanup(self.preparation.stop)
        encoder = patch.object(scheduler_module, "_CPUEncoder",
                               return_value=lambda texts: np.asarray([[1.0, 0.0] for _ in texts]))
        encoder.start()
        self.addCleanup(encoder.stop)

    def test_full_matrix_overlaps_requests_and_drains_each_local_alias_before_switching(self):
        fixture = self.fixture
        self.scheduler.prepare_and_generate()
        self.scheduler.score()
        records = fixture.records()
        self.assertEqual(len(fixture.calls), len(fixture.patients) * len(VARIANTS) * len(ALIASES))
        self.assertEqual(fixture.paid_preparation, fixture.prepared)
        self.assertEqual(fixture.closed_workers, 2)
        self.assertTrue(all(fixture.maximum[alias] <= 2 for alias in ALIASES))
        self.assertTrue(all(fixture.maximum[alias] == 2 for alias in ALIASES[1:]))
        starts = [alias for event, alias, _ in fixture.events if event == "start" and alias != "hosted"]
        phase_order = [alias for index, alias in enumerate(starts) if index == 0 or alias != starts[index - 1]]
        self.assertEqual(phase_order, list(ALIASES[1:]))
        self.assertEqual(fixture.server.unloaded, list(ALIASES[1:]))
        self.assertEqual([record["id"] for record in records], [patient["id"] for patient in fixture.patients])
        for record in records:
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["components"]["evidence"]["outputs"]["example"], "evidence " + record["id"])
            for variant, body in record["variants"].items():
                for alias, output in body["assessment_plans"].items():
                    self.assertEqual(output["status"], "completed")
                    self.assertEqual(output["predictions"][0]["condition"], f"{alias}:{record['id']}/{variant}:1")
                    for evaluation in output["evaluation"]:
                        self.assertEqual(evaluation["condition"]["status"], "scored")
                        self.assertEqual(evaluation["diagnosis_correctness"]["status"], "scored")
                        self.assertEqual(evaluation["plan_correctness"]["reason"], "deferred")
        self.assertEqual([len(batch) for batch in fixture.metrics.batches], [192, 96])
        self.assertEqual(sum(sum(batch["newly_scored_fields_by_patient"].values())
                             for batch in self.scheduler.timings["scoring"]), 288)
        self.assertTrue(all(batch["start_seconds"] <= batch["end_seconds"]
                            for batch in self.scheduler.timings["scoring"]))

    def test_new_raw_reply_is_in_database_before_parser_runs(self):
        fixture = self.fixture
        parse = fixture.models.generate_one.__func__.__globals__["parse_condition_plan"]
        inspected = []
        def checked(content):
            found = [state for _, state in self.scheduler.store.entries("generation")
                     if state["payload"].get("assessment_plan_raw") == content]
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0]["payload"]["_generation_reply_complete"])
            self.assertIsNone(found[0]["payload"]["predictions"])
            inspected.append(content)
            return parse(content)
        with patch("scripts.med_copilot_models.parse_condition_plan", side_effect=checked):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(inspected), len(fixture.calls))

    def test_large_utf8_prompt_is_rejected_before_local_transport(self):
        fixture = self.fixture
        output = deepcopy(self.scheduler.records["SHORT"]["variants"]["llm_only"]["assessment_plans"]["local-a"])
        messages = [{"role": "user", "content": "é" * 9000}]
        with self.assertRaisesRegex(ValueError, "SHORT/llm_only.*context budget"):
            self.scheduler._generate(("SHORT", "llm_only", "local-a", messages, output))
        self.assertEqual(fixture.calls, [])
        self.assertIsNone(self.scheduler.store.load(("generation", "SHORT", "llm_only", "local-a")))

    def test_received_raw_reply_recovery_does_not_reject_large_prompt_or_repeat_transport(self):
        fixture = self.fixture
        output = deepcopy(self.scheduler.records["SHORT"]["variants"]["llm_only"]["assessment_plans"]["local-a"])
        output.update(status="failed_incomplete", _generation_reply_complete=True,
                      assessment_plan_raw=json.dumps({"predictions": [
                          {"condition": "saved" + str(rank), "plan": ["saved action"]} for rank in (1, 2, 3)]}))
        messages = [{"role": "user", "content": "é" * 9000}]
        result = self.scheduler._generate(("SHORT", "llm_only", "local-a", messages, output))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["predictions"][0]["condition"], "saved1")
        self.assertEqual(fixture.calls, [])

    def test_cpu_scoring_overlaps_later_model_generation_without_losing_or_repeating_scores(self):
        fixture = self.fixture
        fixture.experiment.execution["overlap_scoring"] = True
        started = Event()
        score_pairs = fixture.metrics.score_pairs
        def scoring(pairs):
            started.set()
            return score_pairs(pairs)
        def wait_for_scoring(alias, prompt):
            if alias == "local-b":
                self.assertTrue(started.wait(2), "Later local inference should overlap CPU scoring")
        fixture.failure = wait_for_scoring
        with patch.object(fixture.metrics, "score_pairs", side_effect=scoring):
            self.scheduler.prepare_and_generate()
            self.scheduler.score()
        keys = [key for batch in fixture.metrics.batches for key in batch]
        self.assertEqual(len(keys), 3 * len(VARIANTS) * len(ALIASES) * 3 * 2)
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(sum(sum(batch["newly_scored_fields_by_patient"].values())
                             for batch in self.scheduler.timings["scoring"]), len(keys))
        self.assertTrue(all(sum(batch["newly_scored_fields_by_patient"].values()) == 0
                            for batch in self.scheduler.timings["scoring"][-2:]))
        self.assertTrue(all(evaluation["condition"]["status"] == "scored"
                            for record in fixture.records() for body in record["variants"].values()
                            for output in body["assessment_plans"].values() for evaluation in output["evaluation"]))

    def test_hosted_slow_requests_do_not_starve_all_local_model_phases(self):
        fixture = self.fixture
        last_local = Event()
        hosted_unblocked = []
        def wait_for_local(alias, prompt):
            if alias == "hosted":
                hosted_unblocked.append(last_local.wait(2))
            if alias == "local-c" and prompt == "LONG/patient_retrieval_guidelines":
                last_local.set()
        fixture.failure = wait_for_local
        self.scheduler.prepare_and_generate()
        self.assertTrue(last_local.is_set())
        self.assertTrue(all(hosted_unblocked), "Hosted work blocked the local queue")
        self.assertEqual(fixture.maximum["hosted"], 2)
        self.assertEqual(fixture.server.unloaded, list(ALIASES[1:]))

    def test_local_failure_signals_waiting_preparation_before_executor_shutdown(self):
        fixture = self.fixture
        original = fixture.preparer_factory()
        stopped = Event()
        class WaitingWorker(original):
            def prepare(self, item):
                patient, _ = item
                if patient["id"] == "LONG":
                    if self.stop.wait(2):
                        stopped.set()
                    raise RuntimeError("Waiting preparation stopped")
                event = super().prepare(item)
                if patient["id"] == "SHORT":
                    for output in event["result"]["variants"].values():
                        output["assessment_plan_prompts"]["user"] = "é" * 9000
                return event
        with patch.object(scheduler_module, "PreparationWorker", WaitingWorker), \
                self.assertRaisesRegex(ValueError, "context budget"):
            self.scheduler.prepare_and_generate()
        self.assertTrue(stopped.is_set(), "Stop must be set before waiting for preparation cleanup")
        self.assertNotIn("LONG", fixture.paid_preparation)
        self.assertTrue(all(count == 0 for count in fixture.active.values()))

    def test_worker_failure_signals_active_peer_before_waiting_for_its_next_paid_step(self):
        fixture = self.fixture
        original = fixture.preparer_factory()
        peer_started, peer_stopped = Event(), Event()
        class PeerWorker(original):
            def prepare(self, item):
                patient, _ = item
                if patient["id"] == "SHORT":
                    if not peer_started.wait(1):
                        raise RuntimeError("Peer did not start")
                    raise RuntimeError("First worker failed")
                if patient["id"] == "TYPICAL":
                    peer_started.set()
                    if self.stop.wait(0.5):
                        peer_stopped.set()
                    raise RuntimeError("Peer stopped before another paid step")
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", PeerWorker), \
                self.assertRaises(RuntimeError):
            self.scheduler.prepare_and_generate()
        self.assertTrue(peer_stopped.is_set(), "Failed worker must signal peers before bounded_map waits for them")

    def test_worker_failure_preserves_received_preparation_reply_and_resume_avoids_repayment(self):
        fixture = self.fixture
        original = fixture.preparer_factory()
        class BrokenWorker(original):
            def prepare(self, item):
                patient, result = item
                if patient["id"] == "TYPICAL":
                    result["components"]["evidence"]["capture"]["keyword_extraction_raw"] = "paid before worker crash"
                    self.store.save(("preparation", patient["id"], None, None), result)
                    raise RuntimeError("preparation worker crashed")
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", BrokenWorker), \
                self.assertRaises(RuntimeError):
            self.scheduler.prepare_and_generate()
        self.assertTrue(all(count == 0 for count in fixture.active.values()))
        self.scheduler.close()
        resumed = fixture.scheduler()
        self.addCleanup(resumed.close)
        resumed.prepare_and_generate()
        self.assertNotIn("TYPICAL", fixture.paid_preparation)
        self.assertTrue(all(record["status"] == "completed" for record in fixture.records()))

    def test_generation_failure_keeps_siblings_and_resume_retries_only_failed_job(self):
        fixture = self.fixture
        failed = [False]
        def fail(alias, prompt):
            if alias == "local-b" and prompt == "TYPICAL/guidelines" and not failed[0]:
                failed[0] = True
                return RuntimeError("offline-secret rejected")
        fixture.failure = fail
        self.scheduler.prepare_and_generate()
        before = len(fixture.calls)
        failed_slot = self.scheduler.records["TYPICAL"]["variants"]["guidelines"]["assessment_plans"]["local-b"]
        self.assertEqual(failed_slot["status"], "failed_incomplete")
        self.assertEqual(failed_slot["error"]["message"], "[REDACTED] rejected")
        self.scheduler.close()
        resumed = fixture.scheduler()
        self.addCleanup(resumed.close)
        resumed.prepare_and_generate()
        self.assertEqual(fixture.calls[before:], [("local-b", [
            {"role": "system", "content": "System guidelines"}, {"role": "user", "content": "TYPICAL/guidelines"}])])
        self.assertTrue(all(record["status"] == "completed" for record in fixture.records()))

    def test_canonical_completed_plan_scores_are_not_reverted_from_stale_job_payloads(self):
        fixture = self.fixture
        self.scheduler.prepare_and_generate()
        self.scheduler.score()
        records = fixture.records()
        for record in records:
            for body in record["variants"].values():
                for slot in body["assessment_plans"].values():
                    for evaluation in slot["evaluation"]:
                        evaluation["plan_correctness"].update(status="scored", reason="standalone judged",
                                                               score=0.5, weighted_score=0.5 / evaluation["rank"])
        write_results(records, fixture.run_dir / "results.jsonl")
        self.scheduler.close()
        before_calls, before_batches = len(fixture.calls), len(fixture.metrics.batches)
        resumed = fixture.scheduler()
        self.addCleanup(resumed.close)
        resumed.prepare_and_generate()
        resumed.score()
        self.assertEqual(len(fixture.calls), before_calls)
        self.assertEqual(fixture.records(), records)
        # Scorer can inspect completed batches but must not recompute their metrics.
        self.assertTrue(all(not batch for batch in fixture.metrics.batches[before_batches:]))
        self.assertTrue(all(sum(batch["newly_scored_fields_by_patient"].values()) == 0
                            for batch in resumed.timings["scoring"]))

    def test_newer_preparation_checkpoint_overlays_stale_canonical_before_paid_work(self):
        fixture = self.fixture
        self.scheduler.export()  # Canonical initial state exists before worker checkpoints.
        patient_id = fixture.patients[0]["id"]
        received = deepcopy(self.scheduler.records[patient_id])
        received["components"]["evidence"]["capture"]["keyword_extraction_raw"] = "already paid"
        self.scheduler.store.save(("preparation", patient_id, None, None), received)
        self.scheduler.close()
        resumed = fixture.scheduler()
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.records[patient_id]["components"]["evidence"]["capture"]["keyword_extraction_raw"],
                         "already paid")
        resumed.prepare_and_generate()
        self.assertNotIn(patient_id, fixture.paid_preparation)

    def test_durable_scoring_survives_canonical_export_failure_without_recomputation(self):
        fixture = self.fixture
        self.scheduler.prepare_and_generate()
        with patch.object(scheduler_module, "write_results", side_effect=OSError("disk unavailable")), \
                self.assertRaisesRegex(OSError, "disk unavailable"):
            self.scheduler.score()
        self.scheduler.close()
        before = len(fixture.metrics.batches)
        resumed = fixture.scheduler()
        self.addCleanup(resumed.close)
        resumed.score()
        self.assertTrue(all(not batch for batch in fixture.metrics.batches[before:]))
        self.assertTrue(all(evaluation["condition"]["status"] == "scored"
                            for record in fixture.records() for body in record["variants"].values()
                            for slot in body["assessment_plans"].values() for evaluation in slot["evaluation"]))

    def test_failed_received_reply_checkpoint_prevents_parse_and_remaining_work(self):
        fixture = self.fixture
        original = self.scheduler.store.save
        def save(key, payload, status="checkpoint"):
            if key[0] == "generation" and payload.get("_generation_reply_complete"):
                raise RunMetadataError("writer stopped")
            return original(key, payload, status)
        with patch.object(self.scheduler.store, "save", side_effect=save), \
                self.assertRaisesRegex(RunMetadataError, "writer stopped"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(fixture.server.unloaded, [])
        self.assertTrue(all(state["payload"]["predictions"] is None
                            for _, state in self.scheduler.store.entries("generation")))


class PreparationRecoveryTests(unittest.TestCase):
    """Recover process loss through the real durable store and coordinator."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PipelineFixture(Path(self.temporary.name))
        fixture = self.fixture
        fixture.patients = fixture.patients[:1]
        fixture.labels = fixture.labels[:1]
        fixture.experiment.patient_input.write_text(json.dumps(fixture.patients[0]) + "\n")
        fixture.experiment.patient_labels.write_text(json.dumps(fixture.labels[0]) + "\n")
        fixture.experiment.execution["preparation_workers"] = 1
        fixture.experiment.component_plan["guidance"] = {"component": "guidelines", "settings": {}}
        fixture.experiment.variant_components = {
            "llm_only": {}, "patient_retrieval": {"patient_retrieval": "evidence"},
            "guidelines": {"guidelines": "guidance"},
            "patient_retrieval_guidelines": {"patient_retrieval": "evidence", "guidelines": "guidance"},
        }
        fixture.scorer = PatientScorer(fixture.experiment.patient_labels, fixture.patients,
                                       metric_evaluator=fixture.metrics)
        self.scheduler = fixture.create()
        self.addCleanup(self.scheduler.close)
        self.original = fixture.preparer_factory()
        encoder = patch.object(scheduler_module, "_CPUEncoder",
                               return_value=lambda texts: np.asarray([[1.0, 0.0] for _ in texts]))
        encoder.start()
        self.addCleanup(encoder.stop)

    def durable_partial(self, result):
        component = result["components"]["evidence"]
        component.update(status="completed", outputs={"example": "evidence SHORT"})
        component["capture"].update(
            keyword_extraction_raw="already paid keyword reply",
            retrieval_keywords=["cached keyword"],
            retrieved_patient_ids=[{"id": "candidate", "score": 0.75}],
            reranked_patient_ids=[{"id": "winner", "score": 0.91}],
            selected_reference_patient={"id": "winner", "corpus_row": 4},
        )
        component["capture"]["model_responses"]["keyword_extraction"] = [
            {"reported_model": "keyword-model", "response_id": "paid-1",
             "token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}]
        result["variants"]["patient_retrieval"]["assessment_plan_prompts"] = {
            "system": "Already rendered exactly", "user": "SHORT/patient_retrieval"}
        result["components"]["guidance"]["capture"]["guideline_questions"] = ["already screened question"]
        return deepcopy(result)

    def test_replacement_uses_latest_committed_component_captures_rankings_and_prompts(self):
        original, fixture, test = self.original, self.fixture, self
        created, seen, saved = [], [], []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

            def prepare(self, item):
                patient, result = item
                seen.append(deepcopy(result))
                if len(created) == 1:
                    fixture.paid_preparation.append(patient["id"])
                    # A subprocess never mutates the parent's submitted object.
                    child_result = deepcopy(result)
                    saved.append(test.durable_partial(child_result))
                    self.store.save(("preparation", patient["id"], None, None), child_result)
                    raise scheduler_module.PreparationWorkerExited("offline-secret lost after commit before ACK")
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", Worker):
            self.scheduler.prepare_and_generate()
        self.scheduler.score()
        self.assertEqual(len(created), 2)
        self.assertEqual(fixture.closed_workers, 2)
        self.assertEqual(fixture.paid_preparation, ["SHORT"])
        self.assertIsNone(seen[0]["components"]["evidence"]["capture"]["keyword_extraction_raw"])
        self.assertEqual(seen[1]["components"]["evidence"], saved[0]["components"]["evidence"])
        self.assertEqual(seen[1]["components"]["guidance"]["capture"], saved[0]["components"]["guidance"]["capture"])
        self.assertEqual(seen[1]["variants"]["patient_retrieval"]["assessment_plan_prompts"],
                         saved[0]["variants"]["patient_retrieval"]["assessment_plan_prompts"])
        record = fixture.records()[0]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["components"]["evidence"], saved[0]["components"]["evidence"])
        self.assertEqual(len(fixture.calls), len(VARIANTS) * len(ALIASES))
        recovery = fixture.run_file.metadata["preparation_recovery"]
        self.assertEqual(recovery["max_replacements_per_patient"], 2)
        self.assertEqual(recovery["max_startup_replacements_per_worker"], 2)
        self.assertEqual(recovery["budget_scope"], "invocation")
        self.assertEqual(recovery["log_file"], "preparation-recovery.log")
        self.assertTrue(any(event["patient_id"] == "SHORT" and event["attempt"] == 1
                            and event["outcome"] == "replacement" for event in recovery["events"]))
        self.assertTrue(any(event["outcome"] == "recovered" for event in recovery["events"]))
        self.assertNotIn("offline-secret", json.dumps(recovery))
        self.assertNotIn("offline-secret", (fixture.run_dir / "preparation-recovery.log").read_text())
        self.assertEqual(fixture.run_file.metadata["errors"], [])

    def test_unsaved_response_is_not_reused_after_death_before_checkpoint_commit(self):
        original, fixture = self.original, self.fixture
        created, seen = [], []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

            def prepare(self, item):
                patient, result = item
                seen.append(deepcopy(result))
                if len(created) == 1:
                    fixture.paid_preparation.append(patient["id"])
                    child_result = deepcopy(result)
                    child_result["components"]["evidence"]["capture"]["keyword_extraction_raw"] = "unsaved API response"
                    raise scheduler_module.PreparationWorkerExited("lost before checkpoint commit")
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", Worker):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 2)
        self.assertIsNone(seen[1]["components"]["evidence"]["capture"]["keyword_extraction_raw"])
        self.assertEqual(fixture.paid_preparation, ["SHORT", "SHORT"])
        self.assertEqual(fixture.records()[0]["status"], "completed")

    def test_replacement_worker_is_reused_for_later_patients(self):
        root = self.fixture.root / "multiple-patients"
        root.mkdir()
        fixture = PipelineFixture(root)
        fixture.experiment.execution["preparation_workers"] = 1
        coordinator = fixture.create()
        self.addCleanup(coordinator.close)
        original = fixture.preparer_factory()
        created, assignments = [], []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.number = len(created)
                created.append(self)
            def prepare(self, item):
                assignments.append((self.number, item[0]["id"]))
                if self.number == 0:
                    raise scheduler_module.PreparationWorkerExited("original worker died")
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", Worker):
            coordinator.prepare_and_generate()
        self.assertEqual(len(created), 2)
        self.assertEqual(assignments, [(0, "SHORT"), (1, "SHORT"), (1, "TYPICAL"), (1, "LONG")])
        self.assertEqual(fixture.closed_workers, 2)
        self.assertEqual(fixture.paid_preparation, ["SHORT", "TYPICAL", "LONG"])

    def test_recoverable_worker_death_does_not_stop_a_healthy_peer(self):
        root = self.fixture.root / "healthy-peer"
        root.mkdir()
        fixture = PipelineFixture(root)
        coordinator = fixture.create()
        self.addCleanup(coordinator.close)
        original = fixture.preparer_factory()
        recovered, peer_finished = Event(), Event()
        failed = [False]
        class Worker(original):
            def prepare(self, item):
                if item[0]["id"] == "SHORT":
                    if not failed[0]:
                        failed[0] = True
                        raise scheduler_module.PreparationWorkerExited("one worker died")
                    recovered.set()
                if item[0]["id"] == "TYPICAL":
                    if not recovered.wait(2):
                        raise RuntimeError("Replacement did not recover while peer was active")
                    if self.stop.is_set():
                        raise RuntimeError("Recoverable failure stopped healthy peer")
                    peer_finished.set()
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", Worker):
            coordinator.prepare_and_generate()
        self.assertTrue(peer_finished.is_set())
        self.assertEqual(len(fixture.paid_preparation), 3)
        self.assertEqual(len(fixture.calls), 3 * len(VARIANTS) * len(ALIASES))

    def test_simultaneous_worker_recoveries_retain_both_event_histories(self):
        root = self.fixture.root / "simultaneous-recovery"
        root.mkdir()
        fixture = PipelineFixture(root)
        coordinator = fixture.create()
        self.addCleanup(coordinator.close)
        original = fixture.preparer_factory()
        both_started = Barrier(2)
        created, per_slot = [], {}
        class Worker(original):
            def __init__(self, config_path, run_dir, number, *args, **kwargs):
                super().__init__(config_path, run_dir, number, *args, **kwargs)
                with fixture.lock:
                    self.first_in_slot = number not in per_slot
                    per_slot[number] = per_slot.get(number, 0) + 1
                    created.append(self)
            def prepare(self, item):
                if self.first_in_slot:
                    both_started.wait(2)
                    raise scheduler_module.PreparationWorkerExited("simultaneous worker death")
                return super().prepare(item)
        update = fixture.run_file.update
        def delayed_recovery_write(**fields):
            if "preparation_recovery" in fields:
                # Without append serialization both threads can snapshot the
                # same event list before either replacement reaches disk.
                time.sleep(0.005)
            return update(**fields)
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                patch.object(fixture.run_file, "update", side_effect=delayed_recovery_write):
            coordinator.prepare_and_generate()
        self.assertEqual(len(created), 4)
        events = fixture.run_file.metadata["preparation_recovery"]["events"]
        self.assertEqual(len(events), 4)
        self.assertEqual({event["patient_id"] for event in events if event["outcome"] == "replacement"},
                         {"SHORT", "TYPICAL"})
        self.assertEqual({event["patient_id"] for event in events if event["outcome"] == "recovered"},
                         {"SHORT", "TYPICAL"})
        logged = [json.loads(line) for line in (fixture.run_dir / "preparation-recovery.log").read_text().splitlines()]
        self.assertEqual(logged, events)
        self.assertEqual(fixture.closed_workers, 4)
        self.assertEqual(len(fixture.paid_preparation), 3)

    def test_reference_fingerprint_drift_on_replacement_is_fatal_before_prepare(self):
        fixture, original = self.fixture, self.original
        created, preparations = [], []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
                if len(created) == 2:
                    self.metadata["reference_data"]["guidelines"] = {"sha256": "new-reference-fingerprint"}
            def prepare(self, item):
                preparations.append(item[0]["id"])
                raise scheduler_module.PreparationWorkerExited("worker lost before replacement resource check")
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                self.assertRaisesRegex(ValueError, "reference data changed"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 2)
        self.assertEqual(preparations, ["SHORT"])
        self.assertEqual(fixture.closed_workers, 2)
        self.assertEqual(fixture.paid_preparation, [])
        self.assertEqual(fixture.calls, [])

    def test_global_failure_during_dead_worker_cleanup_prevents_replacement(self):
        fixture, original, coordinator = self.fixture, self.original, self.scheduler
        created = []
        failure = ValueError("another pipeline stage failed")
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.crashed = False
                created.append(self)
            def prepare(self, item):
                self.crashed = True
                raise scheduler_module.PreparationWorkerExited("worker transport lost")
            def close(self):
                super().close()
                if self.crashed:
                    coordinator._abort(failure)
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                self.assertRaisesRegex(ValueError, "another pipeline stage failed") as raised:
            coordinator.prepare_and_generate()
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(created), 1)
        self.assertTrue(coordinator._stop.is_set())
        self.assertFalse(any(event["outcome"] == "replacement" for event in
                             fixture.run_file.metadata.get("preparation_recovery", {}).get("events", [])))
        self.assertEqual(fixture.paid_preparation, [])

    def test_preparation_queues_preserve_original_exhaustion_when_peer_shutdown_also_fails(self):
        original = RuntimeError("Preparation recovery exhausted for patient SHORT after 2 replacements")
        self.scheduler._abort(original)
        local_ready, hosted_ready = Queue(), Queue()
        with patch.object(scheduler_module, "bounded_map", side_effect=RuntimeError("peer negative ACK")), \
                self.assertRaises(RuntimeError) as raised:
            self.scheduler._prepare([Mock()], local_ready, hosted_ready)
        self.assertIs(raised.exception, original)
        for ready in (local_ready, hosted_ready):
            self.assertIs(ready.get_nowait(), original)
            self.assertIsNone(ready.get_nowait())

    def test_recovery_does_not_replace_completed_generation_or_scoring_with_stale_preparation(self):
        fixture, original, test = self.fixture, self.original, self
        record = self.scheduler.records["SHORT"]
        prompt = {"system": "System llm_only", "user": "SHORT/llm_only"}
        record["variants"]["llm_only"]["assessment_plan_prompts"] = prompt
        messages = [{"role": key, "content": value} for key, value in prompt.items()]
        self.scheduler._generate(("SHORT", "llm_only", "hosted", messages,
                                  deepcopy(record["variants"]["llm_only"]["assessment_plans"]["hosted"])))
        self.scheduler.score()
        completed = deepcopy(record["variants"]["llm_only"]["assessment_plans"]["hosted"])
        prior_keys = {key for batch in fixture.metrics.batches for key in batch}
        created, received = [], []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

            def prepare(self, item):
                patient, result = item
                if len(created) == 1:
                    stale = new_patient_result(patient["id"], model_ids=ALIASES, variant_ids=VARIANTS,
                        component_plan=fixture.experiment.component_plan,
                        variant_components=fixture.experiment.variant_components)
                    test.durable_partial(stale)
                    self.store.save(("preparation", patient["id"], None, None), stale)
                    raise scheduler_module.PreparationWorkerExited("saved older preparation snapshot")
                received.append(deepcopy(result))
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", Worker):
            self.scheduler.prepare_and_generate()
        self.scheduler.score()
        restored = received[0]["variants"]["llm_only"]
        self.assertEqual(restored["assessment_plan_prompts"], prompt)
        self.assertEqual(restored["assessment_plans"]["hosted"], completed)
        self.assertEqual(fixture.records()[0]["variants"]["llm_only"]["assessment_plans"]["hosted"], completed)
        self.assertEqual(len(fixture.calls), len(VARIANTS) * len(ALIASES))
        all_keys = [key for batch in fixture.metrics.batches for key in batch]
        self.assertTrue(prior_keys)
        self.assertEqual(len(all_keys), len(set(all_keys)))
        self.assertEqual(len(all_keys), len(VARIANTS) * len(ALIASES) * 3 * 2)

    def test_two_replacements_exhaust_cleanly_and_manual_resume_reuses_the_checkpoint(self):
        fixture, original, test = self.fixture, self.original, self
        created = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

            def prepare(self, item):
                patient, result = item
                if result["components"]["evidence"]["capture"]["keyword_extraction_raw"] is None:
                    fixture.paid_preparation.append(patient["id"])
                    child_result = deepcopy(result)
                    test.durable_partial(child_result)
                    self.store.save(("preparation", patient["id"], None, None), child_result)
                raise scheduler_module.PreparationWorkerExited("persistent worker death")
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                self.assertRaisesRegex(RuntimeError, "SHORT.*(?:two|2)|(?:two|2).*SHORT"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 3, "Original worker plus exactly two replacements")
        self.assertEqual(fixture.closed_workers, 3)
        self.assertTrue(self.scheduler._stop.is_set())
        durable = self.scheduler.store.load(("preparation", "SHORT", None, None))
        self.assertEqual(durable["components"]["evidence"]["capture"]["keyword_extraction_raw"],
                         "already paid keyword reply")
        self.assertTrue(any(event["outcome"] == "exhausted" for event in
                            fixture.run_file.metadata["preparation_recovery"]["events"]))
        self.scheduler.export()
        exported = fixture.records()[0]
        self.assertEqual(exported["components"], durable["components"])
        self.assertEqual(exported["variants"]["patient_retrieval"]["assessment_plan_prompts"],
                         durable["variants"]["patient_retrieval"]["assessment_plan_prompts"])
        self.scheduler.close()
        resumed = fixture.scheduler()
        self.addCleanup(resumed.close)
        with patch.object(scheduler_module, "PreparationWorker", original):
            resumed.prepare_and_generate()
        self.assertEqual(fixture.paid_preparation, ["SHORT"])
        self.assertEqual(fixture.records()[0]["status"], "completed")

    def test_replacement_startup_death_consumes_the_same_patient_budget(self):
        fixture, original = self.fixture, self.original
        attempts = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                attempts.append(self)
                if len(attempts) == 2:
                    self.close()
                    raise scheduler_module.PreparationWorkerExited("replacement exited before ready")

            def prepare(self, item):
                if len(attempts) == 1:
                    raise scheduler_module.PreparationWorkerExited("initial process died")
                return super().prepare(item)
        with patch.object(scheduler_module, "PreparationWorker", Worker):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(attempts), 3)
        self.assertEqual(fixture.closed_workers, 3)
        events = fixture.run_file.metadata["preparation_recovery"]["events"]
        self.assertEqual(max(event["attempt"] for event in events if event["patient_id"] == "SHORT"), 2)
        self.assertEqual(fixture.paid_preparation, ["SHORT"])

    def test_initial_startup_exits_are_bounded_and_successful_replacement_can_prepare(self):
        fixture, original = self.fixture, self.original
        created = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
                if len(created) < 3:
                    self.close()
                    raise scheduler_module.PreparationWorkerExited("startup exited before ready")
        with patch.object(scheduler_module, "PreparationWorker", Worker):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 3)
        self.assertEqual(fixture.closed_workers, 3)
        self.assertEqual(fixture.paid_preparation, ["SHORT"])
        events = fixture.run_file.metadata["preparation_recovery"]["events"]
        self.assertEqual(max(event["attempt"] for event in events if event["patient_id"] is None), 2)

    def test_persistent_initial_startup_exit_stops_without_preparation_or_generation(self):
        fixture, original = self.fixture, self.original
        created = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
                self.close()
                raise scheduler_module.PreparationWorkerExited("persistent startup exit")
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                self.assertRaisesRegex(RuntimeError, "2|exhaust"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 3)
        self.assertEqual(fixture.closed_workers, 3)
        self.assertEqual(fixture.paid_preparation, [])
        self.assertEqual(fixture.calls, [])

    def test_deterministic_preparation_and_replacement_startup_errors_are_fatal(self):
        fixture, original = self.fixture, self.original
        created = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
            def prepare(self, item):
                raise ValueError("invalid evidence configuration")
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                self.assertRaisesRegex(ValueError, "invalid evidence configuration"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 1)
        self.assertEqual(fixture.paid_preparation, [])

    def test_checkpoint_error_after_commit_is_fatal_and_never_reclassified_as_worker_death(self):
        fixture, original = self.fixture, self.original
        created = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
        save = self.scheduler.store.save
        def failing_writer(key, payload, status="checkpoint"):
            result = save(key, payload, status)
            if key[0] == "preparation":
                raise RunMetadataError("checkpoint writer failed after commit")
            return result
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                patch.object(self.scheduler.store, "save", side_effect=failing_writer), \
                self.assertRaisesRegex(RunMetadataError, "writer failed after commit"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 1)
        durable = self.scheduler.store.load(("preparation", "SHORT", None, None))
        self.assertEqual(durable["components"]["evidence"]["capture"]["keyword_extraction_raw"],
                         "paid keyword reply SHORT")

    def test_deterministic_replacement_startup_error_does_not_consume_another_retry(self):
        fixture, original = self.fixture, self.original
        created = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
                if len(created) == 2:
                    self.close()
                    raise ValueError("invalid replacement configuration")
            def prepare(self, item):
                raise scheduler_module.PreparationWorkerExited("initial process died")
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                self.assertRaisesRegex(ValueError, "invalid replacement configuration"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 2)
        self.assertEqual(fixture.closed_workers, 2)
        self.assertEqual(fixture.paid_preparation, [])

    def test_comment_only_experiment_change_during_recovery_remains_fatal(self):
        fixture, original = self.fixture, self.original
        created = []
        class Worker(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
            def prepare(self, item):
                fixture.experiment.config_path.write_text(fixture.experiment.config_path.read_text() + "# changed\n")
                raise scheduler_module.PreparationWorkerExited("worker died after config edit")
        with patch.object(scheduler_module, "PreparationWorker", Worker), \
                self.assertRaisesRegex(ValueError, "[Cc]onfig.*changed|changed.*[Cc]onfig"):
            self.scheduler.prepare_and_generate()
        self.assertEqual(len(created), 1)
        self.assertEqual(fixture.paid_preparation, [])


class PreparationWorkerProtocolTests(unittest.TestCase):
    def test_semantically_invalid_complete_frames_without_newline_remain_fatal(self):
        frames = ('{"type":"ready","metadata":{"bad":NaN}}',
                  '{"type":"ready","type":"ready","metadata":{}}',
                  '{"type":"ready","metadata":{"bad":1e999}}')
        for frame in frames:
            with self.subTest(frame=frame):
                worker = scheduler_module.PreparationWorker.__new__(scheduler_module.PreparationWorker)
                worker.process = SimpleNamespace(stdout=StringIO(frame), poll=Mock(return_value=0))
                with self.assertRaises(ValueError):
                    worker._read()

    def test_eof_after_ready_is_typed_as_retryable_worker_exit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            process = SimpleNamespace(stdin=StringIO(),
                stdout=StringIO(json.dumps({"type": "ready", "metadata": {}}) + "\n"),
                poll=Mock(return_value=-9))
            with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                    EXECUTION_DEFAULTS, Mock(), Mock(), root)
            try:
                with self.assertRaises(scheduler_module.PreparationWorkerExited):
                    worker.prepare(({"id": "patient"}, {}))
            finally:
                worker.close()

    def test_unterminated_checkpoint_line_is_a_retryable_exit_without_commit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = Mock()
            process = SimpleNamespace(stdin=StringIO(), stdout=StringIO(
                json.dumps({"type": "ready", "metadata": {}}) + '\n{"type":"checkpoint","result":'),
                poll=Mock(return_value=-9))
            with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                    EXECUTION_DEFAULTS, store, Mock(), root)
            try:
                with self.assertRaises(scheduler_module.PreparationWorkerExited):
                    worker.prepare(({"id": "patient"}, {}))
                store.save.assert_not_called()
            finally:
                worker.close()

    def test_truncated_utf8_is_retryable_only_when_the_process_has_exited(self):
        for status in (-9, None):
            with self.subTest(process_status=status), TemporaryDirectory() as directory:
                root = Path(directory)
                bad_utf8 = UnicodeDecodeError("utf-8", b"\xc3", 0, 1, "unexpected end of data")
                stdout = Mock()
                stdout.readline.side_effect = [json.dumps({"type": "ready", "metadata": {}}) + "\n", bad_utf8]
                process = SimpleNamespace(stdin=StringIO(), stdout=stdout, poll=Mock(return_value=status),
                    terminate=Mock(), kill=Mock(),
                    wait=Mock(side_effect=[subprocess.TimeoutExpired("worker", 0.1), 0]))
                with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                    worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                        EXECUTION_DEFAULTS, Mock(), Mock(), root)
                try:
                    expected = scheduler_module.PreparationWorkerExited if status is not None else UnicodeDecodeError
                    with self.assertRaises(expected):
                        worker.prepare(({"id": "patient"}, {}))
                    if status is None:
                        process.wait.assert_called_once_with(timeout=0.1)
                finally:
                    worker.close()
                self.assertTrue(worker._log.closed)
                stdout.close.assert_called_once_with()

    def test_complete_checkpoint_without_final_newline_is_committed_before_ack_failure(self):
        class ClosedAck(StringIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0
            def flush(self):
                self.flushes += 1
                if self.flushes == 2:
                    raise BrokenPipeError("child exited after complete JSON but before newline")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"id": "patient", "durable_response": "complete paid reply"}
            store = Mock()
            process = SimpleNamespace(stdin=ClosedAck(), stdout=StringIO(
                json.dumps({"type": "ready", "metadata": {}}) + "\n"
                + json.dumps({"type": "checkpoint", "result": payload})), poll=Mock(return_value=-9))
            with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                    EXECUTION_DEFAULTS, store, Mock(), root)
            try:
                with self.assertRaises(scheduler_module.PreparationWorkerExited):
                    worker.prepare(({"id": "patient"}, {}))
                store.save.assert_called_once_with(("preparation", "patient", None, None), payload)
            finally:
                worker.close()

    def test_complete_invalid_json_and_fatal_events_are_not_retryable(self):
        for reply in ('{"not valid":}\n', json.dumps({"type": "fatal", "error": "invalid configuration"}) + "\n"):
            with self.subTest(reply=reply), TemporaryDirectory() as directory:
                root = Path(directory)
                process = SimpleNamespace(stdin=StringIO(), stdout=StringIO(
                    json.dumps({"type": "ready", "metadata": {}}) + "\n" + reply), poll=Mock(return_value=-9))
                with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                    worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                        EXECUTION_DEFAULTS, Mock(), Mock(), root)
                try:
                    with self.assertRaises((ValueError, RuntimeError)) as raised:
                        worker.prepare(({"id": "patient"}, {}))
                    self.assertNotIsInstance(raised.exception, scheduler_module.PreparationWorkerExited)
                finally:
                    worker.close()

    def test_ack_broken_pipe_happens_after_checkpoint_commit_and_is_retryable(self):
        class BrokenAck(StringIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0
            def flush(self):
                self.flushes += 1
                if self.flushes == 2:
                    raise BrokenPipeError("child died before ACK")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"id": "patient", "durable_response": "paid reply"}
            store = Mock()
            process = SimpleNamespace(stdin=BrokenAck(), stdout=StringIO("\n".join(map(json.dumps, [
                {"type": "ready", "metadata": {}}, {"type": "checkpoint", "result": payload}])) + "\n"),
                poll=Mock(return_value=-9))
            with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                    EXECUTION_DEFAULTS, store, Mock(), root)
            try:
                with self.assertRaises(scheduler_module.PreparationWorkerExited):
                    worker.prepare(({"id": "patient"}, {}))
                store.save.assert_called_once_with(("preparation", "patient", None, None), payload)
                commands = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
                self.assertEqual(commands[-1], {"type": "ack"})
            finally:
                worker.close()

    def test_checkpoint_commit_error_is_fatal_even_if_child_also_exited(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"id": "patient", "durable_response": "paid reply"}
            store = Mock()
            store.save.side_effect = RunMetadataError("cannot persist checkpoint")
            process = SimpleNamespace(stdin=StringIO(), stdout=StringIO("\n".join(map(json.dumps, [
                {"type": "ready", "metadata": {}}, {"type": "checkpoint", "result": payload}])) + "\n"),
                poll=Mock(return_value=-9))
            with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                    EXECUTION_DEFAULTS, store, Mock(), root)
            try:
                with self.assertRaisesRegex(RunMetadataError, "cannot persist checkpoint"):
                    worker.prepare(({"id": "patient"}, {}))
                self.assertEqual(len(process.stdin.getvalue().splitlines()), 1, "No ACK before a durable save")
            finally:
                worker.close()

    def test_cancelled_parent_commits_received_checkpoint_then_sends_stop_ack(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stop = Event()
            store, run_file = Mock(), Mock()
            payload = {"id": "patient", "paid_reply": "received"}
            process = SimpleNamespace(stdin=StringIO(), stdout=StringIO("\n".join(map(json.dumps, [
                {"type": "ready", "metadata": {}}, {"type": "checkpoint", "result": payload},
                {"type": "fatal", "error": "Parent stopped"}])) + "\n"), poll=Mock(return_value=0))
            with patch.object(scheduler_module.subprocess, "Popen", return_value=process):
                worker = scheduler_module.PreparationWorker(root / "config.yaml", root, 0,
                    EXECUTION_DEFAULTS, store, run_file, root, stop=stop)
            stop.set()
            try:
                with self.assertRaisesRegex(RuntimeError, "Parent stopped"):
                    worker.prepare(({"id": "patient"}, {}))
                store.save.assert_called_once_with(("preparation", "patient", None, None), payload)
                commands = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
                self.assertEqual(commands[-1], {"type": "ack", "error": "Run is stopping"})
            finally:
                worker.close()

    def test_constructor_closes_log_when_process_cannot_start(self):
        log = StringIO()
        with patch.object(Path, "open", return_value=log), \
                patch.object(scheduler_module.subprocess, "Popen", side_effect=OSError("spawn unavailable")), \
                self.assertRaisesRegex(OSError, "spawn unavailable"):
            scheduler_module.PreparationWorker(Path("config.yaml"), Path("/tmp"), 0,
                EXECUTION_DEFAULTS, Mock(), Mock(), ROOT)
        self.assertTrue(log.closed)

    def test_close_terminates_then_kills_unresponsive_process(self):
        worker = scheduler_module.PreparationWorker.__new__(scheduler_module.PreparationWorker)
        worker._log = StringIO()
        worker.process = SimpleNamespace(poll=Mock(return_value=None), terminate=Mock(), kill=Mock(),
            wait=Mock(side_effect=[subprocess.TimeoutExpired("worker", 10), 0]),
            stdin=StringIO(), stdout=StringIO())
        worker.close()
        worker.process.terminate.assert_called_once_with()
        worker.process.kill.assert_called_once_with()
        self.assertEqual(worker.process.wait.call_args_list[0].kwargs, {"timeout": 10})
        self.assertTrue(worker.process.stdin.closed and worker.process.stdout.closed and worker._log.closed)

    def test_close_ignores_dead_pipe_flush_and_still_closes_all_handles(self):
        worker = scheduler_module.PreparationWorker.__new__(scheduler_module.PreparationWorker)
        worker._log = StringIO()
        stdin = Mock()
        stdin.close.side_effect = BrokenPipeError("flush to dead child")
        worker.process = SimpleNamespace(poll=Mock(return_value=-9), stdin=stdin, stdout=StringIO())
        worker.close()
        stdin.close.assert_called_once_with()
        self.assertTrue(worker.process.stdout.closed)
        self.assertTrue(worker._log.closed)

    def test_other_close_errors_are_fatal_but_do_not_leak_remaining_handles(self):
        worker = scheduler_module.PreparationWorker.__new__(scheduler_module.PreparationWorker)
        worker._log = StringIO()
        stdin = Mock()
        stdin.close.side_effect = OSError("unexpected descriptor failure")
        worker.process = SimpleNamespace(poll=Mock(return_value=-9), stdin=stdin, stdout=StringIO())
        with self.assertRaisesRegex(OSError, "unexpected descriptor failure"):
            worker.close()
        self.assertTrue(worker.process.stdout.closed)
        self.assertTrue(worker._log.closed)


class RunnerEntryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = PipelineFixture(self.root)
        corpus = self.root / "reference/corpus.json"
        artifacts = self.root / "reference/artifacts"
        guidelines = self.root / "guidelines"
        artifacts.mkdir(parents=True)
        corpus.write_text('[{"subjective": "reference", "objective": "evidence"}]')
        for name in ("embeddings.npy", "index.faiss", "matching.json"):
            (artifacts / name).write_text("offline " + name)
        (guidelines / "output").mkdir(parents=True)
        (guidelines / "settings.yaml").write_text(yaml.safe_dump({"models": {
            "default_chat_model": {"type": "chat", "model_provider": "openai", "model": "gpt-4o-mini",
                                   "api_key": "offline-test-key", "requests_per_minute": "auto", "tokens_per_minute": "auto"},
            "default_embedding_model": {"requests_per_minute": None, "tokens_per_minute": None},
        }}))
        for name in GUIDELINE_TABLES:
            (guidelines / "output" / (name + ".parquet")).write_text("offline " + name)
        self.fixture.reference_data = reference_data_metadata(self.root, corpus_file=corpus,
            artifacts_directory=artifacts, artifact_files=[artifacts / name for name in
                ("embeddings.npy", "index.faiss", "matching.json")],
            guideline_directory=guidelines, guideline_settings=guidelines / "settings.yaml",
            guideline_output=guidelines / "output")
        self.config = self.root / "entry.yaml"
        (self.root / "condition_plan_models.yaml").write_bytes(self.fixture.registry.read_bytes())
        self.settings = {
            "selected_models": list(ALIASES), "selected_variants": list(VARIANTS),
            "temperature": 1, "max_retries": 1, "max_patients": None,
            "patient_input": str(self.fixture.experiment.patient_input),
            "patient_labels": str(self.fixture.experiment.patient_labels),
            "experiment_output": str(self.root / "output"), "plan_judge": "local-c",
            "execution": {"manage_ollama": False, "local_concurrency": 2,
                          "hosted_concurrency": 2, "scoring_batch_patients": 2},
        }
        self.write()
        with patch("dotenv.load_dotenv", return_value=False):
            self.namespace = runpy.run_path(str(ROOT / "scripts/run_experiments.py"))
        self.entry = self.namespace["main"]
        self.server = FakeServer()

    def write(self):
        self.config.write_text(yaml.safe_dump(self.settings, sort_keys=False))

    def invoke(self, extra=None, *, report_failure=False, server_failure=False):
        fixture = self.fixture
        def scorer(*args, **kwargs):
            return PatientScorer(*args, **kwargs, metric_evaluator=fixture.metrics)
        def report(directory, **kwargs):
            if report_failure:
                raise OSError("report disk unavailable")
            path = directory / "results_summary.html"
            path.write_text("<html>saved predictions</html>")
            return path
        class BrokenServer:
            def __enter__(self):
                raise RuntimeError("server offline")
            def __exit__(self, *args):
                return None
        server = BrokenServer() if server_failure else self.server
        with patch.dict(os.environ, {"OFFLINE_MODEL_KEY": "offline-secret", "OPENAI_API_KEY": "offline-openai",
                                     "GRAPHRAG_API_KEY": "offline-graph"}), \
                patch("scripts.med_copilot_models.ModelFactory.create_chat_model", side_effect=fixture.model_factory), \
                patch.object(scheduler_module, "PreparationWorker", fixture.preparer_factory()), \
                patch.object(scheduler_module, "_CPUEncoder", return_value=lambda texts: np.asarray([[1.0, 0.0] for _ in texts])), \
                patch.dict(self.entry.__globals__, {
                    "ROOT": self.root, "PatientScorer": scorer, "generate_report": report,
                    "ManagedOllama": lambda *args, **kwargs: server,
                    "git_branch": lambda root: "speedup", "git_commit": lambda root: "offline-commit",
                    "_sources": lambda: [{"file": "scripts/run_experiments.py", "sha256": "offline-source"}],
                }):
            return self.entry(["--config", str(self.config), "--yes", *(extra or [])])

    def saved(self):
        [directory] = (self.root / "output").iterdir()
        return directory, json.loads((directory / "run.json").read_text()), [
            json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]

    def test_entry_runs_real_pipeline_records_all_settings_and_resumes_without_calls(self):
        self.assertEqual(self.invoke(), 0)
        directory, metadata, records = self.saved()
        for field, value in self.settings.items():
            if field == "execution":
                self.assertEqual(metadata["experiment"][field], {**EXECUTION_DEFAULTS, **value})
            else:
                expected = str(Path(value).relative_to(self.root)) if field in PATH_SETTINGS else value
                self.assertEqual(metadata["experiment"][field], expected)
        self.assertEqual(metadata["experiment"]["config_sha256"], hashlib.sha256(self.config.read_bytes()).hexdigest())
        self.assertEqual(metadata["execution_sources"], [{"file": "scripts/run_experiments.py", "sha256": "offline-source"}])
        self.assertEqual(metadata["ollama_runtime"], self.server.metadata())
        self.assertEqual(metadata["schema_version"], 4)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["results_summary"]["status"], "completed")
        self.assertTrue({"patient_id", "variant", "model", "seconds", "status"}
                        <= set(metadata["runtime_timings"]["generation"][0]))
        self.assertEqual({item["model"] for item in metadata["runtime_timings"]["model_phases"]}, set(ALIASES[1:]))
        self.assertEqual(len(records), 3)
        metadata["evaluation"]["correctness"]["standalone_stage_marker"] = "judged later"
        (directory / "run.json").write_text(json.dumps(metadata))
        before = len(self.fixture.calls)
        self.assertEqual(self.invoke(["--resume", str(directory)]), 0)
        self.assertEqual(len(self.fixture.calls), before)
        self.assertEqual(self.saved()[2], records)
        self.assertEqual(len(self.saved()[1]["resumptions"]), 1)
        self.assertEqual(self.saved()[1]["evaluation"]["correctness"]["standalone_stage_marker"], "judged later")

    def test_resume_blocks_even_comment_only_configuration_change_before_rewriting_artifacts(self):
        self.assertEqual(self.invoke(), 0)
        directory, _, _ = self.saved()
        artifacts = {name: (directory / name).read_bytes() for name in ("run.json", "results.jsonl", "jobs.sqlite3")}
        before = len(self.fixture.calls)
        self.config.write_text(self.config.read_text() + "\n# changed after interruption\n")
        with self.assertRaisesRegex(ValueError, "experiment configuration changed"):
            self.invoke(["--resume", str(directory)])
        self.assertEqual(len(self.fixture.calls), before)
        self.assertEqual({name: (directory / name).read_bytes() for name in artifacts}, artifacts)

    def test_invalid_configuration_stops_before_provider_or_artifact_work(self):
        self.settings["execution"]["local_concurrency"] = 65
        self.write()
        with self.assertRaisesRegex(ValueError, "local_concurrency"):
            self.invoke()
        self.assertEqual(self.fixture.calls, [])
        self.assertFalse((self.root / "output").exists())

    def test_resume_rejects_changed_variant_prompt_and_implementation_provenance_without_writes(self):
        self.assertEqual(self.invoke(), 0)
        directory, _, _ = self.saved()
        artifacts = {name: (directory / name).read_bytes() for name in ("run.json", "results.jsonl", "jobs.sqlite3")}
        before = len(self.fixture.calls)
        original = ExperimentConfig.variant_metadata
        for field in ("variant_definitions", "component_definitions", "registry_provenance", "prompt_provenance"):
            def changed(config, root):
                definitions = original(config, root)
                definitions[field] = {"changed": True}
                return definitions
            with self.subTest(field=field), patch.object(ExperimentConfig, "variant_metadata", changed), \
                    self.assertRaisesRegex(ValueError, field):
                self.invoke(["--resume", str(directory)])
            self.assertEqual({name: (directory / name).read_bytes() for name in artifacts}, artifacts)
        self.assertEqual(len(self.fixture.calls), before)

    def test_resume_rejects_changed_correctness_semantics_without_writes(self):
        self.assertEqual(self.invoke(), 0)
        directory, _, _ = self.saved()
        artifacts = {name: (directory / name).read_bytes() for name in ("run.json", "results.jsonl", "jobs.sqlite3")}
        before = len(self.fixture.calls)
        with patch("scripts.med_copilot_judges.BIOLORD_REVISION", "changed-revision"), \
                self.assertRaisesRegex(ValueError, "correctness.diagnosis.model_revision"):
            self.invoke(["--resume", str(directory)])
        self.assertEqual({name: (directory / name).read_bytes() for name in artifacts}, artifacts)
        self.assertEqual(len(self.fixture.calls), before)

    def test_resume_rejects_changed_corpus_index_and_guideline_resources_without_writes(self):
        self.assertEqual(self.invoke(), 0)
        directory, _, _ = self.saved()
        artifacts = {name: (directory / name).read_bytes() for name in ("run.json", "results.jsonl", "jobs.sqlite3")}
        before = len(self.fixture.calls)
        resources = [(self.root / "reference/corpus.json", "patient_retrieval"),
                     *[(self.root / "reference/artifacts" / name, "patient_retrieval")
                       for name in ("embeddings.npy", "index.faiss", "matching.json")],
                     (self.root / "guidelines/settings.yaml", "guidelines"),
                     (self.root / "guidelines/output/entities.parquet", "guidelines")]
        for path, resource in resources:
            old = path.read_bytes()
            try:
                path.write_bytes(old + b"\n# changed\n")
                with self.subTest(resource=path.name), self.assertRaisesRegex(ValueError, resource + " reference data changed"):
                    self.invoke(["--resume", str(directory)])
            finally:
                path.write_bytes(old)
            self.assertEqual({name: (directory / name).read_bytes() for name in artifacts}, artifacts)
        self.assertEqual(len(self.fixture.calls), before)

    def test_numeric_guideline_chat_limits_are_valid_but_block_generation_resume_before_any_work(self):
        self.assertEqual(self.invoke(), 0)
        directory, metadata, _ = self.saved()
        artifacts = {name: (directory / name).read_bytes() for name in ("run.json", "results.jsonl", "jobs.sqlite3")}
        before = (len(self.fixture.calls), len(self.fixture.prepared), len(self.fixture.paid_preparation))
        path = self.root / "guidelines/settings.yaml"
        original = yaml.safe_load(path.read_text())
        changed = deepcopy(original)
        changed["models"]["default_chat_model"].update(requests_per_minute=12, tokens_per_minute=200000)
        path.write_text(yaml.safe_dump(changed))
        accepted = LanguageModelConfig(**changed["models"]["default_chat_model"])
        self.assertEqual((accepted.requests_per_minute, accepted.tokens_per_minute), (12, 200000))
        self.assertEqual(changed["models"]["default_embedding_model"], original["models"]["default_embedding_model"])
        current = reference_data_metadata(self.root, corpus_file=None, artifacts_directory=self.root,
            artifact_files=None, guideline_directory=path.parent, guideline_settings=path,
            guideline_output=path.parent / "output")
        self.assertNotEqual(current["guidelines"]["sha256"], metadata["reference_data"]["guidelines"]["sha256"])
        with self.assertRaisesRegex(ValueError, "guidelines reference data changed"):
            self.invoke(["--resume", str(directory)])
        self.assertEqual({name: (directory / name).read_bytes() for name in artifacts}, artifacts)
        self.assertEqual((len(self.fixture.calls), len(self.fixture.prepared), len(self.fixture.paid_preparation)), before)

    def test_managed_server_rejects_generator_and_deferred_judge_endpoint_mismatches_before_artifacts(self):
        self.settings["execution"]["manage_ollama"] = True
        self.write()
        path = self.root / "condition_plan_models.yaml"
        registry = yaml.safe_load(path.read_text())
        for alias, endpoint in (("local-a", "http://localhost:11434/v1"),
                                ("local-a", "http://127.0.0.2:11435/v1"),
                                ("judge-only", "http://127.0.0.1:11436/v1")):
            changed = deepcopy(registry)
            changed["models"][alias] = {**changed["models"]["local-a"], "api_base": endpoint}
            self.settings["plan_judge"] = alias if alias == "judge-only" else "local-c"
            self.write()
            path.write_text(yaml.safe_dump(changed))
            with self.subTest(alias=alias, endpoint=endpoint), self.assertRaisesRegex(ValueError, alias + ": local api_base"):
                self.invoke()
            self.assertFalse((self.root / "output").exists())
        self.assertEqual(self.fixture.calls, [])

    def test_managed_endpoint_check_allows_localhost_on_the_owned_port(self):
        self.settings["execution"]["manage_ollama"] = True
        self.write()
        path = self.root / "condition_plan_models.yaml"
        registry = yaml.safe_load(path.read_text())
        registry["models"]["local-a"]["api_base"] = "http://localhost:11435/v1/"
        path.write_text(yaml.safe_dump(registry))
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(len(self.fixture.calls), 3 * len(VARIANTS) * len(ALIASES))

    def test_server_startup_failure_marks_run_failed_without_provider_requests(self):
        with self.assertRaisesRegex(RuntimeError, "server offline"):
            self.invoke(server_failure=True)
        [directory] = (self.root / "output").iterdir()
        metadata = json.loads((directory / "run.json").read_text())
        self.assertEqual(metadata["status"], "failed_incomplete")
        self.assertEqual(self.fixture.calls, [])

    def test_report_failure_keeps_completed_results_and_explicit_failure_metadata(self):
        self.assertEqual(self.invoke(report_failure=True), 1)
        _, metadata, records = self.saved()
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["results_summary"]["status"], "failed_incomplete")
        self.assertTrue(all(record["status"] == "completed" for record in records))

    def test_cleanup_export_failure_does_not_mask_original_generation_failure(self):
        export = scheduler_module.ExperimentScheduler.export
        calls = []
        def broken_cleanup(scheduler):
            calls.append(True)
            if len(calls) > 1:
                raise OSError("cleanup export unavailable")
            return export(scheduler)
        with patch.object(scheduler_module.ExperimentScheduler, "export", broken_cleanup), \
                patch.object(scheduler_module.ExperimentScheduler, "prepare_and_generate",
                             side_effect=RuntimeError("original generation failed")), \
                self.assertRaisesRegex(RuntimeError, "original generation failed"):
            self.invoke()
        _, metadata, _ = self.saved()
        self.assertEqual(metadata["status"], "failed_incomplete")
        self.assertEqual(metadata["errors"][-1]["message"], "original generation failed")

    def test_readable_entry_has_top_level_imports_and_recognizable_stages(self):
        source = (ROOT / "scripts/run_experiments.py").read_text()
        tree = ast.parse(source)
        self.assertNotIn("def _main(", source)
        self.assertIn("# Clinical evidence and exact prompts", source)
        self.assertIn("# Batched text metrics", source)
        self.assertIn("# Canonical results", source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertIn(node, tree.body)


if __name__ == "__main__":
    unittest.main()
