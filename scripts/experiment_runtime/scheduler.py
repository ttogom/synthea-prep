"""Bounded, recoverable preparation and model-major execution for experiment runs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Event, Lock
from time import perf_counter, time_ns

from sentence_transformers import SentenceTransformer

from scripts.experiment_runtime.execution import JobStore, bounded_map, write_results, _decode as decode_checkpoint
from scripts.med_copilot_helpers import new_patient_result
from scripts.experiment_runtime.preparation import validate_preparation_result
from scripts.med_copilot_judges import BIOLORD_MODEL, BIOLORD_REVISION, BioLordSimilarity, CorrectnessJudges, JUDGE_FILE
from scripts.med_copilot_run_file import error_scope, pacific_now, pipeline_stage


MAX_PREPARATION_REPLACEMENTS = 2


class PreparationWorkerExited(RuntimeError):
    """The subprocess transport ended unexpectedly, rather than reporting a failure."""

    def __init__(self, message, *, exit_code=None):
        super().__init__(message)
        self.exit_code = exit_code


class PreparationWorker:
    """One persistent interpreter; checkpoint events are acknowledged after commit."""

    def __init__(self, config_path, run_dir, number, settings, store, run_file, root, stop=None):
        self.store, self.run_file, self.root = store, run_file, root
        self.stop = stop
        self._log = (run_dir / f"preparation-{number}.log").open("a", encoding="utf-8")
        self.process = None
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(root / "scripts/experiment_runtime/worker.py"), str(config_path),
                 "--device", settings["encoder_device"], "--threads", str(settings["cpu_threads"])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                text=True, bufsize=1, cwd=root,
            )
            ready = self._read()
            if ready.get("type") != "ready":
                raise RuntimeError(f"Preparation worker failed: {ready.get('error', ready)}")
            self.metadata = ready["metadata"]
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            line = self.process.stdout.readline()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise self._exited("reading its response") from exc
        except UnicodeDecodeError as exc:
            # A signal can cut a UTF-8 character in half. A live worker emitting
            # invalid text is a protocol bug, not a reason to repeat paid work.
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    raise exc
            raise self._exited("reading a truncated UTF-8 response") from exc
        if not line:
            raise self._exited("waiting for its response")
        try:
            event = decode_checkpoint(line)
        except json.JSONDecodeError as exc:
            if not line.endswith("\n"):
                raise self._exited("reading a truncated response") from exc
            raise
        if not isinstance(event, dict):
            raise ValueError("Preparation worker returned a non-object protocol message")
        return event

    def _exited(self, action):
        status = self.process.poll()
        return PreparationWorkerExited(
            f"Preparation worker exited or closed its pipe while {action} (status {status}); see its log",
            exit_code=status)

    def _send(self, message):
        payload = json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise self._exited("sending a command or checkpoint acknowledgement") from exc

    def prepare(self, item):
        patient, result = item
        self._send({"type": "prepare", "patient": patient, "result": result})
        while True:
            event = self._read()
            kind = event.get("type")
            if kind == "checkpoint":
                self.store.save(("preparation", patient["id"], None, None), event["result"])
                self._send({"type": "ack", "error": "Run is stopping"}
                           if self.stop is not None and self.stop.is_set() else {"type": "ack"})
            elif kind == "error":
                self.run_file.add_error(event["error"])
                self._send({"type": "ack"})
            elif kind == "result":
                self.store.save(("preparation", patient["id"], None, None), event["result"], "completed")
                return event
            else:
                raise RuntimeError(f"Preparation worker failed: {event.get('error', event)}")

    @staticmethod
    def _close_pipe(handle):
        try:
            handle.close()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Buffered stdin may flush again after the child has died.

    def close(self):
        with ExitStack() as handles:
            handles.callback(self._log.close)
            if getattr(self, "process", None) is not None:
                for handle in (self.process.stdin, self.process.stdout):
                    if handle is not None:
                        handles.callback(self._close_pipe, handle)
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()


class _JobScope:
    """A completion's checkpoint owner, separate from any current patient."""

    def __init__(self, store, key, output, run_file):
        self.store, self.key, self.output, self.run_file = store, key, output, run_file
        self.redact = run_file.redact
        self.metadata = run_file.metadata

    def checkpoint(self):
        self.store.save(self.key, self.output, self.output.get("status", "checkpoint"))

    def add_error(self, entry):
        self.run_file.add_error(entry)


class _CPUEncoder:
    """Lazy pinned BioLORD on CPU, owned by the sole scoring thread."""

    def __init__(self):
        self.model = None

    def __call__(self, texts):
        if self.model is None:
            self.model = SentenceTransformer(BIOLORD_MODEL, revision=BIOLORD_REVISION, device="cpu")
        return self.model.encode(texts, convert_to_numpy=True)


def _messages(output):
    saved = output.get("assessment_plan_prompts")
    return None if saved is None else [{"role": role, "content": saved[role]} for role in ("system", "user")]


def _scored_fields(record):
    return sum(entry.get(field, {}).get("status") in ("scored", "error", "interrupted")
               for variant in record["variants"].values()
               for output in variant["assessment_plans"].values()
               for entry in output.get("evaluation") or []
               for field in ("condition", "plan"))


class ExperimentScheduler:
    """One artifact owner; isolated preparation, bounded requests, persistent scoring."""

    def __init__(self, experiment, patients, models, scorer, run_file, *, root, server=None):
        self.experiment, self.patients, self.models, self.scorer = experiment, patients, models, scorer
        self.run_file, self.root, self.server = run_file, Path(root), server
        self.settings = experiment.execution
        self._lock = Lock()
        self._recovery_lock = Lock()
        self._recovery_invocation = str(time_ns())
        self._preparation_config_sha256 = run_file.metadata["experiment"]["config_sha256"]
        self._preparation_reference_data = deepcopy(run_file.metadata.get("reference_data"))
        self._stop = Event()
        self._failure = None
        self.records = {}
        self.timings = {"preparation": [], "generation": [], "scoring": []}
        canonical = run_file.run_dir / "results.jsonl"
        if canonical.exists():
            for line in canonical.read_text().split("\n"):
                if not line.strip():
                    continue
                row = decode_checkpoint(line)
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    raise ValueError("Invalid saved patient result")
                if row["id"] in self.records:
                    raise ValueError("Duplicate saved patient result")
                self.records[row["id"]] = row
        expected_ids = {patient["id"] for patient in patients}
        if set(self.records) - expected_ids:
            raise ValueError("Saved results contain patients outside the recorded cohort")
        for record in self.records.values():
            validate_preparation_result(experiment, record)
        self.store = JobStore(run_file.run_dir / "jobs.sqlite3", identity={
            "experiment": experiment.metadata(root),
            "input_sha256": hashlib.sha256(experiment.patient_input.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(experiment.patient_labels.read_bytes()).hexdigest(),
            "models": {k: v for k, v in models.metadata(root).items() if k != "config_sha256"},
        }, redact=run_file.redact)
        for patient in patients:
            patient_id = patient["id"]
            prepared = self.store.load(("preparation", patient_id, None, None))
            if patient_id not in self.records:
                self.records[patient_id] = prepared or new_patient_result(
                    patient_id, model_ids=models.model_ids, variant_ids=experiment.selected_variants,
                    component_plan=experiment.component_plan, variant_components=experiment.variant_components)
            record = self.records[patient_id]
            if prepared is not None:
                for name, component in record["components"].items():
                    if component["status"] != "completed":
                        record["components"][name] = deepcopy(prepared["components"][name])
                for name, output in record["variants"].items():
                    if output.get("assessment_plan_prompts") is None:
                        output["assessment_plan_prompts"] = deepcopy(
                            prepared["variants"][name].get("assessment_plan_prompts"))
            for variant, output in record["variants"].items():
                for model_id, slot in output["assessment_plans"].items():
                    saved = self.store.load(("generation", patient_id, variant, model_id))
                    # Canonical completed slots may contain later standalone Plan scores.
                    if slot["status"] != "completed" and saved is not None:
                        output["assessment_plans"][model_id] = saved
            scored = self.store.load(("scoring", patient_id, None, None))
            if scored is not None:
                for variant, output in record["variants"].items():
                    for model_id, slot in output["assessment_plans"].items():
                        previous = scored["variants"][variant]["assessment_plans"][model_id]
                        if slot.get("predictions") != previous.get("predictions"):
                            continue
                        if slot.get("evaluation") is None:
                            slot["evaluation"] = deepcopy(previous.get("evaluation"))
                        elif isinstance(previous.get("evaluation"), list):
                            for current, saved in zip(slot["evaluation"], previous["evaluation"]):
                                for field in ("condition", "plan", "diagnosis_correctness", "plan_correctness"):
                                    if (current.get(field, {}).get("status") != "scored"
                                            and saved.get(field, {}).get("status") == "scored"):
                                        current[field] = deepcopy(saved[field])
            validate_preparation_result(experiment, record)
        self._clock = perf_counter()
        self._scorer_ready = False
        for _, state in self.store.entries("timing"):
            self.timings["generation"].append(state["payload"])

    def close(self):
        self.store.close()

    def _abort(self, exc):
        with self._lock:
            if self._failure is None:
                self._failure = exc
        self._stop.set()

    def _check_stopped(self):
        if self._stop.is_set():
            raise self._failure or RuntimeError("Experiment execution is stopping")

    def _check_preparation_config(self):
        current = hashlib.sha256(self.experiment.config_path.read_bytes()).hexdigest()
        if current != self._preparation_config_sha256:
            raise ValueError("Cannot replace preparation worker: experiment configuration changed since the run")

    def _new_preparation_worker(self, number):
        self._check_stopped()
        self._check_preparation_config()
        worker = PreparationWorker(self.experiment.config_path, self.run_file.run_dir, number,
            self.settings, self.store, self.run_file, self.root, stop=self._stop)
        try:
            self._check_stopped()
            self._check_preparation_config()
            reference = worker.metadata.get("reference_data")
            if self._preparation_reference_data is not None and reference != self._preparation_reference_data:
                raise ValueError("Cannot replace preparation worker: reference data changed since the run")
            if self._preparation_reference_data is None:
                self._preparation_reference_data = deepcopy(reference)
            return worker
        except BaseException:
            worker.close()
            raise

    def _record_preparation_recovery(self, number, patient_id, attempt, outcome, error=None):
        event = {"timestamp": pacific_now().isoformat(), "invocation": self._recovery_invocation,
                 "worker": number, "patient_id": patient_id, "attempt": attempt, "outcome": outcome,
                 "log_file": f"preparation-{number}.log",
                 "exit_code": getattr(error, "exit_code", None)}
        if error is not None:
            event["error"] = {"type": type(error).__name__, "message": self.run_file.redact(str(error))}
        # Separate from ordinary errors: a successfully recovered death should
        # not make an otherwise successful experiment exit with an error.
        with self._recovery_lock:
            previous = self.run_file.metadata.get("preparation_recovery", {})
            self.run_file.update(preparation_recovery={
                "max_replacements_per_patient": MAX_PREPARATION_REPLACEMENTS,
                "max_startup_replacements_per_worker": MAX_PREPARATION_REPLACEMENTS,
                "budget_scope": "invocation", "log_file": "preparation-recovery.log",
                "events": [*previous.get("events", []), event]})
            with (self.run_file.run_dir / "preparation-recovery.log").open("a", encoding="utf-8") as log:
                log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def _start_preparation_worker(self, number):
        """Startup deaths have a separate bounded slot budget before any patient."""
        for attempt in range(MAX_PREPARATION_REPLACEMENTS + 1):
            try:
                worker = self._new_preparation_worker(number)
            except PreparationWorkerExited as exc:
                self._check_stopped()
                self._check_preparation_config()
                exhausted = attempt == MAX_PREPARATION_REPLACEMENTS
                self._record_preparation_recovery(number, None, attempt if exhausted else attempt + 1,
                                                  "exhausted" if exhausted else "replacement", exc)
                if exhausted:
                    raise RuntimeError(f"Preparation worker {number} exhausted its {MAX_PREPARATION_REPLACEMENTS} startup replacements; "
                                       "fix the worker failure and resume the run") from exc
            else:
                if attempt:
                    self._record_preparation_recovery(number, None, attempt, "recovered")
                return worker

    def _merge_preparation(self, patient_id, prepared):
        """Only evidence and missing prompts may replace the parent's current work."""
        if not isinstance(prepared, dict) or prepared.get("id") != patient_id:
            raise ValueError("Preparation checkpoint belongs to a different patient")
        validate_preparation_result(self.experiment, prepared)
        with self._lock:
            result = deepcopy(self.records[patient_id])
        for name, component in result["components"].items():
            if component["status"] != "completed":
                result["components"][name] = deepcopy(prepared["components"][name])
        for name, output in result["variants"].items():
            if output.get("assessment_plan_prompts") is None:
                output["assessment_plan_prompts"] = deepcopy(prepared["variants"][name].get("assessment_plan_prompts"))
        return result

    def _prepare_patient(self, workers, number, item):
        patient, submitted = item
        patient_id = patient["id"]
        attempt = 0
        replace = False
        while True:
            self._check_stopped()
            self._check_preparation_config()
            # The submitted snapshot can predate a paid response committed just
            # before the old subprocess lost its acknowledgement pipe.
            saved = self.store.load(("preparation", patient_id, None, None))
            result = self._merge_preparation(patient_id, saved if saved is not None else submitted)
            try:
                if replace:
                    workers[number] = self._new_preparation_worker(number)
                event = workers[number].prepare((patient, result))
            except PreparationWorkerExited as exc:
                workers[number].close()
                self._check_stopped()
                self._check_preparation_config()
                saved = self.store.load(("preparation", patient_id, None, None))
                if saved is not None:
                    recovered = self._merge_preparation(patient_id, saved)
                    with self._lock:
                        self.records[patient_id] = recovered
                if attempt == MAX_PREPARATION_REPLACEMENTS:
                    self._record_preparation_recovery(number, patient_id, attempt, "exhausted", exc)
                    raise RuntimeError(f"Preparation recovery exhausted for patient {patient_id}: "
                        f"{MAX_PREPARATION_REPLACEMENTS} worker replacements failed; saved checkpoints remain usable. "
                        "Fix the worker failure and resume the run") from exc
                attempt += 1
                self._record_preparation_recovery(number, patient_id, attempt, "replacement", exc)
                replace = True
                continue
            event["result"] = self._merge_preparation(patient_id, event["result"])
            if attempt:
                self._record_preparation_recovery(number, patient_id, attempt, "recovered")
            return event

    @staticmethod
    def _close_preparation_workers(workers):
        with ExitStack() as cleanup:
            for worker in workers:
                cleanup.callback(worker.close)

    def _prepare(self, workers, ready, hosted_ready):
        available = Queue()
        for number in range(len(workers)):
            available.put(number)

        def prepare(item):
            number = available.get()
            try:
                self._check_stopped()
                return self._prepare_patient(workers, number, item)
            except BaseException as exc:
                self._abort(exc)
                raise
            finally:
                available.put(number)

        items = ((patient, deepcopy(self.records[patient["id"]])) for patient in self.patients
                 if not self._stop.is_set())
        try:
            for item, event, elapsed in bounded_map(prepare, items, workers=len(workers)):
                patient_id = item[0]["id"]
                with self._lock:
                    self.records[patient_id] = event["result"]
                    self.timings["preparation"].append({"patient_id": patient_id, "seconds": elapsed})
                ready.put(patient_id)
                hosted_ready.put(patient_id)
        except BaseException as exc:
            error = self._failure or exc
            ready.put(error)
            hosted_ready.put(error)
            raise error
        finally:
            ready.put(None)
            hosted_ready.put(None)

    def _jobs(self, patient_ids, model_ids):
        for patient_id in patient_ids:
            for variant in self.experiment.selected_variants:
                output = self.records[patient_id]["variants"][variant]
                messages = _messages(output)
                if messages is None:
                    continue
                for model_id in model_ids:
                    slot = output["assessment_plans"][model_id]
                    if slot["status"] != "completed":
                        yield patient_id, variant, model_id, messages, deepcopy(slot)

    def _generate(self, job):
        try:
            self._check_stopped()
            return self._generate_job(job)
        except BaseException as exc:
            self._abort(exc)
            raise

    def _generate_job(self, job):
        patient_id, variant, model_id, messages, output = job
        owner = _JobScope(self.store, ("generation", patient_id, variant, model_id), output, self.run_file)
        start = perf_counter()
        with error_scope(owner, patient_id=patient_id), pipeline_stage("final_generation"):
            if self.models.model_is_local(model_id) and not output.get("_generation_reply_complete"):
                # Byte fallback tokenizers cannot normally need more text tokens
                # than input bytes; reserve extra space for template/special
                # tokens and the configured completion. This conservative guard
                # rejects unusually large prompts instead of silently dropping S/O.
                prompt_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
                completion = self.models.metadata(self.root)["models"][model_id]["config"].get("max_tokens") or 768
                if prompt_bytes + 1024 + completion > self.settings["ollama_context_length"]:
                    raise ValueError(f"Prompt for {patient_id}/{variant} exceeds the conservative local context budget")
            self.models.generate_one(model_id, messages, output)
        elapsed = perf_counter() - start
        timing = {"patient_id": patient_id, "variant": variant,
            "model": model_id, "seconds": elapsed, "status": output["status"],
            "start_seconds": start - self._clock, "end_seconds": perf_counter() - self._clock}
        self.store.save(("timing", patient_id, variant, f"{model_id}:{time_ns()}"), timing, "completed")
        with self._lock:
            self.records[patient_id]["variants"][variant]["assessment_plans"][model_id] = output
            self.timings["generation"].append(timing)
        return output

    def _drain(self, jobs, concurrency):
        for _ in bounded_map(self._generate, jobs, workers=concurrency):
            pass

    def prepare_and_generate(self):
        """Overlap evidence and hosted I/O with the first complete local model pass."""
        local = [model for model in self.models.model_ids if self.models.model_is_local(model)]
        self._stop.clear()
        self._failure = None
        hosted = [model for model in self.models.model_ids if model not in local]
        ready = Queue()
        hosted_ready = Queue()
        started = perf_counter()
        # Completed preparation needs no worker, encoder, index, or additional API call.
        needs_preparation = any(
            any(c["status"] != "completed" for c in r["components"].values())
            or any(v.get("assessment_plan_prompts") is None for v in r["variants"].values())
            for r in self.records.values())
        with ExitStack() as resources:
            workers = []
            resources.callback(self._close_preparation_workers, workers)
            for number in range(self.settings["preparation_workers"] if needs_preparation else 0):
                workers.append(self._start_preparation_worker(number))
            if workers:
                self.run_file.update(reference_data=workers[0].metadata.get("reference_data"),
                                     preparation_runtime=[worker.metadata for worker in workers])
            producer = resources.enter_context(ThreadPoolExecutor(max_workers=2))
            scoring_pool = resources.enter_context(ThreadPoolExecutor(max_workers=1))
            scoring_futures = []
            if workers:
                preparation = producer.submit(self._prepare, workers, ready, hosted_ready)
            else:
                preparation = None
                for patient in self.patients:
                    ready.put(patient["id"])
                    hosted_ready.put(patient["id"])
                ready.put(None)
                hosted_ready.put(None)

            def prepared_ids(queue):
                while True:
                    item = queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item

            hosted_future = (producer.submit(self._drain, self._jobs(prepared_ids(hosted_ready), hosted),
                                              self.settings["hosted_concurrency"]) if hosted else None)
            # This callback runs before executor shutdown on every exit. It
            # bounds additional paid preparation after another stage fails.
            resources.callback(self._stop.set)

            if local:
                phase_start = perf_counter()
                self.run_file.update(active_generation_model=local[0])
                self._drain(self._jobs(prepared_ids(ready), [local[0]]), self.settings["local_concurrency"])
                self.timings.setdefault("model_phases", []).append({"model": local[0],
                                                                    "seconds": perf_counter() - phase_start})
                if self.server is not None:
                    self.server.unload(self.models.metadata(self.root)["models"][local[0]]["config"]["model"])
            else:
                list(prepared_ids(ready))
            if preparation is not None:
                preparation.result()
            for worker in workers:
                worker.close()  # Release CPU encoders once their cohort is prepared.
            if self.settings.get("overlap_scoring", False):
                self._initialize_scorer()
                scoring_futures.append(scoring_pool.submit(self._score_available))
            for model_id in local[1:]:
                phase_start = perf_counter()
                self.run_file.update(active_generation_model=model_id)
                self._drain(self._jobs([p["id"] for p in self.patients], [model_id]),
                            self.settings["local_concurrency"])
                self.timings.setdefault("model_phases", []).append({"model": model_id,
                                                                    "seconds": perf_counter() - phase_start})
                self.export()
                if self.server is not None:
                    self.server.unload(self.models.metadata(self.root)["models"][model_id]["config"]["model"])
                if self.settings.get("overlap_scoring", False):
                    scoring_futures.append(scoring_pool.submit(self._score_available))
            if hosted_future is not None:
                hosted_future.result()
            for future in scoring_futures:
                future.result()
        self.timings["preparation_and_generation_seconds"] = perf_counter() - started
        self.export()

    def score(self):
        """A single persistent CPU scorer batches unfinished fields across patients."""
        self._initialize_scorer()
        start = perf_counter()
        self._score_available()
        self.timings["scoring_seconds"] = perf_counter() - start
        self.timings["scoring_service_seconds"] = sum(batch["seconds"] for batch in self.timings["scoring"])
        self.export()

    def _initialize_scorer(self):
        if self._scorer_ready:
            return
        self.scorer.attach_judges(CorrectnessJudges(
            None, (), self.run_file.run_dir / JUDGE_FILE, plan_judge=self.experiment.plan_judge,
            generator_models=self.models.model_ids, plan_execution="deferred",
            similarity=BioLordSimilarity(encode=_CPUEncoder()),
        ))
        self._scorer_ready = True

    def _score_available(self):
        batch_size = self.settings["scoring_batch_patients"]
        patient_ids = [p["id"] for p in self.patients]
        for offset in range(0, len(patient_ids), batch_size):
            with self._lock:
                batch = [deepcopy(self.records[key]) for key in patient_ids[offset:offset + batch_size]]
            before = {record["id"]: _scored_fields(record) for record in batch}
            owner = _ScoreScope(self.store, batch, self.run_file)
            began = perf_counter()
            try:
                with error_scope(owner), pipeline_stage("scoring"):
                    self.scorer.score_results(batch)
            finally:
                owner.flush()
            with self._lock:
                for record in batch:
                    for variant, output in record["variants"].items():
                        for model, scored in output["assessment_plans"].items():
                            current = self.records[record["id"]]["variants"][variant]["assessment_plans"][model]
                            if (scored.get("evaluation") is not None
                                    and current.get("predictions") == scored.get("predictions")):
                                current["evaluation"] = scored["evaluation"]
                self.timings["scoring"].append({"patient_ids": [r["id"] for r in batch],
                    "seconds": perf_counter() - began, "start_seconds": began - self._clock,
                    "end_seconds": perf_counter() - self._clock,
                    "newly_scored_fields_by_patient": {
                        record["id"]: max(0, _scored_fields(record) - before[record["id"]])
                        for record in batch}})

    def export(self):
        with self._lock:
            for record in self.records.values():
                for output in record["variants"].values():
                    if _messages(output) is not None:
                        output["status"] = ("completed" if all(slot["status"] == "completed"
                            for slot in output["assessment_plans"].values()) else "failed_incomplete")
                record["status"] = ("completed" if all(output["status"] == "completed"
                    for output in record["variants"].values()) else "failed_incomplete")
            records = [deepcopy(self.records[p["id"]]) for p in self.patients]
            timings = deepcopy(self.timings)
        write_results(records, self.run_file.run_dir / "results.jsonl")
        self.run_file.update(runtime_timings=timings)


class _ScoreScope:
    def __init__(self, store, records, run_file):
        self.store, self.records, self.run_file = store, records, run_file
        self.redact, self.metadata = run_file.redact, run_file.metadata

    def checkpoint(self):
        # These are pure local metrics. Commit the finite batch, not a full
        # patient snapshot after every one of its individual scalar fields.
        pass

    def flush(self):
        for record in self.records:
            self.store.save(("scoring", record["id"], None, None), record)

    def add_error(self, entry):
        self.run_file.add_error(entry)
