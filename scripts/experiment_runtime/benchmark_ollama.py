"""Compare isolated Ollama scheduling using saved, unchanged experiment prompts.

This calibration generates local outputs only. It does not retrieve evidence,
call hosted models, judge plans, or modify the source experiment artifacts.
"""

import argparse
from copy import deepcopy
import hashlib
from ipaddress import ip_address
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import RLock
from time import perf_counter
from urllib.parse import urlsplit

# Resolve the vendored completion wrappers before importing model clients. The
# local catalogue avoids an unrelated hosted-model pricing lookup at import.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from scripts.med_copilot_config import ExperimentConfig, read_yaml_config
from scripts.experiment_runtime.execution import _decode, bounded_map
from scripts.med_copilot_helpers import new_model_result, parse_condition_plan
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.experiment_runtime.ollama import ManagedOllama
from scripts.med_copilot_run_file import _fsync_directory


VARIANTS = ("llm_only", "patient_retrieval", "guidelines", "patient_retrieval_guidelines")


def select_prompts(run_dir, count=12):
    """Choose equal per-variant length quantiles from native variant artifacts.

    At the default count, each variant contributes its shortest, median, and
    longest stored prompt. Full message objects are retained without edits.
    """
    if type(count) is not int or count < 4 or count % 4:
        raise ValueError("requests must be a positive multiple of four, at least four")
    path = Path(run_dir).resolve() / "results.jsonl"
    groups = {variant: [] for variant in VARIANTS}
    ids = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        result = _decode(line)
        patient_id = result.get("id") if isinstance(result, dict) else None
        if not isinstance(patient_id, str) or not patient_id.strip() or patient_id in ids:
            raise ValueError("Saved prompt artifact must have unique patient IDs")
        ids.add(patient_id)
        variants = result.get("variants")
        if not isinstance(variants, dict):
            raise ValueError("Saved prompt artifact requires native variants")
        for variant in VARIANTS:
            output = variants.get(variant)
            messages = output.get("assessment_plan_prompts") if isinstance(output, dict) else None
            if messages is None:
                continue
            if isinstance(messages, dict) and set(messages) == {"system", "user"}:
                messages = [{"role": role, "content": messages[role]} for role in ("system", "user")]
            if (not isinstance(messages, list) or not messages or any(
                    not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str) or not message["content"].strip()
                    for message in messages)):
                raise ValueError("Saved assessment_plan_prompts must be nonempty text messages")
            encoded = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            groups[variant].append({
                "patient_id": patient_id, "variant": variant, "messages": deepcopy(messages),
                "prompt_bytes": sum(len(message["content"].encode("utf-8")) for message in messages),
                "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
            })
    per_variant = count // 4
    selected = {}
    for variant, prompts in groups.items():
        prompts.sort(key=lambda prompt: (prompt["prompt_bytes"], prompt["patient_id"]))
        if len(prompts) < per_variant:
            raise ValueError(f"Not enough stored {variant} prompts for {count} requests")
        indexes = ([len(prompts) // 2] if per_variant == 1 else
                   [round(index * (len(prompts) - 1) / (per_variant - 1)) for index in range(per_variant)])
        selected[variant] = [prompts[index] for index in indexes]
    return [selected[variant][index] for index in range(per_variant) for variant in VARIANTS]


def _local_selection(experiment):
    registry, _ = read_yaml_config(experiment.model_config)
    registry = registry.get("models") if isinstance(registry, dict) else None
    if not isinstance(registry, dict):
        raise ValueError("Model registry requires a models mapping")
    selected = []
    for alias in experiment.selected_models:
        entry = registry.get(alias)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing selected model: {alias}")
        endpoint = entry.get("api_base")
        if not isinstance(endpoint, str):
            continue
        url = urlsplit(endpoint)
        try:
            local = url.hostname == "localhost" or ip_address(url.hostname).is_loopback
        except ValueError:
            local = False
        if not local:
            continue
        if endpoint.rstrip("/") != "http://127.0.0.1:11435/v1":
            raise ValueError("Benchmark local models must use the owned http://127.0.0.1:11435/v1 endpoint")
        if not isinstance(entry.get("model"), str) or not entry["model"].strip():
            raise ValueError(f"Missing local model tag: {alias}")
        selected.append(alias)
    if len(selected) != 3:
        raise ValueError("Benchmark requires exactly three selected local generation models")
    return selected


def _write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".ollama-benchmark-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _valid_ranks(raw):
    predictions = parse_condition_plan(raw)["predictions"]
    if predictions is None:
        return 0
    return sum(isinstance(pair["condition"], str) and isinstance(pair["plan"], list)
               for pair in predictions)


def _call(models, alias, prompt):
    started = perf_counter()
    output = new_model_result()
    models.generate_one(alias, prompt["messages"], output)
    valid_ranks = _valid_ranks(output.get("assessment_plan_raw"))
    return {
        **{key: prompt[key] for key in ("patient_id", "variant", "prompt_bytes", "prompt_sha256")},
        "model": alias, "elapsed_seconds": perf_counter() - started,
        "valid_prediction_ranks": valid_ranks,
        "valid_predictions": output["status"] == "completed" and valid_ranks == 3,
        "output": output,
    }


def _percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _summary(requests, elapsed):
    times = [request["elapsed_seconds"] for request in requests]
    return {
        "requests": len(requests), "valid_requests": sum(request["valid_predictions"] for request in requests),
        "elapsed_seconds": elapsed, "p50_seconds": _percentile(times, 0.5),
        "p95_seconds": _percentile(times, 0.95),
        "requests_per_second": len(requests) / elapsed if elapsed else None,
        "valid_requests_per_second": sum(request["valid_predictions"] for request in requests) / elapsed if elapsed else None,
    }


def run_benchmark(experiment, prompts, output_path, *, models_factory=AssessmentPlanModels,
                  server_factory=ManagedOllama):
    """Measure three model-major conditions and one three-model challenger.

    Every condition starts a fresh owned server. One request per model warms it
    and measures cold latency, separately from the timed steady request stream.
    JSON is atomically persisted after every received result and on interruption.
    """
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {output_path}")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Benchmark requires selected stored prompts")
    aliases = _local_selection(experiment)
    if experiment.execution.get("ollama_host") != "127.0.0.1:11435":
        raise ValueError("Benchmark uses only its dedicated 127.0.0.1:11435 server")
    models = models_factory(experiment.model_config, aliases, temperature=experiment.temperature,
                            max_retries=experiment.max_retries)
    settings = {**experiment.execution, "manage_ollama": True}
    metadata = models.metadata(ROOT)
    tags = {alias: metadata["models"][alias]["config"]["model"] for alias in aliases}
    report = {
        "status": "running", "experiment": experiment.metadata(ROOT),
        "models": metadata, "request_count_per_model": len(prompts),
        "prompts": [{key: value for key, value in prompt.items() if key != "messages"} for prompt in prompts],
        "conditions": [],
    }
    lock = RLock()

    def save():
        with lock:
            _write(output_path, report)

    def receive(condition, record):
        with lock:
            condition["requests"].append(record)
            save()

    save()
    try:
        for parallel in (1, 2, 4):
            condition = {"kind": "model_major", "parallel_slots": parallel,
                         "client_concurrency": parallel, "max_loaded_models": 1,
                         "status": "running", "cold_requests": [], "requests": [], "snapshots": []}
            report["conditions"].append(condition)
            save()
            with server_factory(settings, output_path.with_suffix(f".parallel-{parallel}.log"),
                                parallel=parallel, max_loaded_models=1) as server:
                condition["server"] = server.metadata()
                for alias in aliases:
                    cold = _call(models, alias, prompts[0])
                    condition["cold_requests"].append(cold)
                    condition["snapshots"].append({"model": alias, "stage": "warm", "ps": server.ps_snapshot()})
                    save()
                    started = perf_counter()
                    records = []
                    for _, record, _ in bounded_map(lambda prompt: _call(models, alias, prompt),
                                                     prompts, workers=parallel, max_pending=parallel):
                        records.append(record)
                        receive(condition, record)
                    condition.setdefault("model_summaries", {})[alias] = _summary(records, perf_counter() - started)
                    condition["snapshots"].append({"model": alias, "stage": "steady", "ps": server.ps_snapshot()})
                    server.unload(tags[alias])
                    save()
                condition["status"] = "completed"
                condition["steady_elapsed_seconds"] = sum(summary["elapsed_seconds"]
                                                            for summary in condition["model_summaries"].values())
                condition["summary"] = _summary(condition["requests"], condition["steady_elapsed_seconds"])
                save()

        condition = {"kind": "mixed_models", "parallel_slots": 1, "client_concurrency": len(aliases),
                     "max_loaded_models": 3, "status": "running", "cold_requests": [], "requests": []}
        report["conditions"].append(condition)
        save()
        with server_factory(settings, output_path.with_suffix(".mixed.log"), parallel=1,
                            max_loaded_models=3) as server:
            condition["server"] = server.metadata()
            for alias in aliases:
                condition["cold_requests"].append(_call(models, alias, prompts[0]))
                save()
            condition["warm_ps"] = server.ps_snapshot()

            def stream(alias):
                records = []
                started = perf_counter()
                for prompt in prompts:
                    record = _call(models, alias, prompt)
                    records.append(record)
                    receive(condition, record)
                return _summary(records, perf_counter() - started)

            started = perf_counter()
            condition["model_summaries"] = {}
            # Each stream is serial: at most one active request per model, and
            # at most three across all models, without a slow-model barrier.
            for alias, summary, _ in bounded_map(stream, aliases, workers=len(aliases), max_pending=len(aliases)):
                with lock:
                    condition["model_summaries"][alias] = summary
                    save()
            condition["steady_elapsed_seconds"] = perf_counter() - started
            condition["steady_ps"] = server.ps_snapshot()
            condition["summary"] = _summary(condition["requests"], condition["steady_elapsed_seconds"])
            for alias in aliases:
                server.unload(tags[alias])
            condition["status"] = "completed"
            save()
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        report["error"] = models.error_metadata(exc)
        raise
    finally:
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=12)
    args = parser.parse_args(argv)
    prompts = select_prompts(args.prompts_run, args.requests)
    report = run_benchmark(ExperimentConfig(args.config), prompts, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "conditions": [condition["summary"] for condition in report["conditions"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
