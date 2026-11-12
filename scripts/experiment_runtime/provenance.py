"""Finalize compact usage and recovery reports after runner model work stops.

Only canonical component/final-model captures are inspected. Pricing is a recorded
assumption, not a billing lookup; no model, API, or judge is invoked here.
"""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from ipaddress import ip_address
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit

from scripts.med_copilot_run_file import _fsync_directory, pacific_now, redact_shared_credentials


PRICES = {"gpt_4o_mini_input": 0.15, "gpt_4o_mini_output": 0.6,
          "text_embedding_3_small_input": 0.02}
PRICE_SOURCES = ["https://developers.openai.com/api/docs/models/gpt-4o-mini",
                 "https://developers.openai.com/api/docs/models/text-embedding-3-small"]
LIMITATIONS = ["Reported-token pricing estimate, not provider invoice; uncached rates",
               "Responses never durably saved cannot be included",
               "Local model usage is excluded from API-credit costs; electricity/hardware costs are not estimated"]


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _records(path):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _is_local(config):
    hostname = urlsplit(config.get("api_base") or "").hostname
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _group(model, stage):
    if re.fullmatch(r"gpt-4o-mini(?:-\d{4}-\d{2}-\d{2})?", model):
        return "final_gpt" if stage == "final" else "preparation_gpt"
    if model == "text-embedding-3-small" and stage == "guideline_query_embedding":
        return "embedding"
    return None


def _counts(response, embedding):
    usage = response.get("token_usage")
    if not isinstance(usage, dict):
        return None
    values = (usage.get("prompt_tokens"), 0 if embedding else usage.get("completion_tokens"))
    return values if all(type(value) is int and value >= 0 for value in values) else None


def _usage_report(metadata, records, previous):
    prices = previous.get("usd_per_million_tokens", PRICES)
    if (not isinstance(prices, dict) or set(prices) != set(PRICES)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in prices.values())):
        raise ValueError("Invalid recorded USD-per-million-token prices")
    models = metadata.get("assessment_plan_models", {}).get("models", {})
    configs = {name: settings.get("config", {}) for name, settings in models.items()}
    responses, patient_ids = {}, set()
    prepared = hosted = 0

    def capture(response, stage, fallback=None):
        response = response if isinstance(response, dict) else {}
        model = response.get("reported_model") or fallback or "unknown"
        if not isinstance(model, str):
            model = "unknown"
        identifier = response.get("response_id")
        key = ("id", identifier) if isinstance(identifier, str) and identifier.strip() else ("capture", len(responses))
        item = (_group(model, stage), model, _counts(response, stage == "guideline_query_embedding"))
        old = responses.get(key)
        if old is None or old[2] is None:
            responses[key] = item
        elif item[2] is not None:
            if old[2] != item[2] or (old[0] is not None and item[0] is not None
                                    and (old[0] == "embedding") != (item[0] == "embedding")):
                raise ValueError("Conflicting saved usage for one response ID")
            if old[0] is None and item[0] is not None:
                responses[key] = item  # A later capture can identify an unknown model.

    experiment = metadata.get("experiment", {})
    fallbacks = {"keyword_extraction": experiment.get("patient_keyword_model"),
                 "guideline_questions": experiment.get("guideline_question_model")}
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in patient_ids:
            raise ValueError("Provenance requires unique, nonblank patient IDs")
        patient_ids.add(identifier)
        components, variants = record.get("components", {}), record.get("variants", {})
        prepared += bool(variants) and all(c.get("status") == "completed" for c in components.values()) and all(
            v.get("assessment_plan_prompts") is not None for v in variants.values())
        for component in components.values():
            for stage, entries in component.get("capture", {}).get("model_responses", {}).items():
                for response in entries:
                    capture(response, stage, fallbacks.get(stage))
        for variant in variants.values():
            for alias, output in variant.get("assessment_plans", {}).items():
                config = configs.get(alias, {})
                if _is_local(config):
                    continue
                hosted += output.get("status") == "completed"
                for response in output.get("model_responses", []):
                    capture(response, "final", config.get("model"))

    usage = {name: {"responses": 0, "input_tokens": 0, **({} if name == "embedding" else {"output_tokens": 0}),
                    "estimated_usd": 0.0} for name in ("preparation_gpt", "final_gpt", "embedding")}
    missing, unpriced = 0, Counter()
    for group, model, counts in responses.values():
        if counts is None:
            missing += 1
        elif group is None:
            unpriced[model] += 1
        else:
            entry = usage[group]
            entry["responses"] += 1
            entry["input_tokens"] += counts[0]
            if group != "embedding":
                entry["output_tokens"] += counts[1]
    for name, entry in usage.items():
        cost = (entry["input_tokens"] * prices["text_embedding_3_small_input"] if name == "embedding" else
                entry["input_tokens"] * prices["gpt_4o_mini_input"] + entry["output_tokens"] * prices["gpt_4o_mini_output"])
        entry["estimated_usd"] = round(cost / 1_000_000, 12)
    limitations = list(LIMITATIONS)
    if unpriced:
        limitations.append("Responses excluded because their model has no recorded price: " +
                           ", ".join(f"{model} ({count})" for model, count in sorted(unpriced.items())))
    return {"observed_at_pst": pacific_now().isoformat(), "prepared_patient_count": prepared,
            "completed_hosted_output_count": hosted, "usage": usage, "missing_or_malformed_usage": missing,
            "captured_usage_estimated_usd": round(sum(entry["estimated_usd"] for entry in usage.values()), 12),
            "usd_per_million_tokens": deepcopy(prices),
            "price_sources": previous.get("price_sources", list(PRICE_SOURCES)), "limitations": limitations,
            "deduplication": "Canonical preparation captures and final hosted model response captures only; "
                             "nested variants/results ignored; unique response IDs counted once where available; "
                             "ID-less captures counted per canonical occurrence"}


def _recovery_report(metadata, previous):
    events = deepcopy(previous.get("events", []))
    restarts = previous.get("run_restarts", 0)
    resumptions = metadata.get("resumptions", [])
    if (type(restarts) is not int or not 0 <= restarts <= len(resumptions)
            or not isinstance(events, list) or any(not isinstance(event, dict) for event in events)):
        raise ValueError("Invalid existing recovery history")
    events.extend({"event": "experiment_resumed", **deepcopy(resumption)} for resumption in resumptions[restarts:])
    for event in metadata.get("preparation_recovery", {}).get("events", []):
        entry = {"event": "preparation_worker_recovery", **deepcopy(event)}
        if entry not in events:
            events.append(entry)
    return {"run_restarts": len(resumptions), "events": events}


def _write(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".provenance-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_provenance(run_dir, *, metadata=None, records=None, redact=redact_shared_credentials):
    """Write only final usage/cost and recovery JSON under run_dir/provenance.

    Call after all invocation model work/resources have stopped, while holding
    the run lock. Resume recalculates cumulative usage, preserving recorded
    prices and detailed existing recovery events. Source artifacts are read-only.
    """
    run_dir = Path(run_dir).resolve()
    metadata = _read(run_dir / "run.json", {}) if metadata is None else metadata
    if metadata.get("status") not in ("completed", "completed_with_errors", "failed_incomplete"):
        raise ValueError("Finalize provenance only after the runner has stopped")
    directory = run_dir / "provenance"
    usage_file, recovery_file = directory / "reported-usage-cost.json", directory / "recovery-incidents.json"
    usage = _usage_report(metadata, _records(run_dir / "results.jsonl") if records is None else records,
                          _read(usage_file, {}))
    recovery = _recovery_report(metadata, _read(recovery_file, {}))
    reports = []
    for path, data in ((usage_file, usage), (recovery_file, recovery)):
        text = redact(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        json.loads(text)  # Validate both complete reports before replacing either.
        reports.append((path, text))
    directory.mkdir(exist_ok=True)
    _fsync_directory(run_dir)
    for path, text in reports:
        _write(path, text)
    return {"usage": usage_file, "recovery": recovery_file}


@contextmanager
def finalize_provenance(run_file):
    """Finalize on runner exit without masking an existing pipeline failure."""
    pipeline_failed = True
    try:
        yield
        pipeline_failed = False
    finally:
        try:
            write_provenance(run_file.run_dir, metadata=run_file.metadata, redact=run_file.redact)
        except Exception as exc:
            if not pipeline_failed:
                raise
            print(run_file.redact(f"Could not finalize provenance: {type(exc).__name__}: {exc}"), file=sys.stderr)
