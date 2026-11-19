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
               "Missing provider usage is reported as unknown, never estimated; SDK retries with no returned usage are unobservable",
               "Local GraphRAG cache hits are recorded but excluded from new API usage; provider prompt-cache tokens remain included",
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


def _usage_report(metadata, records, previous, *, ledger=None):
    prices = previous.get("usd_per_million_tokens", PRICES)
    if (not isinstance(prices, dict) or set(prices) != set(PRICES)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in prices.values())):
        raise ValueError("Invalid recorded USD-per-million-token prices")
    models = metadata.get("assessment_plan_models", {}).get("models", {})
    configs = {name: settings.get("config", {}) for name, settings in models.items()}
    responses, patient_ids = {}, set()
    prepared = hosted = 0

    def capture(response, stage, context, occurrence, fallback=None):
        response = response if isinstance(response, dict) else {}
        reported_model = response.get("reported_model")
        reported_model = reported_model if isinstance(reported_model, str) and reported_model.strip() else None
        model = reported_model or fallback or "unknown"
        if not isinstance(model, str):
            model = "unknown"
        identifier = response.get("response_id")
        request_id = response.get("request_id")
        key = (("id", identifier) if isinstance(identifier, str) and identifier.strip() else
               ("request", request_id) if isinstance(request_id, str) and request_id.strip() else
               ("capture", *occurrence))
        raw_usage = response.get("token_usage")
        raw_usage = ({key: raw_usage.get(key) for key in
                      ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens") if key in raw_usage}
                     if isinstance(raw_usage, dict) else None)
        context = {**context, "cache_hit": response.get("cache_hit") is True,
                   "capture_index": occurrence[-1]}
        item = {"group": _group(model, stage), "model": model, "reported_model": reported_model,
                "model_source": "provider" if reported_model else "configured" if fallback else "unknown",
                "counts": _counts(response, stage == "guideline_query_embedding"),
                "response_id": identifier, "request_id": request_id,
                "token_usage": raw_usage, "cache_hit": response.get("cache_hit") is True,
                "contexts": [context]}
        old = responses.get(key)
        if old is None:
            responses[key] = item
            return
        if old["counts"] is not None and item["counts"] is not None:
            if old["counts"] != item["counts"] or (old["group"] is not None and item["group"] is not None
                    and (old["group"] == "embedding") != (item["group"] == "embedding")):
                raise ValueError("Conflicting saved usage for one response ID")
        if old["counts"] is None:
            old["counts"] = item["counts"]
        if old["group"] is None and item["group"] is not None:
            old.update(group=item["group"], model=model)
        if old["reported_model"] is None and reported_model is not None:
            old.update(reported_model=reported_model, model=model, model_source="provider")
        if raw_usage is not None:
            if old["token_usage"] is None:
                old["token_usage"] = {}
            for field, count in raw_usage.items():
                saved = old["token_usage"].get(field)
                if saved is not None and count is not None and saved != count:
                    raise ValueError("Conflicting saved usage for one response ID")
                if count is not None:
                    old["token_usage"][field] = count
        old["cache_hit"] = old["cache_hit"] and item["cache_hit"]
        if context not in old["contexts"]:
            old["contexts"].append(context)

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
        for component_id, component in components.items():
            consumers = [name for name, variant in variants.items()
                         if component_id in (variant.get("components") or {}).values()]
            saved_capture = component.get("capture", {})
            stages = dict(saved_capture.get("model_responses", {}))
            # Older agent runs retained only these actual counts. Include them
            # once when canonical per-response agent captures are absent.
            if not stages.get("agent_control") and isinstance(saved_capture.get("agent"), dict):
                stages["agent_control"] = [{"token_usage": usage}
                                           for usage in saved_capture["agent"].get("usage", [])]
            for stage, entries in stages.items():
                fallback = ((component.get("settings") or {}).get("agent_model") if stage == "agent_control"
                            else fallbacks.get(stage))
                context = {"patient_id": identifier, "component": component_id,
                           "stage": stage, "variants": consumers}
                for index, response in enumerate(entries):
                    capture(response, stage, context, (identifier, component_id, stage, index), fallback)
        for variant_id, variant in variants.items():
            for alias, output in variant.get("assessment_plans", {}).items():
                config = configs.get(alias, {})
                if _is_local(config):
                    continue
                hosted += output.get("status") == "completed"
                context = {"patient_id": identifier, "stage": "final", "variants": [variant_id],
                           "model_alias": alias}
                for index, response in enumerate(output.get("model_responses", [])):
                    capture(response, "final", context, (identifier, variant_id, alias, index), config.get("model"))

    usage = {name: {"responses": 0, "input_tokens": 0, **({} if name == "embedding" else {"output_tokens": 0}),
                    "estimated_usd": 0.0} for name in ("preparation_gpt", "final_gpt", "embedding")}
    measured = {"responses": 0, "local_cache_hits": 0, "by_model": {}, "by_stage": {}, "by_variant": {}}

    def add_measured(groups, name, raw_usage):
        entry = groups.setdefault(name, {"responses": 0, "reported_tokens": dict.fromkeys(
            ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"), 0),
            "missing_usage_fields": dict.fromkeys(
            ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"), 0)})
        entry["responses"] += 1
        for field in entry["reported_tokens"]:
            count = (raw_usage or {}).get(field)
            if type(count) is int and count >= 0:
                entry["reported_tokens"][field] += count
            else:
                entry["missing_usage_fields"][field] += 1

    missing, unpriced = 0, Counter()
    for key, response in responses.items():
        group, model, counts = (response[name] for name in ("group", "model", "counts"))
        if ledger is not None:
            ledger.append({"usage_key": list(key), **{field: deepcopy(value) for field, value in response.items()
                           if field not in ("counts", "group")}})
        measured["local_cache_hits"] += sum(context["cache_hit"] for context in response["contexts"])
        if response["cache_hit"]:
            continue
        measured["responses"] += 1
        raw_usage = response["token_usage"]
        add_measured(measured["by_model"], model, raw_usage)
        executed_contexts = [context for context in response["contexts"] if not context["cache_hit"]]
        for stage in {context["stage"] for context in executed_contexts}:
            add_measured(measured["by_stage"], stage, raw_usage)
        for variant in {variant for context in executed_contexts for variant in context["variants"]}:
            add_measured(measured["by_variant"], variant, raw_usage)
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
            "measured_usage": measured, "api_usage_file": "provenance/api-usage.jsonl",
            "price_sources": previous.get("price_sources", list(PRICE_SOURCES)), "limitations": limitations,
            "deduplication": "Canonical preparation captures and final hosted model response captures only; "
                             "nested variants/results ignored; unique response IDs counted once where available; "
                             "provider request IDs used when response IDs are absent; "
                             "ID-less captures counted per stable canonical occurrence; "
                             "per-variant views can overlap for shared responses and are not additive"}


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
    """Write measured per-response usage, a cost summary, and recovery JSON.

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
    ledger = []
    usage = _usage_report(metadata, _records(run_dir / "results.jsonl") if records is None else records,
                          _read(usage_file, {}), ledger=ledger)
    recovery = _recovery_report(metadata, _read(recovery_file, {}))
    reports = []
    for path, data in ((usage_file, usage), (recovery_file, recovery)):
        text = redact(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        json.loads(text)  # Validate both complete reports before replacing either.
        reports.append((path, text))
    directory.mkdir(exist_ok=True)
    _fsync_directory(run_dir)
    # The per-response artifact contains only provider metadata and attribution,
    # including failed attempts; it never includes prompts or resolved credentials.
    ledger_text = redact("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in ledger))
    _write(directory / "api-usage.jsonl", ledger_text)
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
