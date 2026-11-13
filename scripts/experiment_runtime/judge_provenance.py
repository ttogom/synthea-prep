"""Verify actual native judge settings and preserve per-invocation evidence."""

import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from scripts.experiment_runtime.plan_judging import validate_profile
from scripts.med_copilot_run_file import _fsync_directory, pacific_now


def native_settings(log, profile):
    """Fail closed if native evidence is missing, ambiguous, or differs from the profile."""
    profile = validate_profile(profile)
    patterns = {
        "parallel_slots": r"llama_context: n_seq_max\s*=\s*(\d+)",
        "context_length": r"llama_context: n_ctx_seq\s*=\s*(\d+)",
        "total_context": r"llama_context: n_ctx\s*=\s*(\d+)",
        "batch": r"llama_context: n_batch\s*=\s*(\d+)",
        "microbatch": r"llama_context: n_ubatch\s*=\s*(\d+)",
        "flash_attention": r"llama_context: flash_attn\s*=\s*(\w+)",
        "kv_cache_type": r"llama_kv_cache:.*K \((\w+)\):.*V \((\w+)\):",
        "gpu_layers": r"load_tensors: offloaded (\d+)/(\d+) layers to GPU",
        "native_build": r"common_params_print_info: build [^\n]+",
    }
    effective = {}
    evidence = []
    for key, pattern in patterns.items():
        matches = list(re.finditer(pattern, log))
        values = {match.groups() or (match.group(),) for match in matches}
        if len(values) != 1:
            raise ValueError(f"Cannot verify native judge {key}: missing or conflicting evidence")
        value = next(iter(values))
        effective[key] = value if len(value) > 1 else value[0]
        evidence.extend(match.group() for match in matches)
    for key in ("parallel_slots", "context_length", "total_context", "batch", "microbatch"):
        effective[key] = int(effective[key])
    expected = {"parallel_slots": profile["concurrency"], "context_length": profile["context_length"],
                "total_context": profile["concurrency"] * profile["context_length"],
                "flash_attention": "enabled" if profile["flash_attention"] else "disabled",
                "kv_cache_type": (profile["kv_cache_type"], profile["kv_cache_type"])}
    for key, value in expected.items():
        if effective[key] != value:
            raise ValueError(f"Native judge {key}: requested {value}, observed {effective[key]}")
    # Ollama chooses this automatically from available memory. The profile does
    # not request a batch size; retain the native memory decision rather than
    # forcing the benchmark's observed 1024 on a busier machine.
    if effective["batch"] not in (512, 1024) or effective["microbatch"] != effective["batch"]:
        raise ValueError("Cannot verify supported automatic native judge batch/microbatch")
    layers, total = map(int, effective["gpu_layers"])
    if not layers or layers != total:
        raise ValueError(f"Native judge requires full GPU placement; observed {layers}/{total} layers")
    cache_pattern = (rf"prompt cache is enabled, size limit: {profile['host_cache_mib']} MiB"
                     if profile["host_cache_mib"] else r"prompt cache is disabled")
    cache_lines = re.findall(r"prompt cache is (?:enabled, size limit: \d+ MiB|disabled)", log)
    if not cache_lines or any(re.fullmatch(cache_pattern, line) is None for line in cache_lines):
        raise ValueError("Cannot verify native judge host prompt-cache limit")
    effective["host_cache_mib"] = profile["host_cache_mib"]
    evidence.extend(cache_lines)
    return {"settings": effective, "native_evidence": evidence}


def save_verified_runtime(server, models, judge, run_dir, root):
    """Verify before cache repair/judging; durably write a unique provenance file."""
    if not server.enabled:
        raise ValueError("The measured judge profile requires manage_ollama=true on a dedicated endpoint")
    model_config = models.metadata(root)["models"][judge]["config"]
    endpoint = model_config.get("api_base")
    if not isinstance(endpoint, str) or endpoint.rstrip("/") != server.base_url + "/v1":
        raise ValueError("Judge client endpoint differs from the owned server; refusing unverified judging")
    model = model_config["model"]
    server.preload(model)
    log = server.invocation_log()
    verified = native_settings(log, server.profile)
    placement = server.ps_snapshot().get("models", [])
    if (not isinstance(placement, list) or len(placement) != 1 or not isinstance(placement[0], dict)
            or placement[0].get("name") != model or not placement[0].get("digest")
            or type(placement[0].get("size_vram")) is not int or placement[0]["size_vram"] <= 0):
        raise ValueError("Cannot verify the loaded judge model identity/GPU placement")
    runtime = server.metadata()
    document = {"version": 1, "verified_at": pacific_now().isoformat(), "judge_alias": judge,
                "requested": server.profile, "effective": verified["settings"],
                "native_evidence": verified["native_evidence"],
                "model": {key: placement[0].get(key) for key in
                          ("name", "digest", "size", "size_vram", "context_length", "details")},
                "software": {"ollama_version": runtime["version"],
                             "native_build": verified["settings"]["native_build"]},
                "server": {key: runtime[key] for key in ("base_url", "max_loaded_models", "keep_alive")},
                "log_file": server.log_path.name, "log_start_byte": server._log_start,
                "verified_log_sha256": hashlib.sha256(log.encode()).hexdigest()}
    run_dir = Path(run_dir)
    destination = run_dir / "provenance" / f"plan-judge-runtime-{uuid4().hex}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with destination.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(destination.parent)
    _fsync_directory(run_dir)
    return {"file": str(destination.relative_to(run_dir)), "sha256": hashlib.sha256(data).hexdigest()}
