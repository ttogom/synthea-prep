"""Turn the notebook's top config cell into judge_plans_v2.py arguments.

Kept in the package (not the notebook) so offline tests check it against the
real argument parser. Credentials never pass through here: the notebook puts
HF_TOKEN from Colab Secrets into the child process environment only.
"""

import json


JUDGE_KEYS = {"RESULTS_PATH", "PATIENTS_PATH", "LABELS_PATH", "VARIANTS", "GENERATOR_MODELS", "OUTPUT_DIR",
              "RUN_NAME", "BACKEND", "JUDGE_MODEL", "JUDGE_REVISION", "DTYPE", "MAX_MODEL_LEN",
              "GPU_MEMORY_UTILIZATION", "CHAT_TEMPLATE_KWARGS", "ENGINE_ARGS", "BATCH_SIZE", "MAX_ATTEMPTS",
              "LIMIT", "WRITE_CSV", "OWNER", "TAKEOVER_STALE", "DRY_RUN"}


def _engine_args(engine_args):
    return [f"{key}={json.dumps(value)}" for key, value in (engine_args or {}).items()]


def judge_argv(config):
    unknown = set(config) - JUDGE_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    variants = config["VARIANTS"]
    if isinstance(variants, str):
        variants = [variants]
    argv = ["judge", "--results", str(config["RESULTS_PATH"]), "--patients", str(config["PATIENTS_PATH"]),
            "--labels", str(config["LABELS_PATH"]), "--variants", *variants,
            "--out-dir", str(config["OUTPUT_DIR"]), "--backend", config.get("BACKEND", "vllm"),
            "--dtype", config.get("DTYPE", "auto"), "--max-model-len", str(config.get("MAX_MODEL_LEN", 4096)),
            "--gpu-memory-utilization", str(config.get("GPU_MEMORY_UTILIZATION", 0.9)),
            "--chat-template-kwargs", json.dumps(config.get("CHAT_TEMPLATE_KWARGS") or {}),
            "--batch-size", str(config.get("BATCH_SIZE", 32)),
            "--max-attempts", str(config.get("MAX_ATTEMPTS", 3))]
    if config.get("JUDGE_MODEL"):
        argv += ["--model", config["JUDGE_MODEL"]]
    for key, flag in (("JUDGE_REVISION", "--revision"), ("RUN_NAME", "--run-name"), ("OWNER", "--owner")):
        if config.get(key):
            argv += [flag, str(config[key])]
    if config.get("GENERATOR_MODELS"):
        argv += ["--generator-models", *config["GENERATOR_MODELS"]]
    if config.get("LIMIT") is not None:
        argv += ["--limit", str(config["LIMIT"])]
    for pair in _engine_args(config.get("ENGINE_ARGS")):
        argv += ["--engine-arg", pair]
    for key, flag in (("WRITE_CSV", "--csv"), ("TAKEOVER_STALE", "--takeover-stale"), ("DRY_RUN", "--dry-run")):
        if config.get(key):
            argv.append(flag)
    return argv


def validate_argv(candidate, *, cases, out_dir, backend="vllm", max_model_len=4096, batch_size=16,
                  allow_draft=False):
    """candidate: {"model": id, optional "revision", "dtype", "chat_template_kwargs", "engine_args"}."""
    argv = ["validate", "--cases", str(cases), "--model", candidate["model"], "--out-dir", str(out_dir),
            "--backend", backend, "--dtype", candidate.get("dtype", "auto"),
            "--max-model-len", str(max_model_len), "--batch-size", str(batch_size),
            "--chat-template-kwargs", json.dumps(candidate.get("chat_template_kwargs") or {})]
    if candidate.get("revision"):
        argv += ["--revision", candidate["revision"]]
    for pair in _engine_args(candidate.get("engine_args")):
        argv += ["--engine-arg", pair]
    if allow_draft:
        argv.append("--allow-draft")
    return argv
