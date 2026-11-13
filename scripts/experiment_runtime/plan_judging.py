"""Typed runtime profile for the independent treatment-Plan judge."""

from copy import deepcopy
from pathlib import Path

from scripts.med_copilot_config import read_yaml_config


PROFILE_FIELDS = {"concurrency", "context_length", "flash_attention", "kv_cache_type", "host_cache_mib"}


def validate_profile(settings):
    """Reject unknown, missing, or unsupported settings before server startup."""
    if not isinstance(settings, dict) or set(settings) != PROFILE_FIELDS:
        raise ValueError("Plan judge profile must contain exactly: " + ", ".join(sorted(PROFILE_FIELDS)))
    for field, low, high in (("concurrency", 1, 64), ("context_length", 512, 262144),
                             ("host_cache_mib", 0, 8192)):
        value = settings[field]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Plan judge {field} must be an integer from {low} to {high}")
    if type(settings["flash_attention"]) is not bool:
        raise ValueError("Plan judge flash_attention must be boolean")
    if settings["kv_cache_type"] != "f16":
        raise ValueError("Plan judge kv_cache_type must be f16 (the quality-tested profile)")
    return deepcopy(settings)


def load_judge_profile(path, metadata, *, concurrency=None, context_length=None):
    """Freeze a new stage's file settings; resume its saved profile without rereading it."""
    if not isinstance(metadata, dict):
        raise ValueError("Recorded run metadata must be an object")
    stage = metadata.get("plan_evaluation", {})
    stage = {} if stage is None else stage
    if not isinstance(stage, dict):
        raise ValueError("plan_evaluation must be an object")
    saved = stage.get("runtime_profile")
    if saved is None:
        settings, fingerprint = read_yaml_config(path)
        profile = {"settings": validate_profile(settings), "config_file": str(Path(path).resolve()),
                   "config_sha256": fingerprint}
        # Older Plan invocations already allowed explicit C/context overrides.
        for field in ("concurrency", "context_length"):
            if stage.get(field) is not None:
                profile["settings"][field] = stage[field]
    else:
        if (not isinstance(saved, dict) or set(saved) != {"settings", "config_file", "config_sha256"}
                or not isinstance(saved["config_file"], str)
                or not isinstance(saved["config_sha256"], str) or len(saved["config_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in saved["config_sha256"])):
            raise ValueError("Invalid recorded Plan judge runtime_profile")
        profile = deepcopy(saved)
        validate_profile(profile["settings"])
    for field, override in (("concurrency", concurrency), ("context_length", context_length)):
        if override is not None:
            profile["settings"][field] = override
    profile["settings"] = validate_profile(profile["settings"])
    return profile
