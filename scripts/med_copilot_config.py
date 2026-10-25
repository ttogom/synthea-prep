"""Run-wide MED-COPILOT settings, separate from model definitions and credentials."""

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re

import yaml

from scripts.med_copilot_helpers import repo_path


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(keys.count(key) > 1 for key in keys):
            raise ValueError("Duplicate YAML configuration key")
        return super().construct_mapping(node, deep=deep)


def read_yaml_config(path):
    """Read safe YAML and fingerprint the exact same source bytes."""
    source = Path(path).read_bytes()
    try:
        data = yaml.load(source.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, UnicodeDecodeError):
        raise ValueError(f"Invalid configuration YAML: {path}") from None
    return data, hashlib.sha256(source).hexdigest()


class ExperimentConfig:
    """Read run settings; model definitions always come from the fixed sibling file."""

    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        data, self.config_sha256 = read_yaml_config(self.config_path)
        required = {"selected_models", "temperature", "max_retries", "max_patients"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Experiment config requires only selected_models, temperature, max_retries, max_patients "
                             "in experiments/configs/experiment.yaml")
        self.model_config = self.config_path.with_name("condition_plan_models.yaml")
        selection = data["selected_models"]
        if (not isinstance(selection, list) or not selection
                or any(not isinstance(key, str) or not key.strip() for key in selection)
                or len(set(selection)) != len(selection)):
            raise ValueError("selected_models must be a nonempty list of unique model aliases "
                             "in experiments/configs/experiment.yaml")
        temperature = data["temperature"]
        if (type(temperature) not in (int, float) or not 0 <= temperature <= 2
                or not math.isfinite(temperature)):
            raise ValueError("temperature must be a finite number between 0 and 2")
        retries = data["max_retries"]
        if type(retries) is not int or retries < 1:
            raise ValueError("max_retries must be an integer of at least 1")
        max_patients = data["max_patients"]
        if max_patients is not None and (type(max_patients) is not int or max_patients < 1):
            raise ValueError("max_patients must be null or a positive integer "
                             "in experiments/configs/experiment.yaml")
        self.selected_models = deepcopy(selection)
        self.temperature = temperature
        self.max_retries = retries
        self.max_patients = max_patients

    def metadata(self, root):
        """Record the source fingerprint, fixed registry, and effective run settings."""
        return {
            "config_file": repo_path(self.config_path, root),
            "config_sha256": self.config_sha256,
            "model_config": repo_path(self.model_config, root),
            "selected_models": list(self.selected_models),
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "max_patients": self.max_patients,
        }


def configured_model_credentials(registry, selected_models, *, require_shared_credentials=False):
    """Check alias/key configuration together without authenticating or making requests.

    Return private (variable-name, value) pairs. Only variable names may be shown
    in confirmation or recorded metadata. Collect all selection/credential errors
    before constructing any model clients.
    """
    issues = []
    credentials = {}
    missing_keys = set()
    for model_id in selected_models:
        if model_id not in registry:
            issues.append(
                f"Unknown model alias: {model_id}\n"
                "    Correct selected_models in experiments/configs/experiment.yaml.\n"
                "    Available aliases in experiments/configs/condition_plan_models.yaml:\n"
                f"    {', '.join(registry)}")
            continue
        entry = registry[model_id]
        key_env = entry.get("api_key_env")
        if key_env is not None:
            if not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
                issues.append(f"{model_id}: api_key_env must name an environment variable in\n"
                              "    experiments/configs/condition_plan_models.yaml.\n"
                              "    Add the matching API key variable to .env in the repository root.")
                continue
            key = os.environ.get(key_env)
            if not key or not key.strip():
                issues.append(f"API key for {model_id} is missing or empty: {key_env}\n"
                              f"    Add {key_env} to .env in the repository root.\n"
                              "    Its name must match api_key_env in\n"
                              "    experiments/configs/condition_plan_models.yaml.")
                missing_keys.add(key_env)
                continue
        elif entry.get("auth_type") == "azure_managed_identity":
            key = None
        elif entry.get("api_base"):
            # Upstream requires a nonempty placeholder for unauthenticated servers.
            key = "local-no-key"
        else:
            issues.append(f"{model_id}: api_key_env is not configured in\n"
                          "    experiments/configs/condition_plan_models.yaml.\n"
                          "    Set api_key_env to the key's variable name and add that variable\n"
                          "    to .env in the repository root.")
            continue
        credentials[model_id] = (key_env, key)
    if require_shared_credentials:
        for key_env in ("OPENAI_API_KEY", "GRAPHRAG_API_KEY"):
            value = os.environ.get(key_env)
            if (not value or not value.strip()) and key_env not in missing_keys:
                issues.append(f"Shared-stage API key is missing or empty: {key_env}\n"
                              f"    Add {key_env} to .env in the repository root.")
    if issues:
        raise ValueError("\n\n".join(f"  - {issue}" for issue in issues))
    return credentials


def confirm_experiment(experiment, model_metadata, root):
    """Show model identities/key variable names and require an affirmative answer."""
    print("\n========== EXPERIMENT CONFIGURATION ==========\n")
    print(f"Experiment: {repo_path(experiment.config_path, root)}")
    print(f"Model definitions: {repo_path(experiment.model_config, root)}")
    print("\nSelected models:")
    for number, model_id in enumerate(experiment.selected_models, 1):
        settings = model_metadata["models"][model_id]
        config = settings["config"]
        name = config.get("deployment_name") or config["model"]
        print(f"  {number}. {model_id} ({config['model_provider']}/{name})")
        if settings["api_key_env"]:
            print(f"     API key configured: {settings['api_key_env']}")
        elif config.get("auth_type") == "azure_managed_identity":
            print("     Authentication: Azure managed identity")
        else:
            print("     API key not required (unauthenticated endpoint)")
    print("\nTo change the selection, edit selected_models in")
    print("experiments/configs/experiment.yaml and restart the script.")
    print("\nThese are the models selected for this experiment run.")
    try:
        answer = input("Is this correct? Start the run? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("Run cancelled. No experiment output was created.")
        return False
    print("Confirmed. Starting experiment run.")
    return True
