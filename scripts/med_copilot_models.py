"""Assessment/Plan experiments using the vendored GraphRAG model implementation."""

from copy import deepcopy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.factory import ModelFactory
from med_copilot.prompts.diagnosis_plan import CONDITION_PLAN_RESPONSE_FORMAT
from scripts.med_copilot_config import configured_model_credentials, read_yaml_config
from scripts.med_copilot_helpers import model_response_metadata, parse_condition_plan, repo_path


class AssessmentPlanModels:
    """Load selected models once; reuse upstream configuration and wrapped completions.

    Only selected credentials are resolved. The full GraphRAG settings loader is
    intentionally unnecessary: final generation has no embedding/index settings.
    """

    def __init__(self, config_path, selected_models, *, temperature, max_retries,
                 require_shared_credentials=False, required_shared_keys=None):
        self.config_path = Path(config_path).resolve()
        data, self.config_sha256 = read_yaml_config(self.config_path)
        if not isinstance(data, dict) or set(data) != {"models"}:
            raise ValueError("Model registry requires only models; run settings belong in experiment.yaml")
        registry = data["models"]
        if not isinstance(registry, dict) or not registry or any(
                not isinstance(key, str) or not key.strip() or not isinstance(value, dict)
                for key, value in registry.items()):
            raise ValueError("models must be a nonempty mapping of model IDs to settings")
        selection = selected_models
        if (not isinstance(selection, list) or not selection
                or any(not isinstance(key, str) or not key.strip() for key in selection)
                or len(set(selection)) != len(selection)):
            raise ValueError("Select a nonempty list of unique model IDs")
        credentials = configured_model_credentials(
            registry, selection, require_shared_credentials=require_shared_credentials,
            required_shared_keys=required_shared_keys,
        )
        self.model_ids = tuple(selection)
        self._models = {}
        self._settings = {}
        self._secrets = []
        configurations = {}
        # Validate every selected entry before constructing any model clients.
        for model_id in self.model_ids:
            entry = dict(registry[model_id])
            if {"temperature", "max_retries"} & set(entry):
                raise ValueError(f"{model_id}: temperature and max_retries belong in experiment.yaml")
            entry.update(temperature=temperature, max_retries=max_retries)
            if "api_key" in entry:
                raise ValueError(f"{model_id}: use api_key_env instead of a literal api_key")
            key_env, key = credentials[model_id]
            entry.pop("api_key_env", None)
            output_mode = entry.pop("output_mode", "json_schema")
            if output_mode not in ("json_schema", "json_object", "prompt"):
                raise ValueError(f"{model_id}: invalid output_mode")
            if set(entry) - set(LanguageModelConfig.model_fields):
                raise ValueError(f"{model_id}: unknown model configuration field")
            if (entry.get("type") != "chat" or type(entry.get("n", 1)) is not int
                    or entry.get("n", 1) != 1):
                raise ValueError(f"{model_id}: require type: chat and n: 1")
            endpoint = entry.get("api_base")
            if endpoint is not None:
                if not isinstance(endpoint, str):
                    raise ValueError(f"{model_id}: api_base must be an HTTP(S) URL")
                url = urlsplit(endpoint)
                if (url.scheme not in ("http", "https") or not url.hostname
                        or url.username or url.password or url.query or url.fragment):
                    raise ValueError(f"{model_id}: api_base must be an HTTP(S) URL without credentials/query")
            if key_env is not None:
                self._secrets.append(key)
            try:
                config = LanguageModelConfig(**entry, api_key=key)
                effective = config.model_dump(mode="json", exclude={"api_key"})
                json.dumps(effective, allow_nan=False)
            except Exception:
                # Pydantic errors may include the resolved API key in input_value.
                raise ValueError(f"{model_id}: invalid upstream model configuration") from None
            configurations[model_id] = config
            self._settings[model_id] = {
                "api_key_env": key_env, "output_mode": output_mode,
                "config": effective,
            }
        for model_id, config in configurations.items():
            try:
                self._models[model_id] = ModelFactory.create_chat_model(
                    "chat", name=model_id, config=config,
                )
            except Exception:
                raise ValueError(f"{model_id}: upstream model initialization failed") from None

    def metadata(self, root):
        """Snapshot effective upstream settings, preserving no resolved credentials."""
        return {
            "config_file": repo_path(self.config_path, root),
            "config_sha256": self.config_sha256,
            "selected_models": list(self.model_ids),
            "models": deepcopy(self._settings),
        }

    def generate(self, messages, result):
        """Fill preallocated model results independently, retaining partial responses.

        Use the upstream configured completion callable instead of chat(), which
        parses JSON before returning and omits the provider response/usage.
        No provider fallback, output repair, or automatic JSON-mode downgrade.
        """
        for model_id, model in self._models.items():
            output = result["assessment_plans"][model_id]
            output["status"] = "failed_incomplete"
            try:
                mode = self._settings[model_id]["output_mode"]
                options = {}
                if mode != "prompt":
                    options["response_format"] = (deepcopy(CONDITION_PLAN_RESPONSE_FORMAT)
                                                  if mode == "json_schema" else {"type": "json_object"})
                response = model.completion(
                    messages=deepcopy(messages), stream=False, drop_params=False, **options,
                )
                output["model_responses"].append(model_response_metadata(response))
                choice = response.choices[0]
                content = choice.message.content
                output["assessment_plan_raw"] = content
                if getattr(choice.message, "refusal", None):
                    raise ValueError("Assessment/Plan generation was refused")
                finish = getattr(choice, "finish_reason", None)
                if finish not in (None, "stop"):
                    raise ValueError("Assessment/Plan generation did not complete")
                output.update(parse_condition_plan(content))
                output["status"] = "completed"
            except Exception as exc:
                output["error"] = self.error_metadata(exc)
            except BaseException as exc:
                output["error"] = {"type": type(exc).__name__, "message": "Generation interrupted"}
                raise
            # BaseException propagates to ExperimentRun.stop, preserving this slot
            # as failed and later models as not_processed.
        result["status"] = ("completed" if all(
            output["status"] == "completed" for output in result["assessment_plans"].values()
        ) else "failed_incomplete")

    def error_metadata(self, exc):
        """Describe source/model failures without exposing selected or shared keys."""
        message = str(exc)
        secrets = self._secrets + [os.environ.get(name) for name in ("OPENAI_API_KEY", "GRAPHRAG_API_KEY")]
        for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
            message = message.replace(secret, "[REDACTED]")
        return {"type": type(exc).__name__, "message": message}
