"""Run-wide MED-COPILOT settings, separate from model definitions and credentials."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from string import Formatter

import yaml

from med_copilot.prompts import diagnosis_plan
from scripts.med_copilot_helpers import repo_path
from scripts import med_copilot_variants
from scripts.med_copilot_variants import COMPONENTS, VARIANTS, ComponentSpec, VariantSpec, Parameter


# Defaults preserve older experiment files; the shipped YAML spells out each field.
PIPELINE_DEFAULTS = {
    "guidelines": "med_copilot/data/guidelines",
    "experiment_output": "experiments/output",
    "patient_corpus": "med_copilot/upstream/soap_with_metadata.json",
    "patient_retrieval_artifacts_dir": "med_copilot/data/patient_retrieval_so",
    "guideline_question_model": "gpt-4o-mini",
    "patient_embedding_model": "emilyalsentzer/Bio_ClinicalBERT",
    "patient_keyword_model": "gpt-4o-mini",
    "patient_reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "patient_retrieval_alpha": 0.5,
    "patient_retrieval_top_k": 20,
    "patient_rerank_top_k": 5,
    "guideline_community_level": 2,
    "guideline_response_type": "Multiple Paragraphs",
    "guideline_verbose": True,
}
PATH_SETTINGS = ("patient_input", "patient_labels", "guidelines", "experiment_output",
                 "patient_corpus", "patient_retrieval_artifacts_dir")


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


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
        defaults = {
            "selected_variants": ["patient_retrieval_guidelines"],
            "patient_input": "experiments/input/test_sample_so.jsonl",
            "patient_labels": "experiments/input/test_sample_labels.jsonl",
            "presenting_evidence_only": True,
            "exclude_none_recorded_plans": False,
            "variant_settings": {},
            **PIPELINE_DEFAULTS,
        }
        if (not isinstance(data, dict) or not required <= set(data)
                or set(data) - required - set(defaults)):
            raise ValueError("Experiment config requires selected_models, temperature, max_retries, max_patients; "
                             "optional settings are " + ", ".join(defaults) + ". "
                             "Remove unknown settings from experiments/configs/experiment.yaml")
        data = {**defaults, **data}
        variants = data["selected_variants"]
        if (not isinstance(variants, list) or not variants
                or any(not isinstance(v, str) or not v.strip() or v == "not_recorded" or v not in VARIANTS for v in variants)
                or len(set(variants)) != len(variants)):
            raise ValueError("selected_variants must be a nonempty list of unique supported variants")
        for field in PATH_SETTINGS:
            if not isinstance(data[field], str) or not data[field].strip():
                raise ValueError(f"{field} must be a nonempty path string")
        for field in ("presenting_evidence_only", "exclude_none_recorded_plans", "guideline_verbose"):
            if type(data[field]) is not bool:
                raise ValueError(f"{field} must be a boolean")
        for field in ("guideline_question_model", "patient_embedding_model", "patient_keyword_model",
                      "patient_reranker_model", "guideline_response_type"):
            if not isinstance(data[field], str) or not data[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        for field in ("patient_retrieval_top_k", "patient_rerank_top_k"):
            if type(data[field]) is not int or data[field] < 1:
                raise ValueError(f"{field} must be a positive integer")
        level = data["guideline_community_level"]
        if type(level) is not int or level < 0:
            raise ValueError("guideline_community_level must be a nonnegative integer")
        alpha = data["patient_retrieval_alpha"]
        if type(alpha) not in (int, float) or not 0 <= alpha <= 1 or not math.isfinite(alpha):
            raise ValueError("patient_retrieval_alpha must be a finite number between 0 and 1")
        for field in PIPELINE_DEFAULTS:
            setattr(self, field, data[field])
        self.selected_variants = list(variants)
        self.presenting_evidence_only = data["presenting_evidence_only"]
        self.exclude_none_recorded_plans = data["exclude_none_recorded_plans"]
        root = Path(__file__).resolve().parents[1]
        for field in PATH_SETTINGS:
            setattr(self, field, (root / data[field]).resolve())
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
        self._resolve_components(data["variant_settings"])

    def _resolve_components(self, configured):
        if not isinstance(configured, dict) or any(not isinstance(key, str) or not key.strip()
                or key == "not_recorded" or key not in VARIANTS for key in configured):
            raise ValueError("variant_settings must map registered variant IDs to component settings")
        self.component_plan, self.variant_components = {}, {}
        self.variant_settings = {}
        # Validate explicitly configured inactive variants too, without loading resources.
        for variant_id in dict.fromkeys([*self.selected_variants, *configured]):
            variant = VARIANTS[variant_id]
            if (not isinstance(variant, VariantSpec)
                    or not isinstance(variant.components, tuple)
                    or any(not isinstance(name, str) or not name.strip() for name in variant.components)
                    or not isinstance(variant.prompt_key, str)
                    or not isinstance(variant.bindings, dict)
                    or not isinstance(variant.description, str) or not variant.description.strip()
                    or len(set(variant.components)) != len(variant.components)
                    or any(name not in COMPONENTS for name in variant.components)
                    or variant.prompt_key not in diagnosis_plan.VARIANT_TEMPLATES):
                raise ValueError(f"Invalid variant definition: {variant_id}")
            fields = {name for _, name, _, _ in Formatter().parse(
                diagnosis_plan.VARIANT_TEMPLATES[variant.prompt_key]) if name is not None}
            if any(not name.isidentifier() for name in fields):
                raise ValueError(f"Unsupported prompt field in {variant_id}")
            if set(variant.bindings) != fields - {"conditions"} or "conditions" in variant.bindings:
                raise ValueError(f"Prompt bindings do not match template for {variant_id}")
            for name in variant.components:
                component = COMPONENTS[name]
                if (not isinstance(component, ComponentSpec)
                        or not isinstance(component.description, str) or not component.description.strip()
                        or not isinstance(component.outputs, tuple) or not component.outputs
                        or any(not isinstance(output, str) or not output for output in component.outputs)
                        or len(set(component.outputs)) != len(component.outputs)
                        or not isinstance(component.parameters, dict)
                        or any(not isinstance(key, str) or not key or not isinstance(value, Parameter)
                               for key, value in component.parameters.items())
                        or not isinstance(component.resources, tuple)
                        or any(not isinstance(resource, str) or not resource.strip() for resource in component.resources)
                        or not isinstance(component.credentials, tuple)
                        or any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                               for key in component.credentials)
                        or not isinstance(component.implementation, str) or not component.implementation.strip()):
                    raise ValueError(f"Invalid component definition: {name}")
                for key, parameter in component.parameters.items():
                    if (parameter.kind not in ("str", "int", "bool", "number")
                            or parameter.config_field is not None and (
                                not isinstance(parameter.config_field, str) or parameter.config_field not in vars(self))
                            or any(bound is not None and not _finite_number(bound)
                                   for bound in (parameter.minimum, parameter.maximum))
                            or parameter.minimum is not None and parameter.maximum is not None
                            and parameter.minimum > parameter.maximum):
                        raise ValueError(f"Invalid parameter definition: {name}.{key}")
            for binding in variant.bindings.values():
                if (not isinstance(binding, tuple) or len(binding) != 2
                        or not all(isinstance(part, str) for part in binding)
                        or binding[0] not in variant.components
                        or binding[1] not in COMPONENTS[binding[0]].outputs):
                    raise ValueError(f"Invalid output binding for {variant_id}")
            overrides = configured.get(variant_id, {})
            if not isinstance(overrides, dict) or set(overrides) - set(variant.components):
                raise ValueError(f"Invalid component settings for {variant_id}")
            resolved, references = {}, {}
            for name in variant.components:
                component = COMPONENTS[name]
                values = overrides.get(name, {})
                if not isinstance(values, dict) or set(values) - set(component.parameters):
                    raise ValueError(f"Unknown parameter for {variant_id}.{name}")
                defaults = {}
                for key, parameter in component.parameters.items():
                    value = (getattr(self, parameter.config_field) if parameter.config_field
                             else deepcopy(parameter.default))
                    # Shared path settings have already been resolved to Path objects.
                    defaults[key] = str(value) if isinstance(value, Path) else value
                settings = {key: parameter.validate(values.get(key, defaults[key]), f"{variant_id}.{name}.{key}")
                            for key, parameter in component.parameters.items()}
                resolved[name] = settings
                token = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
                instance = name if settings == defaults else f"{name}__{token}"
                references[name] = instance
                if variant_id in self.selected_variants:
                    if instance in self.component_plan and self.component_plan[instance] != {"component": name, "settings": settings}:
                        raise ValueError("Component instance identity collision")
                    self.component_plan.setdefault(instance, {"component": name, "settings": deepcopy(settings)})
            if resolved:
                self.variant_settings[variant_id] = resolved
            if variant_id in self.selected_variants:
                self.variant_components[variant_id] = references

    def needs(self, component):
        """Whether a selected variant consumes this preparation component."""
        return any(item["component"] == component for item in self.component_plan.values())

    def component_instances(self, component):
        """Return the distinct configured executions for one visible runner stage."""
        return [(key, deepcopy(item["settings"])) for key, item in self.component_plan.items()
                if item["component"] == component]

    @property
    def required_resources(self):
        return {resource for item in self.component_plan.values()
                for resource in COMPONENTS[item["component"]].resources}

    @property
    def required_credentials(self):
        return sorted({key for item in self.component_plan.values()
                       for key in COMPONENTS[item["component"]].credentials})

    @property
    def needs_patient_retrieval(self):
        return "patient_retrieval" in self.required_resources

    @property
    def needs_guidelines(self):
        return "guidelines" in self.required_resources

    def variant_metadata(self, root):
        """Identify the actual project templates and fingerprint their exact text."""
        source = Path(diagnosis_plan.__file__)
        return {
            "variant_definitions": {v: {
                "description": VARIANTS[v].description,
                "components": dict(self.variant_components[v]),
                "bindings": {key: list(value) for key, value in VARIANTS[v].bindings.items()},
                "patient_retrieval": any("patient_retrieval" in COMPONENTS[name].resources for name in VARIANTS[v].components),
                "guidelines": any("guidelines" in COMPONENTS[name].resources for name in VARIANTS[v].components),
                "template": f"VARIANT_TEMPLATES[{VARIANTS[v].prompt_key!r}]",
                "sha256": hashlib.sha256(diagnosis_plan.VARIANT_TEMPLATES[VARIANTS[v].prompt_key].encode()).hexdigest(),
            } for v in self.selected_variants},
            "component_definitions": {key: {
                **deepcopy(item), "description": COMPONENTS[item["component"]].description,
                "outputs": list(COMPONENTS[item["component"]].outputs),
                "resources": list(COMPONENTS[item["component"]].resources),
                "credentials": list(COMPONENTS[item["component"]].credentials),
                "implementation": COMPONENTS[item["component"]].implementation,
                "implementation_sha256": (hashlib.sha256((Path(root) / COMPONENTS[item["component"]].implementation).read_bytes()).hexdigest()
                                          if (Path(root) / COMPONENTS[item["component"]].implementation).is_file() else None),
            } for key, item in self.component_plan.items()},
            "registry_provenance": {"file": repo_path(med_copilot_variants.__file__, root),
                                    "sha256": hashlib.sha256(Path(med_copilot_variants.__file__).read_bytes()).hexdigest()},
            "prompt_provenance": {"file": repo_path(source, root),
                                  "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        }

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
            "selected_variants": list(self.selected_variants),
            "patient_input": repo_path(self.patient_input, root),
            "patient_labels": repo_path(self.patient_labels, root),
            "presenting_evidence_only": self.presenting_evidence_only,
            "exclude_none_recorded_plans": self.exclude_none_recorded_plans,
            "variant_settings": deepcopy(self.variant_settings),
            **{field: (repo_path(getattr(self, field), root) if field in PATH_SETTINGS
                       else getattr(self, field)) for field in PIPELINE_DEFAULTS},
        }


def configured_model_credentials(registry, selected_models, *, require_shared_credentials=False, required_shared_keys=None):
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
    shared_keys = (("OPENAI_API_KEY", "GRAPHRAG_API_KEY") if require_shared_credentials
                   else tuple(required_shared_keys or ()))
    for key_env in shared_keys:
        value = os.environ.get(key_env)
        if (not value or not value.strip()) and key_env not in missing_keys:
            issues.append(f"Shared-stage API key is missing or empty: {key_env}\n"
                          f"    Add {key_env} to .env in the repository root.")
    if issues:
        raise ValueError("\n\n".join(f"  - {issue}" for issue in issues))
    return credentials


def confirm_experiment(experiment, model_metadata, root, cohort=None):
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
    print("\nRun settings:")
    for key in ("temperature", "max_retries", "max_patients"):
        value = getattr(experiment, key)
        print(f"  {key}: {'null' if value is None else value}")
    print("\nEvidence variants:")
    for variant in experiment.selected_variants:
        print(f"  {variant} - {VARIANTS[variant].description}")
    print("\nDataset and cohort:")
    settings = experiment.metadata(root)
    for key in ("patient_input", "patient_labels", "presenting_evidence_only", "exclude_none_recorded_plans"):
        value = settings[key]
        print(f"  {key}: {str(value).lower() if type(value) is bool else value}")
    print('    - Excluded label plan value is exactly ["none recorded"] when the filter is enabled.')
    if cohort is not None:
        print()
        for label, key in (("Input patients:", "input_patient_count"),
                           ("After presenting-evidence filter:", "after_presenting_evidence_count"),
                           ("After recorded-plan filter:", "eligible_patient_count"),
                           ("Selected after patient limit:", "selected_patient_count")):
            print(f"  {label:<34}{cohort[key]:>3}")
        count = cohort["selected_patient_count"] * len(experiment.selected_variants) * len(experiment.selected_models)
        print(f"\nPlanned final requests: {count} (before retries; shared evidence calls are additional)")
    print("\nTo change settings, edit experiments/configs/experiment.yaml and restart the script.")
    try:
        answer = input("Is this configuration correct? Start the run? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("Run cancelled. No experiment output was created.")
        return False
    print("Confirmed. Starting experiment run.")
    return True
