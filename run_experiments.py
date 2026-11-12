"""Run restartable, model-major MED-COPILOT experiments.

Settings live in experiments/configs; --config selects a file, --resume selects
an existing run, and --yes allows an explicitly authorized unattended run.
"""

import argparse
from contextlib import ExitStack
import hashlib
from ipaddress import ip_address
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "med_copilot/upstream"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

# Paths and credentials precede all project/provider imports. Retrieval/FAISS
# lives exclusively in separately bootstrapped CPU preparation interpreters.
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from scripts.collate_med_copilot_results import generate_report
from scripts.med_copilot_config import ExperimentConfig, confirm_experiment, read_yaml_config
from scripts.med_copilot_helpers import (
    git_branch, git_commit, guideline_resource_files, load_patient_records, repo_path, sha256_bundle,
)
from scripts.med_copilot_judges import CorrectnessJudges
from scripts.med_copilot_models import AssessmentPlanModels
from scripts.experiment_runtime.ollama import ManagedOllama
from scripts.experiment_runtime.provenance import finalize_provenance
from scripts.med_copilot_run_file import RunFile, make_run_name, pacific_now
from scripts.med_copilot_run_lock import exclusive_run_lock
from scripts.experiment_runtime.scheduler import ExperimentScheduler
from scripts.med_copilot_scoring import PatientScorer


def _fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources():
    paths = ("run_experiments.py", "scripts/experiment_runtime/scheduler.py",
             "scripts/experiment_runtime/execution.py", "scripts/experiment_runtime/preparation.py",
             "scripts/experiment_runtime/worker.py", "scripts/experiment_runtime/worker_stream.py",
             "scripts/experiment_runtime/ollama.py", "scripts/experiment_runtime/provenance.py")
    return [{"file": path, "sha256": _fingerprint(ROOT / path)} for path in paths]


def _validate_resume(metadata, experiment, models, scorer):
    if metadata.get("experiment") != experiment.metadata(ROOT):
        raise ValueError("Cannot resume: experiment configuration changed since the run")
    if metadata.get("input", {}).get("sha256") != _fingerprint(experiment.patient_input):
        raise ValueError("Cannot resume: patient input changed since the run")
    saved = metadata.get("assessment_plan_models", {})
    current = models.metadata(ROOT)
    if {k: v for k, v in saved.items() if k != "config_sha256"} != {
            k: v for k, v in current.items() if k != "config_sha256"}:
        raise ValueError("Cannot resume: selected generation model settings changed")
    for key, value in scorer.metadata(ROOT).items():
        if metadata.get("evaluation", {}).get(key) != value:
            raise ValueError(f"Cannot resume: evaluation.{key} changed")
    for key, value in experiment.variant_metadata(ROOT).items():
        if metadata.get(key) != value:
            raise ValueError(f"Cannot resume: {key} changed")
    CorrectnessJudges.validate_resume(metadata, ROOT, plan_judge=experiment.plan_judge)
    for resource, saved in (metadata.get("reference_data") or {}).items():
        if saved is None:
            continue
        if resource == "patient_retrieval":
            entries = [("corpus", ROOT / saved["corpus_file"])]
            if saved["artifacts_directory"] is not None:
                entries.extend((name, ROOT / saved["artifacts_directory"] / name)
                               for name in ("embeddings.npy", "index.faiss", "matching.json"))
        elif resource == "guidelines":
            directory = ROOT / saved["directory"]
            entries = guideline_resource_files(directory, directory / "settings.yaml", directory / "output")
        else:
            raise ValueError(f"Cannot resume: unknown reference resource {resource}")
        if saved["sha256"] != (sha256_bundle(entries) if entries is not None else None):
            raise ValueError(f"Cannot resume: {resource} reference data changed")


def _validate_managed_endpoints(experiment, models):
    """Keep local requests on the experiment-owned server, including the later judge."""
    if not experiment.execution["manage_ollama"]:
        return
    registry, _ = read_yaml_config(experiment.model_config)
    host = urlsplit("http://" + experiment.execution["ollama_host"])
    aliases = [*models.model_ids, experiment.plan_judge]
    for alias in dict.fromkeys(aliases):
        endpoint = registry["models"].get(alias, {}).get("api_base")
        if not endpoint:
            continue
        address = urlsplit(endpoint)
        try:
            local = ip_address(address.hostname).is_loopback
        except ValueError:
            local = address.hostname == "localhost"
        if not local:
            continue
        if (address.scheme != "http" or address.hostname not in (host.hostname, "localhost")
                or address.port != host.port or address.path.rstrip("/") != "/v1"
                or address.username or address.password or address.query or address.fragment):
            raise ValueError(f"{alias}: local api_base must use http://{experiment.execution['ollama_host']}/v1 "
                             "while manage_ollama is enabled")


def main(argv=None):
    # %% ------------------------------------------------------------------
    # Configuration, cohort selection, and model clients
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/experiment.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--yes", action="store_true", help="Run without an interactive confirmation")
    args = parser.parse_args(argv)
    experiment = ExperimentConfig(args.config)
    patients = load_patient_records(experiment.patient_input)
    scorer = PatientScorer(experiment.patient_labels, patients, experiment.presenting_evidence_only,
                          max_patients=experiment.max_patients,
                          exclude_none_recorded_plans=experiment.exclude_none_recorded_plans)
    models = AssessmentPlanModels(experiment.model_config, experiment.selected_models,
                                 temperature=experiment.temperature, max_retries=experiment.max_retries,
                                 required_shared_keys=experiment.required_credentials)
    _validate_managed_endpoints(experiment, models)
    if not args.yes and not confirm_experiment(experiment, models.metadata(ROOT), ROOT, scorer.metadata(ROOT)):
        return 0
    started = perf_counter()
    branch = git_branch(ROOT)
    run_file = None
    if args.resume:
        directory = args.resume.resolve()
    else:
        run_file = RunFile.begin(experiment.experiment_output,
                                run_name=make_run_name(pacific_now(), experiment.patient_input, branch))
        directory = run_file.run_dir

    with ExitStack() as resources:
        resources.enter_context(exclusive_run_lock(directory))
        if args.resume:
            metadata = json.loads((directory / "run.json").read_text())
            _validate_resume(metadata, experiment, models, scorer)
            run_file = RunFile(directory, metadata)
            run_file.update(resumptions=[*metadata.get("resumptions", []), {
                "started_at": pacific_now().isoformat(), "execution_sources": _sources()}])
        run_file.redact = models.redact
        definitions = experiment.variant_metadata(ROOT)
        if not args.resume:
            run_file.update(schema_version=4, branch=branch, git_commit=git_commit(ROOT),
                input={"file": repo_path(experiment.patient_input, ROOT), "sha256": _fingerprint(experiment.patient_input)},
                experiment=experiment.metadata(ROOT), assessment_plan_models=models.metadata(ROOT),
                evaluation=scorer.metadata(ROOT), execution_sources=_sources(), **definitions)
            manifest = experiment.patient_input.with_suffix(".selection.json")
            if manifest.exists():
                run_file.update(pilot_selection=json.loads(manifest.read_text()))
        run_file.update(status="running", finished_at=None)
        resources.enter_context(finalize_provenance(run_file))
        print(f"Run directory: {directory}", flush=True)

        # %% ------------------------------------------------------------------
        # Dedicated Ollama server and durable job recovery
        # ---------------------------------------------------------------------
        scheduler = None
        try:
            server = resources.enter_context(ManagedOllama(experiment.execution, directory / "ollama-generation.log"))
            server_metadata = server.metadata()
            run_file.update(ollama_runtime=server_metadata)
            if server_metadata.get("managed"):
                installed = {model["name"] for model in server_metadata["models"]}
                missing = [settings["config"]["model"] for model, settings in models.metadata(ROOT)["models"].items()
                           if models.model_is_local(model) and settings["config"]["model"] not in installed]
                if missing:
                    raise ValueError("Local models are not installed: " + ", ".join(missing))
            scheduler = ExperimentScheduler(experiment, scorer.patients, models, scorer, run_file,
                                            root=ROOT, server=server)
            resources.callback(scheduler.close)
            scheduler.export()

            # %% ------------------------------------------------------------------
            # Clinical evidence and exact prompts, once per configuration
            # Model-major local generation; hosted requests run independently
            # ---------------------------------------------------------------------
            scheduler.prepare_and_generate()

            # %% ------------------------------------------------------------------
            # Batched text metrics and CPU diagnosis similarity; Plans deferred
            # ---------------------------------------------------------------------
            scheduler.score()
        except BaseException as exc:
            if scheduler is not None:
                try:
                    scheduler.export()
                except Exception:
                    pass  # The durable job store survives even if export fails.
            run_file.fail(exc)
            raise

        # %% ------------------------------------------------------------------
        # Canonical results, timing summary, and HTML report
        # ---------------------------------------------------------------------
        status = "completed" if (not run_file.metadata["errors"] and all(
            row["status"] == "completed" for row in scheduler.records.values())) else "completed_with_errors"
        evaluation = scorer.metadata(ROOT)
        if args.resume and run_file.metadata.get("evaluation", {}).get("correctness"):
            evaluation["correctness"] = run_file.metadata["evaluation"]["correctness"]
        run_file.update(status=status, finished_at=pacific_now().isoformat(),
                        evaluation=evaluation, elapsed_seconds=perf_counter() - started)
        try:
            report = generate_report(directory, root=ROOT)
        except Exception as exc:
            run_file.update(results_summary={"status": "failed_incomplete", "error": models.error_metadata(exc)})
            run_file.record_fatal(exc, stage="output_saving")
            return 1
        run_file.update(results_summary={"status": "completed", "file": report.name})
        print(f"Finished: {status}; HTML results: {report}", flush=True)
        return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
