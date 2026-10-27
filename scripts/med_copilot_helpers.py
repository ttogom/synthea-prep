"""Helpers for the project-owned MED-COPILOT runner."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from textwrap import indent
from zoneinfo import ZoneInfo

import yaml

from scripts.med_copilot_variants import COMPONENTS, VARIANTS


PACIFIC = ZoneInfo("America/Los_Angeles")
RUN_FILE = "run.json"
RESULTS_FILE = "results.jsonl"
# Tables read by the vendored GraphRAG local search (graphrag/cli/query.py).
GUIDELINE_TABLES = ("communities", "community_reports", "text_units", "relationships", "entities")
OPTIONAL_GUIDELINE_TABLES = ("covariates",)
# Local search queries this embedding in every configured vector store.
GUIDELINE_VECTOR_EMBEDDING = "entity-description"
PATIENT_STATUSES = ("completed", "failed_incomplete", "not_processed")
PATIENT_RESULT_FIELDS = (
    "predictions", "assessment_plan_raw", "selected_reference_patient",
    "guideline_questions", "guideline_evidence", "graphrag_context", "assessment_plan_prompts",
    "keyword_extraction_raw", "retrieval_keywords", "retrieved_patient_ids",
    "reranked_patient_ids", "model_responses", "evaluation",
)
MODEL_RESPONSE_STAGES = (
    "keyword_extraction", "guideline_questions", "guideline_query_embedding",
    "guideline_evidence", "assessment_plan",
)
# Historical helper callers retain these views; schema 4 resolves the live registry.
VARIANT_SOURCES = {name: spec.components for name, spec in VARIANTS.items()}
VARIANT_DESCRIPTIONS = {name: spec.description for name, spec in VARIANTS.items()}
# Legacy Markdown-parser compatibility: a line naming one generated section,
# such as "## Assessment", "## Plan", or a standalone label ("Assessment:",
# "**Plan**") like the reference patient's. Heading level, case, bold, an
# "(A)"/"(P)" suffix, a colon, and closing hashes are tolerated; a label without
# "#" needs a colon or bold, and nothing else may share the line.
SECTION_HEADING = re.compile(
    r" {0,3}(?=#|\*\*|__|.*:)(?:#{1,6}[ \t]+)?(?:\*\*|__)?(assessment|plan)"
    r"(?:[ \t]*\([ap]\))?:?(?:\*\*|__)?:?(?:[ \t]+#+)?[ \t\r]*",
    re.IGNORECASE,
)


def load_patient_records(patient_input):
    """Load and validate all prepared S/O JSONL records, preserving their contents."""
    patient_input = Path(patient_input)
    patients = []
    id_lines = {}
    with patient_input.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{patient_input}:{line_number}: invalid patient JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{patient_input}:{line_number}: patient must be a JSON object")
            patient_id = record.get("id")
            if not isinstance(patient_id, str) or not patient_id.strip():
                raise ValueError(f"{patient_input}:{line_number}: id must be a nonempty string")
            if patient_id in id_lines:
                raise ValueError(
                    f"{patient_input}:{line_number}: duplicate id {patient_id!r} "
                    f"(first used on line {id_lines[patient_id]})"
                )
            id_lines[patient_id] = line_number
            for field in ("subjective", "objective"):
                if not isinstance(record.get(field), str):
                    raise ValueError(
                        f"{patient_input}:{line_number}: {field} must be a string"
                    )
            patients.append(record)
    if not patients:
        raise ValueError(f"{patient_input}: no patient records found")
    return patients


def validate_guideline_tables(guideline_output):
    """Require local-search tables directly in the selected output directory."""
    guideline_output = Path(guideline_output)
    for name in GUIDELINE_TABLES:
        table = guideline_output / f"{name}.parquet"
        if not table.is_file():
            raise FileNotFoundError(errno.ENOENT, "Required guideline table not found", str(table))


# %% ------------------------------------------------------------------
# Run naming and Git identity
# ---------------------------------------------------------------------

def pacific_time(instant):
    """Convert an aware datetime to whole-second America/Los_Angeles time."""
    return instant.astimezone(PACIFIC).replace(microsecond=0)


def pacific_now():
    return pacific_time(datetime.now(timezone.utc))


def path_safe_branch(branch):
    """Make a branch name usable as one directory-name component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip(".-")
    if not safe:
        raise ValueError(f"Branch name {branch!r} has no path-safe characters")
    return safe


def make_run_name(started_at, input_path, branch):
    """<Pacific timestamp>__<input filename stem>__<branch name>."""
    timestamp = started_at.strftime("%Y-%m-%d_%H%M%S") + started_at.tzname()
    return f"{timestamp}__{Path(input_path).stem}__{path_safe_branch(branch)}"


def _git(repo, *args):
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to identify the run's branch, "
                           "but the git executable was not found") from exc


def git_branch(repo):
    """Return the checked-out branch of `repo`, or "HEAD" when detached."""
    repo = Path(repo).resolve()
    toplevel = _git(repo, "rev-parse", "--show-toplevel")
    if toplevel.returncode != 0 or Path(toplevel.stdout.strip()).resolve() != repo:
        detail = toplevel.stderr.strip() or f"top level is {toplevel.stdout.strip()}"
        raise RuntimeError(f"Cannot identify the Git repository at {repo}: {detail}")
    branch = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch.returncode == 0:
        return branch.stdout.strip()
    if branch.returncode == 1:
        return "HEAD"
    raise RuntimeError(f"Cannot determine the Git branch at {repo}: {branch.stderr.strip()}")


def git_commit(repo):
    """Return the full checked-out commit SHA, or None when unavailable."""
    try:
        commit = _git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    except RuntimeError:
        return None
    return commit.stdout.strip() if commit.returncode == 0 and commit.stdout.strip() else None


def create_run_dir(output_root, run_name):
    """Create a new run directory, never reusing an existing one."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_name
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(errno.EEXIST, "Run directory already exists; refusing to overwrite",
                              str(run_dir)) from exc
    return run_dir


# %% ------------------------------------------------------------------
# Resource metadata and fingerprints
# ---------------------------------------------------------------------

def repo_path(path, root):
    """Repository-relative POSIX path inside `root`; absolute path outside it."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bundle(entries):
    """Fingerprint labelled files/directories together; None if any is missing.

    Each entry is (label, path). A directory contributes every file beneath it,
    labelled by its relative path, so renamed or changed members alter the hash.
    """
    digest = hashlib.sha256()
    for label, path in entries:
        path = Path(path)
        if path.is_file():
            members = [(label, path)]
        elif path.is_dir():
            members = sorted(
                (f"{label}/{member.relative_to(path).as_posix()}", member)
                for member in path.rglob("*") if member.is_file()
            )
        else:
            return None
        for name, member in members:
            digest.update(f"{name}\0{sha256_file(member)}\n".encode("utf-8"))
    return digest.hexdigest()


def guideline_resource_files(guideline_dir, settings_file, guideline_output):
    """List the settings, prompt, tables, and vector stores local search uses.

    Mirrors the vendored GraphRAG local search: the prompt is read from
    root/local_search.prompt when present, tables from the selected output
    directory, and each LanceDB store at root/db_uri. Returns None when the
    configuration cannot be resolved to local files.
    """
    guideline_dir, settings_file = Path(guideline_dir), Path(settings_file)
    guideline_output = Path(guideline_output)
    try:
        settings = yaml.safe_load(settings_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(settings, dict) or settings.get("outputs"):
        return None  # Multi-index outputs are not used by this runner.
    files = [("settings", settings_file)]
    prompt = (settings.get("local_search") or {}).get("prompt")
    if prompt and (guideline_dir / prompt).is_file():
        files.append(("local_search_prompt", guideline_dir / prompt))
    tables = list(GUIDELINE_TABLES) + [name for name in OPTIONAL_GUIDELINE_TABLES
                                       if (guideline_output / f"{name}.parquet").is_file()]
    files.extend((f"tables/{name}.parquet", guideline_output / f"{name}.parquet")
                 for name in tables)
    for store_id, store in sorted((settings.get("vector_store") or {}).items()):
        if not isinstance(store, dict) or store.get("type") != "lancedb":
            return None
        db_uri = guideline_dir / (store.get("db_uri") or "output/lancedb")
        table = f"{store.get('container_name') or 'default'}-{GUIDELINE_VECTOR_EMBEDDING}.lance"
        files.append((f"vector_store/{store_id}/{table}", db_uri / table))
    return files


def reference_data_metadata(root, *, corpus_file, artifacts_directory, artifact_files,
                            guideline_directory, guideline_settings, guideline_output):
    """Describe the reference resources actually used by this run.

    Pass the runtime paths: the loaded corpus, the retrieval artifact directory
    and files the retriever loaded or regenerated (None when retrieval did not
    initialize), and the guideline root, settings, and table directory.
    """
    retrieval = None
    if corpus_file is not None:
        entries = [("corpus", corpus_file)]
        if artifact_files is not None:
            entries.extend((repo_path(path, artifacts_directory), path) for path in artifact_files)
        retrieval = {
            "corpus_file": repo_path(corpus_file, root),
            "artifacts_directory": (repo_path(artifacts_directory, root)
                                    if artifact_files is not None else None),
            "sha256": sha256_bundle(entries),
        }
    guidelines = None
    if guideline_directory is not None:
        files = guideline_resource_files(guideline_directory, guideline_settings, guideline_output)
        guidelines = {
            "directory": repo_path(guideline_directory, root),
            "sha256": sha256_bundle(files) if files is not None else None,
        }
    return {"patient_retrieval": retrieval, "guidelines": guidelines}


# %% ------------------------------------------------------------------
# Per-patient outputs
# ---------------------------------------------------------------------

def new_model_result():
    """Fresh final-model slot, including explicit unattempted generation status."""
    return {"status": "not_processed", "error": None, "predictions": None,
            "assessment_plan_raw": None, "model_responses": [], "evaluation": None}


def format_assessment_plans(assessment_plans, model_settings):
    """Render model-grouped ranked condition/Plan pairs without changing results."""
    groups = []
    for model_id, output in assessment_plans.items():
        config = model_settings[model_id]["config"]
        model_name = config.get("deployment_name") or config["model"]
        lines = [f"Model: {model_id} ({config['model_provider']}/{model_name})"]
        if output["status"] != "completed":
            lines.append(f"    Generation status: {output['status']}")
            if output["error"] is not None:
                error = output["error"]
                lines.append(indent(f"Error ({error['type']}): {error['message']}", "    "))
        elif output["predictions"] is None:
            lines.append("    No valid ranked conditions and plans returned.")
        else:
            for rank, pair in enumerate(output["predictions"], 1):
                condition = pair["condition"] if pair["condition"] is not None else "[unavailable]"
                lines.append(indent(f"{rank}. Condition: {condition}", "    "))
                lines.append("        Plan:")
                if pair["plan"] is None:
                    lines.append("            [unavailable]")
                elif not pair["plan"]:
                    lines.append("            No plan items returned.")
                else:
                    for step, action in enumerate(pair["plan"], 1):
                        lines.append(indent(f"{step}. {action}", "            "))
                lines.append("")
        groups.append("\n".join(lines).rstrip("\n"))
    return "\n\n".join(groups)


def new_patient_result(patient_id, status=None, model_ids=None, variant_ids=None, *,
                       component_plan=None, variant_components=None):
    """Fresh patient outputs; provider response lists are independent per stage/patient."""
    if component_plan is not None:
        model_ids, variant_ids = tuple(model_ids), tuple(variant_ids)
        fields = [field for field in PATIENT_RESULT_FIELDS if field not in (
            "predictions", "assessment_plan_raw", "evaluation", "assessment_plan_prompts", "model_responses")]
        return {
            "id": patient_id, "status": status,
            "components": {key: {
                "component": item["component"], "status": "not_processed", "error": None,
                "settings": deepcopy(item["settings"]), "outputs": {},
                "capture": {**dict.fromkeys(fields), "model_responses": {
                    stage: [] for stage in MODEL_RESPONSE_STAGES if stage != "assessment_plan"}},
            } for key, item in component_plan.items()},
            "variants": {variant: {
                "status": "not_processed", "error": None,
                "components": dict(variant_components[variant]), "assessment_plan_prompts": None,
                "assessment_plans": {model: new_model_result() for model in model_ids},
            } for variant in variant_ids},
        }
    if variant_ids is not None:
        model_ids, variant_ids = tuple(model_ids), tuple(variant_ids)
        sources = {source for variant in variant_ids for source in VARIANT_SOURCES[variant]}
        fields = [field for field in PATIENT_RESULT_FIELDS if field not in (
            "predictions", "assessment_plan_raw", "evaluation", "assessment_plan_prompts")]
        return {
            "id": patient_id, "status": status, **dict.fromkeys(fields),
            "model_responses": {stage: [] for stage in MODEL_RESPONSE_STAGES if stage != "assessment_plan"},
            "evidence_status": {source: {"status": "not_processed" if source in sources else "not_requested",
                                         "error": None} for source in ("patient_retrieval", "guidelines")},
            "variants": {variant: {
                "status": "not_processed", "error": None,
                "evidence_sources": list(VARIANT_SOURCES[variant]), "assessment_plan_prompts": None,
                "assessment_plans": {model: new_model_result() for model in model_ids},
            } for variant in variant_ids},
        }
    if model_ids is not None:
        fields = [field for field in PATIENT_RESULT_FIELDS if field not in (
            "predictions", "assessment_plan_raw", "evaluation")]
        return {"id": patient_id, "status": status, **dict.fromkeys(fields),
                "model_responses": {stage: [] for stage in MODEL_RESPONSE_STAGES
                                    if stage != "assessment_plan"},
                "assessment_plans": {model_id: new_model_result() for model_id in model_ids}}
    # Preserve the public helper's historical single-model calling convention.
    return {"id": patient_id, "status": status, **dict.fromkeys(PATIENT_RESULT_FIELDS),
            "model_responses": {stage: [] for stage in MODEL_RESPONSE_STAGES}}


def model_response_metadata(response):
    """Copy only provider-reported model, response ID, and token counts."""
    def value(source, key):
        return source.get(key) if isinstance(source, dict) else getattr(source, key, None)

    model, response_id, usage = (value(response, key) for key in ("model", "id", "usage"))
    counts = None
    if usage is not None:
        counts = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            count = value(usage, key)
            counts[key] = count if isinstance(count, int) and not isinstance(count, bool) else None
    return {"reported_model": model if isinstance(model, str) else None,
            "response_id": response_id if isinstance(response_id, str) else None,
            "token_usage": counts}


def reference_patient_identity(corpus_records, row):
    """Identify one reference case, using its corpus position to disambiguate IDs."""
    row = int(row)
    if not 0 <= row < len(corpus_records):
        raise ValueError(f"Reference corpus row out of range: {row}")
    note_id = corpus_records[row].get("note_id")
    if not isinstance(note_id, str) or not note_id.strip():
        return {"corpus_row": row}
    if sum(record.get("note_id") == note_id for record in corpus_records) > 1:
        return {"note_id": note_id, "corpus_row": row}
    return {"note_id": note_id}


def ranked_reference_patients(corpus_records, candidates):
    """Snapshot case IDs and available unrounded scores in the supplied ranking order."""
    return [{**reference_patient_identity(corpus_records, candidate["idx"]),
             **{score: float(candidate[score]) for score in
                ("embedding_score", "keyword_score", "hybrid_score", "rerank_score")
                if score in candidate}} for candidate in candidates]


def prepare_patient_output_capture(result, corpus_records, retriever, reranker):
    """Bind shared components to this result and reset their retained rankings.

    Called before any patient stage can fail. Each ranking callback snapshots
    IDs/scores into its own result field using this patient's reference corpus.
    Bindings remain attached until the next patient calls this helper.
    """
    extractor = getattr(retriever, "keyword_extractor", None)
    if extractor is not None:
        extractor.capture_result = result
    for component, field in ((retriever, "retrieved_patient_ids"),
                             (reranker, "reranked_patient_ids")):
        if component is not None:
            component.ranked_candidates = None
            def save_ranking(ranking, field=field):
                result[field] = ranked_reference_patients(corpus_records, ranking)
            component.on_ranking = save_ranking


@contextmanager
def capture_component(component, *, error_metadata=None, corpus_records=(), retriever=None,
                      reranker=None, credential_names=()):
    """Capture one scheduled execution, restoring mutable hooks on every exit.

    Ordinary failures remain local. Interruptions propagate after recording the
    partial capture. Only validated published outputs count as completed evidence.
    """
    if component["status"] != "not_processed":
        raise ValueError("Component was already attempted; shared components must be scheduled once")
    previous = []
    extractor = getattr(retriever, "keyword_extractor", None)
    for target, names in ((extractor, ("capture_result",)),
                          (retriever, ("ranked_candidates", "on_ranking", "alpha")),
                          (reranker, ("ranked_candidates", "on_ranking"))):
        if target is not None:
            for name in names:
                previous.append((target, name, hasattr(target, name), getattr(target, name, None)))
    component["status"] = "failed_incomplete"
    try:
        prepare_patient_output_capture(component["capture"], corpus_records, retriever, reranker)
        yield component["capture"]
        expected = COMPONENTS[component["component"]].outputs
        if any(name not in component["outputs"] or not isinstance(component["outputs"][name], str)
               for name in expected):
            raise ValueError("Component did not publish its declared string outputs")
        component["status"] = "completed"
    except Exception as exc:
        error = error_metadata(exc) if error_metadata else {"type": type(exc).__name__, "message": str(exc)}
        secrets = {os.environ.get(key) for key in credential_names} - {None, ""}
        for secret in sorted(secrets, key=len, reverse=True):
            error["message"] = error["message"].replace(secret, "[REDACTED]")
        component["error"] = error
        print(f"{component['component']} failed ({error['type']}): {error['message']}", file=sys.stderr)
    except BaseException as exc:
        component["error"] = {"type": type(exc).__name__, "message": "Evidence preparation interrupted"}
        raise
    finally:
        for target, name, existed, value in reversed(previous):
            if existed:
                setattr(target, name, value)
            elif hasattr(target, name):
                delattr(target, name)


def component_needed(experiment, component_id, result):
    """Skip work only when every selected consumer is already blocked elsewhere."""
    for references in experiment.variant_components.values():
        if component_id in references.values() and not any(
                result["components"][other]["status"] == "failed_incomplete"
                for other in references.values() if other != component_id):
            return True
    return False


def prepare_variant_messages(experiment, variant_id, s_o, result, output, *, templates,
                             error_metadata=None):
    """Resolve component bindings and save exact final messages, or fail this variant."""
    try:
        variant = VARIANTS[variant_id]
        references = experiment.variant_components[variant_id]
        unavailable = [name for name, instance in references.items()
                       if result["components"][instance]["status"] != "completed"]
        if unavailable:
            output.update(status="failed_incomplete", error={
                "type": "EvidenceUnavailable", "message": "Required evidence failed: " + ", ".join(unavailable)})
            return None
        values = {field: result["components"][references[component]]["outputs"][name]
                  for field, (component, name) in variant.bindings.items()}
        prompt = templates[variant.prompt_key].format(conditions=s_o, **values)
        messages = [{"role": "system", "content": "You are a clinical expert."},
                    {"role": "user", "content": prompt}]
        output["assessment_plan_prompts"] = {"system": messages[0]["content"], "user": prompt}
        output["status"] = "failed_incomplete"
        return messages
    except Exception as exc:
        output.update(status="failed_incomplete", error=(error_metadata(exc) if error_metadata else
                      {"type": type(exc).__name__, "message": "Variant prompt preparation failed"}))
        return None


def selected_reference_patient(corpus_records, candidates, reranked):
    """Identify the reference patient the reranker placed first.

    The vendored reranker returns formatted text only, but records each
    candidate's rerank_score in place; repeating its stable descending sort
    recovers the same first candidate and its corpus row (`idx`).
    """
    if not reranked:
        return None
    top = sorted(candidates, key=lambda candidate: candidate["rerank_score"], reverse=True)[0]
    return reference_patient_identity(corpus_records, top["idx"])


def parse_guideline_questions(response):
    """Return the "Key Questions" strings, or None when the reply lacks them."""
    try:
        questions = json.loads(response).get("Key Questions")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return None
    return questions


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError("Nonfinite JSON constant")


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def parse_condition_plan(response):
    """Parse three ranked condition/Plan pairs, retaining valid fields by rank.

    The root must contain only a predictions list of exactly three pairs.
    Malformed JSON, duplicate keys, and nonfinite numbers invalidate the reply.
    Each pair accepts only condition and plan keys; otherwise both fields in
    that rank are null. Valid fields survive an invalid partner unchanged.
    Plans accept zero to three nonblank strings. No list is truncated.
    """
    missing = {"predictions": None}
    if not isinstance(response, str):
        return missing
    try:
        payload = json.loads(
            response, object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant, parse_float=_finite_json_float,
        )
    except (TypeError, ValueError, RecursionError):
        return missing
    if not isinstance(payload, dict) or set(payload) != {"predictions"}:
        return missing
    predictions = payload["predictions"]
    if not isinstance(predictions, list) or len(predictions) != 3:
        return missing
    parsed = []
    for pair in predictions:
        if not isinstance(pair, dict) or set(pair) - {"condition", "plan"}:
            parsed.append({"condition": None, "plan": None})
            continue
        condition, plan = pair.get("condition"), pair.get("plan")
        parsed.append({
            "condition": condition if isinstance(condition, str) and condition.strip() else None,
            "plan": plan if isinstance(plan, list) and len(plan) <= 3 and all(
                isinstance(item, str) and item.strip() for item in plan
            ) else None,
        })
    return {"predictions": parsed}


def split_assessment_plan(response):
    """Split the final reply into its Assessment and Plan section texts.

    Each section runs from its SECTION_HEADING line to the next one and is
    kept verbatim apart from surrounding whitespace. A section that is missing,
    empty, or headed more than once is None; text before the first heading
    belongs to neither section.
    """
    sections = {"assessment": [], "plan": []}
    current = None
    if isinstance(response, str):
        for line in response.split("\n"):
            heading = SECTION_HEADING.fullmatch(line)
            if heading:
                current = sections[heading.group(1).lower()]
                current.append([])
            elif current is not None:
                current[-1].append(line)
    return {name: ("\n".join(bodies[0]).strip() or None) if len(bodies) == 1 else None
            for name, bodies in sections.items()}


def graphrag_context_json(context):
    """Convert every returned context table to {columns, index, data}."""
    return json.loads(json.dumps(
        context, default=lambda table: json.loads(table.to_json(orient="split")),
    ))


def write_run_json(run_dir, metadata):
    """Atomically replace run.json so readers never see a partial file."""
    run_dir = Path(run_dir)
    temporary = run_dir / f".{RUN_FILE}.tmp"
    try:
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
                             + "\n", encoding="utf-8")
        os.replace(temporary, run_dir / RUN_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def run_status(patient_statuses, stopped=False):
    """Aggregate final patient statuses into the run status."""
    if stopped or any(status not in ("completed", "failed_incomplete")
                      for status in patient_statuses):
        return "failed_incomplete"
    if all(status == "completed" for status in patient_statuses):
        return "completed"
    return "completed_with_errors"


class ExperimentRun:
    """Own one run directory: run.json plus one results.jsonl line per patient."""

    def __init__(self, run_dir, metadata, patient_ids, model_ids=None, variant_ids=None,
                 component_plan=None, variant_components=None):
        self.run_dir = Path(run_dir)
        self.metadata = metadata
        self.unsaved = list(patient_ids)
        self.statuses = []
        self.results_writable = True
        self.model_ids = None if model_ids is None else tuple(model_ids)
        self.variant_ids = None if variant_ids is None else tuple(variant_ids)
        self.component_plan = deepcopy(component_plan)
        self.variant_components = deepcopy(variant_components)

    @classmethod
    def start(cls, output_root, *, root, input_path, started_at, branch, commit, patient_ids,
              model_ids=None, variant_ids=None, component_plan=None, variant_components=None):
        model_ids = None if model_ids is None else tuple(model_ids)
        variant_ids = None if variant_ids is None else tuple(variant_ids)
        run_dir = create_run_dir(output_root, make_run_name(started_at, input_path, branch))
        metadata = {
            "run_name": run_dir.name,
            "started_at": started_at.isoformat(),
            "finished_at": None,
            "status": "running",
            "branch": branch,
            "input": {"file": repo_path(input_path, root), "sha256": sha256_file(input_path)},
            "git_commit": commit,
            "reference_data": None,
        }
        if model_ids is not None:
            metadata.update(schema_version=2, assessment_plan_models=None)
        if variant_ids is not None:
            metadata.update(schema_version=3, variant_definitions=None, prompt_provenance=None)
        if component_plan is not None:
            metadata.update(schema_version=4, component_definitions=None, registry_provenance=None)
        (run_dir / RESULTS_FILE).touch(exist_ok=False)
        run = cls(run_dir, metadata, patient_ids, model_ids, variant_ids, component_plan, variant_components)
        run.update()
        return run

    def update(self, **fields):
        self.metadata.update(fields)
        write_run_json(self.run_dir, self.metadata)

    def save_result(self, result):
        """Append one final patient record as a single JSONL line."""
        if result["status"] not in PATIENT_STATUSES:
            raise ValueError(f"Invalid patient status {result['status']!r}")
        if result["id"] not in self.unsaved:
            raise ValueError(f"Patient {result['id']!r} is not awaiting a result")
        if not self.results_writable:
            raise RuntimeError("results.jsonl is no longer writable after an earlier failure")
        line = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            with (self.run_dir / RESULTS_FILE).open("a", encoding="utf-8") as results:
                results.write(line)
        except BaseException:
            # A failed append may leave a partial line; never append after it.
            self.results_writable = False
            raise
        self.unsaved.remove(result["id"])
        self.statuses.append(result["status"])

    def finish(self, finished_at):
        """Finalize a run whose every patient has a saved result."""
        if self.unsaved:
            raise RuntimeError(f"{len(self.unsaved)} patients have no saved result")
        status = run_status(self.statuses)
        self.update(status=status, finished_at=finished_at.isoformat())
        return status

    def stop(self, finished_at, current=None):
        """Best-effort finalization after a fatal error or interruption.

        The in-flight patient becomes failed_incomplete (keeping partial
        outputs when they serialize) and the rest become not_processed.
        """
        if current is not None and current["id"] in self.unsaved and self.results_writable:
            current = {**current, "status": "failed_incomplete"}
            for record in (current, new_patient_result(
                    current["id"], "failed_incomplete", self.model_ids, self.variant_ids,
                    component_plan=self.component_plan, variant_components=self.variant_components)):
                try:
                    self.save_result(record)
                    break
                except Exception:
                    if not self.results_writable:
                        break
        for patient_id in list(self.unsaved):
            if not self.results_writable:
                break
            try:
                self.save_result(new_patient_result(patient_id, "not_processed", self.model_ids, self.variant_ids,
                                 component_plan=self.component_plan, variant_components=self.variant_components))
            except Exception:
                break
        try:
            self.update(status=run_status(self.statuses, stopped=True),
                        finished_at=finished_at.isoformat())
        except Exception:
            pass


class JSONKeywordExtractor:
    """Keep the upstream extraction prompt, with enforced JSON and stable ordering."""

    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.capture_result = None

    def extract(self, text):
        capture = self.capture_result
        if capture is not None:
            capture["keyword_extraction_raw"] = None
            capture["retrieval_keywords"] = None
        prompt = f"""
You are a clinical NLP expert.
From the following clinical note, extract medically meaningful entities.

IMPORTANT:
- Each item MUST be a verbatim span copied from the text.
- Do NOT paraphrase or infer.
Return ONLY valid JSON:
{{
  "symptoms": [],
  "diagnoses": [],
  "medications": []
}}

Text:
{text}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        if capture is not None:
            capture["model_responses"]["keyword_extraction"].append(model_response_metadata(response))
            choices = getattr(response, "choices", None)
            content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
            if isinstance(content, str):
                capture["keyword_extraction_raw"] = content
        if not getattr(response, "choices", None):
            raise ValueError("Keyword extraction returned no choices")
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise ValueError("Keyword extraction was refused")
        if choice.finish_reason != "stop":
            raise ValueError(
                f"Keyword extraction did not complete (finish_reason={choice.finish_reason})"
            )
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Keyword extraction returned empty or non-text content")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Keyword extraction returned invalid JSON: {exc.msg}") from exc
        if not isinstance(result, dict):
            raise ValueError("Keyword extraction must return a JSON object")
        keywords = []
        for category in ("symptoms", "diagnoses", "medications"):
            terms = result.get(category)
            if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
                raise ValueError(f"Keyword extraction {category} must be a list of strings")
            keywords.extend(term.strip() for term in terms)
        keywords = list(dict.fromkeys(keywords))
        if capture is not None:
            capture["retrieval_keywords"] = list(keywords)
        return keywords
