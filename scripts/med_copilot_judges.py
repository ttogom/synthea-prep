"""Preliminary single-model midterm correctness metrics for diagnoses and treatment plans.

PRELIMINARY: this is the provisional midterm metric, to be revised after group
review. The diagnosis score is a BioLORD-2023 clinical concept similarity score: the
embedding cosine between the labeled condition and the generated diagnosis, clipped
to 0-1 (no LLM judge). It is not an accuracy or a probability. Plan correctness
applies the treatment-plan rubric with the experiment's configured Plan judge:
one scalar 0/0.5/1 call per reference action, then counts unsupported extra
actions (0.5 deducted each) unless all reference credits are zero. The judge sees
only the labeled condition, the patient's S/O, and the reference and generated
actions; the generator, evidence variant, and rank stay hidden.

Every completed judge call is appended to judge_responses.jsonl in the run
directory before it is used. Valid judgements are reused, so a resumed
evaluation never repeats them; malformed or refused replies stay in the file for
auditing but are requested again.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from threading import Lock, RLock
import time
import weakref

from med_copilot.prompts.correctness_judges import PLAN_ACTION_JUDGE_TEMPLATE, PLAN_EXTRAS_JUDGE_TEMPLATE
from scripts.med_copilot_helpers import model_response_metadata, repo_path
from scripts.med_copilot_run_file import RunMetadataError, pacific_now, record_error, _fsync_directory


METRIC_STATUS = "preliminary single-model midterm metric"
JUDGE_FILE = "judge_responses.jsonl"
# A judge that fails the balanced validation sample (correct vs off-target vs placeholder
# Plans) is marked EXPERIMENTAL_UNVALIDATED: its scores are saved per prediction for review,
# excluded from every reported aggregate, and never presented as accuracy.
EXPERIMENTAL_UNVALIDATED = "experimental_unvalidated"
PLAN_VALIDATION_STATUS = "passed_balanced_sample"
PLAN_VALIDATION_EVIDENCE = (
    "2026-11-06 balanced sample, 3 cohort patients (essential hypertension, streptococcal sore throat, "
    "chronic congestive heart failure) x 6 Plans: 2 correct (reference verbatim; hand-written equivalent), "
    "2 off-target (other patients' references), 2 placeholder. Pass = all valid, mean correct >= 0.7, "
    "placeholder <= 0.1, off-target <= 0.2, and every correct Plan above every other Plan per patient. "
    "ollama_qwen2_5_14b passed: 18/18 valid; correct 1.00, off-target 0.00, placeholder 0.083. "
    "ollama_llama3 failed: 18/18 valid; correct 1.00, off-target 0.15, placeholder 0.83. "
    "Small sample; placeholder credit was sometimes given and offset by the extras deduction.")
JUDGE_TEMPERATURE = 0
BIOLORD_MODEL = "FremyCompany/BioLORD-2023"
BIOLORD_REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
# Deterministic expansion applied to both texts before BioLORD encoding: each
# abbreviation is replaced only as a whole word, in any case. BioLORD otherwise
# scores these abbreviations far below their expansions (e.g. HTN 0.32).
DIAGNOSIS_ABBREVIATIONS = {
    "HTN": "hypertension",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
}
DIAGNOSIS_NORMALIZATION_RULE = (
    "before encoding, both texts expand each abbreviation as a whole word in any "
    "case, then trim and collapse whitespace; originals are kept as reference/"
    "generated and the encoded texts as normalized_reference/normalized_generated")
_ABBREVIATION = re.compile(r"\b(" + "|".join(DIAGNOSIS_ABBREVIATIONS) + r")\b", re.IGNORECASE)
CREDIT_VALUES = (0, 0.5, 1)
DEDUCTION_PER_UNSUPPORTED_EXTRA = 0.5
RECORD_ARTIFACT_MARKER = "(record artifact)"
# Any case and spacing, e.g. "( Record  Artifact )".
RECORD_ARTIFACT = re.compile(r"\(\s*record\s+artifact\s*\)", re.IGNORECASE)
PLAN_RUBRIC = Path(__file__).resolve().parents[1] / "med_copilot/docs/treatment_plan_evaluation.md"


def rank_weight(rank):
    """Reciprocal-rank weight: 1, 0.5, and 1/3 for ranks 1-3."""
    return 1 / rank


def plan_evaluation_protocol(root, *, plan_judge):
    """Snapshot Plan semantics independently of generation-time scoring provenance.

    Bump the version for incompatible semantics not captured by these settings.
    Runtime concurrency and provider settings remain separate from the protocol.
    """
    if not isinstance(plan_judge, str) or not plan_judge.strip():
        raise ValueError("plan_judge must be a nonempty registry alias")
    return {
        "version": 1,
        "plan": {
            "judge_model": plan_judge,
            "judge_temperature": JUDGE_TEMPERATURE,
            "rubric_file": repo_path(PLAN_RUBRIC, root),
            "rubric_sha256": hashlib.sha256(PLAN_RUBRIC.read_bytes()).hexdigest(),
            "action_prompt_template_sha256": hashlib.sha256(PLAN_ACTION_JUDGE_TEMPLATE.encode()).hexdigest(),
            "extras_prompt_template_sha256": hashlib.sha256(PLAN_EXTRAS_JUDGE_TEMPLATE.encode()).hexdigest(),
            "credit_values": list(CREDIT_VALUES),
            "deduction_per_unsupported_extra": DEDUCTION_PER_UNSUPPORTED_EXTRA,
        },
        "rank_weighting": {
            "formula": "1 / rank",
            "weights": {str(rank): rank_weight(rank) for rank in (1, 2, 3)},
        },
    }


def unavailable_plan_reason(plan):
    """Why a reference Plan cannot be judged, or None when it can."""
    if not isinstance(plan, list) or not plan or not any(item.strip() for item in plan):
        return "missing_reference"
    if plan == ["none recorded"]:
        return "no_recorded_plan"
    if any(RECORD_ARTIFACT.search(item) for item in plan):
        return "record_artifact_plan"
    return None


def _numbered(actions):
    return "\n".join(f"{number}. {action}" for number, action in enumerate(actions, 1))


def plan_action_messages(condition, subjective, objective, reference_action, generated_actions):
    """One reference action against the numbered generated actions."""
    prompt = PLAN_ACTION_JUDGE_TEMPLATE.format(
        condition=condition, subjective=subjective, objective=objective,
        reference_action=reference_action, generated_plan=_numbered(generated_actions))
    return [{"role": "user", "content": prompt}]


def plan_extras_messages(condition, subjective, objective, reference_actions, generated_actions):
    """Count unsupported extras among the numbered generated actions."""
    prompt = PLAN_EXTRAS_JUDGE_TEMPLATE.format(
        condition=condition, subjective=subjective, objective=objective,
        reference_plan=_numbered(reference_actions), generated_plan=_numbered(generated_actions))
    return [{"role": "user", "content": prompt}]


def request_key(judge_model, messages):
    """Identify one judge request; identical requests to the same judge are asked once."""
    payload = json.dumps({"judge_model": judge_model, "temperature": JUDGE_TEMPERATURE,
                          "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeResponseError(ValueError):
    """A judge replied, but not with a valid judgement."""


class JudgeCallError(RuntimeError):
    """A judge request raised before any reply was obtained."""


class SimilarityError(RuntimeError):
    """The BioLORD-2023 similarity could not be computed."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JudgeResponseError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise JudgeResponseError(f"Nonfinite JSON constant: {value}")


def _judgement(content):
    if not isinstance(content, str) or not content.strip():
        raise JudgeResponseError("Judge returned empty or non-text content")
    try:
        value = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"Judge returned invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise JudgeResponseError("Judge JSON must be an object")
    return value


def _credit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in CREDIT_VALUES:
        raise JudgeResponseError(f"{name} must be 0, 0.5, or 1")
    return float(value)


def parse_action_judgement(content):
    """Return the judge's 0/0.5/1 credit for one reference action; extra keys stay in the raw reply."""
    judgement = _judgement(content)
    if "credit" not in judgement:
        raise JudgeResponseError("Action judgement requires credit")
    return {"credit": _credit(judgement["credit"], "credit")}


def parse_extras_judgement(content):
    """Return a nonnegative integer count of unsupported extras, including bundled actions."""
    judgement = _judgement(content)
    if "unsupported_extra_count" not in judgement:
        raise JudgeResponseError("Extras judgement requires unsupported_extra_count")
    count = judgement["unsupported_extra_count"]
    if type(count) is not int or count < 0:
        raise JudgeResponseError("unsupported_extra_count must be a nonnegative integer")
    return {"unsupported_extra_count": count}


def plan_score(credits, deduction):
    """max(0, (sum of item credits - deduction) / number of reference actions)."""
    return max(0.0, (math.fsum(credits) - deduction) / len(credits))


def normalize_diagnosis(text):
    """Expand HTN, CHF, and COPD (whole words, any case), then trim and collapse whitespace."""
    expanded = _ABBREVIATION.sub(lambda match: DIAGNOSIS_ABBREVIATIONS[match.group(1).upper()], text)
    return " ".join(expanded.split())


def diagnosis_score(cosine):
    """Clinical concept similarity score: BioLORD cosine clipped to [0, 1]; negative counts as 0."""
    return min(1.0, max(0.0, cosine))


class BioLordSimilarity:
    """Cosine similarity of BioLORD-2023 sentence embeddings; the model loads on first use.

    encode(texts) may be supplied (tests); otherwise the pinned BioLORD-2023
    revision is loaded with sentence-transformers. Embeddings are cached by text.
    """

    def __init__(self, encode=None):
        self._encode = encode
        self._vectors = {}

    def _encoder(self):
        if self._encode is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(BIOLORD_MODEL, revision=BIOLORD_REVISION)
            self._encode = lambda texts: model.encode(texts, convert_to_numpy=True)
        return self._encode

    def cosine(self, reference, generated):
        missing = [text for text in dict.fromkeys((reference, generated)) if text not in self._vectors]
        if missing:
            vectors = self._encoder()(missing)
            if len(vectors) != len(missing):
                raise SimilarityError("BioLORD returned the wrong number of embeddings")
            for text, vector in zip(missing, vectors):
                self._vectors[text] = [float(value) for value in vector]
        first, second = self._vectors[reference], self._vectors[generated]
        norms = math.sqrt(math.fsum(x * x for x in first)) * math.sqrt(math.fsum(y * y for y in second))
        cosine = math.fsum(x * y for x, y in zip(first, second)) / norms if norms else math.nan
        if len(first) != len(second) or not math.isfinite(cosine):
            raise SimilarityError("BioLORD embeddings are empty, mismatched, or nonfinite")
        return cosine


def _close_judge_index(connection, directory):
    try:
        connection.close()
    finally:
        directory.cleanup()


class JudgeCache(Mapping):
    """Durable JSONL replies with a disposable, disk-backed SQLite offset index.

    Rebuilding on every open deliberately validates every complete audit line
    and avoids a second persistent source of truth. SQLite retains only offsets
    and checksums, with a 2 MiB page-cache target and no memory mapping. A crash
    after audit fsync but before indexing is recovered by the next rebuild.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        self._lock = RLock()
        self._write_error = None
        self._identity = None
        self._indexed_end = 0
        # A real temporary file guarantees disk backing regardless of SQLite's
        # compile-time TEMP_STORE setting. Never reuse scratch indexes on restart.
        directory = TemporaryDirectory(prefix="medcopilot-judge-index-")
        try:
            self._index = sqlite3.connect(Path(directory.name) / "index.sqlite3", check_same_thread=False)
        except BaseException as exc:
            directory.cleanup()
            if isinstance(exc, sqlite3.Error):
                raise OSError("Could not create judge response index") from exc
            raise
        self._cleanup = weakref.finalize(self, _close_judge_index, self._index, directory)
        try:
            self._index.execute("PRAGMA cache_size=-2048")
            self._index.execute("PRAGMA cache_spill=ON")
            self._index.execute("PRAGMA mmap_size=0")
            # Only the disposable index may lose buffered writes; JSONL fsync
            # remains authoritative and rebuilds every index after a crash.
            self._index.execute("PRAGMA synchronous=OFF")
            self._index.execute("CREATE TABLE replies (key TEXT PRIMARY KEY, offset INTEGER, length INTEGER, sha256 TEXT)")
            if not self.path.exists():
                return
            complete_bytes = 0
            torn_tail = False
            with self._index, self.path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                self._identity = (stat.st_dev, stat.st_ino)
                for number, line in enumerate(handle, 1):
                    if not line.endswith(b"\n"):
                        torn_tail = True
                        break
                    if line.decode("utf-8").strip():
                        entry = json.loads(line.decode("utf-8"))
                        if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                            raise ValueError(f"{self.path}:{number}: invalid judge response entry")
                        self._index_entry(entry, complete_bytes, line)
                    complete_bytes += len(line)
            # Validate ALL complete lines before repairing a crash-truncated tail.
            # A complete line can survive a process crash that interrupted its
            # writer before fsync. Make recovered evidence durable before use.
            with self.path.open("r+b") as handle:
                if torn_tail:
                    handle.truncate(complete_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.path.parent)
            self._indexed_end = complete_bytes
        except BaseException as exc:
            self._cleanup()
            if isinstance(exc, sqlite3.Error):
                raise OSError("Could not build judge response index") from exc
            raise

    @property
    def entries(self):
        """Compatibility view; iterating it does not materialize reply payloads."""
        return self

    def __len__(self):
        with self._lock:
            return self._index.execute("SELECT count(*) FROM replies").fetchone()[0]

    def __iter__(self):
        with self._lock:
            cursor = self._index.execute("SELECT key FROM replies")
            try:
                for row in cursor:
                    yield row[0]
            finally:
                cursor.close()

    def __getitem__(self, key):
        entry = self.get(key)
        if entry is None:
            raise KeyError(key)
        return entry

    def _index_entry(self, entry, offset, line):
        if entry.get("valid") is True:
            self._index.execute("INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?)",
                                (entry["key"], offset, len(line), hashlib.sha256(line).hexdigest()))

    def _check_available(self):
        if self._write_error is not None:
            raise OSError("Judge response cache has a failed write; restart evaluation") from self._write_error
        if self._identity is not None:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self._identity or stat.st_size < self._indexed_end:
                raise OSError("Judge response audit changed; restart evaluation")

    def get(self, key, default=None):
        """Load one valid reply on demand; index/read errors never become misses."""
        with self._lock:
            try:
                self._check_available()
                row = self._index.execute("SELECT offset, length, sha256 FROM replies WHERE key=?", (key,)).fetchone()
                if row is None:
                    return default
                offset, length, checksum = row
                with self.path.open("rb") as handle:
                    size = os.fstat(handle.fileno()).st_size
                    if (type(offset) is not int or type(length) is not int or offset < 0
                            or length <= 0 or offset + length > size):
                        raise OSError("Judge response index offset is outside the audit")
                    handle.seek(offset)
                    # A damaged length must not cause a whole-file allocation.
                    line = handle.readline(length)
                if len(line) != length or hashlib.sha256(line).hexdigest() != checksum:
                    raise OSError("Judge response audit/index mismatch; restart evaluation")
                entry = json.loads(line.decode("utf-8"))
                if not isinstance(entry, dict) or entry.get("key") != key or entry.get("valid") is not True:
                    raise OSError("Judge response index refers to an invalid reply")
                return {field: entry[field] for field in ("content", "model_response", "elapsed_seconds") if field in entry}
            except (OSError, sqlite3.Error, ValueError) as exc:
                self._write_error = exc
                raise OSError("Could not read judge response cache; restart evaluation") from exc

    def add(self, entry):
        if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
            raise ValueError("Judge response entry must have a string key")
        line = (json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with self._lock:
            try:
                self._check_available()
                with self.path.open("ab") as handle:
                    offset = handle.tell()
                    if handle.write(line) != len(line):
                        raise OSError("Incomplete judge response append")
                    handle.flush()
                    os.fsync(handle.fileno())
                    stat = os.fstat(handle.fileno())
                _fsync_directory(self.path.parent)
                # Publication follows durable audit persistence. An index failure
                # poisons this owner; restart rebuilds and reuses the saved reply.
                with self._index:
                    self._index_entry(entry, offset, line)
                self._identity = (stat.st_dev, stat.st_ino)
                self._indexed_end = offset + len(line)
            except BaseException as exc:
                self._write_error = exc
                if isinstance(exc, (OSError, sqlite3.Error)):
                    raise OSError("Could not persist/index judge response; restart evaluation") from exc
                raise


class CorrectnessJudges:
    """Preliminary single-model midterm metric: BioLORD diagnoses, one local Plan judge.

    complete(model_id, messages) sends one judge request and returns the
    provider response; only plan_judge is ever asked. Valid judgements are
    reused by request. Failed calls, malformed or refused replies, and
    similarity failures are saved in run.json errors (once per patient,
    request, and message); judge requests are asked again when evaluation resumes.
    """

    def __init__(self, complete, model_ids, cache_path, *, plan_judge, similarity=None,
                 generator_models=None, judge_model_config=None, plan_execution="inline", redact=None):
        self.complete = complete
        self.model_ids = tuple(model_ids)
        if not isinstance(plan_judge, str) or not plan_judge.strip():
            raise ValueError("plan_judge must be a nonempty registry alias")
        if plan_execution not in ("inline", "deferred"):
            raise ValueError("plan_execution must be inline or deferred")
        if redact is not None and not callable(redact):
            raise ValueError("redact must be callable or None")
        self.plan_judge = plan_judge
        self.plan_execution = plan_execution
        self._redact = redact
        self.generator_models = None if generator_models is None else tuple(generator_models)
        self.judge_model_config = judge_model_config
        self.similarity = similarity if similarity is not None else BioLordSimilarity()
        self._lock = RLock()
        self._request_locks = {}
        self._cache_path = cache_path
        # Generation/diagnosis-only runs must not read or repair an existing
        # Plan cache. The separate evaluation stage opens it when needed.
        self._cache = JudgeCache(cache_path) if plan_execution == "inline" else None
        self.calls = 0
        self.cache_hits = 0
        self._request_timing = {"attempted": 0, "failed": 0, "interrupted": 0, "invalid": 0,
                                "total_seconds": 0.0, "max_seconds": 0.0}

    def request_timing(self):
        """Snapshot actual provider work; cached replies never count as model time.

        Durations include provider-wrapper retries, but exclude cache waiting,
        parsing and persistence. Concurrent durations overlap and must not be
        interpreted as stage wall time.
        """
        with self._lock:
            return dict(self._request_timing)

    @property
    def cache(self):
        """Lazily open the durable Plan cache, under one owner lock."""
        with self._lock:
            if self._cache is None:
                self._cache = JudgeCache(self._cache_path)
            return self._cache

    @contextmanager
    def _request_lock(self, key):
        """Coalesce equal requests without serializing unrelated model calls."""
        with self._lock:
            pending = self._request_locks.setdefault(key, [Lock(), 0])
            pending[1] += 1
        try:
            with pending[0]:
                yield
        finally:
            with self._lock:
                pending[1] -= 1
                if pending[1] == 0:
                    del self._request_locks[key]

    @staticmethod
    def validate_resume(metadata, root, *, plan_judge):
        """Reject changed scoring semantics before opening models, caches, or output files."""
        correctness = (metadata.get("evaluation") or {}).get("correctness")
        if correctness is None:
            return
        protocol = plan_evaluation_protocol(root, plan_judge=plan_judge)
        expected = {
            "diagnosis": {
                "model": BIOLORD_MODEL,
                "model_revision": BIOLORD_REVISION,
                "normalization": {"rule": DIAGNOSIS_NORMALIZATION_RULE,
                                  "abbreviations": dict(DIAGNOSIS_ABBREVIATIONS)},
            },
            "plan": protocol["plan"],
            "rank_weighting": protocol["rank_weighting"],
        }
        for group, settings in expected.items():
            saved = correctness.get(group) if isinstance(correctness, dict) else None
            for key, value in settings.items():
                if not isinstance(saved, dict) or saved.get(key) != value:
                    raise ValueError(f"Cannot resume: correctness.{group}.{key} changed since the run")

    def metadata(self, root):
        return {
            "status": METRIC_STATUS,
            "method": ("preliminary single-model midterm metric: BioLORD-2023 clinical concept similarity "
                       "score for diagnoses (not an accuracy or probability); one local judge model for Plans"),
            "selected_models": list(self.model_ids),
            "hidden_from_judges": ["generating_model", "evidence_variant", "rank"],
            "diagnosis": {
                "method": ("BioLORD-2023 clinical concept similarity score: embedding cosine clipped to 0-1 "
                           "(no LLM judge); not an accuracy or probability"),
                "model": BIOLORD_MODEL,
                "model_revision": BIOLORD_REVISION,
                "compared": ["labeled condition (underscores replaced by spaces)", "generated diagnosis"],
                "normalization": {
                    "rule": DIAGNOSIS_NORMALIZATION_RULE,
                    "abbreviations": dict(DIAGNOSIS_ABBREVIATIONS),
                },
                "score": "cosine similarity clipped to [0, 1]; the raw cosine is kept as cosine_similarity",
            },
            "plan": {
                "execution": self.plan_execution,
                "judge_loading": ("deferred" if self.plan_execution == "deferred" else
                                  "loaded" if self.plan_judge in self.model_ids else "unavailable"),
                "validation_status": PLAN_VALIDATION_STATUS,
                "validation_evidence": PLAN_VALIDATION_EVIDENCE,
                "reporting": ("excluded from reported aggregate results; per-prediction scores are kept "
                              "for review and are not an accuracy measure"
                              if PLAN_VALIDATION_STATUS == EXPERIMENTAL_UNVALIDATED else "reported"),
                "judge_model": self.plan_judge,
                "judge_model_config": self.judge_model_config,
                "judge_count": 1,
                "judge_available": self.plan_judge in self.model_ids,
                "generator_models": None if self.generator_models is None else list(self.generator_models),
                "self_judging": (None if self.generator_models is None else
                                 "the judge also generated Plans and judges its own outputs"
                                 if self.plan_judge in self.generator_models else
                                 "the judge is not a generating model"),
                "judge_temperature": JUDGE_TEMPERATURE,
                "response_format": "json_object unless the model's output_mode is prompt",
                "responses_file": JUDGE_FILE,
                "resumption": ("valid judgements are reused by request key; failed calls and malformed or "
                               "refused replies are kept in responses_file but requested again"),
                "rubric_file": repo_path(PLAN_RUBRIC, root),
                "rubric_sha256": hashlib.sha256(PLAN_RUBRIC.read_bytes()).hexdigest(),
                "credit_values": list(CREDIT_VALUES),
                "deduction_per_unsupported_extra": DEDUCTION_PER_UNSUPPORTED_EXTRA,
                "implementation": ("one scalar 0/0.5/1 call per reference action; count unsupported extras "
                                   "unless all reference credits are valid zeros; deduction = 0.5 x that count"),
                "zero_reference_coverage": ("scores 0 without an extras call; unsupported_extra_count and "
                                            "deduction stay null because they were not measured"),
                "implementation_reason": ("a single call returning the full credits array, with the exact "
                                          "length stated and a JSON example of that length, returned valid "
                                          "arrays but local judges copied the example credits instead of "
                                          "grading"),
                "strength_mismatch_items": "not collected (they never change credit)",
                "formula": "max(0, (sum of action credits - 0.5 x unsupported extras) / number of reference actions)",
                "actions": "generated actions stay separate, numbered in order",
                "shown_to_judge": {
                    "action_call": ["labeled_condition", "subjective", "objective",
                                    "one_reference_action", "generated_actions"],
                    "extras_call": ["labeled_condition", "subjective", "objective",
                                    "reference_actions", "generated_actions"],
                },
                "unavailable_references": ['["none recorded"]', "missing or empty plan",
                                           f"any action containing {RECORD_ARTIFACT_MARKER!r} "
                                           "(any case or spacing)"],
                "empty_generated_plan": "scores 0 without judge calls when a reference exists",
                "action_prompt_template": PLAN_ACTION_JUDGE_TEMPLATE,
                "action_prompt_template_sha256": hashlib.sha256(PLAN_ACTION_JUDGE_TEMPLATE.encode()).hexdigest(),
                "extras_prompt_template": PLAN_EXTRAS_JUDGE_TEMPLATE,
                "extras_prompt_template_sha256": hashlib.sha256(PLAN_EXTRAS_JUDGE_TEMPLATE.encode()).hexdigest(),
            },
            "rank_weighting": {
                "formula": "1 / rank",
                "weights": {str(rank): rank_weight(rank) for rank in (1, 2, 3)},
                "applied_to": ["weighted diagnosis score = clinical concept similarity score / rank",
                               "weighted plan score = plan score / rank"],
                "reported_aggregate": ("rank-weighted mean = average of score / rank over scored ranks 1-3; "
                                       "not normalized, maximum (1 + 1/2 + 1/3) / 3 = 0.6111"),
                "combined_score": None,
            },
        }

    def _ask(self, judge, task, messages, parse):
        """Return one judge record, reusing a saved valid judgement for this request."""
        key = request_key(judge, messages)
        with self._request_lock(key):
            record = self._redacted(self._ask_locked(key, judge, task, messages, parse))
            record["request_key"] = key
            return record

    def _redacted(self, value):
        """Sanitize provider-controlled text before it enters result/cache artifacts."""
        if self._redact is None:
            return value
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {key: self._redacted(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redacted(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redacted(item) for item in value)
        return value

    def _ask_locked(self, key, judge, task, messages, parse):
        record = {"judge_model": judge, "status": "pending", "request_key": key,
                  "raw_response": None, "model_response": None, "error": None}
        try:
            entry = self.cache.get(key)
        except OSError as exc:
            raise RunMetadataError("Could not read judge response cache") from exc
        try:
            parsed = parse(entry["content"]) if entry is not None else None
        except JudgeResponseError:
            parsed = None  # Saved under an older parser; request it again.
        if parsed is not None:
            with self._lock:
                self.cache_hits += 1
            record.update(status="scored", raw_response=entry["content"],
                          model_response=entry["model_response"], **parsed)
            if "elapsed_seconds" in entry:
                # This is the original request duration, not work done on resume.
                record["elapsed_seconds"] = entry["elapsed_seconds"]
            return record
        label = f"{task} judge {judge} (request {key[:12]})"
        started = time.monotonic()
        outcome = None
        try:
            try:
                response = self.complete(judge, messages)
            finally:
                # Stop the provider timer before parsing or saving errors.
                elapsed = time.monotonic() - started
                record["elapsed_seconds"] = elapsed
        except Exception as exc:
            outcome = "failed"
            record.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            self._record(JudgeCallError(f"{label} failed: {type(exc).__name__}: {exc}"), exc)
            return record
        except BaseException as exc:
            outcome = "interrupted"
            record.update(status="interrupted", error={"type": type(exc).__name__,
                                                        "message": "Judge call interrupted"})
            raise
        finally:
            with self._lock:
                self._request_timing["attempted"] += 1
                if outcome is not None:
                    self._request_timing[outcome] += 1
                self._request_timing["total_seconds"] += elapsed
                self._request_timing["max_seconds"] = max(self._request_timing["max_seconds"], elapsed)
        with self._lock:
            self.calls += 1
        choices = getattr(response, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        content, refusal = getattr(message, "content", None), getattr(message, "refusal", None)
        entry = {"key": key, "task": task, "judge_model": judge,
                 "recorded_at": pacific_now().isoformat(), "messages": messages,
                 "content": content if isinstance(content, str) else None,
                 "refusal": refusal if isinstance(refusal, str) else None,
                 "elapsed_seconds": elapsed,
                 "model_response": model_response_metadata(response), "valid": False, "error": None}
        record.update(raw_response=entry["content"], model_response=entry["model_response"])
        try:
            if entry["refusal"] or (refusal is not None and not isinstance(refusal, str)):
                raise JudgeResponseError("Judge refused the request")
            parsed = parse(entry["content"])
        except JudgeResponseError as exc:
            with self._lock:
                self._request_timing["invalid"] += 1
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self._save_response(entry)
            record.update(status="invalid_response", error=entry["error"])
            self._record(JudgeResponseError(f"{label} replied invalidly: {exc}"), exc)
            return record
        entry["valid"] = True
        self._save_response(entry)
        record.update(status="scored", **parsed)
        return record

    def _save_response(self, entry):
        """Stop the run if a completed judge response cannot be saved durably."""
        try:
            sanitized = self._redacted(entry)
            # Request identity always describes the actual request, even when
            # stored prompt/provider text has had a credential removed.
            sanitized["key"] = entry["key"]
            self.cache.add(sanitized)
        except (OSError, TypeError, ValueError) as exc:
            raise RunMetadataError("Could not save judge response") from exc

    @staticmethod
    def _record(error, cause):
        """Save a scoring failure under the active patient and scoring stage, once."""
        error.__cause__ = cause
        record_error(error, unique=True)

    @contextmanager
    def _keep_interrupted(self, entry, judges):
        """On interruption, keep the judge records obtained so far, then re-raise."""
        try:
            yield
        except BaseException as exc:
            if not isinstance(exc, Exception):
                entry.update(status="interrupted", judges=judges,
                             error={"type": type(exc).__name__, "message": "Judging interrupted"})
            raise

    def score_diagnosis(self, entry):
        """Fill a pending diagnosis_entry in place with the BioLORD clinical concept similarity score and its weight."""
        if entry["status"] != "pending":
            return entry
        try:
            cosine = self.similarity.cosine(entry["normalized_reference"], entry["normalized_generated"])
        except Exception as exc:
            entry.update(status="error", reason="similarity_failed",
                         error={"type": type(exc).__name__, "message": str(exc)})
            self._record(SimilarityError(f"BioLORD diagnosis similarity failed: {type(exc).__name__}: {exc}"),
                         exc)
            return entry
        score = diagnosis_score(cosine)
        entry.update(status="scored", cosine_similarity=cosine, score=score,
                     weighted_score=score * entry["rank_weight"])
        return entry

    def judge_plan(self, entry, condition, patient):
        """Judge each reference action, then extras unless valid zero credits fix the score at zero.

        All action calls are attempted so valid replies are saved even when another fails.
        """
        if entry["status"] != "pending":
            return entry
        if self.plan_execution == "deferred":
            entry["reason"] = "deferred"
            return entry
        if self.plan_judge not in self.model_ids:
            entry.update(status="unavailable", reason="judge_unavailable")
            return entry
        reference, generated = entry["reference_actions"], entry["generated_actions"]
        context = (condition, patient["subjective"], patient["objective"])
        judges = []
        with self._keep_interrupted(entry, judges):
            for number, action in enumerate(reference, 1):
                record = self._ask(self.plan_judge, "plan_action",
                                   plan_action_messages(*context, action, generated), parse_action_judgement)
                judges.append({"call": "action", "reference_item": number, **record})
            if all(record["status"] == "scored" and record["credit"] == 0 for record in judges):
                # Any nonnegative extras deduction leaves the floored score at zero.
                entry.update(status="scored", reason="zero_reference_coverage", score=0.0, weighted_score=0.0,
                             credits=[record["credit"] for record in judges], judges=judges,
                             unsupported_extra_count=None, deduction=None)
                return entry
            record = self._ask(self.plan_judge, "plan_extras",
                               plan_extras_messages(*context, reference, generated),
                               parse_extras_judgement)
            judges.append({"call": "extras", "reference_item": None, **record})
        entry["judges"] = judges
        if any(record["status"] != "scored" for record in judges):
            entry.update(status="error", reason="judge_failed", error={
                "type": "JudgeFailed", "message": "The Plan judge did not return every valid judgement"})
            return entry
        credits = [record["credit"] for record in judges[:-1]]
        extras = judges[-1]["unsupported_extra_count"]
        deduction = DEDUCTION_PER_UNSUPPORTED_EXTRA * extras
        score = plan_score(credits, deduction)
        entry.update(status="scored", score=score, weighted_score=score * entry["rank_weight"],
                     credits=credits, unsupported_extra_count=extras, deduction=deduction)
        return entry


def diagnosis_entry(rank, reference, generated, problem=None):
    """A pending BioLORD diagnosis entry, or a final one when scoring cannot apply."""
    entry = {"status": "pending", "method": "biolord_2023_cosine", "reason": None,
             "cosine_similarity": None, "score": None, "weighted_score": None,
             "rank_weight": rank_weight(rank), "reference": reference, "generated": generated,
             "normalized_reference": normalize_diagnosis(reference) if isinstance(reference, str) else None,
             "normalized_generated": normalize_diagnosis(generated) if isinstance(generated, str) else None,
             "error": None}
    if problem is not None:
        entry.update(status="unavailable", reason=problem)
    elif generated is None or not generated.strip():
        entry.update(status="unavailable", reason="missing_generation")
    return entry


def plan_entry(rank, reference, generated, problem=None):
    """A pending plan-correctness entry; unavailable references are never judged.

    An explicitly empty generated Plan scores 0 without judge calls when the
    reference is available, as the rubric specifies.
    """
    entry = {"status": "pending", "method": "single_local_judge", "validation": PLAN_VALIDATION_STATUS,
             "reason": None,
             "score": None, "weighted_score": None,
             "rank_weight": rank_weight(rank), "reference_actions": list(reference),
             "generated_actions": list(generated) if isinstance(generated, list) else None,
             "credits": None, "unsupported_extra_count": None, "deduction": None, "judges": [],
             "error": None}
    reason = unavailable_plan_reason(reference)
    if reason is not None:
        entry.update(status="unavailable", reason=reason)
    elif problem is not None:
        entry.update(status="unavailable", reason=problem)
    elif generated is None:
        entry.update(status="unavailable", reason="missing_generation")
    elif not generated:
        entry.update(status="scored", reason="empty_generated_plan", score=0.0, weighted_score=0.0,
                     credits=[0.0] * len(reference), unsupported_extra_count=0, deduction=0.0)
    return entry
