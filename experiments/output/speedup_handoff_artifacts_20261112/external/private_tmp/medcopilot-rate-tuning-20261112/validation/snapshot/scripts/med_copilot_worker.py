"""Isolated, persistent evidence worker speaking acknowledged JSON lines.

This is an internal subprocess entry point, not an experiment runner. Library
output goes to stderr; stdout is reserved for parent/worker protocol messages.
"""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from time import perf_counter

# Subprocesses start without the repository on sys.path. Credentials and physical
# CPU encoder initialization must precede retrieval's FAISS import on Apple.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "med_copilot/upstream"))
sys.path.insert(0, '/private/tmp/medcopilot-rate-tuning-20261112/validation')
PROTOCOL_STDOUT = sys.stdout

from dotenv import load_dotenv

load_dotenv('/Users/rasikaramanan/Documents/usc/by_semester/f26/csci566/project/synthea-prep/.env', override=False)

from scripts.med_copilot_config import ExperimentConfig
from scripts.med_copilot_run_file import RunMetadataError, redact_secrets
from scripts.med_copilot_worker_stream import RedactingStream


PARSER = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
PARSER.add_argument("config", type=Path)
PARSER.add_argument("--device", default="cpu")
PARSER.add_argument("--threads", type=int, default=2)
ARGS = PARSER.parse_args() if __name__ == "__main__" else None
BOOTSTRAP_ERROR = None
ENCODER = None
EXPERIMENT = None
# Install this before third-party imports create logging handlers. Retaining
# partial lines also prevents a key split across multiple writes from leaking.
DIAGNOSTICS = RedactingStream(sys.stderr, [os.environ.get(name) for name in (
    "OPENAI_API_KEY", "GRAPHRAG_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")])
sys.stderr = DIAGNOSTICS
try:
    with redirect_stdout(sys.stderr):
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if ARGS is not None:
            if ARGS.threads < 1:
                raise ValueError("Worker threads must be positive")
            torch.set_num_threads(ARGS.threads)
            EXPERIMENT = ExperimentConfig(ARGS.config)
            if EXPERIMENT.needs_patient_retrieval:
                ENCODER = SentenceTransformer(EXPERIMENT.patient_embedding_model, device=ARGS.device)

        # The encoder above is physically initialized before this import loads
        # FAISS. Each process owns its encoder, retriever, and reranker instances.
        from scripts.med_copilot_retrieval import CrossEncoderReranker, HybridRetriever
        from graphrag.cli.query import run_local_search
        from scripts.med_copilot_preparation import EvidencePreparer
except Exception as exc:
    BOOTSTRAP_ERROR = exc


def _redact(text):
    names = ("OPENAI_API_KEY", "GRAPHRAG_API_KEY") if EXPERIMENT is None else EXPERIMENT.required_credentials
    return redact_secrets(text, [os.environ.get(name) for name in (*names, "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")])


def _emit(value):
    PROTOCOL_STDOUT.write(_redact(json.dumps(value, ensure_ascii=False, allow_nan=False)) + "\n")
    PROTOCOL_STDOUT.flush()


def _acknowledged_event(kind, **payload):
    _emit({"type": kind, **payload})
    line = sys.stdin.readline()
    if not line:
        raise RunMetadataError("Parent exited before acknowledging the worker checkpoint")
    reply = json.loads(line)
    if not isinstance(reply, dict) or reply.get("type") != "ack" or reply.get("error"):
        raise RunMetadataError("Parent could not durably acknowledge the worker checkpoint")


def _checkpoint(result):
    _acknowledged_event("checkpoint", result=result)


def _error(error):
    _acknowledged_event("error", error=error)


def _reranker(model_name):
    # The inherited constructor chooses MPS automatically. Allocate its single
    # model explicitly on the requested device, retaining the project reranker.
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = CrossEncoder(model_name, device=ARGS.device)
    return reranker


from prep_profile import install
install(globals())

def main():
    if BOOTSTRAP_ERROR is not None:
        _emit({"type": "fatal", "error": {"type": type(BOOTSTRAP_ERROR).__name__,
                                           "message": str(BOOTSTRAP_ERROR)}})
        return 1
    with redirect_stdout(sys.stderr):
        preparer = EvidencePreparer(
            ARGS.config, device=ARGS.device, checkpoint=_checkpoint, error_callback=_error,
            encoder=ENCODER, retriever_factory=HybridRetriever, reranker_factory=_reranker,
            guideline_search=run_local_search,
        )
        metadata = preparer.metadata()
        metadata["cpu_threads"] = ARGS.threads
    _emit({"type": "ready", "metadata": metadata})
    for line in sys.stdin:
        command = json.loads(line)
        if command.get("type") == "shutdown":
            return 0
        if command.get("type") != "prepare":
            raise ValueError("Worker expected a prepare or shutdown command")
        started = perf_counter()
        with redirect_stdout(sys.stderr):
            result = preparer.prepare(command["patient"], command["result"])
        _emit({"type": "result", "result": result, "metadata": metadata,
               "errors": preparer.errors, "duration_seconds": perf_counter() - started})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _emit({"type": "fatal", "error": {"type": type(exc).__name__, "message": str(exc)}})
        raise SystemExit(1) from None
    finally:
        DIAGNOSTICS.finish()
