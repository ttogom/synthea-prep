"""Judge rank-1 Plans for selected variants of one results.jsonl (protocol 2).

The model is loaded once and only if some variant has pending calls. Each
variant is judged under its own lease, with rows appended and fsynced after
every batch, so an interrupted run resumes where it stopped.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time

from scripts.plan_judge_v2 import protocol
from scripts.plan_judge_v2.backends import Request, build_backend, parse_engine_args, redact
from scripts.plan_judge_v2.inputs import describe_input, file_signature, load_by_id, scan_results, sha256_file
from scripts.plan_judge_v2.store import (
    CSV_FILE, MANIFEST, MANIFEST_VERSION, ROW_VERSION, JudgmentStore, Lease, LeaseHeld, LeaseLost,
    OutputConflict, atomic_write_json, atomic_write_text, held_lease, latest_rows, new_writer_id,
    open_manifest, summarize, utc_now,
)


EXIT_OK, EXIT_FAILED, EXIT_INCOMPLETE, EXIT_LEASE, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
CSV_COLUMNS = ["variant", "run_id", "patient_id", "generator_model", "rank", "status", "reason", "reasons",
               "variant_status", "generation_status", "coverage", "harmful_count", "penalized", "credits", "matches", "harmful_actions",
               "attempt", "error_type", "error_message", "judge_model", "judge_revision", "protocol_version"]


def default_run_name(results_path, sha256):
    """<run directory>__<sha12> for a run's results.jsonl, else <file stem>__<sha12>."""
    path = Path(results_path)
    stem = path.parent.name if path.name == "results.jsonl" and path.parent.name else path.stem
    return f"{stem}__{sha256[:12]}"


def _check_paths(inputs, out_dir):
    out_dir = Path(out_dir).resolve()
    for path in inputs:
        resolved = Path(path).resolve()
        if resolved == out_dir or out_dir in resolved.parents:
            raise ValueError(f"Output directory must not contain the input file {path}")


def _row(manifest, writer_id, item, status, reason, **fields):
    row = {"row_version": ROW_VERSION, "run_id": manifest["run_id"], "writer_id": writer_id,
           "patient_id": item.patient_id, "generator_model": item.generator_model,
           "rank": protocol.JUDGED_RANK, "status": status, "reason": reason,
           "reasons": list(item.reasons), "source": item.source, "attempt": 0,
           "prompt_sha256": None, "raw_response": None, "finish_reason": None, "parsed": None,
           "credits": None, "coverage": None, "harmful_count": None, "penalized": None,
           "error": None, "judged_at": utc_now()}
    row.update(fields)
    return row


def _no_call_row(manifest, writer_id, item):
    if item.status == "unavailable":
        return _row(manifest, writer_id, item, "unavailable", item.reason)
    credits = [0.0] * len(item.reference_actions)
    return _row(manifest, writer_id, item, "scored", item.reason, credits=credits, coverage=0.0,
                harmful_count=0, penalized=0.0)


def _call_row(manifest, writer_id, item, messages, result, attempt):
    fields = {"attempt": attempt, "prompt_sha256": protocol.messages_sha256(messages),
              "raw_response": result.text, "finish_reason": result.finish_reason}
    if result.error is not None:
        return _row(manifest, writer_id, item, "error", "request_failed", error=result.error, **fields)
    try:
        parsed = protocol.parse_response(result.text, len(item.reference_actions), len(item.generated_actions))
    except protocol.JudgeResponseError as exc:
        reason = "truncated_response" if result.finish_reason == "length" else "invalid_response"
        return _row(manifest, writer_id, item, "invalid_response", reason,
                    error={"type": "JudgeResponseError", "message": str(exc)}, **fields)
    return _row(manifest, writer_id, item, "scored", None, parsed=parsed, **protocol.score(parsed), **fields)


def _messages(item, patients):
    patient = patients[item.patient_id]
    return protocol.render_messages(item.condition, patient["subjective"], patient["objective"],
                                    item.reference_actions, item.generated_actions)


def _counts(rows, items, max_attempts):
    final, attempts, duplicates = summarize(rows, max_attempts)
    counts = {"plans": len(items), "scored": 0, "scored_without_call": 0, "unavailable": 0,
              "unavailable_by_reason": {}, "failed_retryable": 0, "failed_exhausted": 0, "not_attempted": 0,
              "duplicate_final_rows": duplicates}
    for item in items:
        row = final.get(item.key)
        if row is not None:
            counts[row["status"]] += 1
            if row["status"] == "unavailable":
                by_reason = counts["unavailable_by_reason"]
                by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
            elif row["reason"] == "empty_generated_plan":
                counts["scored_without_call"] += 1
        elif item.key in attempts:
            counts["failed_exhausted" if attempts[item.key] >= max_attempts else "failed_retryable"] += 1
        else:
            counts["not_attempted"] += 1
    return counts


def export_csv(directory, manifest, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    judge = manifest["judge"]
    for key, row in sorted(latest_rows(rows).items()):
        parsed = row.get("parsed") or {}
        error = row.get("error") or {}
        source = row.get("source") or {}
        writer.writerow({
            "variant": manifest["variant"], "run_id": manifest["run_id"], "patient_id": row["patient_id"],
            "generator_model": row["generator_model"], "rank": row["rank"], "status": row["status"],
            "reason": row["reason"] or "", "reasons": ";".join(row.get("reasons") or []),
            "variant_status": source.get("variant_status") or "",
            "generation_status": source.get("generation_status") or "", "coverage": "" if row["coverage"] is None else row["coverage"],
            "harmful_count": "" if row["harmful_count"] is None else row["harmful_count"],
            "penalized": "" if row["penalized"] is None else row["penalized"],
            "credits": "" if row["credits"] is None else json.dumps(row["credits"]),
            "matches": json.dumps([i["match"] for i in parsed.get("reference_items", [])]) if parsed else "",
            "harmful_actions": json.dumps(parsed.get("harmful_actions", []), ensure_ascii=False) if parsed else "",
            "attempt": row["attempt"], "error_type": error.get("type", ""), "error_message": error.get("message", ""),
            "judge_model": judge["model_id"], "judge_revision": judge["revision_resolved"],
            "protocol_version": manifest["protocol"]["version"],
        })
    atomic_write_text(Path(directory) / CSV_FILE, buffer.getvalue())


class _LazyBackend:
    """Load model weights the first time a variant actually needs a call."""

    def __init__(self, backend):
        self.backend, self.loaded, self.load_seconds = backend, False, None

    def get(self):
        if not self.loaded:
            started = time.monotonic()
            self.backend.load()
            self.load_seconds = time.monotonic() - started
            self.loaded = True
        return self.backend


def judge_variant(variant, items, patients, *, inputs, selection, backend, lazy, out_root, writer_id,
                  owner=None, batch_size=32, max_attempts=3, limit=None, write_csv=False,
                  takeover_stale=False, lease_options=None, log=print):
    directory = Path(out_root) / variant
    directory.mkdir(parents=True, exist_ok=True)
    proposed = {"manifest_version": MANIFEST_VERSION, "protocol": protocol.protocol_identity(),
                "variant": variant, "inputs": inputs, "selection": selection, "judge": backend.identity(),
                "outputs": {"judgments": "judgments.jsonl", "csv": CSV_FILE}}
    lease = Lease(directory, writer_id, owner, takeover_stale=takeover_stale, **(lease_options or {}))
    with held_lease(lease):
        manifest = open_manifest(directory, proposed)
        store = JudgmentStore(directory, manifest["run_id"])
        rows = store.load()
        final, attempts, _ = summarize(rows, max_attempts)
        session = {"writer_id": writer_id, "owner": owner, "started_at": utc_now(), "finished_at": None,
                   "status": "running", "batch_size": batch_size, "max_attempts": max_attempts, "limit": limit,
                   "repaired_tail_bytes": store.repaired_bytes, "calls": 0, "elapsed_seconds": None,
                   "runtime": None, "counts": None}
        manifest["sessions"].append(session)
        atomic_write_json(directory / MANIFEST, manifest)
        started = time.monotonic()
        try:
            no_call = [_no_call_row(manifest, writer_id, item) for item in items
                       if item.status != "pending" and item.key not in final]
            lease.verify()
            store.append(no_call)
            rows.extend(no_call)
            pending = [item for item in sorted(items, key=lambda i: i.key)
                       if item.status == "pending" and item.key not in final
                       and attempts.get(item.key, 0) < max_attempts]
            if limit is not None:
                pending = pending[:limit]
            work = [(item, _messages(item, patients)) for item in pending]
            # Similar lengths per batch reduce padding (transformers) and scheduler churn (vLLM).
            work.sort(key=lambda pair: len(pair[1][0]["content"]), reverse=True)
            log(f"[{variant}] {len(items)} rank-1 Plans: {len(final)} already final, "
                f"{len(no_call)} recorded without a call, {len(work)} to judge")
            if work:
                model = lazy.get()
                session["runtime"] = model.runtime()
                if lazy.load_seconds is not None:
                    session["model_load_seconds"] = lazy.load_seconds
            for start in range(0, len(work), batch_size):
                chunk = work[start:start + batch_size]
                requests = [Request(messages, protocol.response_schema(len(item.reference_actions),
                                                                       len(item.generated_actions)))
                            for item, messages in chunk]
                results = model.generate(requests)
                batch = [_call_row(manifest, writer_id, item, messages, result, attempts.get(item.key, 0) + 1)
                         for (item, messages), result in zip(chunk, results)]
                lease.verify()  # Never append after another writer took the variant over.
                store.append(batch)
                rows.extend(batch)
                session["calls"] += len(batch)
                done = start + len(chunk)
                rate = done / max(time.monotonic() - started, 1e-9)
                log(f"[{variant}] judged {done}/{len(work)} ({rate:.2f} Plans/s); "
                    f"invalid {sum(r['status'] == 'invalid_response' for r in batch)}, "
                    f"errors {sum(r['status'] == 'error' for r in batch)} in this batch")
            session["status"] = "completed"
        except KeyboardInterrupt:
            session["status"] = "interrupted"
            raise
        except LeaseLost:
            session["status"] = "lease_lost"
            raise
        except BaseException as exc:
            session["status"] = "failed"
            session["error"] = {"type": type(exc).__name__, "message": redact(str(exc))[:2000]}
            raise
        finally:
            session["finished_at"] = utc_now()
            session["elapsed_seconds"] = round(time.monotonic() - started, 3)
            session["counts"] = _counts(rows, items, max_attempts)
            if session["status"] != "lease_lost":
                manifest["counts"] = session["counts"]
                atomic_write_json(directory / MANIFEST, manifest)
                if write_csv:
                    export_csv(directory, manifest, rows)
    return session["counts"]


def build_parser():
    parser = argparse.ArgumentParser(prog="judge_plans_v2.py judge", description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", required=True, type=Path, help="Base or agentic results.jsonl (read-only)")
    parser.add_argument("--patients", type=Path, default=Path("experiments/input/test_so.jsonl"),
                        help="Patient S/O JSONL used for generation")
    parser.add_argument("--labels", type=Path, default=Path("experiments/input/test_labels.jsonl"),
                        help="Gold label JSONL used for generation")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variant names to judge, or 'all' for every variant in the file")
    parser.add_argument("--generator-models", nargs="+", help="Only judge these final-model aliases")
    parser.add_argument("--out-dir", required=True, type=Path, help="Root for per-variant output directories")
    parser.add_argument("--run-name", help="Output subdirectory name (default: <run dir>__<results sha12>)")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--model", help="Hugging Face model id of the judge")
    parser.add_argument("--revision", help="Hub branch, tag, or commit; resolved to a commit and recorded")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, or float16 (T4 GPUs need float16)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM only")
    parser.add_argument("--chat-template-kwargs", default="{}",
                        help='JSON, e.g. {"enable_thinking": false} for hybrid-reasoning models')
    parser.add_argument("--engine-arg", action="append", default=[], metavar="KEY=JSON",
                        help="Extra vLLM/transformers load argument; part of the judge identity")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per Plan for invalid/error replies")
    parser.add_argument("--limit", type=int, help="Judge at most N pending Plans per variant (smoke tests)")
    parser.add_argument("--csv", action="store_true", help="Also write judgments.csv per variant")
    parser.add_argument("--owner", help="Name shown to teammates if they hit this run's lease")
    parser.add_argument("--takeover-stale", action="store_true",
                        help="Take over a lease whose heartbeat is older than 15 minutes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and report pending work; no model, network, or output writes")
    parser.add_argument("--skip-final-hash", action="store_true",
                        help="Skip re-hashing results.jsonl at the end (size/mtime are still checked)")
    return parser


def run(argv=None, *, backend_factory=None, log=print, lease_options=None):
    args = build_parser().parse_args(argv)
    for name in ("batch_size", "max_attempts", "max_model_len"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be nonnegative")
    try:
        chat_kwargs = json.loads(args.chat_template_kwargs)
        engine_args = parse_engine_args(args.engine_arg)
        if not isinstance(chat_kwargs, dict):
            raise ValueError("--chat-template-kwargs must be a JSON object")
        input_paths = [args.results, args.patients, args.labels]
        _check_paths(input_paths, args.out_dir)
        hashes = {"patients": sha256_file(args.patients), "labels": sha256_file(args.labels)}
        patients = load_by_id(args.patients, ("subjective", "objective"))
        labels = load_by_id(args.labels, ("condition", "plan"))
        scan = scan_results(args.results, labels, patients, args.variants, args.generator_models)
    except (OSError, ValueError) as exc:
        log(f"Cannot read inputs: {exc}")
        return EXIT_FAILED
    inputs = {"results": describe_input(args.results, scan.sha256),
              "patients": describe_input(args.patients, hashes["patients"]),
              "labels": describe_input(args.labels, hashes["labels"])}
    selection = {"rank": protocol.JUDGED_RANK,
                 "generator_models": sorted(args.generator_models) if args.generator_models else None,
                 "generator_models_expected": sorted(scan.generator_models),
                 "patients_in_results": scan.patients}
    out_root = args.out_dir / (args.run_name or default_run_name(args.results, scan.sha256))
    log(f"Protocol {protocol.PROTOCOL_NAME} v{protocol.PROTOCOL_VERSION}; results sha256 {scan.sha256[:12]}; "
        f"variants in file: {', '.join(scan.variants_found)}")
    log(f"Denominator per variant: {scan.patients} patients x {len(scan.generator_models)} generators "
        f"({', '.join(scan.generator_models)})")
    if scan.patients_not_in_results:
        log(f"Note: {scan.patients_not_in_results} patient(s) in {args.patients.name} have no results record; "
            "only patients in the results file are judged and counted.")
    if args.dry_run:
        for variant, items in scan.items.items():
            by_status, by_reason = {}, {}
            for item in items:
                by_status[item.status] = by_status.get(item.status, 0) + 1
                if item.reason:
                    by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
            log(f"[{variant}] {len(items)} rank-1 Plans by status: {json.dumps(by_status, sort_keys=True)}; "
                f"by reason: {json.dumps(by_reason, sort_keys=True)}")
        example = next((item for items in scan.items.values() for item in items if item.status == "pending"), None)
        if example is not None:
            log("Example prompt:\n" + _messages(example, patients)[0]["content"])
        log(f"Dry run: outputs would go to {out_root}")
        return EXIT_OK
    if not args.model:
        log("--model is required unless --dry-run is set")
        return EXIT_FAILED
    factory = backend_factory or build_backend
    try:
        options = {"revision": args.revision, "dtype": args.dtype, "max_model_len": args.max_model_len,
                   "chat_template_kwargs": chat_kwargs, "engine_args": engine_args}
        if args.backend == "vllm":
            options["gpu_memory_utilization"] = args.gpu_memory_utilization
        backend = factory(args.backend, args.model, **options)
    except Exception as exc:  # noqa: BLE001 - e.g. gated model without access, no network
        log(f"Cannot prepare judge {args.model}: {type(exc).__name__}: {redact(str(exc))}")
        return EXIT_FAILED
    lazy, writer_id, status = _LazyBackend(backend), new_writer_id(), EXIT_OK
    try:
        for variant, items in scan.items.items():
            counts = judge_variant(
                variant, items, patients, inputs=inputs, selection=selection, backend=backend, lazy=lazy,
                out_root=out_root, writer_id=writer_id, owner=args.owner, batch_size=args.batch_size,
                max_attempts=args.max_attempts, limit=args.limit, write_csv=args.csv,
                takeover_stale=args.takeover_stale, lease_options=lease_options, log=log)
            log(f"[{variant}] counts: {json.dumps(counts, sort_keys=True)}")
            if counts["failed_retryable"] or counts["failed_exhausted"] or counts["not_attempted"]:
                status = EXIT_INCOMPLETE
    except KeyboardInterrupt:
        log("Interrupted. Completed batches are saved; rerun the same command to resume.")
        return EXIT_INTERRUPTED
    except LeaseHeld as exc:
        log(str(exc))
        return EXIT_LEASE
    except (LeaseLost, OutputConflict) as exc:
        log(f"Stopped: {exc}")
        return EXIT_FAILED
    if file_signature(args.results) != scan.signature or (
            not args.skip_final_hash and sha256_file(args.results) != scan.sha256) or any(
            sha256_file(getattr(args, name)) != hashes[name] for name in ("patients", "labels")):
        log("An input file changed during judging; treat these outputs as invalid.")
        return EXIT_FAILED
    log(f"Inputs unchanged. Outputs: {out_root}")
    return status


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
