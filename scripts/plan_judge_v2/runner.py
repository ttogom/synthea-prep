"""Judge rank-1 Plans for selected variants of one results.jsonl (protocol 2).

The model is loaded once and only if some variant has pending calls. Each
variant is judged under its own lease, with rows appended and fsynced after
every batch, so an interrupted run resumes where it stopped.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

from scripts.plan_judge_v2 import protocol
from scripts.plan_judge_v2.backends import (
    Request, build_backend, chat_token_ids, load_tokenizer, parse_engine_args, redact,
)
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
           "prompt_tokens": None, "output_tokens": None,
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
              "raw_response": result.text, "finish_reason": result.finish_reason,
              "prompt_tokens": result.prompt_tokens, "output_tokens": result.output_tokens}
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


def token_summary(values):
    """count/mean/p50/p95/max of the known values (None entries are skipped)."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]  # noqa: E731
    return {"count": len(values), "mean": round(sum(values) / len(values), 1), "p50": pick(0.5),
            "p95": pick(0.95), "max": values[-1]}


def _timing(session, rows_this_session, durations):
    """Separate model load, first (warm-up) batch, and steady-state generation.

    durations are the unrounded generate() times, one per entry in session["batches"].
    """
    batches = session["batches"]
    generation = sum(durations)
    steady = batches[1:]
    steady_seconds = sum(durations[1:])
    output_tokens = sum(b["output_tokens"] or 0 for b in batches)
    finish = {}
    for row in rows_this_session:
        finish[row["finish_reason"]] = finish.get(row["finish_reason"], 0) + 1
    return {
        "model_load_seconds": session.get("model_load_seconds"),
        "generation_seconds": round(generation, 3),
        "generation_plans_per_second": round(session["calls"] / generation, 4) if generation else None,
        "first_batch_seconds": batches[0]["seconds"] if batches else None,
        "steady_state_plans_per_second": (round(sum(b["plans"] for b in steady) / steady_seconds, 4)
                                          if steady_seconds else None),
        "output_tokens_per_second": round(output_tokens / generation, 1) if generation else None,
        "wall_plans_per_second_including_load": (round(session["calls"] / session["elapsed_seconds"], 4)
                                                 if session.get("elapsed_seconds") else None),
        "prompt_tokens": token_summary(r["prompt_tokens"] for r in rows_this_session),
        "output_tokens": token_summary(r["output_tokens"] for r in rows_this_session),
        "output_tokens_at_limit": sum(1 for r in rows_this_session
                                      if r["output_tokens"] is not None
                                      and r["output_tokens"] >= protocol.DECODING["max_new_tokens"]),
        "finish_reasons": finish,
    }


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


def sample_order(item, seed):
    """Stable pseudo-random rank of a Plan; the same seed always selects the same subset."""
    return hashlib.sha256(f"{seed}\0{item.patient_id}\0{item.generator_model}".encode()).hexdigest()


def judge_variant(variant, items, patients, *, inputs, selection, backend, lazy, out_root, writer_id,
                  owner=None, batch_size=32, max_attempts=3, limit=None, write_csv=False,
                  takeover_stale=False, lease_options=None, log=print, sample=None, sample_seed="benchmark"):
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
                   "sample": None if sample is None else {"size": sample, "seed": sample_seed},
                   "repaired_tail_bytes": store.repaired_bytes, "calls": 0, "elapsed_seconds": None,
                   "runtime": None, "counts": None, "batches": [], "timing": None}
        called, durations = [], []
        manifest["sessions"].append(session)
        atomic_write_json(directory / MANIFEST, manifest)
        started = time.monotonic()
        try:
            no_call = [_no_call_row(manifest, writer_id, item) for item in items
                       if item.status != "pending" and item.key not in final]
            lease.verify()
            store.append(no_call)
            rows.extend(no_call)
            if sample is not None:
                # A fixed pseudo-random subset of all pending Plans (finished ones keep their place, so a
                # resumed benchmark completes the same subset instead of drawing new Plans).
                chosen = {item.key for item in sorted((i for i in items if i.status == "pending"),
                                                       key=lambda i: sample_order(i, sample_seed))[:sample]}
            pending = [item for item in sorted(items, key=lambda i: i.key)
                       if item.status == "pending" and item.key not in final
                       and attempts.get(item.key, 0) < max_attempts
                       and (sample is None or item.key in chosen)]
            if limit is not None:
                pending = pending[:limit]
            work = [(item, _messages(item, patients)) for item in pending]
            # Similar lengths per batch reduce padding (transformers) and scheduler churn (vLLM).
            work.sort(key=lambda pair: len(pair[1][0]["content"]), reverse=True)
            log(f"[{variant}] {len(items)} rank-1 Plans: {len(final)} already final, "
                f"{len(no_call)} recorded without a call, {len(work)} to judge")
            if work:
                was_loaded = lazy.loaded
                model = lazy.get()
                session["runtime"] = model.runtime()
                if not was_loaded and lazy.load_seconds is not None:
                    session["model_load_seconds"] = round(lazy.load_seconds, 3)
                    log(f"[{variant}] model ready in {lazy.load_seconds:.1f}s (download, weight load, engine "
                        "profiling and CUDA graph capture; excluded from generation throughput)")
            for start in range(0, len(work), batch_size):
                chunk = work[start:start + batch_size]
                requests = [Request(messages, protocol.response_schema(len(item.reference_actions),
                                                                       len(item.generated_actions)))
                            for item, messages in chunk]
                tick = time.monotonic()
                results = model.generate(requests)
                seconds = max(time.monotonic() - tick, 1e-9)
                batch = [_call_row(manifest, writer_id, item, messages, result, attempts.get(item.key, 0) + 1)
                         for (item, messages), result in zip(chunk, results)]
                lease.verify()  # Never append after another writer took the variant over.
                store.append(batch)
                rows.extend(batch)
                called.extend(batch)
                durations.append(seconds)
                session["calls"] += len(batch)
                output_tokens = sum(r["output_tokens"] or 0 for r in batch)
                session["batches"].append({
                    "plans": len(batch), "seconds": round(seconds, 3),
                    "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in batch) or None,
                    "output_tokens": output_tokens or None})
                done = start + len(chunk)
                label = "first batch, includes warm-up" if len(session["batches"]) == 1 else "steady state"
                log(f"[{variant}] batch {len(session['batches'])} ({label}): {len(batch)} Plans in {seconds:.1f}s "
                    f"= {len(batch) / seconds:.2f} Plans/s, {output_tokens / seconds:.0f} output tokens/s; "
                    f"judged {done}/{len(work)}; invalid {sum(r['status'] == 'invalid_response' for r in batch)}, "
                    f"errors {sum(r['status'] == 'error' for r in batch)}")
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
            if session["calls"]:
                session["timing"] = timing = _timing(session, called, durations)
                rate = lambda value: "n/a" if value is None else f"{value:.3f} Plans/s"  # noqa: E731
                load = timing["model_load_seconds"]
                log(f"[{variant}] timing: model load {'n/a (already loaded)' if load is None else f'{load:.1f}s'}; "
                    f"generation {timing['generation_seconds']:.1f}s for {session['calls']} Plans = "
                    f"{rate(timing['generation_plans_per_second'])}; steady state after batch 1: "
                    f"{rate(timing['steady_state_plans_per_second']) if len(session['batches']) > 1 else 'n/a (one batch)'}; "
                    f"wall {session['elapsed_seconds']:.1f}s including load and I/O")
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
                        help='JSON passed to the chat template; leave {} for Qwen3-4B-Instruct-2507 (non-thinking only)')
    parser.add_argument("--engine-arg", action="append", default=[], metavar="KEY=JSON",
                        help="Extra vLLM/transformers load argument; part of the judge identity")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per Plan for invalid/error replies")
    parser.add_argument("--limit", type=int, help="Judge at most N pending Plans per variant (smoke tests)")
    parser.add_argument("--sample", type=int,
                        help="Benchmark: judge a fixed pseudo-random subset of N pending Plans per variant "
                             "(representative across conditions, unlike --limit, which takes the first N by id)")
    parser.add_argument("--sample-seed", default="benchmark", help="Seed for --sample (default: benchmark)")
    parser.add_argument("--csv", action="store_true", help="Also write judgments.csv per variant")
    parser.add_argument("--owner", help="Name shown to teammates if they hit this run's lease")
    parser.add_argument("--takeover-stale", action="store_true",
                        help="Take over a lease whose heartbeat is older than 15 minutes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and report pending work; no model, network, or output writes")
    parser.add_argument("--count-tokens", action="store_true",
                        help="With --dry-run and --model: load only the judge tokenizer, report prompt-token "
                             "counts, and exit 1 if any prompt exceeds --max-model-len minus the output budget")
    parser.add_argument("--skip-final-hash", action="store_true",
                        help="Skip re-hashing results.jsonl at the end (size/mtime are still checked)")
    return parser


def count_prompt_tokens(scan, patients, tokenizer, chat_kwargs, max_model_len, log):
    """Exact chat-template token counts for every pending prompt; returns how many do not fit."""
    budget = max_model_len - protocol.DECODING["max_new_tokens"]
    over = 0
    for variant, items in scan.items.items():
        lengths = []
        for item in items:
            if item.status == "pending":
                length = len(chat_token_ids(tokenizer, _messages(item, patients), chat_kwargs))
                lengths.append(length)
                if length > budget:
                    over += 1
                    log(f"[{variant}] prompt too long: {item.patient_id} / {item.generator_model} has {length} "
                        f"tokens; budget {budget}")
        log(f"[{variant}] prompt tokens: {json.dumps(token_summary(lengths))}; budget {budget} "
            f"(max_model_len {max_model_len} - {protocol.DECODING['max_new_tokens']} output); "
            f"over budget: {sum(n > budget for n in lengths)}")
    return over


def run(argv=None, *, backend_factory=None, log=print, lease_options=None, tokenizer_factory=None):
    args = build_parser().parse_args(argv)
    for name in ("batch_size", "max_attempts", "max_model_len"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be nonnegative")
    if args.sample is not None and (args.sample < 1 or args.limit is not None):
        raise SystemExit("--sample must be positive and cannot be combined with --limit")
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
                 "patients_in_results": scan.patients,
                 # Part of the output identity: a resume with another sample size or seed is refused.
                 "sample": None if args.sample is None else {"size": args.sample, "seed": args.sample_seed}}
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
        if args.count_tokens:
            if not args.model:
                log("--count-tokens needs --model (the judge whose tokenizer and chat template are used)")
                return EXIT_FAILED
            try:
                tokenizer = (tokenizer_factory or load_tokenizer)(args.model, args.revision)
            except Exception as exc:  # noqa: BLE001 - e.g. gated tokenizer without access
                log(f"Cannot load the {args.model} tokenizer: {type(exc).__name__}: {redact(str(exc))}")
                return EXIT_FAILED
            if count_prompt_tokens(scan, patients, tokenizer, chat_kwargs, args.max_model_len, log):
                log("Some prompts do not fit; raise --max-model-len or shorten inputs before judging.")
                return EXIT_FAILED
        elif args.model:
            log("Prompt tokens not counted; add --count-tokens to check every prompt fits --max-model-len.")
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
                sample=args.sample, sample_seed=args.sample_seed,
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
