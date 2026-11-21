"""Pilot comparison of candidate judges on a small manually reviewed case set.

PILOT, not definitive validation: about 48 balanced cases support preliminary
model selection only. Run one model per process (the notebook does this) so GPU
memory is released between candidates, then `compare` the saved reports.

Case file: JSONL, one object per case:
  case_id, category, condition, subjective, objective, reference_actions,
  generated_actions, source {kind, patient_id, generator_model, variant, run},
  gold {matches: ["full"|"partial"|"none" per reference action], harmful: [generated numbers]},
  review {status: "draft"|"reviewed"|"spot_checked", reviewer, notes}
Gold labels follow protocol 2's rubric. Unreviewed (draft) labels are refused
unless --allow-draft is given, and such reports are marked as draft.
"""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import time

from scripts.plan_judge_v2 import protocol
from scripts.plan_judge_v2.backends import Request, build_backend, parse_engine_args
from scripts.plan_judge_v2.inputs import sha256_file
from scripts.plan_judge_v2.store import atomic_write_json, atomic_write_text, utc_now


CATEGORIES = ("exact", "valid_substitute", "more_specific", "less_specific_or_strength", "conditional",
              "off_target", "placeholder", "correct_plus_harmful")
# Categories whose gold coverage should rank above the "wrong" ones.
CORRECT = ("exact", "valid_substitute", "more_specific")
WRONG = ("off_target", "placeholder")
REVIEWED = ("reviewed", "spot_checked")
LEVEL = {"none": 0, "partial": 1, "full": 2}


def load_cases(path):
    cases, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            where = f"{path}:{number}"
            for key in ("case_id", "category", "condition", "subjective", "objective",
                        "reference_actions", "generated_actions", "gold", "review"):
                if key not in case:
                    raise ValueError(f"{where}: missing {key}")
            if case["case_id"] in seen:
                raise ValueError(f"{where}: duplicate case_id {case['case_id']}")
            seen.add(case["case_id"])
            if case["category"] not in CATEGORIES:
                raise ValueError(f"{where}: unknown category {case['category']}")
            gold, n, m = case["gold"], len(case["reference_actions"]), len(case["generated_actions"])
            if n < 1 or m < 1:
                raise ValueError(f"{where}: reference and generated actions must be nonempty")
            if len(gold.get("matches", [])) != n or any(g not in protocol.MATCH_CREDITS for g in gold["matches"]):
                raise ValueError(f"{where}: gold.matches needs one full/partial/none per reference action")
            if any(type(h) is not int or not 1 <= h <= m for h in gold.get("harmful", [])):
                raise ValueError(f"{where}: gold.harmful must list generated item numbers")
            cases.append(case)
    return cases


def review_status(cases):
    statuses = {case["review"].get("status") for case in cases}
    return "reviewed" if statuses <= set(REVIEWED) else "draft"


def quadratic_kappa(pairs, levels=3):
    """Cohen's quadratic-weighted kappa for ordinal labels 0..levels-1."""
    if not pairs:
        return None
    observed = [[0] * levels for _ in range(levels)]
    for gold, predicted in pairs:
        observed[gold][predicted] += 1
    total = len(pairs)
    rows = [sum(row) for row in observed]
    cols = [sum(observed[i][j] for i in range(levels)) for j in range(levels)]
    weight = lambda i, j: (i - j) ** 2 / (levels - 1) ** 2  # noqa: E731
    disagreement = sum(weight(i, j) * observed[i][j] for i in range(levels) for j in range(levels)) / total
    expected = sum(weight(i, j) * rows[i] * cols[j] for i in range(levels) for j in range(levels)) / total ** 2
    return None if expected == 0 else 1 - disagreement / expected


def _mean(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def metrics(cases, outcomes):
    """Agreement, reliability, separation, harmful-flag accuracy, and self-preference."""
    valid = [(case, out) for case, out in zip(cases, outcomes) if out["status"] == "scored"]
    pairs = [(LEVEL[g], LEVEL[p["match"]]) for case, out in valid
             for g, p in zip(case["gold"]["matches"], out["parsed"]["reference_items"])]
    gold_coverage = lambda case: protocol.score({"reference_items": [{"match": g} for g in case["gold"]["matches"]],  # noqa: E731
                                                 "harmful_actions": []})["coverage"]
    by_category = {}
    for case, out in valid:
        by_category.setdefault(case["category"], []).append(out["coverage"])
    category_means = {name: _mean(values) for name, values in sorted(by_category.items())}
    correct = _mean([v for name in CORRECT for v in by_category.get(name, [])])
    wrong = [category_means[name] for name in WRONG if name in category_means]
    flagged = gold_flags = true_flags = 0
    clean_cases = clean_flagged = 0
    for case, out in valid:
        predicted = {h["generated"] for h in out["parsed"]["harmful_actions"]}
        gold = set(case["gold"].get("harmful", []))
        flagged += len(predicted)
        gold_flags += len(gold)
        true_flags += len(predicted & gold)
        if not gold:
            clean_cases += 1
            clean_flagged += bool(predicted)
    bias = {}
    for case, out in valid:
        family = _family(case.get("source", {}).get("generator_model"))
        bias.setdefault(family, []).append(out["coverage"] - gold_coverage(case))
    return {
        "cases": len(cases),
        "valid_json_rate": len(valid) / len(cases) if cases else None,
        "invalid_responses": sum(o["status"] == "invalid_response" for o in outcomes),
        "request_errors": sum(o["status"] == "error" for o in outcomes),
        "item_exact_agreement": (sum(g == p for g, p in pairs) / len(pairs)) if pairs else None,
        "item_quadratic_kappa": quadratic_kappa(pairs),
        "coverage_mae": _mean([abs(out["coverage"] - gold_coverage(case)) for case, out in valid]),
        "coverage_by_category": category_means,
        "separation_correct_minus_worst_wrong": (correct - max(wrong)) if correct is not None and wrong else None,
        "harmful_precision": true_flags / flagged if flagged else None,
        "harmful_recall": true_flags / gold_flags if gold_flags else None,
        "harmful_false_positive_case_rate": clean_flagged / clean_cases if clean_cases else None,
        "coverage_bias_by_generator_family": {name: _mean(values) for name, values in sorted(bias.items())},
    }


def _family(alias):
    alias = (alias or "").lower()
    for family in ("gemma", "llama", "ministral", "qwen", "gpt"):
        if family in alias:
            return family
    return "other" if alias else "constructed"


def evaluate(cases, backend, *, batch_size=16):
    """One call per case; no retries, so reliability reflects first-attempt behaviour."""
    started = time.monotonic()
    backend.load()
    load_seconds = time.monotonic() - started
    outcomes, generate_seconds = [], 0.0
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        messages = [protocol.render_messages(c["condition"], c["subjective"], c["objective"],
                                             c["reference_actions"], c["generated_actions"]) for c in chunk]
        requests = [Request(m, protocol.response_schema(len(c["reference_actions"]), len(c["generated_actions"])))
                    for m, c in zip(messages, chunk)]
        tick = time.monotonic()
        results = backend.generate(requests)
        generate_seconds += time.monotonic() - tick
        for case, result in zip(chunk, results):
            outcome = {"case_id": case["case_id"], "raw_response": result.text,
                       "finish_reason": result.finish_reason, "status": None, "parsed": None,
                       "coverage": None, "harmful_count": None, "error": result.error}
            if result.error is None:
                try:
                    parsed = protocol.parse_response(result.text, len(case["reference_actions"]),
                                                     len(case["generated_actions"]))
                except protocol.JudgeResponseError as exc:
                    outcome.update(status="invalid_response", error={"type": "JudgeResponseError",
                                                                    "message": str(exc)})
                else:
                    scores = protocol.score(parsed)
                    outcome.update(status="scored", parsed=parsed, coverage=scores["coverage"],
                                   harmful_count=scores["harmful_count"])
            else:
                outcome["status"] = "error"
            outcomes.append(outcome)
    timing = {"load_seconds": round(load_seconds, 3), "generate_seconds": round(generate_seconds, 3),
              "plans_per_second": len(cases) / generate_seconds if generate_seconds else None}
    return outcomes, timing


def _slug(model_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def validate(args, *, backend_factory=None, log=print):
    cases = load_cases(args.cases)
    status = review_status(cases)
    if status != "reviewed" and not args.allow_draft:
        log("Cases still contain draft labels. Review them first, or pass --allow-draft "
            "(the report will be marked draft).")
        return 1
    options = {"revision": args.revision, "dtype": args.dtype, "max_model_len": args.max_model_len,
               "chat_template_kwargs": json.loads(args.chat_template_kwargs),
               "engine_args": parse_engine_args(args.engine_arg)}
    if args.backend == "vllm":
        options["gpu_memory_utilization"] = args.gpu_memory_utilization
    backend = (backend_factory or build_backend)(args.backend, args.model, **options)
    outcomes, timing = evaluate(cases, backend, batch_size=args.batch_size)
    report = {"kind": "plan_judge_v2_pilot", "pilot_note": "Preliminary model selection on a small pilot set; "
              "not definitive validation.", "labels_status": status, "created_at": utc_now(),
              "protocol": protocol.protocol_identity(),
              "cases_file": {"path": str(args.cases), "sha256": sha256_file(args.cases)},
              "judge": backend.identity(), "runtime": backend.runtime(), "timing": timing,
              "metrics": metrics(cases, outcomes), "outcomes": outcomes}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{_slug(args.model)}.json"
    atomic_write_json(path, report)
    log(json.dumps({"model": args.model, "labels_status": status, **report["metrics"], **timing}, indent=2))
    log(f"Report: {path}")
    return 0


SUMMARY_COLUMNS = ["model", "revision", "labels_status", "cases", "valid_json_rate", "item_exact_agreement",
                   "item_quadratic_kappa", "coverage_mae", "separation_correct_minus_worst_wrong",
                   "harmful_precision", "harmful_recall", "harmful_false_positive_case_rate",
                   "plans_per_second", "load_seconds", "gpu"]


def compare(args, *, log=print):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.reports.glob("*.json"))
               if path.name != "comparison.json"]
    reports = [r for r in reports if r.get("kind") == "plan_judge_v2_pilot"]
    if not reports:
        log(f"No pilot reports in {args.reports}")
        return 1
    if len({r["cases_file"]["sha256"] for r in reports}) != 1:
        log("Reports were made from different case files; rerun them on the same cases.")
        return 1
    rows = []
    for report in reports:
        m, t = report["metrics"], report["timing"]
        gpu = (report.get("runtime") or {}).get("gpu") or {}
        rows.append({"model": report["judge"]["model_id"], "revision": report["judge"]["revision_resolved"],
                     "labels_status": report["labels_status"], **{k: m.get(k) for k in SUMMARY_COLUMNS if k in m},
                     "plans_per_second": t.get("plans_per_second"), "load_seconds": t.get("load_seconds"),
                     "gpu": gpu.get("name")})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(args.reports / "comparison.csv", buffer.getvalue())
    atomic_write_json(args.reports / "comparison.json", {"pilot_note": reports[0]["pilot_note"], "rows": rows,
                                                        "bias_by_generator_family": {
                                                            r["judge"]["model_id"]:
                                                                r["metrics"]["coverage_bias_by_generator_family"]
                                                            for r in reports}})
    log(buffer.getvalue())
    return 0


def build_parsers(subparsers):
    run = subparsers.add_parser("validate", help="Pilot one candidate judge on the reviewed case set")
    run.add_argument("--cases", type=Path, default=Path("experiments/validation/plan_judge_v2_pilot_cases.jsonl"))
    run.add_argument("--model", required=True)
    run.add_argument("--revision")
    run.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    run.add_argument("--dtype", default="auto")
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    run.add_argument("--chat-template-kwargs", default="{}")
    run.add_argument("--engine-arg", action="append", default=[], metavar="KEY=JSON")
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--allow-draft", action="store_true")
    summary = subparsers.add_parser("compare", help="Summarize pilot reports in one directory")
    summary.add_argument("--reports", type=Path, required=True)
