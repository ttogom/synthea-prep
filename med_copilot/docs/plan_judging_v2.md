# Plan judging protocol 2 (single call, rank 1)

Protocol 2 replaces the per-action Plan judge for new analyses. It judges only the
**rank-1** Plan of each patient, variant and generator model, with **one model call
per Plan**, using a 3–4B Hugging Face model on a Colab GPU. Protocol 1
(`judge_treatment_plans.py`, [treatment_plan_evaluation.md](treatment_plan_evaluation.md))
is unchanged, and its saved results remain valid under protocol 1. Do not
edit `treatment_plan_evaluation.md` for protocol 2: protocol 1 records that file's
hash and refuses to resume when it changes.

## Why it changed

On the 2026-11-12 run, protocol 1 scored 87.5% of judged Plans as 0:

- 80% of the zeros were Plans where every reference action got 0, so the extras call was skipped.
- 18% were Plans whose partial credit was cancelled by the 0.5-per-extra deduction. With one
  reference action, a single extra turns 0.5 into 0. Appropriate care such as acetaminophen,
  hydration or follow-up was often counted as an extra.
- Ranks 2 and 3 were judged against the labeled condition's reference, even though their
  diagnoses rarely match it. Their zero rates were 92% and 96%.

## Rubric (one call)

The judge sees the labeled condition, S, O, the numbered reference actions, and the
numbered rank-1 generated actions. It never sees the generator model, variant, rank or
predicted diagnosis. For each reference action it returns a label:

| Label | Credit | Meaning |
| --- | --- | --- |
| `full` | 1 | Same action, a valid more specific version, or an interchangeable substitute for this patient |
| `partial` | 0.5 | Relevant but less specific, a partial substitute or fallback, a missing or different strength of the same drug, or a conditional action whose prerequisite the reference lacks |
| `none` | 0 | Missing, wrong target, a different action, a placeholder, or asking instead of counselling |

Each `full`/`partial` label must cite the generated item numbers that match, and `none`
cites nothing. Separately, the judge lists **harmful** generated items: unsafe,
contraindicated by the notes, or clearly aimed at the wrong condition. Symptom relief,
hydration, rest, monitoring, follow-up, education and confirmatory testing are not
harmful unless the notes contraindicate them.

The prompt is `PROMPT_TEMPLATE` in [scripts/plan_judge_v2/protocol.py](../../scripts/plan_judge_v2/protocol.py).
Changing the prompt, response schema, credits or scoring requires a new protocol version.

## Scores (computed in code, not by the model)

| Field | Formula | Role |
| --- | --- | --- |
| `coverage` | mean credit over reference actions | **Primary** score |
| `harmful_count` | number of distinct harmful items | Reported separately; never subtracted from coverage |
| `penalized` | max(0, (sum of credits − 0.5 × harmful_count) / reference count) | Secondary score |

## Denominator and unavailable Plans

Every patient in `results.jsonl` × selected variant × expected generator model gets exactly one
final row. The expected generators are every model slot that appears anywhere in the file, or
the `--generator-models` list. Nothing is dropped silently: a Plan that can't be judged gets an
`unavailable` row giving the reason.

| `reason` | Status | Cause |
| --- | --- | --- |
| `variant_not_completed` | `unavailable` | The variant was blocked before generation (for example `EvidenceUnavailable`), is absent from the record, or has no slot for this generator |
| `generation_failed` | `unavailable` | The generator's slot is `failed_incomplete` |
| `generation_not_processed` | `unavailable` | The slot never ran although the variant completed |
| `missing_plan` | `unavailable` | A completed slot has no rank-1 Plan, or a completed variant lacks this generator's slot |
| `invalid_generation` | `unavailable` | The slot or its Plan is malformed |
| `missing_reference`, `no_recorded_plan`, `record_artifact_plan` | `unavailable` | The reference Plan is unusable (the same rules as protocol 1) |
| `empty_generated_plan` | `scored`, coverage 0 | The generated Plan has no non-blank items |

- `reason` is the leading cause. Variant and generation causes come before reference causes,
  except that an empty Plan against an unusable reference is `unavailable` for the reference.
- `reasons` lists every cause, so a failed generation for a COPD patient keeps both
  `generation_failed` and `record_artifact_plan`.
- `source` copies the variant status, slot status and error type from `results.jsonl`.
- The manifest `counts` give `unavailable_by_reason` and `scored_without_call`, and
  `scored + unavailable = plans` once a variant is complete.
- Only patients present in the results file count. The S/O file holds the whole cohort, and
  an evaluation run covers a filtered subset (2,378 patients in the 2026-11-12 run). The judge
  logs how many S/O patients have no results record, but doesn't add rows for them.
- Agentic and base inputs may have any filename. The default output folder is
  `<run directory>__<sha12>` for a file named `results.jsonl`, otherwise `<file stem>__<sha12>`,
  so different inputs get different folders. Reusing a folder for different content is
  refused, because the manifest hashes must match. Copies of the same file under two names get
  two folders, so share one copy (see "One writer per variant").

## Reply validation

vLLM constrains decoding to a JSON schema built for each Plan, sized to its reference and
generated counts. The code still rejects any reply that:

- is not one JSON object, or has duplicate keys or NaN values;
- has the wrong number of reference items, or items out of order;
- cites a generated item that doesn't exist, or cites the same item twice;
- labels a reference item `none` but cites a generated item;
- labels a reference item `full` or `partial` but cites nothing;
- lists a harmful item twice or without a reason.

A whole-reply ```` ```json ```` fence is accepted, because the transformers fallback has no
guided decoding. Rejected replies are saved as `invalid_response` and retried on a later
run, up to `--max-attempts`.

## Running it

```bash
# Validate inputs and print one prompt; no model, network, or output writes
python judge_plans_v2.py judge --results RUN/results.jsonl --variants llm_only agentic_both \
  --out-dir OUT --dry-run
# Also count every prompt with the judge tokenizer (downloads the tokenizer only, no weights);
# exits 1 if any prompt exceeds --max-model-len minus the 384-token output budget
python judge_plans_v2.py judge --results RUN/results.jsonl --variants all --out-dir OUT \
  --model Qwen/Qwen3-4B-Instruct-2507 --dry-run --count-tokens
# Judge on a GPU (Colab: notebooks/plan_judge_v2_colab.ipynb builds this command)
python judge_plans_v2.py judge --results RUN/results.jsonl --variants all --out-dir OUT \
  --model Qwen/Qwen3-4B-Instruct-2507 --csv --owner "Your name"
python judge_plans_v2.py --version
```

`--results` accepts base or agentic `run_experiments.py` output (schema 4, with `variants`).
`--limit N` judges the first N pending Plans by patient id, which suits smoke tests but samples
a single condition. `--sample N` judges a fixed pseudo-random subset of N pending Plans per
variant (seeded by `--sample-seed`), representative across conditions, for benchmarks; a
resumed run completes the same subset. The sample size and seed are part of the output identity
(`selection.sample` in the manifest), so resuming with another sample, or none, is refused; give
each benchmark its own `--out-dir`. Both options leave the rest pending, so the run exits 2.
`--patients` and `--labels` default to `experiments/input/test_so.jsonl` and
`test_labels.jsonl`, the files the runs were generated from.

Exit status:

| Code | Meaning |
| --- | --- |
| 0 | Every selected Plan is final |
| 1 | Unusable inputs or outputs, or an input changed during judging |
| 2 | Some Plans are still unjudged; rerun the same command |
| 3 | Another writer holds a variant |
| 130 | Interrupted; completed batches are saved |

### Colab

Open [notebooks/plan_judge_v2_colab.ipynb](../../notebooks/plan_judge_v2_colab.ipynb) on a GPU
runtime. An L4 or A100 is recommended. A T4 lacks bfloat16, so set `DTYPE = "float16"`.
Tested on 2026-11-20: T4, Python 3.13.16, `vllm==0.34.0`, `Qwen/Qwen3-4B-Instruct-2507`,
20/20 Plans judged with guided JSON and 0 invalid replies. `vllm==0.10.1.1` does not install
on Colab's Python 3.13. Qwen3-4B-Instruct-2507 supports only non-thinking mode, so keep
`CHAT_TEMPLATE_KWARGS = {}` (do not pass `enable_thinking`).

1. Add a Colab Secret named `HF_TOKEN`. MedGemma is gated: accept its licence on Hugging
   Face with the same account first.
2. Edit only the top config cell and run all cells. The notebook mounts Drive, clones the
   public repository at `REPO_REF`, installs
   [requirements-colab-judge.txt](../../requirements-colab-judge.txt), and runs the script in a
   child process.
3. The token is passed only through that child process's environment. It is never written
   to outputs, and `HF_TOKEN` values are redacted from saved error messages.

`results.jsonl` files are large Git LFS objects, so put a copy on Drive and point
`RESULTS_PATH` at it. One way to get it: `git lfs pull --include=experiments/output/<run>/results.jsonl`.

## Outputs, resume and safety

```text
OUT/<run directory>__<results sha12>/<variant>/
  manifest.json    run_id, protocol, input paths+hashes, variant, selection, judge identity, sessions, counts
  judgments.jsonl  one row per attempt; references manifest run_id
  judgments.csv    optional (--csv): one row per (patient, generator model)
  .lease           present while a writer owns the variant
```

- **The manifest holds the shared metadata once.**
  - Protocol identity: version, prompt template hash, credits and decoding settings.
  - Inputs: path, sha256 and size of `results.jsonl`, the patient file and the label file.
  - Variant and selection: rank 1, the generator-model filter, the expected generators and
    the number of patients in `results.jsonl`.
  - Judge: backend, Hugging Face model id, requested revision and resolved commit, dtype,
    `max_model_len`, guided decoding, chat-template and engine arguments.
  - One `sessions` entry per invocation: writer, owner, status, batch size, call count,
    GPU and library versions, `batches` (Plans, seconds and tokens per `generate` call),
    elapsed time, counts and `timing`.
  - `timing` separates startup from throughput: `model_load_seconds` (download, weight load,
    vLLM profiling and CUDA graph capture; only the session that loaded the model reports
    it), `generation_seconds` and `generation_plans_per_second` (all `generate` calls),
    `first_batch_seconds`, `steady_state_plans_per_second` (batches after the first, which
    carries warm-up), `output_tokens_per_second`, `wall_plans_per_second_including_load`,
    prompt- and output-token summaries, `output_tokens_at_limit` and `finish_reasons`.
    The log prints the same split per batch and per variant.
- **Each row in `judgments.jsonl` holds only the per-Plan data.**
  - Fields: `run_id`, `writer_id`, `patient_id`, `generator_model`, `rank`, `status`, `reason`,
    `reasons`, `source`, `attempt`, `prompt_sha256`, `raw_response`, `finish_reason`,
    `prompt_tokens`, `output_tokens`, `parsed`, `credits`, `coverage`, `harmful_count`,
    `penalized`, `error` and `judged_at`. Token counts come from the backend (vLLM's
    prompt and completion token ids) and are `null` for rows recorded without a call.
  - The first `scored` or `unavailable` row for a key is final. `invalid_response` and
    `error` rows count as attempts.
- **Results can't mix across runs.** On resume, the saved manifest must match the protocol,
  variant, selection, judge identity and every input hash. Paths may differ between users'
  Drive mounts. Any mismatch is refused, so use a new `--out-dir` or `--run-name` instead.
  Every row's `run_id` must match the manifest.
- **The checkpoint survives interruption.** Rows are appended and fsynced after every batch.
  An interrupted append leaves an incomplete last line, which is trimmed on resume and
  recorded as `repaired_tail_bytes`. A corrupt line elsewhere stops the run for manual
  inspection. `manifest.json` and `judgments.csv` are replaced atomically (temporary file,
  fsync, rename).
- **Inputs are never modified.** All inputs are opened read-only and hashed while being read.
  At the end, size, mtime and hashes are checked again; `--skip-final-hash` skips re-hashing
  `results.jsonl`. If anything changed, the run exits 1.
- **One writer per variant.**
  - `.lease` is created exclusively, re-read to confirm ownership, refreshed every 60 s, and
    checked before every append.
  - A live lease stops a second writer with exit 3, naming `--owner`. A lease whose heartbeat
    is older than 15 minutes is taken over only with `--takeover-stale`.
  - Google Drive can't guarantee exclusive creation across machines. The re-read, the
    per-batch ownership check and per-row `writer_id` are the remaining protection, and
    `duplicate_final_rows` in the counts would show if two writers finalized the same Plan.
  - Agree on who judges each variant.

## Model selection pilot

Before the full experiment, compare candidate 3–4B judges on
[experiments/validation/plan_judge_v2_pilot_cases.jsonl](../../experiments/validation/plan_judge_v2_pilot_cases.jsonl)
(see [its README](../../experiments/validation/README.md)). The candidates start with
`google/medgemma-4b-it` and `Qwen/Qwen3-4B-Instruct-2507`.

```bash
python judge_plans_v2.py validate --model google/medgemma-4b-it \
  --engine-arg 'limit_mm_per_prompt={"image": 0}' --out-dir REPORTS
python judge_plans_v2.py validate --model Qwen/Qwen3-4B-Instruct-2507 --out-dir REPORTS
python judge_plans_v2.py compare --reports REPORTS
```

Each report records:

- **Reply reliability:** valid-JSON rate on first attempt, without retries.
- **Agreement with the gold labels:** item exact agreement, quadratic-weighted kappa, and
  coverage mean absolute error.
- **Separation:** mean coverage by category, and the margin between correct and wrong categories.
- **Harmful-item accuracy:** precision and recall, false-positive items (flags on items
  the gold labels do not mark harmful), and the number and rate of cases with no gold harmful
  items that were flagged anyway.
- **Invalid responses and request errors**, and the `excluded_cases` (disputed) that were left out.
- **Bias:** coverage bias by generator family. MedGemma shares the Gemma 3 base with the
  `gemma3_4b` generator, so watch for self-preference.
- **Runtime:** throughput, load time and GPU.

The pilot is a preliminary model-selection step on about 48 cases, not definitive
validation. `validate` refuses `draft` labels unless `--allow-draft` is given, and marks
such reports `draft`. It accepts `preliminary` labels and marks the report `preliminary`.
It does not run `disputed` cases and leaves them out of every statistic. Currently 47
cases are preliminary and `less-chf` is disputed; see the
[pilot README](../../experiments/validation/README.md).

## Tests

```bash
python tests/test_plan_judge_v2.py   # offline: fake backend, no model, GPU or network
```

This branch is based on `origin/main@cc8017a`, where the whole offline suite passes
(`python tests/<file>.py` for each of the 32 test files). The 5 config and runner test
failures present at `92b4dfe` were fixed on `main` before this branch was rebased.

## Known limitations

- Protocol 2 has had one 20-Plan GPU smoke test (T4, `vllm==0.34.0`), all acute bacterial
  sinusitis Plans with one reference action, the easiest case. Throughput is not yet measured
  on a representative sample. The vLLM adapter handles both the `GuidedDecodingParams` and
  `StructuredOutputsParams` APIs.
- Prompt sizes, counted with the Qwen3-4B-Instruct-2507 tokenizer for all 36,970 pending
  Plans of the four base variants of the 2026-11-12 run: median 1,130 tokens, p99 1,642,
  maximum 2,004. Every prompt fits `max_model_len` 4,096 with the 384-token output budget.
  Agentic inputs have not been counted; run `--dry-run --count-tokens` on them first.
- The transformers backend has no guided decoding, so expect more invalid replies.
- Rank-1 judging does not assess the plans for rank-2 or rank-3 diagnoses.
