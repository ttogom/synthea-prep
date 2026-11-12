# synthea-prep

Generates a reproducible synthetic EHR dataset using [Synthea](https://github.com/synthetichealth/synthea) for a clinical-reasoning fine-tuning project.

```bash
# Generate the full cohort (10,000 patients, default seed)
./scripts/generate_synthea.sh 10000 20260916

# Summarize the run
python3 scripts/summarize_dataset.py data/pop10000-seed20260916/
```

See [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) for prerequisites, settings rationale, and reproducibility details.

## MED-COPILOT pipeline

Install Python 3.11 and [Git LFS](https://git-lfs.com/), then run these commands from the repository root:

```bash
git lfs install
git lfs pull

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4

cp -n .env.example .env
# Edit .env: replace the OPENAI_API_KEY placeholder and add any selected provider keys.
ollama pull gemma3:4b
ollama pull llama3.2:3b
ollama pull ministral-3:3b
ollama pull qwen2.5:14b

python tests/test_med_copilot_variants.py
python tests/test_med_copilot_config.py
python tests/test_med_copilot_startup.py
python tests/test_med_copilot_run_file.py
python tests/test_med_copilot_helpers.py
python tests/test_med_copilot_retrieval.py
python tests/test_med_copilot_observability.py
python tests/test_med_copilot_judges.py
python tests/test_med_copilot_plan_evaluation.py
python tests/test_med_copilot_scoring.py
python tests/test_med_copilot_models.py
python tests/test_collate_med_copilot_results.py
python tests/test_run_med_copilot.py
python tests/test_med_copilot_execution.py
python tests/test_med_copilot_preparation.py
python tests/test_med_copilot_ollama.py
python tests/test_med_copilot_pilot.py
python tests/test_run_experiments.py
python tests/test_benchmark_ollama.py
python run_experiments.py
# Select another experiment configuration, or skip its interactive confirmation:
python run_experiments.py --config experiments/configs/experiment.yaml --yes
# Continue an interrupted run with its unchanged configuration:
python run_experiments.py --config experiments/configs/experiment.yaml --resume experiments/output/<run-directory>
# After generation finishes, judge Plans; rerun this same command after interruption:
python judge_treatment_plans.py experiments/output/<run-directory>
# Resume that same Plan stage with different local runtime settings:
python judge_treatment_plans.py experiments/output/<run-directory> --concurrency 8 --context-length 4096
```

`run_experiments.py` is the concurrent runner. It reuses four isolated CPU evidence
workers, overlaps hosted requests with preparation, and drains each local model's
whole-cohort queue with four simultaneous requests before switching models.
One persistent CPU scorer batches eight patients at a time and overlaps scoring
with later local model phases. The original `run_med_copilot.py` remains the
serial runner, including its existing `--presenting-evidence-only` override;
use the same runner and configuration when resuming a run.

GraphRAG's guideline `settings.yaml` sets its chat model to 12 requests/minute and
200,000 tokens/minute per preparation worker. The four preparation workers therefore
have combined local ceilings of 48 RPM / 800,000 TPM; other API clients use
separate limiters. Embedding limits remain `null`.

The runner calculates text metrics and BioLORD diagnosis similarity, leaving
Plan correctness pending with reason `deferred`. The separate evaluator loads
only the recorded `plan_judge`, preserves generation and existing text/diagnosis
scores, and regenerates the report after judging. Download the configured
judge before starting that stage. The first invocation uses the run's saved
`plan_judge_concurrency` (shipped value 8; older default 2; range 1–64) and judge
context length. Optional `--concurrency` and `--context-length` override only the
Plan stage; omitted options reuse its last saved settings. Both the request pool
and owned Ollama server use those effective settings. Valid saved judge replies
are reused in the same run directory. The original experiment YAML and
`run.json.experiment` stay unchanged. Actual parallel inference also depends on
the provider/server capacity.

Each scalar judge response is saved durably for reuse after crashes. The
evaluator exports results once per invocation or controlled interruption rather
than rewriting the full results file after every patient. Both commands lock
the run directory to prevent simultaneous writers. Generation must finish before
starting Plan evaluation; its separate status is recorded in
`run.json.plan_evaluation`. Earlier invocations, including their effective runtime
settings, start/finish times, call counts, and status, remain in
`run.json.plan_evaluation_history`.

`--resume` continues a schema-4 run in its existing directory. It preserves
completed evidence, exact final prompts, completed model responses, successful
scores, and valid judge calls; failed or unstarted work is retried. Keep the
experiment settings, selected model definitions, input/label files, prompts,
scoring rules, and reference resources unchanged. Any edit to `experiment.yaml`,
including comments or whitespace, blocks a resume. Plan-judge selection is
checked by the `plan_judge` alias; judge registry definitions are retained for
provenance without a separate comparison. Adding an unused model alias is
allowed. If the original run used a cohort CLI override, supply the same
override when resuming. Validation and cancellation leave the existing run
untouched. Original provenance is retained; `run.json.resumptions` records each
resume's code identity, current source/model fingerprints, and correctness configuration.

The concurrent runner saves independent preparation, generation, and scoring jobs
in `jobs.sqlite3`, using durable SQLite transactions. Workers wait for the parent
to save evidence checkpoints before continuing. Received final replies are saved
before parsing, and score batches are saved before canonical export.
`results.jsonl` remains the public artifact and is replaced atomically at phase
boundaries and controlled stops. Keep the SQLite database and its WAL/SHM files
with the run directory: a hard crash may leave newer work there than in JSONL.
An interrupted request without a durably received response can be sent again.

An unexpected preparation subprocess exit or broken pipe automatically replaces
that worker and retries its interrupted patient from the latest durable checkpoint.
Checkpointed evidence components, captured replies, prompts, and ranking metadata
are retained; completed steps, generation, and scoring are skipped. Each patient has
at most two automatic worker replacements per invocation. Exhaustion stops the
run with its usable checkpoints intact; after addressing the cause, use the same
`--resume` command to continue with a fresh recovery budget. Configuration,
validation, protocol, and checkpoint-writing failures remain fatal. Recovery
attempts are recorded in `run.json.preparation_recovery` and `preparation-recovery.log` with
secrets redacted. Responses not durably saved before a worker died may be
requested again; automatic recovery does not promise exactly-once API execution.
Deaths before initial readiness have the same two-replacement budget per worker
slot; subprocess diagnostics append to `preparation-N.log` for that slot.

The original serial runner uses an atomic `.patient_checkpoint.json` between completed
stages, model calls, and score fields. This file is removed after the patient row is
flushed to disk. File replacements and checkpoint removal also sync their
directory entries. A recovered checkpoint is committed before execution can
replace it with another patient's checkpoint, preserving work across repeated
abrupt exits. Keyword extraction and completed, nonblank screened guideline
queries are also reused when their evidence component needs retrying. Refused,
empty, or incomplete question replies are audited and requested again on resume.
A request interrupted before its
response was durably saved may run again. Do not run multiple processes against
the same output directory. `judge_treatment_plans.py` judges only Plans and
preserves the saved text metrics and BioLORD scores.

Besides the six text metrics, each prediction gets a **preliminary single-model midterm** correctness metric (to be revised after group review). The diagnosis score is a BioLORD-2023 **clinical concept similarity score** (embedding cosine clipped to 0–1; not an accuracy or probability) between the labeled condition and the generated diagnosis (`FremyCompany/BioLORD-2023`, about 420 MB, downloaded from Hugging Face on first use). Abbreviations HTN, CHF, and COPD are expanded before encoding. Plan correctness uses one local judge, `ollama_qwen2_5_14b_speedup` (selected by `plan_judge` in `experiments/configs/experiment.yaml`; run `ollama pull qwen2.5:14b`, about 9 GB, and allow roughly 10 GB of free memory): one 0/0.5/1 call per reference action plus one call counting unsupported extra actions. The judge is loaded separately from the generating models; if it cannot be loaded, Plans are marked `judge_unavailable` and diagnoses are still scored. No OpenAI judge requests are made. Both scores are reported unweighted and weighted by 1 / rank, never combined. One patient makes at most 48 × (reference actions + 1) local judge requests; identical requests are sent once. See the preliminary correctness section of [med_copilot/docs/output_schema.md](med_copilot/docs/output_schema.md).

The runner automatically loads `.env` from the repository root before initializing model clients, even when launched from another directory. [`.env.example`](.env.example) lists the supported credential-variable names. Set `OPENAI_API_KEY` there; `GRAPHRAG_API_KEY=${OPENAI_API_KEY}` reuses it for guidelines, or you can supply a separate key. Add other provider keys only when selecting models that need them. Existing exported environment variables take precedence over `.env`. The local `.env` file is gitignored; keep actual tokens out of the model YAML.

[requirements.txt](requirements.txt) contains the Python dependencies and the `en_core_sci_md` model. Keep it updated as the pipeline changes. BioClinicalBERT, the cross-encoder, and BERTScore's default RoBERTa-large encoder download from Hugging Face on first use, so initial setup requires internet access. The unchanged upstream evaluator also attempts four NLTK resource downloads at import. The offline tests replace those download calls and pretrained BERTScore; real-tokenizer parity tests require locally provisioned NLTK data.

Configure `patient_input` and `patient_labels` in [experiments/configs/experiment.yaml](experiments/configs/experiment.yaml); relative paths resolve from the repository root. The shipped config selects `experiments/input/test_so.jsonl` and `experiments/input/test_labels.jsonl`; older configs that omit these settings retain the sample-file defaults. Each input line has a nonempty, unique string `id` and prepared `subjective` and `objective` strings, used directly without an LLM normalization call. Labels are required and joined by exact `id`, independently of file order:

```json
{"id": "patient-id", "condition": "strep_throat", "plan": ["Treatment", "Follow-up"], "has_presenting_evidence": true}
```

Every input ID must have one label before filtering. Duplicate IDs, invalid condition/plan types, or a non-boolean evidence flag fail preflight before per-patient API calls. A label file may contain extra IDs for other input subsets. `bin` and other label metadata do not enter prompts. The shipped test input contains 4,959 rows; both eligibility filters retain 2,378 patients. The shipped `max_patients: null` runs all 2,378 eligible patients (38,048 final requests before retries). A positive `max_patients` selects a seeded sample after filtering.

`presenting_evidence_only: true` selects patients whose joined label has `has_presenting_evidence: true`. `exclude_none_recorded_plans: true` then excludes patients whose label `plan` is exactly `["none recorded"]`, preserving case and whitespace distinctions. Set either YAML flag to `false` to disable that filter. The serial runner's existing `--presenting-evidence-only true|false` option overrides only its matching YAML flag; the concurrent runner uses YAML. Input order is preserved; excluded patients have no result rows and do not become `not_processed`. An empty selected cohort fails before a run directory is created. The filter applies only to experiment patients, not the similar-patient corpus.

Keep all runner configuration files in [experiments/configs/](experiments/configs/README.md). The fixed [experiment.yaml](experiments/configs/experiment.yaml) groups settings by dataset/cohort, evidence variants, patient retrieval, clinical guidelines, final generation, and results. It contains dataset/reference/output paths, model selections, retrieval/search tuning, filters, and generation settings. The fixed [condition_plan_models.yaml](experiments/configs/condition_plan_models.yaml) contains model definitions only. The current experiment selects `gpt_4o_mini`, `ollama_gemma3_4b` (`gemma3:4b`), `ollama_llama3_2_3b` (`llama3.2:3b`), and `ollama_ministral_3_3b` (`ministral-3:3b`), with all four variants, temperature 1, a retry limit of 3, and `max_patients: null`. `ollama_qwen2_5_14b_speedup` (`qwen2.5:14b`) is the separate preliminary Plan judge. Optional hosted and local Llama 3.3 70B definitions remain registered; the hosted Groq entry requires enterprise access.

Edit `selected_models` in `experiments/configs/experiment.yaml` to choose models in execution order. Add their credential variables to the repository-root `.env` (see `.env.example`), then run:

```bash
python run_experiments.py --config experiments/configs/experiment.yaml
```

Startup checks the selected aliases and nonblank API-key variables. Patient retrieval needs `OPENAI_API_KEY`; guidelines need `OPENAI_API_KEY` and `GRAPHRAG_API_KEY`. Unselected evidence sources require no credentials or resource initialization; `llm_only` with keyless local final models needs neither shared key nor retrieval-model downloads. Configuration errors name the affected settings and terminate preflight before the concurrent runner creates output. Failures after its run directory is created are saved in `run.json.errors` with status `failed_incomplete` (see `med_copilot/docs/output_schema.md`). These checks do not authenticate credentials or verify account/model access. Valid settings display grouped model identities/key-variable names, run settings, explained evidence variants, dataset/cohort settings, vertically aligned filter counts, and planned final requests. Additional resource/output paths and evidence-stage model/tuning settings are configured in YAML and recorded in metadata without expanding this confirmation display. Confirm with `y` or `yes` to start; any other response, EOF, or Ctrl+C cancels without creating experiment output or making generation requests. All effective experiment settings and defaults remain recorded in `run.json.experiment`, alongside the config fingerprint.

Set `max_patients` in `experiments/configs/experiment.yaml` to a positive integer N to sample up to N eligible patients without replacement using seed 42. Selected patients run in input-file order. Use `max_patients: null` to run all eligible patients. The limit applies after presenting-evidence filtering and recorded-plan filtering, in that order. All input records and labels are still validated before selection; values such as 0, negative numbers, booleans, strings and floats are rejected. `run.json.experiment.max_patients` records the configured limit, and `run.json.evaluation` records that limit, `patient_sampling_seed`, the intermediate cohort counts, selected/excluded patient IDs, and exclusion reasons (`missing_presenting_evidence`, `no_recorded_plan`, or `max_patients`). Excluded patients have no `results.jsonl` rows, including when the run is interrupted.

Model overrides belong in YAML. The concurrent runner accepts `--config PATH`,
`--resume RUN_DIRECTORY`, and `--yes`; the original runner retains its existing
CLI. The shared `temperature` and `max_retries` are passed into every selected
final model; putting either in a registry entry is rejected.

Each model definition uses upstream GraphRAG `LanguageModelConfig` fields (`type: chat`, `model_provider`, `model`, optional `api_base`, and other provider settings), plus `api_key_env` and `output_mode`. For Anthropic or Gemini, add an entry with `model_provider: anthropic` or `gemini`, the provider's model ID and its credential environment-variable name. Do not put literal API keys in YAML. An explicit endpoint may omit `api_key_env` for an unauthenticated local server; the adapter supplies the nonsecret placeholder required by upstream validation. This connects to an already running model server; it does not download or host weights. Credentials for the selected shared evidence stages are still needed when final models are served elsewhere.

The selected local models use the experiment-owned Ollama server at
`http://127.0.0.1:11435/v1`, with four parallel slots, one loaded model, a 16,384-token
context, and a 768-token generation cap. The concurrent runner starts and stops
only this server; the GUI daemon on 11434 is untouched. Download the configured
weights beforehand. The separate Qwen judge uses the same owned endpoint with
its effective Plan-stage concurrency/context (initially eight slots and 4,096
tokens), and a 64-token cap.
Local entries request JSON schema output and need no API-key variable; shared
evidence stages and `gpt_4o_mini` still require OpenAI credentials.

The project-owned [model adapter](scripts/med_copilot_models.py) constructs models through the unchanged upstream GraphRAG factory and uses `LitellmChatModel.completion`, including its configured retry/rate-limit wrappers. `output_mode` is explicitly `json_schema` (default), `json_object`, or `prompt` (JSON requested in the shared prompt without an API formatting option). Unsupported request parameters produce a model-specific error; there is no automatic model fallback, mode downgrade, or repair request. The master experiment supplies temperature and retry limits; other unspecified settings retain upstream defaults. Effective settings are recorded per model in run metadata. The default experiment retains temperature 1 from the previous model definitions. Input, label, output and reference-resource paths all live in the experiment YAML.

Select any nonempty subset of `selected_variants` in YAML:

| Variant | Final prompt evidence |
| --- | --- |
| `llm_only` | Prepared patient S/O |
| `patient_retrieval` | S/O + similar-patient example |
| `guidelines` | S/O + clinical guidelines |
| `patient_retrieval_guidelines` | S/O + both sources (previous behavior) |

The lightweight [variant registry](scripts/med_copilot_variants.py) declares each variant's component dependencies, prompt template, and bindings from component outputs to prompt fields. Required similar-patient retrieval and guideline retrieval execute once per patient and their component outputs are reused across all consuming variants/models. Each variant renders one prompt; all selected final models receive independent copies of those exact messages. The concurrent runner schedules local requests by model across the cohort, while the serial runner retains its per-patient loop. [diagnosis_plan.py](med_copilot/prompts/diagnosis_plan.py) derives `VARIANT_TEMPLATES` by removing only the unused evidence sections from the existing project prompt, including the reference-specific fallback when reference evidence is absent. The combined prompt's wording/order and shared output instructions remain unchanged. The original runner keeps visible requirement-based `if` blocks for the two components followed by one final variant loop; the concurrent runner delegates these stages to isolated preparation workers. Configuration resolves requirements; shared helpers handle captures, dependency failures, and exact prompt construction. Add variants using [the extension guide](med_copilot/docs/adding_a_new_variant.md); combinations of existing components or prompt-only variants need no runner branch. Optional `variant_settings` in the experiment YAML supplies registered parameters and their effective values are saved in run metadata. [med_copilot_startup.py](scripts/med_copilot_startup.py) handles conditional encoder initialization before FAISS import for Apple Silicon compatibility, preserving encoder errors for dependent variants. The YAML encoder ID controls both preloading and cache identity; changing it after importing the runner requires restarting the script. Each model retains its own raw reply, parsed predictions, provider metadata, generation status/error and evaluation. One model's failure does not prevent the remaining models or patients from running.

The original serial runner prints a `RANKED CONDITIONS AND PLANS` section grouped by model alias and provider/model name. Each model contains conditions numbered 1–3, with a separately numbered Plan indented under each condition. Failed models, unavailable fields and empty Plans are labeled explicitly. Console formatting leaves the raw responses and saved predictions unchanged.

Every final model is asked for the same JSON output with a `predictions` array of exactly three condition/Plan pairs, ordered from most to least likely. Each pair contains a concise `condition` string and its own `plan` array of at most three nonblank action strings. Plan items retain their original order, wording, duplicates, and internal commas; they are joined only for scoring.

After generation, [scripts/med_copilot_scoring.py](scripts/med_copilot_scoring.py) uses the project metric adapter, preserving upstream lexical defaults and batching BERTScore for each field of each ranked pair. Each condition is compared to `label["condition"].replace("_", " ")`; each generated Plan is joined with `", "` and compared to `", ".join(label["plan"])`. All ranks use the same patient labels. Each variant/model's `variants[variant_id].assessment_plans[model_id].evaluation` array contains three entries with `rank` (1–3), `condition`, and `plan`; each field has six metric keys and an explicit status. There is no rank weighting or combined score. Exactly `["none recorded"]` in the **label** skips Plan scoring at every rank; an empty joined reference Plan is unavailable. An empty generated Plan remains `[]` and receives `missing_generation`. Unavailable/error scores are JSON `null`, not zero or `NaN`. Ordinary scoring errors preserve other fields and ranks and do not change the generation status or run exit code.

Scores preserve upstream defaults and retain one result per patient/rank/field.
BLEU is **singleton-corpus BLEU**; averaging saved BLEUs does not reproduce
dataset corpus BLEU, and exact texts shorter than four tokens can score zero.
BERTScore loads one persistent scorer and batches up to 64 pending pairs,
retaining each pair's F1 and the encoder's existing token-limit truncation.
There is no chunking, clinical normalization, or composite score. Metadata records
the metric adapter, label/helper hashes, cohort, package versions, and defaults;
it does not establish the weight revision used by the original study.

The experiment YAML selects the patient embedding, keyword-extraction and cross-encoder models, hybrid embedding weight, candidate count and retained reranking count. Only the first reranked reference enters the final prompt. It also selects the guideline-question model, community hierarchy level, answer description and diagnostic verbosity. `guideline_response_type` is free-form text (for example, `Single Paragraph`, `Single Sentence` or `List of 3-7 Points`), matching upstream GraphRAG. Reference matching uses S/O only for embeddings, keyword overlap, and cross-encoder reranking, through [scripts/med_copilot_retrieval.py](scripts/med_copilot_retrieval.py). The selected reference's complete SOAP is still supplied to final generation. S/O embeddings and the FAISS index are built on first use under the configured `patient_retrieval_artifacts_dir` (default `med_copilot/data/patient_retrieval_so/`), separate from the upstream S/O/A artifacts. A `matching.json` manifest binds the cache to the ordered S/O texts and encoder identity and checks artifact hashes; missing, stale, or corrupt caches are rebuilt. Upstream source and artifacts are left intact.

Each new run writes a new directory; `--resume` reuses its existing directory:

```text
experiments/output/<Pacific timestamp>__<input stem>__<branch>/
├── run.json             # configuration, cohort, provenance, status and resource fingerprints
├── results.jsonl        # one line per patient: components, then variants → models → outputs/scores
├── provenance/
│   ├── reported-usage-cost.json # final captured API tokens and estimated USD
│   └── recovery-incidents.json # resumptions and preparation-worker recovery history
├── jobs.sqlite3         # concurrent runner's durable job recovery; retain accompanying WAL/SHM files
└── results_summary.html # generated automatically after a completed batch, including batches with errors
```

The concurrent runner writes the two `provenance/` reports only on exit, after all
model calls and resource cleanup finish. Resuming recalculates cumulative usage
from canonical captures, deduplicating response IDs and retaining recorded prices
and detailed recovery events. Usage with missing/invalid token counts is explicit;
models without a recorded price are excluded with a limitation. Local inference
does not incur API-credit costs. These are reported-token estimates, not invoices;
they include no budget or prior-work allowance. `prepared_patient_count` counts
prepared patients; `completed_hosted_output_count` counts completed hosted outputs,
not API attempts. `usd_per_million_tokens` records the price assumptions.
No `command.txt` or `launch.json` is generated. A hard kill can prevent finalization;
a successful resume refreshes the reports. Provenance-save failures are surfaced
without overwriting canonical results or masking an existing pipeline failure.

Git exposes only `run.json`, `results.jsonl`, `results_summary.html`, and these two
provenance reports for new output directories. Checkpoints, logs, PID files, and
other operational files remain ignored. Keep reusable monitoring/graphics tools
and progress histories outside the repository. Existing tracked historical outputs
are unaffected by the ignore rules.

`run.json.experiment` records every effective YAML setting/default plus config path/SHA-256 and registry reference. Schema version 4 stores component captures once per patient and variant references to their required captures. Run metadata snapshots selected definitions, effective settings, component/prompt identities and source fingerprints. `assessment_plan_models` retains registry path/SHA-256 and effective per-model upstream settings once per run, excluding resolved keys. Only selected and successfully initialized evidence resources are fingerprinted; unused/failed resources are `null`. Historical schemas remain readable by the collator.

Each result row stores evidence, rankings, traces, provider responses, statuses and errors inside its component captures, then stores `variants[variant_id]`. Each variant references the captures it consumes and retains its own status/error, exact rendered `assessment_plan_prompts`, and `assessment_plans[model_id]`. Shared capture bindings are scoped and restored between component calls. Model slots retain `status`, `error`, `predictions`, `assessment_plan_raw`, `model_responses`, and three-rank `evaluation`. Evidence failures block only dependent variants. A successful empty retrieval still completes its source and uses the existing no-reference fallback. Other variants/models and later patients continue.

Patient status is `completed` when all selected variant/model generations complete, `failed_incomplete` if any fails, or `not_processed` for an unstarted selected patient. Run statuses remain `running`, `completed`, `completed_with_errors`, and `failed_incomplete`. Failures retain any obtained raw reply/metadata; failed models are not scored. Ordinary metric errors remain field/rank/model-local and do not change generation status. An obtained but malformed JSON response retains the existing `completed` semantics and explicit missing-generation scores. Interruptions preserve outputs/scores, mark the run stopped, and retain every unattempted variant/model/patient slot. Excluded patients never receive result rows or placeholders.

The runner's `RESULTS` block automatically calls [the existing collator](scripts/collate_med_copilot_results.py). Terminal and HTML summaries show each model's mean **top-ranked condition BERTScore_F1** by variant, choose the highest mean (ties retain config order), and report that variant's separate top-ranked condition/Plan mean, population standard deviation, min/max and scored count. Missing/skipped/error scores remain unavailable; counts expose differing scored subsets. The HTML retains all three ranks, six metrics per condition/Plan, patient details and paired comparisons against `llm_only`; it creates no combined condition/Plan score. Report generation needs only saved results and the fingerprint-matching label file, and can be repeated with `python scripts/collate_med_copilot_results.py experiments/output/<run-directory>`. A report-write failure preserves completed JSON artifacts, records `run.json.results_summary` failure metadata, and exits nonzero; interruptions preserve JSON without generating a report. The command exits successfully only for a completed run or user cancellation.

Every response goes through the existing parser: malformed JSON, duplicate keys, nonfinite numbers, unexpected top-level keys, or a prediction count other than three produce `predictions: null`. Invalid fields become `null` without discarding other valid fields/ranks; no arrays are truncated and Markdown is not a fallback. Historical result files are not rewritten. Approved result/provenance files under `experiments/output/` remain visible to Git. The runner locates its default model config and guideline artifacts independently of the launch directory. Offline tests mock external requests; live generation and provider compatibility remain unverified by those tests.

The [patient output schema](med_copilot/docs/output_schema.md) documents exact keyword replies, processed retrieval keywords, retained hybrid and cross-encoder IDs/scores, and per-stage provider response metadata. Captures survive subsequent failures and interruptions. They do not enter clinical prompts.
