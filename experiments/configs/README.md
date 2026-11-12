# MED-COPILOT runner configuration

Edit `experiment.yaml` before each run. Model definitions live in the fixed sibling
`condition_plan_models.yaml`; tokens stay in the repository-root `.env` (see
`.env.example`). Paths resolve from the repository root, independently of the
launch directory. The concurrent runner selects a YAML file with `--config`;
model and scheduling overrides belong in that file. The original serial runner
retains its existing CLI and fixed default config.

```bash
python run_experiments.py --config experiments/configs/experiment.yaml
python run_experiments.py --config experiments/configs/experiment.yaml --resume RUN_DIRECTORY
python run_experiments.py --config experiments/configs/experiment.yaml --yes
python judge_treatment_plans.py RUN_DIRECTORY
# Optional runtime-only overrides; valid saved replies are reused:
python judge_treatment_plans.py RUN_DIRECTORY --concurrency 8 --context-length 4096
```

| Experiment setting | Meaning |
| --- | --- |
| `selected_models` | Nonempty, unique registry aliases, in execution order. |
| `plan_judge` | Nonblank registry alias used for Plan correctness. Reuses a selected generating model, or loads the judge separately. Defaults to `ollama_qwen2_5_14b` for older configs. |
| `plan_judge_concurrency` | Initial simultaneous requests in the separate Plan stage, integer 1–64; older configs default to 2. The judge CLI can override this stage's runtime. |
| `temperature` | Shared final-model temperature, finite number from 0 to 2. |
| `max_retries` | Shared upstream final-model retry setting, integer at least 1. |
| `max_patients` | Sample up to N eligible patients without replacement using seed 42; positive integer or `null` for all. Selected patients run in input order. |
| `selected_variants` | Nonempty, unique registered variants; every final model runs each one. |
| `variant_settings` | Optional mapping of registered variant IDs to declared parameter values; omitted parameters use registered defaults. |
| `patient_input` | Prepared patient S/O JSONL path. |
| `patient_labels` | Gold-label JSONL path, joined by exact patient ID. |
| `presenting_evidence_only` | Keep labels with `has_presenting_evidence: true`. |
| `exclude_none_recorded_plans` | Exclude label `plan` exactly `["none recorded"]`. |
| `patient_corpus` | Reference SOAP JSON corpus used by patient retrieval. |
| `patient_retrieval_artifacts_dir` | Directory for the S/O embeddings, FAISS index and cache manifest. |
| `patient_embedding_model` | Encoder for corpus/query vectors and cache identity. |
| `patient_keyword_model` | OpenAI model extracting retrieval keywords. |
| `patient_reranker_model` | Cross-encoder scoring query/reference pairs. |
| `patient_retrieval_alpha` | Embedding weight in [0, 1]; keyword weight is 1 minus this value. |
| `patient_retrieval_top_k` | Positive number of hybrid candidates sent to reranking. |
| `patient_rerank_top_k` | Positive number of reranked cases retained; only the first enters generation. |
| `guidelines` | GraphRAG project root containing `settings.yaml` and `output/`. |
| `guideline_question_model` | OpenAI model producing the shared guideline-search questions. |
| `guideline_community_level` | Nonnegative community hierarchy level passed to GraphRAG. |
| `guideline_response_type` | Free-form answer description, e.g. `Single Paragraph`, `Single Sentence`, `List of 3-7 Points`; any nonblank text is accepted upstream. |
| `guideline_verbose` | Boolean controlling GraphRAG diagnostic output. |
| `experiment_output` | Parent directory for timestamped run artifacts and the generated HTML report. |
| `execution` | Concurrent runner scheduling and isolated Ollama settings, detailed below. |

The shipped `execution` values are:

| Setting | Value and purpose |
| --- | --- |
| `preparation_workers` | `4` persistent, isolated evidence processes. |
| `local_concurrency` | `4` final requests for the active local model. |
| `hosted_concurrency` | `4` hosted final requests, overlapping preparation/local inference. |
| `scoring_batch_patients` | `8` patients per durable scoring batch. |
| `overlap_scoring` | `true`; score available outputs while later local model phases run. |
| `cpu_threads` | `2` Torch threads per preparation worker. |
| `encoder_device` | `cpu`; keep retrieval and scoring work off Ollama's GPU. |
| `manage_ollama` | `true`; start/stop an experiment-owned server. Older configs default to `false`. |
| `ollama_host` | `127.0.0.1:11435`; never adopt or stop the GUI daemon on 11434. |
| `ollama_parallel_slots` | `4`; allocated before model loading, at least `local_concurrency`. |
| `ollama_context_length` | `16384` tokens per generation slot. |
| `ollama_judge_context_length` | Initially `4096` tokens per separate judge slot; the judge CLI can override the stage's context. |

Local generation drains one model across the cohort before loading the next.
The owned server keeps one model resident. Two evidence workers prepare exact
prompts once per patient, and one persistent scorer reuses its BERTScore model
with batches of up to 64 text pairs. `jobs.sqlite3` checkpoints independent jobs;
keep it and any WAL/SHM sidecars with the run directory when resuming.

Unexpected preparation subprocess exits and broken pipes trigger replacement
and retry from the interrupted patient's latest durable checkpoint. The fixed
policy allows two replacements per patient per invocation; there is no new YAML
or CLI setting. Completed durable work is reused. Configuration, validation,
protocol, and checkpoint-writing failures stop immediately. Exhaustion also
stops cleanly; after fixing the cause, explicit `--resume` starts a fresh recovery
budget. Recovery attempts are recorded in `run.json.preparation_recovery` and
`preparation-recovery.log`; worker diagnostics append to `preparation-N.log`.
Initial startup deaths also have a two-replacement budget per worker slot.
A response lost before its checkpoint committed can be requested
again. The existing exact-config resume checks still apply.

The guideline project's `med_copilot/data/guidelines/settings.yaml` separately
sets `default_chat_model.requests_per_minute: 12` and
`tokens_per_minute: 200000`. These are per-worker limits: four preparation workers
have nominal combined ceilings of 48 RPM / 800,000 TPM. Other API clients do not
share these local buckets; retain quota headroom if adding traffic or workers.
Both `default_embedding_model` limits remain `null`.

For Plan judging, explicit CLI overrides take precedence over the last saved
`run.json.plan_evaluation.concurrency` and `context_length`, which take precedence
over the original experiment defaults. The client pool and owned server receive
the same effective values. A subsequent invocation without overrides reuses the
last Plan-stage values and valid scalar replies in the same run directory.
These overrides do not edit `experiment.yaml` or generation provenance in
`run.json.experiment`. Previous stages, with settings, start/finish times, counts,
and status, are retained in `run.json.plan_evaluation_history`.

The YAML groups fields by pipeline stage and documents each one. The presenting-evidence filter runs first, the recorded-plan filter second, followed by `max_patients`. All input
and label rows are validated before selection, including excluded rows. An empty
eligible cohort is an error. The existing `--presenting-evidence-only true|false`
option in the original serial runner overrides its matching YAML flag; its effective value is saved.

The four variants are `llm_only` (S/O), `patient_retrieval` (S/O + similar patient),
`guidelines` (S/O + guidelines), and `patient_retrieval_guidelines` (both sources).
The component and variant definitions live in `scripts/med_copilot_variants.py`. Required components are prepared once per patient and reused across consuming variants/models. Variant definitions select existing prompt templates and bind component outputs to template fields. Registered settings belong in `variant_settings`; they do not create arbitrary runner overrides. See [Adding a new variant](../../med_copilot/docs/adding_a_new_variant.md) for the always/conditional edit workflow.
Only evidence sections differ in final prompts; the combined project prompt and
shared diagnosis/Plan instructions remain unchanged. Source failures block their
dependent variants while independent variants continue.

The shipped config selects all four variants and these final models:

| Alias | Model tag |
| --- | --- |
| `gpt_4o_mini` | `gpt-4o-mini` |
| `ollama_gemma3_4b` | `gemma3:4b` |
| `ollama_llama3_2_3b` | `llama3.2:3b` |
| `ollama_ministral_3_3b` | `ministral-3:3b` |

The Ollama entries reuse the upstream OpenAI-compatible adapter at
`http://127.0.0.1:11435/v1` and request `json_schema` output, capped at 768 tokens.
Download these weights before running; no key variable is needed. `ollama_qwen2_5_14b_speedup`
(`qwen2.5:14b`) is registered as the preliminary Plan-correctness judge
(the shipped `plan_judge` in `experiment.yaml`); it is loaded separately
because it is not a selected generating model. Optional hosted/local
Llama 3.3 70B entries remain registered. The judge has a 64-token cap and the
shipped `plan_judge_concurrency` is 8. The shipped full test input has 4,959 rows,
of which 2,378 pass both filters. The shipped `max_patients: null` selects all
2,378 (38,048 final requests). A positive value selects a seeded eligible sample.
Shared component calls are additional and are not repeated per variant/model.

Patient retrieval needs `OPENAI_API_KEY`. Guidelines need `OPENAI_API_KEY` and
`GRAPHRAG_API_KEY`. Shared credentials/resources are checked only for selected
sources: keyless local `llm_only` needs neither shared key, reference artifacts,
nor retrieval encoder initialization. Selected final-provider credentials are
always checked. These are configuration checks; real requests verify access.

Before generation or output creation, the terminal displays grouped model/run/
variant/dataset settings, variant explanations, the exact sentinel note, aligned
cohort counts, and planned final requests. Additional resource/output paths and
evidence-model/retrieval/search tuning stay in YAML and metadata without adding
fields to this confirmation display. Confirm with `y` or `yes`. Any other
response, EOF or Ctrl+C cancels without run artifacts or generation calls.

`run.json.experiment` records every setting and its effective value (including
applied defaults), config path/source SHA-256 and model-registry reference. The
registry has its own fingerprint and effective upstream settings snapshot.
Every change to `experiment.yaml`, including comments, whitespace, or spelling
out an existing default, blocks a full runner resume because its exact source
SHA-256 must match. Plan-judge selection is checked by alias; its model-registry
settings are saved for provenance but are not separately compared for the judge.
Resolved keys are never saved. Keep validation, behavior, metadata and tests in
sync when adding settings. Older four-setting configs default to the combined
variant, sample paths, presenting-evidence filtering enabled, and recorded-plan
filtering disabled. New path/model/tuning settings also default to the previous
pipeline values (spelled out in the shipped YAML), preserving older config files.
The shipped YAML explicitly selects the new matrix/filters.

Each completed batch automatically creates `results_summary.html` and prints the
best variant per model using mean top-ranked condition BERTScore_F1, with separate
condition/Plan statistics and counts. All variants/models and all three ranks
remain available in the report and version-4 JSONL. Component captures are canonical; variants reference the outputs they consume. Historical schemas remain readable without rewriting old artifacts.

The original runner's encoder startup is isolated in `scripts/med_copilot_startup.py`. It reads the
config before retrieval imports FAISS, loads only a requested encoder, and retains
loading errors for dependent variants. The same configured model ID reaches the
preloader and retrieval manifest; source/encoder changes invalidate the cache.
Restart the runner after changing the encoder setting. Upstream implementations
remain unchanged. Streaming/JSON contracts and prompt wording remain in code;
other GraphRAG settings continue to belong to the guideline project's YAML.
The concurrent runner uses `scripts/experiment_runtime/worker.py` to load each worker's
physical CPU encoder before retrieval imports FAISS, retaining its retriever and
reranker across patients. Runtime metadata records effective devices and owned
Ollama settings/model identities. Plan judging runs separately and preserves
generation, evidence, text metrics, and diagnosis scores.
