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

python tests/test_med_copilot_config.py
python tests/test_med_copilot_startup.py
python tests/test_med_copilot_helpers.py
python tests/test_med_copilot_retrieval.py
python tests/test_med_copilot_observability.py
python tests/test_med_copilot_scoring.py
python tests/test_med_copilot_models.py
python tests/test_collate_med_copilot_results.py
python tests/test_run_med_copilot.py
python scripts/run_med_copilot.py
# Include patients without presenting evidence as well:
python scripts/run_med_copilot.py --presenting-evidence-only false
```

The runner automatically loads `.env` from the repository root before initializing model clients, even when launched from another directory. [`.env.example`](.env.example) lists the supported credential-variable names. Set `OPENAI_API_KEY` there; `GRAPHRAG_API_KEY=${OPENAI_API_KEY}` reuses it for guidelines, or you can supply a separate key. Add other provider keys only when selecting models that need them. Existing exported environment variables take precedence over `.env`. The local `.env` file is gitignored; keep actual tokens out of the model YAML.

[requirements.txt](requirements.txt) contains the Python dependencies and the `en_core_sci_md` model. Keep it updated as the pipeline changes. BioClinicalBERT, the cross-encoder, and BERTScore's default RoBERTa-large encoder download from Hugging Face on first use, so initial setup requires internet access. The unchanged upstream evaluator also attempts four NLTK resource downloads at import. The offline tests replace those download calls and pretrained BERTScore; real-tokenizer parity tests require locally provisioned NLTK data.

Configure `patient_input` and `patient_labels` in [experiments/configs/experiment.yaml](experiments/configs/experiment.yaml); relative paths resolve from the repository root. Their defaults are `experiments/input/test_sample_so.jsonl` and `experiments/input/test_sample_labels.jsonl`. Each input line has a nonempty, unique string `id` and prepared `subjective` and `objective` strings, used directly without an LLM normalization call. Labels are required and joined by exact `id`, independently of file order:

```json
{"id": "patient-id", "condition": "strep_throat", "plan": ["Treatment", "Follow-up"], "has_presenting_evidence": true}
```

Every input ID must have one label before filtering. Duplicate IDs, invalid condition/plan types, or a non-boolean evidence flag fail preflight before per-patient API calls. A label file may contain extra IDs for other input subsets. `bin` and other label metadata do not enter prompts. The configured sample contains 10 matched patients: the evidence filter retains 7, the recorded-plan filter retains 5, and the one-patient limit selects 1. To use the larger `patient_records_so.jsonl` input later, configure **both** paths and supply its matching label file.

`presenting_evidence_only: true` selects patients whose joined label has `has_presenting_evidence: true`. `exclude_none_recorded_plans: true` then excludes patients whose label `plan` is exactly `["none recorded"]`, preserving case and whitespace distinctions. Set either YAML flag to `false` to disable that filter. The existing `--presenting-evidence-only true|false` option overrides only its matching YAML flag. Input order is preserved; excluded patients have no result rows and do not become `not_processed`. An empty selected cohort fails before a run directory is created. The filter applies only to experiment patients, not the similar-patient corpus.

Keep all runner configuration files in [experiments/configs/](experiments/configs/README.md). The fixed [experiment.yaml](experiments/configs/experiment.yaml) groups settings by dataset/cohort, evidence variants, patient retrieval, clinical guidelines, final generation, and results. It contains dataset/reference/output paths, model selections, retrieval/search tuning, filters, and generation settings. The fixed [condition_plan_models.yaml](experiments/configs/condition_plan_models.yaml) contains model definitions only. The current experiment selects `gpt_4o_mini` and three downloaded Ollama models: `ollama_healthcare` (`Jayasimma/healthcare:latest`), `ollama_llama3` (`llama3:latest`), and `ollama_gemma2_2b` (`gemma2:2b`), with all four variants, temperature 1, a retry limit of 3, and a one-patient limit (16 final requests before retries). Optional hosted and local Llama 3.3 70B definitions remain registered; the hosted Groq entry requires enterprise access.

Edit `selected_models` in `experiments/configs/experiment.yaml` to choose models in execution order. Add their credential variables to the repository-root `.env` (see `.env.example`), then run:

```bash
python scripts/run_med_copilot.py
# Include patients without presenting evidence:
python scripts/run_med_copilot.py --presenting-evidence-only false
```

Startup checks the selected aliases and nonblank API-key variables. Patient retrieval needs `OPENAI_API_KEY`; guidelines need `OPENAI_API_KEY` and `GRAPHRAG_API_KEY`. Unselected evidence sources require no credentials or resource initialization; `llm_only` with keyless local final models needs neither shared key nor retrieval-model downloads. Configuration errors name the affected settings and files and terminate the run. These checks do not authenticate credentials or verify account/model access. Valid settings display grouped model identities/key-variable names, run settings, explained evidence variants, dataset/cohort settings, vertically aligned filter counts, and planned final requests. Additional resource/output paths and evidence-stage model/tuning settings are configured in YAML and recorded in metadata without expanding this confirmation display. Confirm with `y` or `yes` to start; any other response, EOF, or Ctrl+C cancels without creating experiment output or making generation requests. All effective experiment settings and defaults remain recorded in `run.json.experiment`, alongside the config fingerprint.

Set `max_patients: 1` in `experiments/configs/experiment.yaml` to run just one patient, or another positive integer to run the first N eligible patients in input-file order. The current `max_patients: 1` runs one patient; `max_patients: null` runs all eligible patients. The limit applies after presenting-evidence filtering and recorded-plan filtering, in that order. All input records and labels are still validated before selection; values such as 0, negative numbers, booleans, strings and floats are rejected. `run.json.experiment.max_patients` records the configured limit, and `run.json.evaluation` records that limit and the intermediate cohort counts, selected/excluded patient IDs, and exclusion reasons (`missing_presenting_evidence`, `no_recorded_plan`, or `max_patients`). Excluded patients have no `results.jsonl` rows, including when the run is interrupted.

Model/config CLI overrides are unavailable: change the fixed YAML files and restart. The shared `temperature` and `max_retries` are passed into every selected final model; putting either in a registry entry is rejected.

Each model definition uses upstream GraphRAG `LanguageModelConfig` fields (`type: chat`, `model_provider`, `model`, optional `api_base`, and other provider settings), plus `api_key_env` and `output_mode`. For Anthropic or Gemini, add an entry with `model_provider: anthropic` or `gemini`, the provider's model ID and its credential environment-variable name. Do not put literal API keys in YAML. An explicit endpoint may omit `api_key_env` for an unauthenticated local server; the adapter supplies the nonsecret placeholder required by upstream validation. This connects to an already running model server; it does not download or host weights. Credentials for the selected shared evidence stages are still needed when final models are served elsewhere.

The three `ollama_*` entries use your running local Ollama server at `http://localhost:11434/v1`, retain the exact installed model tags, and request JSON schema output. They omit `api_key_env`; the adapter supplies the nonsecret placeholder required by the upstream client. Ollama supports this through its [OpenAI-compatible endpoint](https://docs.ollama.com/api/openai-compatibility). Ensure Ollama is running before confirming a run. Shared evidence stages and `gpt_4o_mini` still use your OpenAI credentials.

The project-owned [model adapter](scripts/med_copilot_models.py) constructs models through the unchanged upstream GraphRAG factory and uses `LitellmChatModel.completion`, including its configured retry/rate-limit wrappers. `output_mode` is explicitly `json_schema` (default), `json_object`, or `prompt` (JSON requested in the shared prompt without an API formatting option). Unsupported request parameters produce a model-specific error; there is no automatic model fallback, mode downgrade, or repair request. The master experiment supplies temperature and retry limits; other unspecified settings retain upstream defaults. Effective settings are recorded per model in run metadata. The default experiment retains temperature 1 from the previous model definitions. Input, label, output and reference-resource paths all live in the experiment YAML.

Select any nonempty subset of `selected_variants` in YAML:

| Variant | Final prompt evidence |
| --- | --- |
| `llm_only` | Prepared patient S/O |
| `patient_retrieval` | S/O + similar-patient example |
| `guidelines` | S/O + clinical guidelines |
| `patient_retrieval_guidelines` | S/O + both sources (previous behavior) |

Required retrieval, guideline questions and GraphRAG execute once per patient and are reused across all variants/models. Each variant renders one prompt; all selected final models receive independent copies of those exact messages and run sequentially. [diagnosis_plan.py](med_copilot/prompts/diagnosis_plan.py) derives `VARIANT_TEMPLATES` by removing only the unused evidence sections from the existing project prompt, including the reference-specific fallback when reference evidence is absent. The combined prompt's wording/order and shared output instructions remain unchanged. The runner keeps visible `if` blocks for the two evidence sources followed by one final variant loop. [med_copilot_startup.py](scripts/med_copilot_startup.py) handles conditional encoder initialization before FAISS import for Apple Silicon compatibility, preserving encoder errors for dependent variants. The YAML encoder ID controls both preloading and cache identity; changing it after importing the runner requires restarting the script. Each model retains its own raw reply, parsed predictions, provider metadata, generation status/error and evaluation. One model's failure does not prevent the remaining models or patients from running.

The runner prints a `RANKED CONDITIONS AND PLANS` section grouped by model alias and provider/model name. Each model contains conditions numbered 1–3, with a separately numbered Plan indented under each condition. Failed models, unavailable fields and empty Plans are labeled explicitly. Console formatting leaves the raw responses and saved predictions unchanged.

Every final model is asked for the same JSON output with a `predictions` array of exactly three condition/Plan pairs, ordered from most to least likely. Each pair contains a concise `condition` string and its own `plan` array of at most three nonblank action strings. Plan items retain their original order, wording, duplicates, and internal commas; they are joined only for scoring.

After generation, [scripts/med_copilot_scoring.py](scripts/med_copilot_scoring.py) invokes the unchanged upstream metric helper separately for each field of each ranked pair. Each condition is compared to `label["condition"].replace("_", " ")`; each generated Plan is joined with `", "` and compared to `", ".join(label["plan"])`. All ranks use the same patient labels. Each variant/model's `variants[variant_id].assessment_plans[model_id].evaluation` array contains three entries with `rank` (1–3), `condition`, and `plan`; each field has six metric keys and an explicit status. There is no rank weighting or combined score. Exactly `["none recorded"]` in the **label** skips Plan scoring at every rank; an empty joined reference Plan is unavailable. An empty generated Plan remains `[]` and receives `missing_generation`. Unavailable/error scores are JSON `null`, not zero or `NaN`. Ordinary scoring errors preserve other fields and ranks and do not change the generation status or run exit code.

Scores preserve upstream defaults and operate on one patient/rank/field at a time. Thus BLEU is **singleton-corpus BLEU**; averaging saved BLEUs does not reproduce dataset corpus BLEU, and exact texts shorter than four tokens can score zero. BERTScore reloads its model for each helper call and truncates at its encoder's token limit; this implementation adds no cache, chunking, clinical normalization, or composite score. The run metadata records the label/helper hashes, selected cohort, installed package versions, and configured scoring defaults; it does not establish the model-weight revision used by the original study.

The experiment YAML selects the patient embedding, keyword-extraction and cross-encoder models, hybrid embedding weight, candidate count and retained reranking count. Only the first reranked reference enters the final prompt. It also selects the guideline-question model, community hierarchy level, answer description and diagnostic verbosity. `guideline_response_type` is free-form text (for example, `Single Paragraph`, `Single Sentence` or `List of 3-7 Points`), matching upstream GraphRAG. Reference matching uses S/O only for embeddings, keyword overlap, and cross-encoder reranking, through [scripts/med_copilot_retrieval.py](scripts/med_copilot_retrieval.py). The selected reference's complete SOAP is still supplied to final generation. S/O embeddings and the FAISS index are built on first use under the configured `patient_retrieval_artifacts_dir` (default `med_copilot/data/patient_retrieval_so/`), separate from the upstream S/O/A artifacts. A `matching.json` manifest binds the cache to the ordered S/O texts and encoder identity and checks artifact hashes; missing, stale, or corrupt caches are rebuilt. Upstream source and artifacts are left intact.

Each execution writes a new directory, never overwriting an existing one:

```text
experiments/output/<Pacific timestamp>__<input stem>__<branch>/   # e.g. 2026-10-17_153045PDT__test_sample_so__integrate-medcopilot
├── run.json             # configuration, cohort, provenance, status and resource fingerprints
├── results.jsonl        # one line per patient: shared evidence, then variants → models → outputs/scores
└── results_summary.html # generated automatically after a completed batch, including batches with errors
```

`run.json.experiment` records every effective YAML setting/default plus config path/SHA-256 and registry reference. Schema version 3 adds variant evidence definitions, exact template hashes, and the project prompt file fingerprint. `assessment_plan_models` retains registry path/SHA-256 and effective per-model upstream settings once per run, excluding resolved keys. Only selected and successfully initialized evidence resources are fingerprinted; unused/failed resources are `null`. Historical schemas remain readable by the collator.

Each result row retains shared evidence, rankings, traces and evidence-stage provider responses once, adds `evidence_status` (`not_requested`, `not_processed`, `completed`, `failed_incomplete` plus error), then stores `variants[variant_id]`. Each variant has its own status/error, evidence-source names, exact rendered `assessment_plan_prompts`, and `assessment_plans[model_id]`. Model slots retain `status`, `error`, `predictions`, `assessment_plan_raw`, `model_responses`, and three-rank `evaluation`. Evidence failures block only dependent variants. A successful empty retrieval still completes its source and uses the existing no-reference fallback. Other variants/models and later patients continue.

Patient status is `completed` when all selected variant/model generations complete, `failed_incomplete` if any fails, or `not_processed` for an unstarted selected patient. Run statuses remain `running`, `completed`, `completed_with_errors`, and `failed_incomplete`. Failures retain any obtained raw reply/metadata; failed models are not scored. Ordinary metric errors remain field/rank/model-local and do not change generation status. An obtained but malformed JSON response retains the existing `completed` semantics and explicit missing-generation scores. Interruptions preserve outputs/scores, mark the run stopped, and retain every unattempted variant/model/patient slot. Excluded patients never receive result rows or placeholders.

The runner's `RESULTS` block automatically calls [the existing collator](scripts/collate_med_copilot_results.py). Terminal and HTML summaries show each model's mean **top-ranked condition BERTScore_F1** by variant, choose the highest mean (ties retain config order), and report that variant's separate top-ranked condition/Plan mean, population standard deviation, min/max and scored count. Missing/skipped/error scores remain unavailable; counts expose differing scored subsets. The HTML retains all three ranks, six metrics per condition/Plan, patient details and paired comparisons against `llm_only`; it creates no combined condition/Plan score. Report generation needs only saved results and the fingerprint-matching label file, and can be repeated with `python scripts/collate_med_copilot_results.py experiments/output/<run-directory>`. A report-write failure preserves completed JSON artifacts, records `run.json.results_summary` failure metadata, and exits nonzero; interruptions preserve JSON without generating a report. The command exits successfully only for a completed run or user cancellation.

Every response goes through the existing parser: malformed JSON, duplicate keys, nonfinite numbers, unexpected top-level keys, or a prediction count other than three produce `predictions: null`. Invalid fields become `null` without discarding other valid fields/ranks; no arrays are truncated and Markdown is not a fallback. Historical result files are not rewritten. `experiments/output/` remains gitignored. The runner locates its default model config and guideline artifacts independently of the launch directory. Offline tests mock external requests; live generation and provider compatibility remain unverified.

The [patient output schema](med_copilot/docs/output_schema.md) documents exact keyword replies, processed retrieval keywords, retained hybrid and cross-encoder IDs/scores, and per-stage provider response metadata. Captures survive subsequent failures and interruptions. They do not enter clinical prompts.
