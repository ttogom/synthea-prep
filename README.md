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
python tests/test_med_copilot_helpers.py
python tests/test_med_copilot_retrieval.py
python tests/test_med_copilot_observability.py
python tests/test_med_copilot_scoring.py
python tests/test_med_copilot_models.py
python tests/test_run_med_copilot.py
python scripts/run_med_copilot.py
# Include patients without presenting evidence as well:
python scripts/run_med_copilot.py --presenting-evidence-only false
```

The runner automatically loads `.env` from the repository root before initializing model clients, even when launched from another directory. [`.env.example`](.env.example) lists the supported credential-variable names. Set `OPENAI_API_KEY` there; `GRAPHRAG_API_KEY=${OPENAI_API_KEY}` reuses it for guidelines, or you can supply a separate key. Add other provider keys only when selecting models that need them. Existing exported environment variables take precedence over `.env`. The local `.env` file is gitignored; keep actual tokens out of the model YAML.

[requirements.txt](requirements.txt) contains the Python dependencies and the `en_core_sci_md` model. Keep it updated as the pipeline changes. BioClinicalBERT, the cross-encoder, and BERTScore's default RoBERTa-large encoder download from Hugging Face on first use, so initial setup requires internet access. The unchanged upstream evaluator also attempts four NLTK resource downloads at import. The offline tests replace those download calls and pretrained BERTScore; real-tokenizer parity tests require locally provisioned NLTK data.

Configure `PATIENT_INPUT` and `PATIENT_LABELS` independently in [scripts/run_med_copilot.py](scripts/run_med_copilot.py). For now, their defaults are `experiments/input/test_sample_so.jsonl` and `experiments/input/test_sample_labels.jsonl`. Each input line has a nonempty, unique string `id` and prepared `subjective` and `objective` strings, used directly without an LLM normalization call. Labels are required and joined by exact `id`, independently of file order:

```json
{"id": "patient-id", "condition": "strep_throat", "plan": ["Treatment", "Follow-up"], "has_presenting_evidence": true}
```

Every input ID must have one label before filtering. Duplicate IDs, invalid condition/plan types, or a non-boolean evidence flag fail preflight before per-patient API calls. A label file may contain extra IDs for other input subsets. `bin` and other label metadata do not enter prompts. The configured sample contains 10 matched patients: the default evidence filter selects 7, and `--presenting-evidence-only false` includes all 10. To use the larger `patient_records_so.jsonl` input later, configure **both** paths and supply its matching label file.

`--presenting-evidence-only` defaults to `true` and selects patients whose joined label has `has_presenting_evidence: true`. Pass `--presenting-evidence-only false` to process every input patient. Input order is preserved; excluded patients have no result rows and do not become `not_processed`. An empty selected cohort fails before a run directory is created. The filter applies only to experiment patients, not the similar-patient corpus.

Keep all runner configuration files in [experiments/configs/](experiments/configs/README.md). The fixed [experiment.yaml](experiments/configs/experiment.yaml) contains `selected_models`, the shared `temperature`, `max_retries`, and `max_patients`. The fixed [condition_plan_models.yaml](experiments/configs/condition_plan_models.yaml) contains model definitions only. The current experiment selects `gpt_4o_mini` and three downloaded Ollama models: `ollama_healthcare` (`Jayasimma/healthcare:latest`), `ollama_llama3` (`llama3:latest`), and `ollama_gemma2_2b` (`gemma2:2b`), with temperature 1, a retry limit of 3, and a one-patient limit. Optional hosted and local Llama 3.3 70B definitions remain registered; the hosted Groq entry requires enterprise access.

Edit `selected_models` in `experiments/configs/experiment.yaml` to choose models in execution order. Add their credential variables to the repository-root `.env` (see `.env.example`), then run:

```bash
python scripts/run_med_copilot.py
# Include patients without presenting evidence:
python scripts/run_med_copilot.py --presenting-evidence-only false
```

Startup checks the selected aliases and nonblank API-key variables, including `OPENAI_API_KEY` and `GRAPHRAG_API_KEY` for shared stages. Configuration errors name the affected settings and files and terminate the run. These checks do not authenticate credentials or verify account/model access. Valid settings produce an ordered model list with provider/model identities and credential variable names. Confirm with `y` or `yes` to start; any other response, EOF, or Ctrl+C cancels without creating experiment output or making generation requests. Temperature, retry settings, and shared-key status are omitted from this prompt; effective settings remain recorded in `run.json`.

Set `max_patients: 1` in `experiments/configs/experiment.yaml` to run just one patient, or another positive integer to run the first N eligible patients in input-file order. The current `max_patients: 1` runs one patient; `max_patients: null` runs all eligible patients. The limit applies after `--presenting-evidence-only` filtering. All input records and labels are still validated before selection; values such as 0, negative numbers, booleans, strings and floats are rejected. `run.json.experiment.max_patients` records the configured limit, and `run.json.evaluation` records that limit and the resulting input/selected counts and selected/excluded patient IDs. Excluded patients have no `results.jsonl` rows, including when the run is interrupted.

Model/config CLI overrides are unavailable: change the fixed YAML files and restart. The shared `temperature` and `max_retries` are passed into every selected final model; putting either in a registry entry is rejected.

Each model definition uses upstream GraphRAG `LanguageModelConfig` fields (`type: chat`, `model_provider`, `model`, optional `api_base`, and other provider settings), plus `api_key_env` and `output_mode`. For Anthropic or Gemini, add an entry with `model_provider: anthropic` or `gemini`, the provider's model ID and its credential environment-variable name. Do not put literal API keys in YAML. An explicit endpoint may omit `api_key_env` for an unauthenticated local server; the adapter supplies the nonsecret placeholder required by upstream validation. This connects to an already running model server; it does not download or host weights. Existing `OPENAI_API_KEY` and `GRAPHRAG_API_KEY` are still needed by the shared evidence stages, even when every final model is served elsewhere.

The three `ollama_*` entries use your running local Ollama server at `http://localhost:11434/v1`, retain the exact installed model tags, and request JSON schema output. They omit `api_key_env`; the adapter supplies the nonsecret placeholder required by the upstream client. Ollama supports this through its [OpenAI-compatible endpoint](https://docs.ollama.com/api/openai-compatibility). Ensure Ollama is running before confirming a run. Shared evidence stages and `gpt_4o_mini` still use your OpenAI credentials.

The project-owned [model adapter](scripts/med_copilot_models.py) constructs models through the unchanged upstream GraphRAG factory and uses `LitellmChatModel.completion`, including its configured retry/rate-limit wrappers. `output_mode` is explicitly `json_schema` (default), `json_object`, or `prompt` (JSON requested in the shared prompt without an API formatting option). Unsupported request parameters produce a model-specific error; there is no automatic model fallback, mode downgrade, or repair request. The master experiment supplies temperature and retry limits; other unspecified settings retain upstream defaults. Effective settings are recorded per model in run metadata. The default experiment retains temperature 1 from the previous model definitions. Input/output paths remain in the runner for now.

Retrieval, guideline questions, GraphRAG and final prompt construction execute once per patient. All selected final models receive independent copies of those exact messages and run sequentially. Each model retains its own raw reply, parsed predictions, provider metadata, generation status/error and evaluation. One model's failure does not prevent the remaining models or patients from running.

The runner prints a `RANKED CONDITIONS AND PLANS` section grouped by model alias and provider/model name. Each model contains conditions numbered 1–3, with a separately numbered Plan indented under each condition. Failed models, unavailable fields and empty Plans are labeled explicitly. Console formatting leaves the raw responses and saved predictions unchanged.

Every final model is asked for the same JSON output with a `predictions` array of exactly three condition/Plan pairs, ordered from most to least likely. Each pair contains a concise `condition` string and its own `plan` array of at most three nonblank action strings. Plan items retain their original order, wording, duplicates, and internal commas; they are joined only for scoring.

After generation, [scripts/med_copilot_scoring.py](scripts/med_copilot_scoring.py) invokes the unchanged upstream metric helper separately for each field of each ranked pair. Each condition is compared to `label["condition"].replace("_", " ")`; each generated Plan is joined with `", "` and compared to `", ".join(label["plan"])`. All ranks use the same patient labels. Each model's `assessment_plans[model_id].evaluation` array contains three entries with `rank` (1–3), `condition`, and `plan`; each field has six metric keys and an explicit status. There is no rank weighting or combined score. Exactly `["none recorded"]` in the **label** skips Plan scoring at every rank; an empty joined reference Plan is unavailable. An empty generated Plan remains `[]` and receives `missing_generation`. Unavailable/error scores are JSON `null`, not zero or `NaN`. Ordinary scoring errors preserve other fields and ranks and do not change the generation status or run exit code.

Scores preserve upstream defaults and operate on one patient/rank/field at a time. Thus BLEU is **singleton-corpus BLEU**; averaging saved BLEUs does not reproduce dataset corpus BLEU, and exact texts shorter than four tokens can score zero. BERTScore reloads its model for each helper call and truncates at its encoder's token limit; this implementation adds no cache, chunking, clinical normalization, or composite score. The run metadata records the label/helper hashes, selected cohort, installed package versions, and configured scoring defaults; it does not establish the model-weight revision used by the original study.

Reference matching uses S/O only for embeddings, keyword overlap, and cross-encoder reranking, through [scripts/med_copilot_retrieval.py](scripts/med_copilot_retrieval.py). The selected reference's complete SOAP is still supplied to final generation. S/O embeddings and the FAISS index are built on first use under `med_copilot/data/patient_retrieval_so/`, separate from the upstream S/O/A artifacts. A `matching.json` manifest binds the cache to the ordered S/O texts and encoder identity and checks artifact hashes; missing, stale, or corrupt caches are rebuilt. Upstream source and artifacts are left intact.

Each execution writes a new directory, never overwriting an existing one:

```text
experiments/output/<Pacific timestamp>__<input stem>__<branch>/   # e.g. 2026-10-17_153045PDT__test_sample_so__integrate-medcopilot
├── run.json        # run status, times, branch, commit, input and reference-resource fingerprints
└── results.jsonl   # one line per patient: shared evidence plus assessment_plans keyed by model alias
```

`run.json.experiment` records the master config path/SHA-256, resolved registry, effective selection, temperature, retry limit and `max_patients`. `run.json` also records `schema_version: 2` and the registry path/SHA-256 and effective per-model upstream settings (excluding API keys) under `assessment_plan_models`. Run statuses remain `running`, `completed`, `completed_with_errors`, and `failed_incomplete`. Each `results.jsonl` row has shared evidence/prompts and an `assessment_plans` mapping. Each model slot contains `status`, `error`, `predictions`, `assessment_plan_raw`, `model_responses`, and `evaluation`; final fields no longer appear at the patient root. Shared `model_responses` contains only evidence-stage responses.

Patient status is `completed` when every selected model completes, `failed_incomplete` if a shared stage or any final model fails, or `not_processed` for an unstarted patient. Model slots start `not_processed`, become `completed` after a consumed response, or `failed_incomplete` on API/response failure. Failures retain any obtained raw reply/metadata; failed models are not scored. Ordinary metric errors remain field/rank/model-local and do not change generation status. An obtained but malformed JSON response retains the existing `completed` semantics and explicit missing-generation scores. Interruptions preserve available outputs/scores and mark the run stopped; later model/patient slots remain unattempted. The command exits nonzero unless the run is `completed`.

Every response goes through the existing parser: malformed JSON, duplicate keys, nonfinite numbers, unexpected top-level keys, or a prediction count other than three produce `predictions: null`. Invalid fields become `null` without discarding other valid fields/ranks; no arrays are truncated and Markdown is not a fallback. Historical result files are not rewritten. `experiments/output/` remains gitignored. The runner locates its default model config and guideline artifacts independently of the launch directory. Offline tests mock external requests; live generation and provider compatibility remain unverified.

The [patient output schema](med_copilot/docs/output_schema.md) documents exact keyword replies, processed retrieval keywords, retained hybrid and cross-encoder IDs/scores, and per-stage provider response metadata. Captures survive subsequent failures and interruptions. They do not enter clinical prompts.
