# MED-COPILOT runner configuration

Edit `experiment.yaml` before each run. Model definitions live in the fixed sibling
`condition_plan_models.yaml`; tokens stay in the repository-root `.env` (see
`.env.example`). Paths resolve from the repository root, independently of the
launch directory. There are no model/config CLI overrides.

| Experiment setting | Meaning |
| --- | --- |
| `selected_models` | Nonempty, unique registry aliases, in execution order. |
| `temperature` | Shared final-model temperature, finite number from 0 to 2. |
| `max_retries` | Shared upstream final-model retry setting, integer at least 1. |
| `max_patients` | First N eligible patients in input order; positive integer or `null` for all. |
| `selected_variants` | Nonempty, unique evidence variants; every model runs each one. |
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

The YAML groups fields by pipeline stage and documents each one. The presenting-evidence filter runs first, the recorded-plan filter second, followed by `max_patients`. All input
and label rows are validated before selection, including excluded rows. An empty
eligible cohort is an error. The existing `--presenting-evidence-only true|false`
option overrides its matching YAML flag; its effective value is saved.

The four variants are `llm_only` (S/O), `patient_retrieval` (S/O + similar patient),
`guidelines` (S/O + guidelines), and `patient_retrieval_guidelines` (both sources).
Required evidence is prepared once per patient and reused across variants/models.
Only evidence sections differ in final prompts; the combined project prompt and
shared diagnosis/Plan instructions remain unchanged. Source failures block their
dependent variants while independent variants continue.

The shipped config selects all four variants and these final models:

| Alias | Model tag |
| --- | --- |
| `gpt_4o_mini` | `gpt-4o-mini` |
| `ollama_healthcare` | `Jayasimma/healthcare:latest` |
| `ollama_llama3` | `llama3:latest` |
| `ollama_gemma2_2b` | `gemma2:2b` |

The Ollama entries reuse the upstream OpenAI-compatible adapter at
`http://localhost:11434/v1` and request `json_schema` output. Keep that server
running; no key variable is needed for these entries. Optional hosted/local
Llama 3.3 70B entries remain registered. The shipped one-patient limit means 16
final requests before retries; shared evidence calls are additional. The sample
cohort has 10 input patients, 7 after presenting-evidence filtering, 5 after the
recorded-plan filter, and 1 after the limit.

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
Resolved keys are never saved. Keep validation, behavior, metadata and tests in
sync when adding settings. Older four-setting configs default to the combined
variant, sample paths, presenting-evidence filtering enabled, and recorded-plan
filtering disabled. New path/model/tuning settings also default to the previous
pipeline values (spelled out in the shipped YAML), preserving older config files.
The shipped YAML explicitly selects the new matrix/filters.

Each completed batch automatically creates `results_summary.html` and prints the
best variant per model using mean top-ranked condition BERTScore_F1, with separate
condition/Plan statistics and counts. All variants/models and all three ranks
remain available in the report and version-3 JSONL.

Encoder startup is isolated in `scripts/med_copilot_startup.py`. It reads the
config before retrieval imports FAISS, loads only a requested encoder, and retains
loading errors for dependent variants. The same configured model ID reaches the
preloader and retrieval manifest; source/encoder changes invalidate the cache.
Restart the runner after changing the encoder setting. Upstream implementations
remain unchanged. Streaming/JSON contracts and prompt wording remain in code;
other GraphRAG settings continue to belong to the guideline project's YAML.
