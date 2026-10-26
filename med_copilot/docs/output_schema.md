# MED-COPILOT patient output schema (version 3)

Each run appends one JSON object per selected input patient to `results.jsonl`. JSONL
escaping is ordinary JSON escaping: after `json.loads`, captured strings retain
their original characters, including whitespace, Unicode, and embedded newlines.
`run.json.schema_version` is `3` for variant/model runs. Historical files retain their historical schema. The patient keys are written in this order:

| Field | Value |
|---|---|
| `id` | Prepared input patient ID. |
| `status` | `completed`, `failed_incomplete`, or `not_processed`. |
| `selected_reference_patient` | Minimal identity of the reference whose formatted full SOAP example enters final generation, or `null`. |
| `guideline_questions` | Parsed `Key Questions` list, or `null`; the raw question reply remains the GraphRAG query. |
| `guideline_evidence` | GraphRAG answer text, or `null`. |
| `graphrag_context` | Returned context tables as `{columns, index, data}`, or `null`. |
| `keyword_extraction_raw` | Exact string from the keyword API response's `choices[0].message.content`, captured before validation. Malformed JSON, duplicates, category structure, empty strings, and whitespace are preserved. `null` means extraction was not attempted or no string content was obtained. |
| `retrieval_keywords` | Actual keyword list returned by successful extraction: categories flattened in symptoms/diagnoses/medications order, terms trimmed, duplicates removed in first-occurrence order. `[]` means successful extraction with no terms. `null` means no valid list was produced. These are distinct from guideline questions. |
| `retrieved_patient_ids` | Retained hybrid ranking, normally up to 20 references, before cross-encoder reranking. Each entry contains a minimal identity and its unrounded `embedding_score`, `keyword_score`, and `hybrid_score`. |
| `reranked_patient_ids` | Retained cross-encoder ranking, up to five references, containing minimal identity and the preceding scores plus `rerank_score`. Ties preserve hybrid order. When a reference is selected, its identity equals the first entry's identity. |
| `model_responses` | Shared evidence-stage provider metadata lists, described below. |
| `evidence_status` | Per-source status/error for `patient_retrieval` and `guidelines`, as described below. |
| `variants` | Selected variants, in config order; each contains its own prompts and model outputs. |


## Final-model outputs and configuration

Each `variants[variant_id]` contains:

| Field | Value |
|---|---|
| `status` | `not_processed`, `completed`, or `failed_incomplete`; completed only when every selected final model completes generation. |
| `error` | `{type, message}` for unavailable required evidence, or `null`; model errors remain in their model slots. |
| `evidence_sources` | `[]`, `["patient_retrieval"]`, `["guidelines"]`, or `["patient_retrieval", "guidelines"]`. Names reference shared patient evidence. |
| `assessment_plan_prompts` | Exact final `{system, user}` messages before generation, or `null` for an unattempted/blocked variant. |
| `assessment_plans` | Model slots keyed by selected registry aliases, in selection order. |

Each `variants[variant_id].assessment_plans[model_id]` contains these fields in order:

| Field | Value |
|---|---|
| `status` | `not_processed`, `completed`, or `failed_incomplete`. This describes generation, independently of metric statuses. |
| `error` | Generation error `{type, message}`, with selected credential values redacted, or `null`. |
| `predictions` | Exactly three ranked `{condition, plan}` pairs, or `null` when structurally invalid/unavailable. |
| `assessment_plan_raw` | Exact consumed final content, including malformed content, or `null` before content is obtained. |
| `model_responses` | List of response metadata objects for this model; empty when no response is obtained. |
| `evaluation` | Three ordered `{rank, condition, plan}` score entries, or `null` when scoring has not started. |

For example (shared capture fields and evaluation arrays omitted here):

```json
{
  "id": "prepared-patient-id",
  "status": "failed_incomplete",
  "evidence_status": {
    "patient_retrieval": {"status": "not_requested", "error": null},
    "guidelines": {"status": "completed", "error": null}
  },
  "variants": {
    "llm_only": {
      "status": "failed_incomplete", "error": null,
      "evidence_sources": [],
      "assessment_plan_prompts": {"system": "You are a clinical expert.", "user": "exact S/O-only prompt"},
      "assessment_plans": {
        "gpt_4o_mini": {
          "status": "completed", "error": null,
          "predictions": [{"condition": "A", "plan": []}, {"condition": "B", "plan": []}, {"condition": "C", "plan": []}],
          "assessment_plan_raw": "{...}", "model_responses": [], "evaluation": null
        },
        "llama_3_3_70b": {
          "status": "failed_incomplete",
          "error": {"type": "RuntimeError", "message": "provider request failed"},
          "predictions": null, "assessment_plan_raw": null,
          "model_responses": [], "evaluation": null
        }
      }
    },
    "guidelines": {
      "status": "not_processed", "error": null,
      "evidence_sources": ["guidelines"], "assessment_plan_prompts": null,
      "assessment_plans": {
        "gpt_4o_mini": {"status": "not_processed", "error": null, "predictions": null, "assessment_plan_raw": null, "model_responses": [], "evaluation": null},
        "llama_3_3_70b": {"status": "not_processed", "error": null, "predictions": null, "assessment_plan_raw": null, "model_responses": [], "evaluation": null}
      }
    }
  }
}
```

The complete patient record also contains all shared fields in the table above.
Required retrieval, questions and guideline evidence are computed once per patient.
Each selected variant renders its prompt once; every selected model gets an
independent copy of that variant's exact messages. Final raw replies, predictions, metadata and scores exist only in
the model slots; no default model is implicitly copied to patient-level fields.

Console presentation is owned by the runner: its `RANKED CONDITIONS AND PLANS`
heading precedes model alias/provider/model labels, three numbered conditions,
and each condition's indented, numbered Plan items. Failed or unavailable outputs
are labeled within their model group. The formatter does not alter saved results
or invent missing Plan items; the generation adapter does not print final replies.

Models execute sequentially in configured selection order. Ordinary model
failures do not stop sibling models or subsequent patients. Refused/truncated
responses retain obtained content/metadata but fail generation and are not
scored. An obtained but malformed JSON reply retains the prior `completed`
generation semantics and is evaluated as missing generation by field/rank.
A patient is `completed` only if every selected variant/model generation completes; otherwise
it is `failed_incomplete`. The existing run-status aggregation/exit code applies.
`evidence_status[source]` has `status` and `error`. Status is `not_requested` when
no selected variant needs the source, `not_processed` before an attempt,
`completed` on success (including empty retrieval), or `failed_incomplete` on
initialization/runtime failure. Errors record type and sanitized message. Only
dependent variants fail with `EvidenceUnavailable`; their model slots stay
`not_processed`, prompts stay `null`, and independent variants still run. Empty
successful retrieval uses the existing no-reference fallback. Guideline calls
are skipped if their only dependent variant is already blocked by retrieval.
Source errors preserve all captures obtained before failure.
An interruption retains earlier replies and any completed scores, marks the
current patient/run failed, and leaves later variant/model/patient slots unattempted. Scoring
starts after all final generation calls, so a generation interruption can retain
an earlier reply whose evaluation has not yet started.

All runner configuration files belong in `experiments/configs/`.
`experiment.yaml` is the fixed master config with `selected_models`,
`temperature`, `max_retries`, `max_patients`, `selected_variants`, `patient_input`,
`patient_labels`, `presenting_evidence_only`, and `exclude_none_recorded_plans`,
plus the path/model/tuning settings documented in
[`experiments/configs/README.md`](../../experiments/configs/README.md). The two shared generation settings are passed
into every selected final model before upstream configuration/construction.
Temperature must be a finite number from 0 to 2; max_retries must be an integer
of at least 1, matching the upstream configuration contract. Booleans and numeric
strings are rejected. The default experiment retains temperature 1 and retry
limit 3. Shared retrieval/question/GraphRAG stages keep their existing settings.

`condition_plan_models.yaml` contains only a `models` registry of upstream
GraphRAG `LanguageModelConfig` fields, plus `api_key_env` and `output_mode`.
Model entries cannot redefine temperature or max_retries: these belong in the
master config. Both YAML paths are fixed under `experiments/configs/` relative
to the repository root, independently of the launch directory. Model selection
and config-file command-line overrides are unavailable; edit the YAML and restart.
Only selected credentials are resolved. Invalid experiment fields/values, duplicate
YAML keys, missing files, malformed registries, unknown/duplicate selections and
missing selected credentials fail before a run directory or per-patient requests.

`run.json.experiment` records `config_file`, its exact source-byte `config_sha256`,
the resolved `model_config` path and all effective experiment settings, including
applied defaults (listed below). The registry path is provenance, not a YAML setting. Dataset, output and reference-resource paths are configured in the experiment YAML
and resolve from the repo root; guideline settings/table paths follow the guideline root.

`max_patients` must be `null` (meaning all eligible patients) or a
positive integer. It selects the first N patients in input-file order after the
presenting-evidence and exact recorded-plan filters. All input records and labels, including those outside
the limit, are validated before selection. Zero, negatives, booleans, strings and
floats are rejected at startup. The configured limit is recorded in
`run.json.experiment.max_patients` and `run.json.evaluation.max_patients`.
Evaluation `input_patient_count` still counts the full input;
`selected_patient_count` and `selected_patient_ids` describe the limited cohort.
`excluded_patient_ids` includes patients excluded by either evidence filtering
or recorded-plan filtering or the limit. Generation, scoring, console numbering, result rows and interrupted
run placeholders use only the selected cohort. A limit larger than the eligible
cohort runs all eligible patients; an empty eligible cohort remains an error.

The current experiment selects GPT-4o mini and the downloaded Ollama tags
`Jayasimma/healthcare:latest`, `llama3:latest` and `gemma2:2b`, using the aliases
`ollama_healthcare`, `ollama_llama3` and `ollama_gemma2_2b`. It retains a one-patient
limit. Ollama entries use `model_provider: openai`,
`api_base: http://localhost:11434/v1` and `output_mode: json_schema`, without
`api_key_env`. Existing configuration, generation and metadata paths handle these
entries; no runner source changes are required. Each selected Ollama tag and its
endpoint appear in the per-model effective settings in `run.json`; every patient
gets independent model result/scoring slots. The unselected Groq Llama entry
requires enterprise access; it is not an installed Ollama model.

At startup the runner loads the repository-root `.env` using the already-required
`python-dotenv`, before dependency/client initialization. Exported environment
variables take precedence. This supplies both shared-stage credentials and the
variables named by selected models' `api_key_env`; `${OPENAI_API_KEY}` expansion
can supply `GRAPHRAG_API_KEY` as shown in `.env.example`. Loading is independent
of the launch directory, and an absent `.env` preserves environment-only setup.
The local `.env` is gitignored and its contents are not copied into run artifacts.

Before creating experiment output or making generation requests, startup checks
all selected aliases and their nonblank key variables, plus shared-stage
`OPENAI_API_KEY` for patient retrieval and additionally `GRAPHRAG_API_KEY` for
guidelines, only when
selected variants require those sources. Local `llm_only` needs neither shared
key nor reference resources/encoder initialization. Unknown aliases and credential errors are
reported together with the affected YAML files and repository-root `.env`;
configuration failures exit nonzero. Keyless local endpoints and Azure managed
identity retain their configured authentication modes. These are configuration
checks, not authentication/model-access requests. A valid setup displays the
ordered aliases, provider/model identities and key-variable names and requires
`y` or `yes` to proceed. Any other answer, EOF or Ctrl+C cancels successfully
without run artifacts. The prompt groups model, run, variant and dataset/cohort settings, shows every
model/run/variant/dataset value, variant descriptions, a hyphenated exact sentinel note,
aligned intermediate cohort counts, and planned final requests. Additional resource/output
paths and evidence-model/search tuning are recorded in metadata without expanding
this confirmation display. `run.json` still records temperature and max_retries in the experiment
and effective per-model settings.

`run.json.assessment_plan_models` records `config_file`, the exact source-byte
`config_sha256`, ordered `selected_models`, and a `models` mapping. Each selected
entry records its credential environment-variable name, `output_mode`, and a
`config` snapshot of the effective upstream settings excluding `api_key`.
Literal API keys in model entries and credentials/query parameters in endpoint
URLs are rejected. Upstream settings/defaults, retries, timeouts and optional
rate limits are reused through `ModelFactory` and `LitellmChatModel.completion`.
No upstream source is modified. The full GraphRAG settings loader is unnecessary
for this registry because it also requires default chat/embedding definitions.

`output_mode` is `json_schema`, `json_object`, or `prompt`. The first uses the
existing strict condition/Plan response schema, the second requests valid JSON
without API schema enforcement, and the third relies on the same prompt's JSON
instructions. Every reply still goes through the same local parser. Unsupported
parameters fail explicitly (`drop_params=false`); there is no mode downgrade,
model fallback or repair request. Hosted Llama 3.3 70B and an OpenAI-compatible
local server are included as registry examples. Local servers must already be
running. Shared question/retrieval models retain their existing configuration.


## Variant provenance, cohort and reports

`run.json.experiment` includes all 23 effective experiment settings and defaults,
plus `config_file`, `config_sha256`, and the model-registry reference. Old configs
with only the original four required fields default to the combined variant,
sample paths, presenting-evidence filtering enabled, and recorded-plan filtering
disabled. The shipped YAML explicitly enables all four variants and both filters.

Resource/output path fields are `guidelines`, `experiment_output`, `patient_corpus`,
and `patient_retrieval_artifacts_dir`. Evidence model fields are
`guideline_question_model`, `patient_embedding_model`, `patient_keyword_model`,
and `patient_reranker_model`. Tuning fields are `patient_retrieval_alpha`,
`patient_retrieval_top_k`, `patient_rerank_top_k`, `guideline_community_level`,
`guideline_response_type`, and `guideline_verbose`. Each field has an effective
default and is always included in `run.json.experiment`. Response type is a
free-form description with examples in the YAML comments, not an enum.

The runner delegates config-aware encoder loading to `PatientRetrievalStartup`
before importing FAISS-dependent retrieval code. It skips unused encoders,
retains loading errors for dependent variants, and passes the same model identity
to the preloaded encoder and cache. A changed encoder setting after import is
rejected with a restart instruction before any cache can be mislabeled. This
retains the Apple Silicon initialization-order workaround outside the pipeline.

The optional existing CLI filter override is saved as the effective YAML setting;
the original source fingerprint is retained.

`variant_definitions[variant_id]` records `patient_retrieval`, `guidelines`, the
actual `VARIANT_TEMPLATES['variant_id']` identity, and its rendered-template source
SHA-256. `prompt_provenance` records the project `diagnosis_plan.py` path and exact
file SHA-256. All variants derive from the existing `EVALUATE_TEMPLATE_KEYINFO`
by removing whole unused evidence blocks, including the reference fallback when
reference evidence is omitted. The combined template is unchanged. Upstream
source is unchanged; the final-model factory/completion wrappers and evaluator
remain reused.

Evaluation metadata adds `exclude_none_recorded_plans`,
`excluded_plan_value: ["none recorded"]`, ordered `filter_order`,
`after_presenting_evidence_count`, `eligible_patient_count`, and
`exclusion_reasons` keyed by excluded patient ID (first applicable reason:
`missing_presenting_evidence`, `no_recorded_plan`, `max_patients`). Existing
input/selected counts, selected/excluded IDs and data/scoring fingerprints remain.
Only selected, successfully initialized evidence resources are fingerprinted;
`reference_data` entries for unused/failed sources are `null`. Model snapshots
are stored once per run. No resolved credentials are written.

After the batch finishes (including ordinary failures), the runner's `RESULTS`
block invokes the existing collator and writes `results_summary.html` alongside
the two JSON artifacts. `results_summary` in run metadata records success and
filename or a sanitized failure. Report failure preserves completed JSON outputs
and returns nonzero. Interrupted batches preserve JSON but skip report generation.
The collator continues to support historical schemas without modifying them.

Terminal and HTML summaries show each model's per-variant mean rank-1 condition
and Plan `BERTScore_F1`, then select its highest mean rank-1 **condition** variant.
Ties retain selected-variant order. That variant gets separate condition/Plan
mean, population standard deviation, min, max and scored count. Only available
scored values enter statistics, including real zeros; skipped/error/missing
scores do not. Counts make different scored subsets visible. There is no
condition/Plan composite. Existing HTML details retain all three ranks, six
metrics per field, and paired comparisons against `llm_only` when present.

Regenerate a report without model calls:

```bash
python scripts/collate_med_copilot_results.py experiments/output/<run-directory>
```

This requires the saved label path to remain available with the recorded hash.
The collator reads saved artifacts, never current model/config files, and writes
only its HTML report atomically.


## Condition and Plan evaluation

Final calls configured with `output_mode: json_schema` use the project-owned
`CONDITION_PLAN_RESPONSE_FORMAT` JSON schema: a single `predictions` array with exactly three objects, each with
required `condition: string` and `plan: array[string]` fields. Each Plan has at
most three items; no additional properties are allowed at either object level.
The final prompt requests three distinct concise diagnoses ordered from most to
least likely and a separate Plan tailored to each suspected condition, without
rationale or Markdown. Array position determines rank; the model does not emit
rank numbers or weights. Retrieval and guideline questions keep their behavior.

The generated shape is:

```json
{
  "predictions": [
    {"condition": "most likely condition", "plan": ["action", "another action"]},
    {"condition": "second most likely condition", "plan": ["action"]},
    {"condition": "third most likely condition", "plan": []}
  ]
}
```

`parse_condition_plan` preserves valid strings, rank order, and Plan items
verbatim. A missing/blank/nonstring condition becomes `null`. A Plan of the wrong
type, longer than three items, or containing nonstring/blank items becomes
`null`; `[]` is valid. An individually invalid field does not erase its valid
partner or other ranks. A non-object pair or one with unexpected keys becomes
`{condition: null, plan: null}` at that rank. Malformed JSON, fenced Markdown, a
non-object root, unexpected top-level keys, duplicate keys or nonfinite numbers
anywhere, or an incorrectly typed/sized `predictions` array produce
`predictions: null`. No arrays are truncated and there is no Markdown fallback.
The exact response stays in each model slot's `assessment_plan_raw`. Each variant stores its
`assessment_plan_prompts`; final response metadata
lives in each slot's `model_responses` list.
Generation statuses do not themselves certify response-format validity.

Gold labels are joined to the complete input by exact ID before cohort selection.
YAML `presenting_evidence_only` defaults to `true`, optionally overridden by the
existing `--presenting-evidence-only true|false` flag. YAML
`exclude_none_recorded_plans` excludes label `plan` exactly `["none recorded"]`
when enabled, then `max_patients` selects the first eligible IDs in input order. Only selected IDs are registered in the run. Excluded patients
have no result row, including when an interruption marks remaining selected
patients `not_processed`. `run.json.evaluation` records the option, input and
selected counts, selected/excluded IDs, label and evaluator fingerprints,
installed metric-library versions and configured settings.

Each prediction is scored separately against the same patient labels:
`variants[variant_id].assessment_plans[model_id].evaluation[i].condition` compares that model's `predictions[i].condition` to the label's
`condition` with underscores replaced by spaces, and `variants[variant_id].assessment_plans[model_id].evaluation[i].plan`
compares the generated and reference Plan arrays after joining each with
`", "`. The generated Plan itself stays an array. Each evaluation entry records
`rank: i + 1`. There is no rank weighting, averaging, maximum, or combined score.
No additional text cleaning, sorting, deduplication, or comma splitting is
applied. Labels never enter retrieval or generation prompts. Historical runs
retain their historical schema; scoring does not fall back to legacy top-level
`condition`, `assessment`, or `plan` fields.

After scoring starts for a model, its `evaluation` contains exactly three entries with ranks
1–3, even when predictions are unavailable. One entry has the following shape;
all six metric keys are always present in each field:

```json
{
  "rank": 1,
  "condition": {
    "status": "scored",
    "metrics": {
      "BLEU": 0.0,
      "METEOR": 0.5,
      "ROUGE-1": 0.6,
      "ROUGE-2": 0.3,
      "ROUGE-L": 0.6,
      "BERTScore_F1": 0.8
    },
    "error": null
  },
  "plan": {
    "status": "skipped_no_recorded_plan",
    "metrics": {
      "BLEU": null,
      "METEOR": null,
      "ROUGE-1": null,
      "ROUGE-2": null,
      "ROUGE-L": null,
      "BERTScore_F1": null
    },
    "error": null
  }
}
```

Each entry's status is one of:

- `scored`: the unchanged upstream helper returned six finite numeric scores.
- `skipped_no_recorded_plan`: the original Plan label is exactly
  `["none recorded"]`; no Plan metric call is made. Other spellings, whitespace,
  or lists with additional items do not trigger this sentinel rule.
- `missing_reference`: the joined Plan is empty or whitespace-only.
- `missing_generation`: the generated condition is null/blank, or the generated
  Plan is null or joins to empty/whitespace-only text (including `[]`).
- `error`: scoring or score validation failed. `error` contains `{type, message}`.
- `interrupted`: scoring raised a fatal interruption, recorded before re-raising.
- `pending`: the field/rank was not attempted before an interruption.

Except for `scored`, all six metric values are `null`. Condition scoring is
independent of the skipped Plan sentinel, which skips every rank's Plan. An
ordinary error in one field/rank does not discard successful scores elsewhere,
predictions, prompts, or evidence, and
does not change the generation status or exit code. Interruptions retain the
existing stopped-run semantics. Null/blank references are not replaced with
empty strings or zeros. When both are unavailable, missing-reference status takes
precedence over missing-generation; the exact Plan sentinel takes precedence over
either.

Every invocation compares singleton lists, separately for condition and Plan at
each rank. Rank order is preserved for future analysis without weighting.
BLEU uses corpus statistics for that singleton; METEOR and ROUGE/BERTScore F1
retain the upstream reductions. These are not cohort-level scores, and averaging
per-patient BLEUs is not corpus BLEU. Raw BERTScore is not a correctness
probability. Encoder truncation, upstream import-time NLTK download attempts,
and per-call BERTScore model initialization remain upstream behavior.

Both ranked lists are `null` until their order is known and `[]` if their stage
completes with an empty ranking. An empty hybrid ranking skips reranking, so
`reranked_patient_ids` remains `null` in that case. The lists capture the retained
rankings, not the larger initial FAISS pool, and contain no SOAP text or subject
IDs. Their scores come from structured candidates, before display rounding.

Minimal identity follows the same convention for the selected reference and
both lists:

- `{"note_id": "..."}` for a unique, nonblank string ID, preserved verbatim.
- `{"corpus_row": N}` when no usable `note_id` exists.
- `{"note_id": "...", "corpus_row": N}` when the ID is duplicated.

`corpus_row` is a zero-based position in the loaded reference JSON array, not a
DataFrame label or a `subject_id`. Multiple records can belong to one subject.

## Provider responses

Each patient receives a fresh dictionary with independent lists in this order:

```json
{
  "keyword_extraction": [],
  "guideline_questions": [],
  "guideline_query_embedding": [],
  "guideline_evidence": []
}
```

Each final-model slot has its own response list using the same object format.
Every consumed response adds an entry containing only these keys:

```json
{
  "reported_model": "provider-reported-model",
  "response_id": "provider-response-id",
  "token_usage": {
    "prompt_tokens": 123,
    "completion_tokens": 45,
    "total_tokens": 168
  }
}
```

For shared stages, model and ID come from the provider's `model` and `id`, never
the configured model or HTTP request ID. Final stages use the full upstream
LiteLLM-normalized response; reported fields reflect that adapter's values,
which may differ from vendor-native fields. Configured provider/model identity
is stored separately in run metadata. Each is `null` if unavailable. `token_usage` is `null`
when the provider supplied no usage; when usage exists, an unavailable count is
`null`, including embedding completion counts. Reported zero counts stay zero.
No counts are estimated or derived from other counts. Empty stage lists mean no
response was obtained, including stages never reached. Local BioClinicalBERT and
cross-encoder execution add no entries.

Metadata is recorded before response validation/parsing. Failed, refused, or
truncated replies therefore retain available metadata. A streaming response
creates one entry when its first SDK chunk arrives; later chunks fill available
metadata on that entry. Repeated usage chunks are not summed or counted as new
responses. Separate responses append separate entries, even if their IDs match.
If a stream breaks before usage arrives, usage stays `null` unless earlier chunks
supplied it. An API exception before a response, or a stream yielding no chunks,
adds no entry.

## Capture boundaries and limitations

`JSONKeywordExtractor` captures raw content and metadata immediately after its
API call, and the processed list immediately before returning it. Hybrid and
cross-encoder adapters publish their structured retained rankings before phrase
attribution and formatting. The retriever's and reranker's callbacks are the sole
sources of `retrieved_patient_ids` and `reranked_patient_ids`; the runner does not
reconstruct rankings from returned candidates, scores, or formatted-output counts.
At the start of every `process_patient` call, the
runner calls `prepare_patient_output_capture` in `scripts/med_copilot_helpers.py`
to replace the extractor's result binding and both ranking callbacks and
reset both components' retained rankings to `null`, before constructing S/O or
calling any pipeline stage. These bindings remain attached after the call and
are replaced at the next patient's start. Patient results, provider-stage lists and all selected final-model slots are
freshly initialized for every selected input patient/variant. There is no outer
end-of-patient cleanup or `try/finally` around the pipeline. GraphRAG's own
scoped instrumentation still restores its hooks when its context exits.
Completed captures remain in `failed_incomplete` records. Unstarted patients have shared raw/list fields `null`, shared provider lists
empty, and a fresh `not_processed` slot for every selected variant/model.

GraphRAG's non-streaming CLI path internally streams its answer. The project-owned
`scripts/med_copilot_observability.py` temporarily observes the configured FNLLM
OpenAI provider at SDK response parsing and SDK streaming boundaries. Scoped
ContextVars cover its background embedding event loop; hooks also cover instances
already cached by GraphRAG's model manager. FNLLM cache hits retain their original
raw provider metadata and are recorded when consumed, without making a new API
request. Those token counts describe that cached response, not a new billable
request. The schema intentionally adds no cache-status field.

The only request-option change is `stream_options.include_usage=true` on streamed
chat requests. Query text, prompts, model parameters, embeddings, streaming, and
selection are otherwise preserved; there are no recovery requests. Metadata is
observed before FNLLM can discard it, rather than replaced with configured values.
Embedding responses commonly have no provider response ID; they then record
`null`. Providers may omit usage or the final usage chunk. FNLLM can reject partial
usage objects; available metadata survives that existing failure.

The instrumentation targets the pinned FNLLM 0.4.1 OpenAI implementation classes
also used by its Azure backend. Alternate GraphRAG providers require their own
adapter. Offline integration tests exercise the vendored local-search streaming
method, synchronous embedding bridge, provider wrappers, model manager, real
FNLLM implementations, and mocked SDK clients, plus persisted runner JSONL. Live
provider behavior, Azure endpoint support for usage reporting, and actual remote
usage availability have not been verified. All four offline suites also passed
with the requirements-pinned OpenAI SDK 2.8.0 and FNLLM 0.4.1: 98 helper tests,
21 retrieval tests, 74 runner tests, and 14 observability tests (207 total),
including start-of-patient reset regressions. Shared-component tests cover
success/failure/success sequences at 16 pipeline failure boundaries, retained
raw replies, processed keywords, changing rankings, provider metadata, persisted
JSONL, failures before extraction or reranking, and reuse after `KeyboardInterrupt`
and `SystemExit`. Live integration and concurrent patient execution are outside
this sequential runner verification.

This compatibility check used a temporary virtual environment inheriting
`medcopilot`'s other dependencies, including pandas 3.0.6, with SDK 2.8.0 installed
locally in that environment. Outgoing IPv4/IPv6 connections were blocked during
all four suites; each reported zero connection attempts. No compatibility fixes
were needed. The original `medcopilot` environment remains on SDK 1.99.9, where
the previous 197-test suite passed. Live provider integration remains unverified.

## Variant/model offline verification

Direct helper/config/scoring/model/collator tests cover effective defaults,
validation and complete saved experiment settings; fresh nested slots and stopped
run placeholders (including serialization fallback); exact evidence-section
removal; sentinel filtering and filter order; model/source failure isolation;
and report normalization, missing combinations, ties, zeros, scored counts,
population spread, atomic writes and HTML escaping. Existing model factory,
completion/retry wrappers, parser and evaluator implementations are reused.

Full runner integration tests mock only external model/API/retrieval dependencies
and exercise real configuration, scoring orchestration and collator code. They
verify the complete four-variant matrix, shared evidence reuse, identical messages
across models per variant, keyless local baselines, conditional resources/keys,
source and model failures, cancellation, dataset paths, cohort counts/exclusions,
all effective metadata fields, generated HTML/terminal statistics, report failure,
and generation/scoring interruptions. Existing pipeline/capture regressions are
preserved through an explicit test-only projection of the combined variant.
Historical JSON artifacts are not rewritten. No live inference is part of the
offline suite; provider/schema compatibility, model-server availability and
clinical quality remain unverified.

Verified on 2026-10-26 in `medcopilot`: all 495 offline tests passed (34 config,
9 startup, 132 helper, 25 model, 52 scoring, 21 retrieval, 14 observability,
153 runner, 55 collator). The required direct helper and runner commands also
passed. Configuration tests check all 23 effective fields/defaults in metadata;
pipeline tests exercise custom model/tuning values and relocated resources/output,
conditional initialization, unchanged confirmation, encoder/cache identity and
startup failure isolation. The original combined/question prompt values and
response schema remain unchanged. Live inference remains unverified.
