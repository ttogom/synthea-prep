# MED-COPILOT patient output schema (version 4)

Each run writes `run.json`, one JSON object per selected patient in
`results.jsonl`, and an automatically generated `results_summary.html` after batch
completion. New component-based runs use `run.json.schema_version: 4`.
Historical schema 1–3 files keep their original layout and remain readable by the
collator; they are not migrated or rewritten.

JSONL escaping is ordinary JSON escaping: after `json.loads`, captured strings
retain their original characters, whitespace, Unicode and embedded newlines.

## Patient record

| Field | Value |
| --- | --- |
| `id` | Prepared input patient ID. |
| `status` | `completed`, `failed_incomplete`, or `not_processed`. |
| `components` | Canonical preparation captures keyed by configured component instance ID. |
| `variants` | Selected variants in configuration order, each referencing the captures it consumes and retaining final-model outputs. |

Evidence is stored once inside component records. There are no duplicated
patient-level `guideline_evidence`, selected-reference or ranking fields in schema
4. A variant using an existing component references its instance; it does not
rerun preparation or copy its capture. Each selected final model receives an
independent copy of the variant's exact rendered messages.

## Component records

Each `components[instance_id]` contains:

| Field | Value |
| --- | --- |
| `component` | Logical component ID from the lightweight registry. |
| `status` | `not_processed`, `completed`, or `failed_incomplete`. |
| `error` | Sanitized `{type, message}`, or `null`. |
| `settings` | Effective validated parameters for this configured instance, including defaults. |
| `outputs` | Declared string outputs consumed through prompt bindings. Empty until published. |
| `capture` | Intermediate/raw evidence, rankings, traces and provider metadata obtained while preparing the component. |

A logical component and identical effective settings share one configured
instance across all consuming variants/models for the patient. Different
settings create distinct instance IDs; variant references are authoritative and
consumers must not infer the instance ID from a naming convention. Required
resources can still be shared between instances. Preparation is sequential; the
existing wrappers change process working directories and use mutable callbacks.

The existing logical components publish these outputs:

| Component | Output | Meaning |
| --- | --- | --- |
| `patient_retrieval` | `example` | First reranked full reference SOAP example, or the existing `No reference patient available.` fallback after a successful empty retrieval. |
| `guidelines` | `Key_info` | GraphRAG guideline answer text. |

Components may publish different declared string outputs in future variants.
An output is final accepted evidence, not merely a successful intermediate tool
response. Failed or interrupted preparation retains its capture but does not
become usable evidence accidentally. A successful empty retrieval completes; an
unavailable component blocks its dependent variants with `EvidenceUnavailable`.
Independent variants continue, including standalone guidelines when patient
retrieval fails and standalone patient retrieval when guidelines fail.

### Existing capture fields

The common capture keeps evidence fields separate from prompt outputs. The
existing patient and guideline stages populate the fields applicable to them;
other fields remain `null` or empty provider lists until available.

| Capture field | Value |
| --- | --- |
| `selected_reference_patient` | Minimal identity of the selected reference, or `null`. |
| `guideline_questions` | Parsed `Key Questions` list, or `null`; the exact raw question reply remains the GraphRAG query. |
| `guideline_evidence` | GraphRAG answer text, or `null`. |
| `graphrag_context` | Returned context tables serialized as `{columns, index, data}`, or `null`. |
| `keyword_extraction_raw` | Exact keyword API content captured before validation, including malformed JSON and whitespace. |
| `retrieval_keywords` | Successfully extracted terms flattened in symptoms/diagnoses/medications order, trimmed and deduplicated in first-occurrence order. `[]` is valid; `null` means unavailable. |
| `retrieved_patient_ids` | Retained hybrid ranking with minimal identities and unrounded `embedding_score`, `keyword_score`, and `hybrid_score`. |
| `reranked_patient_ids` | Retained cross-encoder ranking with those scores plus `rerank_score`; ties preserve hybrid order. |
| `model_responses` | Independent provider metadata lists for evidence-stage responses. |

Matching remains S/O-only for corpus/query embeddings, keyword overlap and
cross-encoder scoring. Full reference SOAP is generation evidence after
selection. Labels/scoring never enter component preparation or final prompts.
Keyword model extraction and guideline-question generation are separate stages.

## Variant and final-model records

Each `variants[variant_id]` contains:

| Field | Value |
| --- | --- |
| `status` | `not_processed`, `completed`, or `failed_incomplete`. Completed only when every selected final model completes generation. |
| `error` | `{type, message}` for unavailable required evidence or prompt preparation, or `null`; model errors remain in model slots. |
| `components` | Mapping from each required logical component ID to the configured instance ID in this patient's `components`. Empty for `llm_only`. |
| `assessment_plan_prompts` | Exact `{system, user}` messages, or `null` before preparation or when blocked. |
| `assessment_plans` | Independent final-model slots keyed by selected model aliases. |

Each `variants[variant_id].assessment_plans[model_id]` retains:

| Field | Value |
| --- | --- |
| `status` | `not_processed`, `completed`, or `failed_incomplete`, independently of metric statuses. |
| `error` | Generation error `{type, message}` with credential values redacted, or `null`. |
| `predictions` | Exactly three ranked `{condition, plan}` pairs, or `null` when structurally invalid/unavailable. |
| `assessment_plan_raw` | Exact consumed reply content, including malformed content, or `null`. |
| `model_responses` | Response metadata for this final model. |
| `evaluation` | Three ordered `{rank, condition, plan}` score entries, or `null` before scoring. |

The nesting separates reusable evidence from distinct generated outputs:

```text
patient
├── components
│   ├── patient_retrieval → one capture and accepted example
│   └── guidelines       → one capture and accepted guideline answer
└── variants
    ├── llm_only                    → no component references → model outputs
    ├── patient_retrieval           → patient instance        → model outputs
    ├── guidelines                  → guideline instance      → model outputs
    └── patient_retrieval_guidelines→ both instances           → model outputs
```

For example, a patient with one completed retrieval component and one selected
final model has this shape (capture details and evaluation values abbreviated):

```json
{
  "id": "prepared-patient-id",
  "status": "completed",
  "components": {
    "patient_retrieval": {
      "component": "patient_retrieval",
      "status": "completed",
      "error": null,
      "settings": {"alpha": 0.5, "top_k": 20, "rerank_top_k": 5},
      "outputs": {"example": "Exact selected reference example"},
      "capture": {"selected_reference_patient": {"note_id": "reference-id"}}
    }
  },
  "variants": {
    "patient_retrieval": {
      "status": "completed",
      "error": null,
      "components": {"patient_retrieval": "patient_retrieval"},
      "assessment_plan_prompts": {
        "system": "You are a clinical expert.",
        "user": "Exact rendered patient-retrieval prompt"
      },
      "assessment_plans": {
        "gpt_4o_mini": {
          "status": "completed",
          "error": null,
          "predictions": [
            {"condition": "A", "plan": ["Action A"]},
            {"condition": "B", "plan": ["Action B"]},
            {"condition": "C", "plan": []}
          ],
          "assessment_plan_raw": "Exact raw JSON reply",
          "model_responses": [],
          "evaluation": null
        }
      }
    }
  }
}
```

The full capture contains the fields described above, not only the selected
reference shown in this abbreviated example. Exact raw responses are captured
separately from parsed predictions; the example's descriptive strings are not a
sample live model response.

## Status, failures and interruption

Ordinary component failures block only consuming variants. Their final-model
slots remain `not_processed` and prompts remain `null`; completed component
captures survive. Ordinary model failures remain local to that model, and other
models/variants/patients continue. Refused or truncated replies retain available
content/metadata but fail generation. Obtained malformed JSON retains existing
`completed` generation semantics and produces missing-generation scores.

The patient completes only when every selected variant/model generation
completes. Otherwise it is `failed_incomplete`; an unstarted selected patient is
`not_processed`. Run statuses remain `running`, `completed`,
`completed_with_errors`, and `failed_incomplete`. Scoring failures are field/rank
local and do not change generation status or the run exit code.

Captures are attached before execution. Component-scoped context handling
restores mutable extractor/ranking bindings after success, failure or
interruption; working-directory and GraphRAG hook scopes restore on exit.
Caught interruptions retain the current patient's obtained captures, prompts,
replies and scores, and add explicit unattempted placeholders for remaining
selected patients/variants/models. Excluded patients receive no result rows or
placeholders. Persistence is at patient boundaries plus best-effort interruption
finalization; this schema does not add resume-existing-run support or protection
against uncatchable process termination.

## Configuration and provenance

The fixed `experiments/configs/experiment.yaml` selects models, variants, cohort,
paths and tuning. `variant_settings` nests overrides by variant ID, component ID,
and declared parameter. Registered component defaults may reference shared
experiment fields. Unknown fields, invalid types/ranges, bad bindings/templates,
duplicate selections and missing selected credentials fail during preflight.
`llm_only` with a keyless local final model requires no retrieval resources or
shared API keys. The presenting-evidence CLI filter override remains available;
no additional runner command-line arguments are introduced.

`run.json.experiment` records every effective YAML setting and applied default,
config path/source SHA-256 and the final-model registry reference. `variant_definitions` and
`component_definitions` snapshot the actual dependencies, prompt bindings,
implementation identities and effective parameters used; `registry_provenance`
records the registry source path and SHA-256. Prompt provenance
records the project `diagnosis_plan.py` path/source fingerprint and exact selected
template hashes. Only accessed/successfully initialized resources are
fingerprinted; unused/failed resources remain `null`. Shared reference corpus,
S/O cache/manifest, guideline settings, active prompt, tables and vector stores
retain their existing fingerprints. Resolved credentials are never saved.

`condition_plan_models.yaml` continues to contain only model definitions, with
credential variable names and `output_mode`. Shared final `temperature` and
`max_retries` come from the experiment; other upstream defaults, retries,
timeouts and rate limits remain reused. `run.json.assessment_plan_models` saves
selected effective settings excluding API keys. Literal keys in model entries
and credentials/query parameters in endpoint URLs are rejected. Exported
credentials take precedence over the repository-root `.env`.

Input records and labels are validated before cohort selection. The presenting-
evidence filter runs first, then optional exact label-Plan exclusion
`["none recorded"]`, then `max_patients`. `null` selects all eligible patients;
positive integers select the first N in input order. Evaluation metadata records
the label/helper hashes, installed scoring versions/defaults, filter order,
intermediate counts, selected/excluded IDs and reasons. The shipped sample has
10 input patients, 7 after presenting-evidence filtering and 5 after the recorded-
plan filter; `max_patients: null` retains all five. Additional output/resource paths
and evidence tuning remain configured/saved without expanding confirmation.

## Reporting and historical compatibility

The runner's RESULTS block invokes the collator after the batch, including
ordinary failures. It writes `results_summary.html` and records success/filename
or a sanitized report failure in `run.json.results_summary`. Report failure
preserves saved JSON and returns nonzero; interrupted batches preserve JSON and
skip report generation.

Reports use saved variant definitions/labels so a historical variant need not
still exist in today's registry. Schema 1–3 normalization remains supported.
Terminal/HTML summaries show each model's per-variant rank-1 condition and Plan
BERTScore_F1 means, choose the highest rank-1 condition mean (ties retain config
order), and report separate mean, population SD, min, max and scored count.
Available zero values are included; missing/skipped/error values are not. Paired
comparisons use the same patients scored in both a variant and `llm_only`.
All three ranks and six metrics per field remain available. There is no composite
condition/Plan score and no new agent-trace viewer.

```bash
python scripts/collate_med_copilot_results.py experiments/output/<run-directory>
```

Regeneration needs the saved label path with its recorded hash and never makes
model requests or rewrites JSON artifacts. The HTML report is written atomically.

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

Inside the patient-retrieval component capture, both ranked lists are `null` until their order is known and `[]` if their stage
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

Each component's `capture.model_responses` contains independent evidence-stage
lists. Existing stages are `keyword_extraction`, `guideline_questions`,
`guideline_query_embedding` and `guideline_evidence`. Each final-model slot has
its own `model_responses` list. Every obtained response records:

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

Values come from the provider response, not the configured model or HTTP request
ID. Final calls use LiteLLM-normalized metadata; shared GraphRAG capture observes
FNLLM's OpenAI/Azure SDK boundaries. Unknown fields/counts are `null`, reported
zeros remain zero, and no usage counts are estimated. `token_usage` is `null` if
no usage was provided. Local encoder/cross-encoder work adds no API entries.

Metadata/raw content is captured before validation. A streaming response creates
one entry at its first SDK chunk and later chunks fill available fields; repeated
usage chunks are not counted again. Broken streams retain metadata received so
far. A request failing before a response produces no entry. Separate responses
append separate entries even if their response IDs match.

GraphRAG's non-streaming CLI path internally streams its answer. Scoped
instrumentation restores hooks on exit, covers cached model instances and its
background embedding event loop, and requests `stream_options.include_usage=true`
for streamed chat. FNLLM cache hits record consumed cached response metadata;
those counts do not imply a new billable request. Alternate providers require
an appropriate capture adapter. Live provider usage availability is not
established by offline mocks.

Hybrid/reranker callbacks publish structured rankings before display formatting
or attribution. Each component invocation gets its own scoped capture sink;
completed captures cannot leak into later patients/instances. Original raw
keyword content is preserved independently from validated/flattened terms.
Guideline response text and context tables remain separate; only the text enters
the existing final prompt binding.

## Offline verification

Direct registry/config/helper tests cover declarations, parameter defaults and
validation, output bindings, effective metadata, fresh component/model records,
scoped capture restoration, failure isolation and stopped-run placeholders.
Runner integration tests mock external APIs/models/retrieval and verify actual
messages, configured component sharing, exact existing prompts, persisted JSON,
resource/credential selection, source/model failures, interruptions and reports.
Scoring and startup regressions preserve their existing contracts. Collator
cases cover schema 4, historical files, saved variant identities, missing slots,
paired scores, ties and unavailable values. No live inference is part of the
offline suite; provider access, model-server compatibility and clinical quality
remain unverified. See [Adding a new variant](adding_a_new_variant.md) for the
extension workflow and required checks.
