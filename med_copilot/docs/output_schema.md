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
| `guidelines` | `Key_info` | GraphRAG answer formatted as text, matching the original final prompt's conversion. |

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
| `guideline_questions` | Parsed `Key Questions` list from the screened reply, or `null`; that screened raw reply is the GraphRAG query. |
| `guideline_query` | Exact nonblank screened query text, present only after a completed, non-refused question reply. Reused when guideline retrieval resumes, even if the parsed questions are unavailable. Legacy null or blank values are regenerated. |
| `guideline_question_attempts` | Question replies in request order as `{content, refusal, finish_reason}`, including failed or refused replies. Failed attempts do not publish a reusable query. |
| `guideline_question_screen` | `null` when the first question reply names no study diagnosis absent from the S/O. Otherwise `{first_reply, first_terms, retry_reply, retry_terms}`: the flagged reply and names, then the single retry and any names it still contains, which are replaced with "these findings" before GraphRAG. |
| `guideline_evidence` | Raw GraphRAG answer, including a non-string JSON value if returned; `null` also represents an unavailable capture. |
| `graphrag_context` | Returned context tables serialized as `{columns, index, data}`, or `null`. |
| `keyword_extraction_raw` | Exact keyword API content captured before validation, including malformed JSON and whitespace. |
| `retrieval_keywords` | Successfully extracted terms flattened in symptoms/diagnoses/medications order, trimmed and deduplicated in first-occurrence order. `[]` is valid; `null` means unavailable. |
| `_keyword_query` | Exact input text for a completed keyword extraction. Together with validated raw content and terms, permits reuse during an evidence retry. |
| `retrieved_patient_ids` | Retained hybrid ranking with minimal identities and unrounded `embedding_score`, `keyword_score`, and `hybrid_score`. |
| `reranked_patient_ids` | Retained cross-encoder ranking with those scores plus `rerank_score`; ties preserve hybrid order. |
| `model_responses` | Independent provider metadata lists for evidence-stage responses. |

Matching remains S/O-only for corpus/query embeddings, keyword overlap and
cross-encoder scoring. Full reference SOAP is generation evidence after
selection. Labels/scoring never enter component preparation or final prompts.
Keyword model extraction and guideline-question generation are separate stages.

Guideline answers are requested as text. If a successful search returns a
non-string value, preserve it in `capture.guideline_evidence` and format it into
the string `outputs.Key_info` as the original prompt did (`None` becomes
`"None"`, for example). This does not block dependent variants; search exceptions
still fail the component and block its consumers.

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
| `_generation_reply_complete` | Optional recovery marker: a non-refused, non-truncated provider reply was received, but local parsing did not finish. Resume reparses `assessment_plan_raw` without another provider request. Removed after parsing succeeds. |
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
      "capture": {"selected_reference_patient": {"id": "reference-id"}}
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

Ordinary per-patient component failures block only consuming variants
(shared-resource initialization failures stop the run; see Run errors). Their final-model
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

### Run errors

Launching `scripts/run_med_copilot.py` creates the run directory and `run.json`
(`status: running`, `errors: []`) before any heavy import, using the
standard-library-only `scripts/med_copilot_run_file.py`. Until the experiment
config and Git branch are known, the directory is provisional,
`<timestamp>__starting-<suffix>` under `experiments/output/`; it is then renamed
to `<timestamp>__<input stem>__<branch>` under `experiment_output`. Cancelling at
the confirmation prompt (or a successful exit such as `--help`) is not an error:
the run directory, and any output directory created only for it, is removed.

Each failure is appended to `errors` and saved (atomically, through a temporary
file and `os.replace()`) as soon as it is handled, before the run continues or
stops:

| Field | Meaning |
|---|---|
| `timestamp` | Pacific time the failure was recorded. |
| `patient_id` | Patient being processed, or `null` for run-level failures. |
| `stage` | `startup`, `input_loading`, `corpus_loading`, `retrieval_initialization`, `keyword_extraction`, `patient_retrieval`, `reranking`, `guideline_question_generation`, `guideline_retrieval`, `final_generation`, `scoring`, or `output_saving`. |
| `exception_type` | Exception class name, e.g. `RuntimeError` or `KeyboardInterrupt`. |
| `message` | `str()` of the exception, with configured credentials replaced by `[REDACTED]`. |
| `traceback` | Full formatted traceback including chained causes, also redacted. |

Fatal failures are recorded once, then the run is marked `failed_incomplete`
with `finished_at` and the script exits nonzero with the normal console
traceback (configuration errors print their summary instead). Fatal failures are:

- startup: imports, dependency/encoder initialization, experiment/model
  configuration and credential preflight, Git lookup, or client creation;
- input loading: malformed or missing patient input or labels;
- shared resources: the patient corpus (`corpus_loading`), the retriever,
  reranker, or guideline tables (`retrieval_initialization`); unfinished
  patients are saved as `not_processed`;
- any other unhandled failure or interruption (`KeyboardInterrupt`), recorded
  with the in-flight patient and stage.

Component, final-model and variant-prompt failures are isolated to their patient:
they are recorded once with that patient, the batch continues, and the run ends
`completed_with_errors`. An empty reference corpus or no matching reference
remains the supported fallback, not an error. A results-report failure is
recorded without changing the run status.

If `run.json` itself cannot be written, the original failure and the write
failure are both printed to stderr and the run stops unsuccessfully. Forced
termination (e.g. `SIGKILL`) or power loss can prevent the final update, but
errors saved earlier remain readable.

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
positive integers sample up to N without replacement using seed 42. Selected
patients run in input order. Evaluation metadata records `patient_sampling_seed`,
the label/helper hashes, installed scoring versions/defaults, filter order,
intermediate counts, selected/excluded IDs and reasons. The sample dataset has
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
when enabled, then `max_patients` applies the seeded sample. Only selected IDs
are registered in the run. Excluded patients
have no result row, including when an interruption marks remaining selected
patients `not_processed`. `run.json.evaluation` records the option, input and
selected counts, selected/excluded IDs, label and evaluator fingerprints,
installed metric-library versions and configured settings.

Each prediction is scored separately against the same patient labels:
`variants[variant_id].assessment_plans[model_id].evaluation[i].condition` compares that model's `predictions[i].condition` to the label's
`condition` with underscores replaced by spaces, and `variants[variant_id].assessment_plans[model_id].evaluation[i].plan`
compares the generated and reference Plan arrays after joining each with
`", "`. The generated Plan itself stays an array. Each evaluation entry records
`rank: i + 1`. The six text metrics have no rank weighting, averaging, maximum,
or combined score; the preliminary correctness scores below add a separate 1 / rank weight.
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
each rank. Text-metric rank order is preserved without weighting.
BLEU uses corpus statistics for that singleton; METEOR and ROUGE/BERTScore F1
retain the upstream reductions. These are not cohort-level scores, and averaging
per-patient BLEUs is not corpus BLEU. Raw BERTScore is not a correctness
probability. Encoder truncation, upstream import-time NLTK download attempts,
and per-call BERTScore model initialization remain upstream behavior.

### Preliminary correctness metric (single-model midterm)

> **Preliminary single-model midterm metric.** This is a provisional metric so
> the group can start using results; it will be revised after review. It
> replaces the earlier three-judge panel design. The diagnosis score is a
> BioLORD-2023 clinical concept similarity score (no LLM judge; not an accuracy
> or probability), and Plan correctness uses
> one judge model, selected by `plan_judge` in `experiments/configs/experiment.yaml`
> (currently `ollama_qwen2_5_14b`, Ollama tag `qwen2.5:14b`, about 9 GB). The
> judge is a registry model, not a generating model; the runner and resume
> command load it on their own. If it cannot be loaded, every Plan entry is
> `unavailable` with reason `judge_unavailable` and diagnoses are still scored.

The runner also scores every completed prediction for correctness. Each
evaluation entry gains two keys beside `condition` and `plan`; the six text
metrics are unchanged:

```json
{
  "rank": 2,
  "condition": {"status": "scored", "metrics": {"...": 0.0}, "error": null},
  "plan": {"status": "scored", "metrics": {"...": 0.0}, "error": null},
  "diagnosis_correctness": {
    "status": "scored", "method": "biolord_2023_cosine", "reason": null,
    "cosine_similarity": 0.7930, "score": 0.7930, "weighted_score": 0.3965,
    "rank_weight": 0.5, "reference": "essential hypertension", "generated": "HTN",
    "normalized_reference": "essential hypertension",
    "normalized_generated": "hypertension", "error": null
  },
  "plan_correctness": {
    "status": "scored", "method": "single_local_judge",
    "validation": "passed_balanced_sample", "reason": null,
    "score": 0.5, "weighted_score": 0.25, "rank_weight": 0.5,
    "reference_actions": ["Throat culture (procedure)"],
    "generated_actions": ["Throat culture", "Antibacterial therapy"],
    "credits": [1.0], "unsupported_extra_count": 1, "deduction": 0.5,
    "judges": [
      {"call": "action", "reference_item": 1, "judge_model": "ollama_qwen2_5_14b",
       "status": "scored", "credit": 1.0, "request_key": "<sha256>",
       "raw_response": "{\"credit\": 1}", "model_response": {"...": null}, "error": null},
      {"call": "extras", "reference_item": null, "judge_model": "ollama_qwen2_5_14b",
       "status": "scored", "unsupported_extra_count": 1, "request_key": "<sha256>",
       "raw_response": "{\"unsupported_extra_count\": 1}", "model_response": {"...": null},
       "error": null}
    ],
    "error": null
  }
}
```

- **Diagnosis (clinical concept similarity score).** The labeled condition (underscores replaced by
  spaces) and the generated diagnosis are normalized, then embedded with
  `FremyCompany/BioLORD-2023` (sentence-transformers, pinned revision
  `167aab527b238a50ca65224e6319215d2ff4fc9f`). Normalization is deterministic:
  `HTN` → `hypertension`, `CHF` → `congestive heart failure`, and `COPD` →
  `chronic obstructive pulmonary disease`, each only as a whole word in any case
  (`uncontrolled htn` expands; `HTNX` does not), then whitespace is trimmed and
  collapsed. `reference`/`generated` keep the original text and
  `normalized_reference`/`normalized_generated` the encoded text.
  `cosine_similarity` is the raw cosine; `score` is that cosine clipped to
  `[0, 1]`. It is a clinical concept similarity score, not an accuracy, a probability, or a
  graded clinical judgement: related but different conditions can still score high (acute viral
  pharyngitis vs streptococcal sore throat: 0.69). Original sanity pairs
  (unchanged by normalization): heart failure ↔ congestive heart failure 0.918,
  acute viral pharyngitis ↔ acute pharyngitis 0.851, essential hypertension ↔
  hypertension 0.793, and the incorrect heart failure ↔ anemia 0.253.
  Abbreviation pairs, before → after normalization: HTN 0.324 → 0.793, CHF
  0.585 → 0.918, COPD 0.833 → 1.000, and Hypertension 0.793 (unchanged), each
  against its labeled condition. `tests/test_med_copilot_judges.py` checks the
  orderings (every related pair above the incorrect pair; normalization never
  lowers an abbreviation's score), not these exact values, and runs only when
  the pinned revision is already cached.
- **Plan.** Applies `med_copilot/docs/treatment_plan_evaluation.md` with one
  local judge, at temperature 0, in JSON object mode (no response format when
  the model's `output_mode` is `prompt`). The rubric is split into one scalar
  call per reference action (`{"credit": 0 | 0.5 | 1}`, the rubric's scale) and
  one call counting unsupported extra actions
  (`{"unsupported_extra_count": N}`, 0 to the number of generated actions);
  prompts are in `med_copilot/prompts/correctness_judges.py`. Generated actions
  are passed as separate numbered items. `deduction = 0.5 × unsupported_extra_count`
  and `score = max(0, (sum(credits) - deduction) / len(reference_actions))`.
  A single call returning the whole credits array was tried first, with the
  exact length stated and a JSON example of that length; the local judges
  returned valid arrays but copied the example credits instead of grading.
  Strength-mismatch items are not collected (they never change credit). An
  explicitly empty generated Plan scores 0 with reason `empty_generated_plan`
  and no judge calls. Prompts never name the generating model, evidence variant,
  or rank. `run.json` records whether the judge is also a generating model
  (`self_judging`); the shipped judge is not.
- **Plan validation.** Every Plan entry carries `validation`, the judge's result
  on a balanced sample of correct, off-target, and placeholder Plans
  (`PLAN_VALIDATION_STATUS`, with the evidence in `run.json`). The shipped judge
  is `passed_balanced_sample`: on 3 cohort patients × 6 Plans, qwen2.5:14b scored
  correct Plans 1.00, off-target 0.00, and placeholders 0.083, every correct Plan
  above every other Plan per patient (llama3 failed: placeholders 0.83). The
  sample is small, and placeholder credit was sometimes offset only by the
  extras deduction. A judge marked `experimental_unvalidated` keeps its
  per-prediction scores for review, but the report shows them as
  experimental, excludes them from every aggregate (cards, statistics, console
  summary), and adds a notice that they are not a measure of accuracy.
- **Rank weight.** `weighted_score = score / rank` (weights 1, 0.5, 1/3), for
  the concept similarity score and the Plan score separately. Diagnosis and Plan scores,
  weighted or not, are never combined.

`status` is one of `scored`, `unavailable`, `error`, `interrupted`, or `pending`;
`score` and `weighted_score` are `null` unless `scored`. `reason` explains the
status:

- `unavailable`: `no_recorded_plan` (label exactly `["none recorded"]`),
  `record_artifact_plan` (any reference action containing `(record artifact)`
  in any case or spacing, such as `( Record  Artifact )`),
  `missing_reference` (empty or blank reference Plan), `missing_generation`,
  `invalid_generation` (malformed prediction pair or field), or
  `judge_unavailable` (the Plan judge model is not among `selected_models`).
  Reference reasons take precedence over generation reasons.
- `error`: `similarity_failed` (BioLORD could not be loaded or returned unusable
  embeddings; saved in `run.json.errors` as `SimilarityError`) or
  `judge_failed` — a Plan judge call raised, or its reply was refused or not
  valid for its schema. Every call for the Plan is still attempted. Each judge
  record keeps its own `status` (`scored`, `invalid_response`, or `error`), the
  raw reply, and `error: {type, message}`. Each such failure is also saved in
  `run.json.errors` with stage `scoring` and the patient's ID, as
  `JudgeCallError` (the call raised; the original exception is the chained
  cause) or `JudgeResponseError` (refused or malformed reply). The message
  names the task, judge model, and request key. An entry with the same patient,
  stage, type, and message is saved only once, so resuming adds no duplicates.
- `interrupted`: judging stopped mid-Plan; the judge records obtained so far are
  kept. Later ranks remain `pending`.

Correctness scoring only runs for model slots whose generation `completed`.
Records from runs without it have no correctness keys, and the report shows
them as not recorded rather than zero.

**Full runner resumption.** `python scripts/run_med_copilot.py --resume RUN_DIRECTORY`
continues a schema-4 run using the same experiment settings and verified inputs,
models, prompts, scoring definitions, and reference resources. Completed
components, generations (including completed malformed replies), and successful
score fields are preserved. Failed or unstarted work is retried. A restored
variant uses its exact saved messages. Patient rows are replaced atomically in
original cohort order, without duplicates; a repeated interruption retains
already-saved later patients too. `run.json.experiment.plan_judge` records the
configured registry alias, including its default for older configs. Every edit
to `experiment.yaml` blocks a resume: both its effective settings and exact
source SHA-256 must match. The Plan judge is checked by alias; its effective
registry configuration remains in provenance without a separate judge check.
Unrelated model-registry changes are allowed. Cancelled or rejected resumes leave the run
untouched. Original metadata remains; `run.json.resumptions` adds resume time,
branch, commit, current model/source fingerprints, and correctness metadata. Missing original
correctness metadata is filled when judging is first attached.

`.patient_checkpoint.json` is an atomic, flushed snapshot of the current patient
after each completed component, keyword request, screened query, model call, or
score field. It carries `_base_results_sha256` so a newer checkpoint can override
its original row while a stale snapshot cannot roll back a subsequently saved
row. Recovered checkpoint work is committed to `results.jsonl` at the first
patient selection, before execution can overwrite the snapshot. Fresh appends
flush and `fsync` the result file before removing the snapshot; atomic file
replacements and checkpoint deletion also `fsync` their directory entries.
`run.json` replacements likewise sync both file data and directory entries.
If an in-memory patient becomes unserializable, shutdown preserves its last
durable checkpoint rather than replacing it with an empty record. Without a
checkpoint, it retains any previously saved row for that patient. Invalid saved
execution structures are rejected before resumed work or artifact updates.
The snapshot is removed after that patient is saved. An interrupted
provider request without a durable response may repeat; simultaneous runner or
Plan-evaluator writers to the same run directory are rejected by an exclusive lock.

**Auditing and evaluation resumption.** Every completed judge call is appended (and
`fsync`ed) to `judge_responses.jsonl` in the run directory before its reply is
used, with the request key, task (`plan_action` or `plan_extras`), judge model,
full messages, raw content, refusal, provider metadata, `valid`, and the parse
`error`. Only `valid: true` replies are reused: identical requests are then
asked once. Malformed or refused replies stay in the file for auditing, but
they are requested again, as are calls that raised (which have no reply to
save). The runner calculates text metrics and BioLORD diagnosis similarity;
nonempty judgeable Plans remain `pending` with reason `deferred`, and no Plan
model or response cache is loaded by the runner. Empty generated Plans can
still receive the rubric's deterministic zero without a model call.

`python scripts/judge_treatment_plans.py RUN_DIRECTORY` finishes only
Plan correctness after generation finishes. It rebuilds the recorded cohort
(refusing changed input or labels), loads only the recorded Plan judge, and
preserves generation, evidence, text metrics, and diagnosis scores. Bounded
concurrency comes from the saved `experiment.plan_judge_concurrency` (default
2 for older runs; valid range 1–64). Identical concurrent valid requests share
one response. Scalar replies are the durable incremental checkpoint;
`results.jsonl` is exported atomically once on completion or controlled stop,
then the report is regenerated after a successful invocation. Following an
abrupt crash, aggregate Plan scores are reconstructed using saved scalar replies.

Both commands acquire the same exclusive `.run.lock`; the OS releases ownership
after a crash and the lock file remains. The evaluator records its independent
status, timing, concurrency, counts, and judge provenance in
`run.json.plan_evaluation`, preserving the generation run's status and timestamps.
It exits 0 when every applicable Plan entry finished, 1 when inputs cannot be
used or some Plan entries still failed (rerun to retry them), and 130 when
interrupted. Neither generation nor completed text/diagnosis scoring is repeated.

`run.json.evaluation.correctness` records `status: "preliminary single-model
midterm metric"`, the method, `selected_models` (the models the judge can
call), hidden fields, the BioLORD model, revision, and normalization rule, the
Plan judge model, its registry settings (`judge_model_config`), whether it
loaded, the generating models and `self_judging`, the Plan
`validation_status` and `validation_evidence`, judge temperature and response format, the rubric file and
SHA-256, the per-action implementation and why it replaced the full-array call,
credit values, deduction, formula, unavailable-reference rules, both Plan
prompt templates (with SHA-256), and `rank_weighting`
(`{"formula": "1 / rank", "weights": {"1": 1.0, "2": 0.5, "3": 0.3333333333333333}, "combined_score": null}`).

The report labels these scores as the preliminary single-model midterm metric
in a notice, adds `Score (diagnosis: concept similarity; Plan: judge)` and `Weighted (score ÷ rank)` columns for every
prediction, rank-1 `Diagnosis concept similarity` and `Plan judge` means on each model
card, and unweighted and weighted correctness statistics for every rank. The
console summary adds, per model and variant, the unweighted rank-1 mean and the
**rank-weighted mean**: the average of `score / rank` over every scored rank
1–3. That aggregate is not normalized, so even perfect predictions give at most
`(1 + 1/2 + 1/3) / 3 ≈ 0.611`. A single weighted score is at most `1 / rank`.
Missing scores are excluded, never counted as zero.

Inside the patient-retrieval component capture, both ranked lists are `null` until their order is known and `[]` if their stage
completes with an empty ranking. An empty hybrid ranking skips reranking, so
`reranked_patient_ids` remains `null` in that case. The lists capture the retained
rankings, not the larger initial FAISS pool, and contain no SOAP text or subject
IDs. Their scores come from structured candidates, before display rounding.

Minimal identity follows the same convention for the selected reference and
both lists:

- `{"id": "..."}` for a unique, nonblank string ID, preserved verbatim.
- `{"corpus_row": N}` when no usable `id` exists.
- `{"id": "...", "corpus_row": N}` when the ID is duplicated.

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
