# MED-COPILOT patient output schema

Each run appends one JSON object per selected input patient to `results.jsonl`. JSONL
escaping is ordinary JSON escaping: after `json.loads`, captured strings retain
their original characters, including whitespace, Unicode, and embedded newlines.
The keys are written in this order:

| Field | Value |
|---|---|
| `id` | Prepared input patient ID. |
| `status` | `completed`, `failed_incomplete`, or `not_processed`. |
| `assessment` | Parsed Assessment section, or `null` when unavailable/invalid. |
| `plan` | Parsed Plan section, or `null` when unavailable/invalid. |
| `assessment_plan_raw` | Complete final reply, as returned, or `null`. |
| `selected_reference_patient` | Minimal identity of the reference whose formatted full SOAP example enters final generation, or `null`. |
| `guideline_questions` | Parsed `Key Questions` list, or `null`; the raw question reply remains the GraphRAG query. |
| `guideline_evidence` | GraphRAG answer text, or `null`. |
| `graphrag_context` | Returned context tables as `{columns, index, data}`, or `null`. |
| `assessment_plan_prompts` | Exact final `{system, user}` messages, captured before the final request, or `null`. |
| `keyword_extraction_raw` | Exact string from the keyword API response's `choices[0].message.content`, captured before validation. Malformed JSON, duplicates, category structure, empty strings, and whitespace are preserved. `null` means extraction was not attempted or no string content was obtained. |
| `retrieval_keywords` | Actual keyword list returned by successful extraction: categories flattened in symptoms/diagnoses/medications order, terms trimmed, duplicates removed in first-occurrence order. `[]` means successful extraction with no terms. `null` means no valid list was produced. These are distinct from guideline questions. |
| `retrieved_patient_ids` | Retained hybrid ranking, normally up to 20 references, before cross-encoder reranking. Each entry contains a minimal identity and its unrounded `embedding_score`, `keyword_score`, and `hybrid_score`. |
| `reranked_patient_ids` | Retained cross-encoder ranking, up to five references, containing minimal identity and the preceding scores plus `rerank_score`. Ties preserve hybrid order. When a reference is selected, its identity equals the first entry's identity. |
| `model_responses` | Provider metadata lists by stage, described below. |
| `evaluation` | Independent `condition` and `plan` score entries, or `null` if generation never reached evaluation. |

## Condition and Plan evaluation

Gold labels are joined to the complete input by exact ID before cohort selection.
The `--presenting-evidence-only` option defaults to `true`; `false` includes all
input patients. Only selected IDs are registered in the run. Excluded patients
have no result row, including when an interruption marks remaining selected
patients `not_processed`. `run.json.evaluation` records the option, input and
selected counts, selected/excluded IDs, label and evaluator fingerprints,
installed metric-library versions and configured settings.

The generated field remains named `assessment`. Its score entry is named
`condition` to match the label. The references are the label's `condition` with
underscores replaced by spaces, and the original Plan items joined with `", "`.
No additional text cleaning is applied to either field. Labels never enter
retrieval or generation prompts.

An evaluated patient has the following shape; all six metric keys are always
present in each entry:

```json
{
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
- `missing_generation`: the parsed generated section is null or blank.
- `error`: scoring or score validation failed. `error` contains `{type, message}`.
- `interrupted`: scoring raised a fatal interruption, recorded before re-raising.
- `pending`: the field was not attempted before an interruption in the other field.

Except for `scored`, all six metric values are `null`. Condition scoring is
independent of the skipped Plan sentinel. An ordinary error in one field does not
discard successful scores in the other, predictions, prompts, or evidence, and
does not change the generation status or exit code. Interruptions retain the
existing stopped-run semantics. Null/blank references are not replaced with
empty strings or zeros. When both are unavailable, missing-reference status takes
precedence over missing-generation; the exact Plan sentinel takes precedence over
either.

Every invocation compares singleton lists, separately for condition and Plan.
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
  "guideline_evidence": [],
  "assessment_plan": []
}
```

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

The model and ID come from the provider's `model` and `id`, never the configured
model or HTTP request ID. Each is `null` if unavailable. `token_usage` is `null`
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
are replaced at the next patient's start. Patient results and provider-stage
lists are freshly initialized for every input patient. There is no outer
end-of-patient cleanup or `try/finally` around the pipeline. GraphRAG's own
scoped instrumentation still restores its hooks when its context exits.
Completed captures remain in `failed_incomplete` records. Unstarted patients
have all four new raw/list fields `null` and every provider-stage list empty.

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
