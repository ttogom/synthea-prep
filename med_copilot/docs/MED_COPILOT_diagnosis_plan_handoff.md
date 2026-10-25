# MED-COPILOT Follow-Up Implementation Handoff

**Status:** The project-owned pilot requests Assessment and Plan from target Subjective and Objective (S/O). It uses the released 1,000-record reference corpus with project-owned S/O-only matching and preserves full reference SOAP for generation. Integration of the full project reference corpus remains pending.

**Scope:** Current pilot behavior, setup, verification limits, and the remaining integration work. This document does not establish faithful reproduction of the MED-COPILOT study.

---

## Current Working Baseline

The current runner is:

[scripts/run_med_copilot.py](../../scripts/run_med_copilot.py). Paths below are relative to the repository root.

Its current Assessment-and-Plan workflow is:

1. Prepared S/O patient input
    - `PATIENT_INPUT` selects `experiments/input/patient_records_so.jsonl`: one JSON object per line with a nonempty, unique string `id` plus prepared `subjective` and `objective` strings. The loader validates every record, with file and line diagnostics, before a run starts.
    - Encounter selection, S/O preparation, and diagnosis-related scrubbing belong in input preparation, owned by the data team. The runner uses the prepared strings directly, without an LLM normalization call.
    - Optional evaluation labels belong in `experiments/input/patient_records_so_labels.jsonl`, joined by `id`. The runner does not read them.
2. Patient loop
    - Every record is processed sequentially. The retriever, reranker, and guideline tables are initialized once; a failure there stops the run. A failure in one patient's stages is recorded for that patient and the batch continues.
    - The S/O string is used consistently for patient search, reranking, question generation, and final generation. `PATIENT_CASE_TEMPLATE` and `PATIENT_CASE_SYSTEM_TEMPLATE` in [med_copilot/prompts/diagnosis_plan.py](../prompts/diagnosis_plan.py) are not used by the runner.
3. Retrieval of top 20 similar patient candidates
    - Uses the 1,000 full-SOAP records in `med_copilot/upstream/soap_with_metadata.json`, with the project-owned `HybridRetriever` subclass in `scripts/med_copilot_retrieval.py`. Upstream source remains unchanged.
    - One shared formatter constructs target and reference S/O text. Reference embeddings and keyword-overlap scoring exclude reference A/P. Retrieval retains upstream normalized-keyword-mean query embeddings (full-query fallback), BioClinicalBERT, keyword overlap, and `alpha=0.5`; keywords are extracted using `gpt-4o-mini`.
    - Artifacts are `embeddings.npy`, `index.faiss`, and `matching.json` under `med_copilot/data/patient_retrieval_so/`. The manifest identifies ordered S/O texts, model name/revision when available, sequence limit, count/dimension, and file hashes. Invalid or missing bundles are rebuilt. A/P-only edits reuse the matching cache while the current full SOAP payload is loaded afresh. The runner reuses its preloaded encoder, temporarily scopes the working directory to the artifact directory, and fingerprints all three files after initialization.
4. Cross-encoder reranking and reference selection
    - The project-owned reranker subclass uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to compare target S/O with reference S/O and keeps the top 5. The first reranked reference's full SOAP, rerank score, and dominant-term information are supplied as the final prompt's example; its `note_id` (or corpus row) is saved as `selected_reference_patient`. Full-SOAP phrase attribution is retained for display/generation evidence and is not used for matching scores.
    - An empty corpus, no candidates, or no reranked results produces `No reference patient available.`; the final prompt supports generation without a reference. Retrieval exceptions still propagate.
5. Key-question generation
    - `gpt-4o-mini` is prompted to identify at most 4 diagnostic and management questions from target S/O and describe possible diagnoses as hypotheses. The returned question JSON is forwarded verbatim as one GraphRAG query; the runner does not query each question separately. The `"Key Questions"` strings are saved as `guideline_questions`, or `null` when the reply lacks that list.
    - The prompt is `KEY_QUESTIONS_TEMPLATE` in `med_copilot/prompts/diagnosis_plan.py`.
6. GraphRAG guideline retrieval
    - Uses local search with `med_copilot/data/guidelines/settings.yaml`, the Parquet tables under `med_copilot/data/guidelines/output/`, and the entity-description LanceDB table. Community level is 2 and response type is `Multiple Paragraphs`.
    - The configured chat model is `gpt-4o-mini`; the query embedding model is `text-embedding-3-small`. This is separate from BioClinicalBERT patient retrieval.
    - The active answer prompt is `med_copilot/data/guidelines/prompts/local_search_system_prompt.txt`, adapted to answer open-ended diagnostic and management questions.
    - Guideline coverage and indexing fidelity relative to the published study remain unverified. The existence of usable GraphRAG artifacts does not establish completeness or exact study reproduction.
7. Final prompt for Assessment and Plan generation
    - target patient S/O + selected reference SOAP (if available) + GraphRAG guideline response text → final prompt
    - The prompt is `EVALUATE_TEMPLATE_KEYINFO` in `med_copilot/prompts/diagnosis_plan.py`.
    - The GraphRAG return-value issue is fixed: the runner unpacks the response as `key_info, graphrag_context`, saves response text as `guideline_evidence` and the complete context separately as `graphrag_context`, and passes only response text into the final prompt.
8. LLM Assessment-and-Plan generation
    - The project runner calls `gpt-4o-mini`. Its prompt requests Markdown headings `## Assessment` and `## Plan`, with the most likely diagnosis in Assessment and investigations, treatment, and follow-up in Plan.
    - The runner saves the complete reply exactly as returned as `assessment_plan_raw`. `split_assessment_plan` in `scripts/med_copilot_helpers.py` splits it at its Assessment and Plan headings, and the runner saves each section's text, unchanged apart from surrounding whitespace, as `assessment` and `plan`. A heading is either a Markdown heading (`#` to `######`, as the prompt requests) or a standalone label like the reference patient's (`Assessment:`, `**Plan**`); a label without `#` needs a colon or bold. Case, bold, an `(A)`/`(P)` suffix, and a trailing colon are tolerated, but nothing else may share the line: `Assessment: pneumonia`, a bare `Plan`, list items, and combined headings such as `Assessment and Plan:` are not section headings. A section runs to the next Assessment or Plan heading, so subheadings stay inside it, and text before the first heading appears only in the raw reply. A section that is missing, empty, or headed more than once is `null` while the other section and the raw reply are still saved; the patient remains `completed`. When no reply was generated, all three fields are `null`. The exact system and user prompts are saved as `assessment_plan_prompts`. Further output-format validation is planned as a separate post-hoc script.

## Saved Run Outputs

Each execution creates `experiments/output/<Pacific timestamp>__<input stem>__<branch>/` (for example `2026-10-17_153045PDT__patient_records_so__integrate-medcopilot`), never reusing an existing directory. It contains only:

| File | Contents |
|---|---|
| `run.json` | Run name, Pacific start/finish times, run status (`running`, `completed`, `completed_with_errors`, `failed_incomplete`), exact branch, input path and SHA-256, Git commit, and the reference-patient and guideline resources used with their fingerprints |
| `results.jsonl` | One line per input patient: `id`, `status` (`completed`, `failed_incomplete`, `not_processed`), `assessment`, `plan`, `assessment_plan_raw`, `selected_reference_patient`, `guideline_questions`, `guideline_evidence`, `graphrag_context`, `assessment_plan_prompts`, `keyword_extraction_raw`, `retrieval_keywords`, `retrieved_patient_ids`, `reranked_patient_ids`, `model_responses` |

Patient results are appended as each patient finishes, so an interruption keeps earlier results. Unavailable scalar/ranking fields are `null`; response-stage lists start empty. A failed patient keeps captures already obtained, including responses rejected by validation and rankings obtained before attribution/formatting failures. See the [output schema](output_schema.md) for ordering, identity rules, null/empty semantics, and provider limitations. The command exits nonzero unless the run status is `completed`. `experiments/output/` is gitignored.

## Setup and Verification

Follow the MED-COPILOT setup commands in the [root README](../../README.md). The root [requirements.txt](../../requirements.txt) supplies the Python dependencies and scientific-language model. Git LFS supplies tracked large artifacts; both `OPENAI_API_KEY` and `GRAPHRAG_API_KEY` must be available for live execution. The runner validates required guideline tables and scopes local search to their configured output directory, restoring the launch directory afterward.

The runner initializes BioClinicalBERT before importing FAISS as an Apple-Silicon compatibility workaround.

The offline runner suite is [tests/test_run_med_copilot.py](../../tests/test_run_med_copilot.py); its external model/API/retrieval dependencies are mocked. It includes production S/O matching through the final request and saved outputs. Direct matching/cache tests are in [tests/test_med_copilot_retrieval.py](../../tests/test_med_copilot_retrieval.py), with fake models, API, and FAISS and real NumPy scoring/file I/O. The observability suite, [tests/test_med_copilot_observability.py](../../tests/test_med_copilot_observability.py), exercises real FNLLM adapters and vendored GraphRAG wrappers/streaming with mocked SDK responses, including cached/shared models and the background embedding loop. The runner suite verifies persisted JSONL captures, partial failures, and interruption handling. No live API calls are used. A separate offline smoke test used two temporary reference records with cached BioClinicalBERT, the real cross-encoder, spaCy attribution, and real FAISS; A/P-only edits preserved all matching scores and selection, reused the S/O cache, and updated the full-SOAP example. Keyword extraction was mocked. These checks do not verify live API/GraphRAG responses, full-cohort output quality, clinical correctness, or faithful reproduction of the published study.

## Remaining Integration and Accepted Pilot Decisions

- Integrate the full project reference corpus using [reference_corpus_integration.md](reference_corpus_integration.md). Its standalone builder and passage-based adapters remain proposed; the current `scripts/med_copilot_retrieval.py` implements S/O-only matching for the existing corpus. Passage handling and full-corpus integration remain separate work.
- The data team is responsible for keeping target and reference patients disjoint; the current runner does not enforce patient exclusion.
- Implement and test the separate post-hoc Assessment/Plan format checker when that work is taken up.
- No generation prompt-length policy is implemented. Over-limit API requests fail that patient, which is recorded as `failed_incomplete`.
- The main experiment still needs its evaluation definitions. Keep evaluation diagnosis labels outside model inputs; plan evaluation needs its own rubric or reference.

## Repository Boundary

The vendored MED-COPILOT and GraphRAG baseline is under:

`med_copilot/upstream/`

Do **not** modify source code inside `med_copilot/upstream/`. Put experimental adaptations in project-owned modules. Runtime regeneration of upstream embedding and FAISS artifacts is allowed, but the planned full-reference-corpus integration uses separate project-owned artifact paths.
