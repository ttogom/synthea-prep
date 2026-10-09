# MED-COPILOT Follow-Up Implementation Handoff

**Status:** The project-owned pilot requests Assessment and Plan from target Subjective and Objective (S/O). It still uses the released 1,000-record reference corpus and S/O/A retrieval implementation. Integration of the full project reference corpus with S/O-only matching is documented but pending.

**Scope:** Current pilot behavior, setup, verification limits, and the remaining integration work. This document does not establish faithful reproduction of the MED-COPILOT study.

---

## Current Working Baseline

The current runner is:

[scripts/run_med_copilot.py](../../scripts/run_med_copilot.py). Paths below are relative to the repository root.

Its current Assessment-and-Plan workflow is:

1. free-text patient input
    - `NOTE` currently selects `data_run_med_copilot/input/Peter292_Gleichner915_0db4522b-544a-43b1-ad63-3caeb72be2ab.txt`. The runner reads the complete file without selecting an encounter or cutting at `# Assessment and Plan`.
    - Encounter selection and diagnosis-related scrubbing belong in input preparation, owned by the data team. S/O formatting alone does not remove diagnoses already stated in the supplied record.
2. MED-COPILOT patient normalization
    - Normalization currently remains inside the pilot and uses `gpt-4o-mini`, with `PATIENT_CASE_TEMPLATE` and `PATIENT_CASE_SYSTEM_TEMPLATE` in [med_copilot/prompts/diagnosis_plan.py](../prompts/diagnosis_plan.py).
    - The prompts request factual `subjective` and `objective` JSON strings, preserve relevant supplied findings, and prohibit inferred diagnoses or treatment recommendations. The runner parses the JSON and accesses these fields without an additional schema validator; the normalization boundary is trusted. This prompt instruction is not a factual-accuracy check.
    - The resulting S/O string is used consistently for patient search, reranking, question generation, and final generation.
3. Retrieval of top 20 similar patient candidates
    - Uses the 1,000 full-SOAP records in `med_copilot/upstream/soap_with_metadata.json`, with `HybridRetriever` from `med_copilot/upstream/create_embeddings.py`.
    - The target query contains S/O, but reference embeddings and keyword-overlap scoring still use reference S/O/A. Retrieval combines BioClinicalBERT dense scores and keyword overlap with `alpha=0.5`; keywords are extracted using `gpt-4o-mini`.
    - The current embedding and FAISS artifacts are `med_copilot/upstream/embeddings.npy` and `med_copilot/upstream/index.faiss`. The runner temporarily changes to the upstream directory so the retriever can load those relative paths, then restores the original working directory.
4. Cross-encoder reranking and reference selection
    - `cross-encoder/ms-marco-MiniLM-L-6-v2` compares target S/O with reference S/O/A and keeps the top 5. The first reranked reference's full SOAP, rerank score, and dominant-term information are supplied as the final prompt's example.
    - An empty corpus, no candidates, or no reranked results produces `No reference patient available.`; the final prompt supports generation without a reference. Retrieval exceptions still propagate.
5. Key-question generation
    - `gpt-4o-mini` is prompted to identify at most 4 diagnostic and management questions from target S/O and describe possible diagnoses as hypotheses. The returned question JSON is forwarded verbatim as one GraphRAG query; the runner does not validate the question list or query each question separately.
    - The prompt is `KEY_QUESTIONS_TEMPLATE` in `med_copilot/prompts/diagnosis_plan.py`.
6. GraphRAG guideline retrieval
    - Uses local search with `med_copilot/data/guidelines/settings.yaml`, the Parquet tables under `med_copilot/data/guidelines/output/`, and the entity-description LanceDB table. Community level is 2 and response type is `Multiple Paragraphs`.
    - The configured chat model is `gpt-4o-mini`; the query embedding model is `text-embedding-3-small`. This is separate from BioClinicalBERT patient retrieval.
    - The active answer prompt is `med_copilot/data/guidelines/prompts/local_search_system_prompt.txt`, adapted to answer open-ended diagnostic and management questions.
    - Guideline coverage and indexing fidelity relative to the published study remain unverified. The existence of usable GraphRAG artifacts does not establish completeness or exact study reproduction.
7. Final prompt for Assessment and Plan generation
    - target patient S/O + selected reference SOAP (if available) + GraphRAG guideline response text → final prompt
    - The prompt is `EVALUATE_TEMPLATE_KEYINFO` in `med_copilot/prompts/diagnosis_plan.py`.
    - The GraphRAG return-value issue is fixed: the runner unpacks the response as `key_info, graphrag_context`, saves response text to `answers_from_guidelines_db.txt` and context separately to `graphrag_context.json`, and passes only response text into the final prompt.
8. LLM Assessment-and-Plan generation
    - The project runner calls `gpt-4o-mini`. Its prompt requests Markdown headings `## Assessment` and `## Plan`, with the most likely diagnosis in Assessment and investigations, treatment, and follow-up in Plan.
    - The runner saves the returned text to `final_output.txt` without checking that either section exists or contains content. Output-format validation is planned as a separate post-hoc script.

## Saved Pilot Artifacts

The fixed output directory is `data_run_med_copilot/output/`:

| Artifact | Contents |
|---|---|
| `case_SO.txt` | Normalized target S/O, also used as the patient retrieval query |
| `top_similar_patient.txt` | Selected reference text and scores/terms, or the no-reference message |
| `questions_for_guidelines_db.txt` | Raw question JSON and the exact GraphRAG query |
| `answers_from_guidelines_db.txt` | GraphRAG response text |
| `graphrag_context.json` | Separate GraphRAG context |
| `final_prompt.txt` | Exact final-generation prompt |
| `final_output.txt` | Unvalidated model response |

The pilot does not copy the accepted input into its output directory, save a separate reference patient ID, create per-run directories, or maintain a run-status record. Main experiment input/output handling is deferred; the pilot's fixed directories are retained.

## Setup and Verification

Follow the MED-COPILOT setup commands in the [root README](../../README.md). The root [requirements.txt](../../requirements.txt) supplies the Python dependencies and scientific-language model. Git LFS supplies tracked large artifacts; both `OPENAI_API_KEY` and `GRAPHRAG_API_KEY` must be available for live execution. Launch from the repository root because the vendored Parquet loader searches beneath the current working directory.

The runner initializes BioClinicalBERT before importing FAISS as an Apple-Silicon compatibility workaround.

The offline runner suite is [tests/test_run_med_copilot.py](../../tests/test_run_med_copilot.py); its external model/API/retrieval dependencies are mocked. Earlier verification also completed an isolated-checkout run with real local models, retrieval, and GraphRAG context construction, while substituting remote API responses. A separate fresh installation of the root requirements passed dependency checks and model-client initialization. These checks establish local execution and setup behavior; they do not verify live API responses, clinical correctness, or faithful reproduction of the published study.

## Remaining Integration and Accepted Pilot Decisions

- Integrate the full project reference corpus using [reference_corpus_integration.md](reference_corpus_integration.md). Its builder and adapters are proposed files, not currently implemented. The chosen design matches target S/O against reference S/O and supplies the selected reference's full SOAP to final generation.
- The data team is responsible for keeping target and reference patients disjoint; the current runner does not enforce patient exclusion.
- Implement and test the separate post-hoc Assessment/Plan format checker when that work is taken up.
- No generation prompt-length policy is implemented. Over-limit API requests are allowed to fail visibly; previous output files can remain after a failed run.
- The main experiment will need its own input/output handling and evaluation definitions. Keep evaluation diagnosis labels outside model inputs; plan evaluation needs its own rubric or reference.

## Repository Boundary

The vendored MED-COPILOT and GraphRAG baseline is under:

`med_copilot/upstream/`

Do **not** modify source code inside `med_copilot/upstream/`. Put experimental adaptations in project-owned modules. Runtime regeneration of upstream embedding and FAISS artifacts is allowed, but the planned full-reference-corpus integration uses separate project-owned artifact paths.
