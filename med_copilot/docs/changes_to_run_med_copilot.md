
## 1. GraphRAG Return-Value Boundary (Completed)

### Original smoke-test issue

`run_local_search(...)` returns:

``` python
(response_text, context_data)
```

The smoke-test output shows the second object contains DataFrames for:

-   reports
-   entities
-   sources
-   other retrieved context

The original smoke-test runner assigned the entire tuple to `key_info`
and inserted it into the final prompt. This has been corrected in
`scripts/run_med_copilot.py`.

Previously, the final LLM could receive a Python string representation of
both the generated guideline answer **and** the raw GraphRAG context
object/DataFrames.

The current runner sends only the generated guideline answer to final
generation and saves the raw context separately for provenance/debugging.

### Implementation and reference files

**Primary:** - `scripts/run_med_copilot.py`

**Reference:** - `med_copilot/upstream/evaluater.py` -
`med_copilot/upstream/graphrag/cli/query.py`

### Current implementation

The runner explicitly unpacks:

``` python
key_info, graphrag_context = run_local_search(...)
```

It saves `key_info` to `answers_from_guidelines_db.txt` and passes it as
`Key_info` when formatting the final prompt. It saves `graphrag_context`
separately to `graphrag_context.json`.

The frozen upstream source remains unchanged. The explicit response/context
boundary is implemented in the project-owned runner.

### Acceptance test

The runner saves `final_prompt.txt`. Existing runner tests verify that final
generation receives guideline text and that the complete GraphRAG context is
saved separately, rather than inserting the response/context tuple into the
prompt.

------------------------------------------------------------------------

## 2. Make the Runner Accept Any Patient File

### Problem

The current pilot runner hard-codes:

-   one repository-relative patient input path in `NOTE`
-   one specific patient

It reads the entire input file without selecting an encounter or cutting
at `# Assessment and Plan`. Encounter selection and Synthea cleaning belong
in input preparation, outside the runner.

The hardcoded input is sufficient for the pilot. Configurable input/output
paths are deferred to the main experiment script; the interface proposal
below describes that future work.

This item concerns only the **pipeline interface**, not Synthea
cleaning.

### Files to modify

-   `scripts/run_med_copilot.py`

Potentially add:

-   `med_copilot/io.py`

### Required implementation

Accept a prepared patient example through a command-line argument, for
example:

``` bash
python scripts/run_med_copilot.py --input /path/to/prepared_case.txt
```

The runner should treat the prepared input as opaque clinical evidence.
Synthea-specific scrubbing/cleaning should remain outside this pipeline.

### Acceptance test

Two different prepared case files can be run without editing Python
source.

------------------------------------------------------------------------

## 3. Add Structured Experiment Output and Provenance

### Problem

The successful smoke test proves execution but its results exist
primarily as terminal output. That is insufficient for comparing
experimental variants.

### Files to modify/add

Modify:

-   `scripts/run_med_copilot.py`

Add output directory:

``` text
results/med_copilot/
```

Recommended per-case JSON record containing:

``` text
case_id
experiment_name
timestamp
model
input identifier
patient representation
retrieval query
retrieved candidate IDs/scores
selected similar patient
key questions
GraphRAG guideline response
GraphRAG source/context metadata
final prompt version
predicted diagnosis
predicted differential
predicted plan
runtime/errors
```

Also record the Git commit of `synthea-prep` and the MED-COPILOT
upstream commit when practical.

### Acceptance test

Every run produces a machine-readable artifact sufficient to reconstruct
what evidence the model received.

------------------------------------------------------------------------

## 4. Reduce Debug Output Before Batch Runs

### Problem observed in the smoke test

A single case printed hundreds of candidate/debug lines plus GraphRAG
internals. This is useful during bring-up but will make batch
experiments difficult to inspect and store.

### Files to modify/wrap

Debug output originates primarily from:

-   `med_copilot/upstream/create_embeddings.py`
-   modified vendored GraphRAG code under
    `med_copilot/upstream/graphrag/`

Prefer controlling/suppressing this from project-owned runners rather
than editing frozen upstream immediately.

### Required implementation

Provide normal and verbose modes:

``` bash
--verbose
```

Default experimental runs should save important structured information
to result files rather than dumping every candidate to stdout.

### Acceptance test

A normal one-case run has concise stage-level output; `--verbose`
retains detailed retrieval diagnostics.
