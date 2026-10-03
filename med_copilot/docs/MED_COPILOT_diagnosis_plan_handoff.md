# MED-COPILOT Follow-Up Implementation Handoff

**Status:** The released MED-COPILOT treatment-plan pipeline now runs
end-to-end inside `synthea-prep`.

**Scope of this handoff:** Changes needed to turn the working
reproduction into the project's diagnosis + treatment-plan experimental
baseline. Synthea note cleaning, diagnosis scrubbing,
longitudinal-window construction, and ground-truth extraction are
intentionally excluded because those are being handled separately by the
data-preparation team.

------------------------------------------------------------------------

## 1. Current Working Baseline

The current runner is:

`synthea-prep/scripts/run_med_copilot.py`

It successfully executes:

1.  free-text patient input
2.  MED-COPILOT patient normalization
3.  Subjective / Objective / Assessment generation
4.  similar-patient retrieval
5.  cross-encoder reranking
6.  key-question generation
7.  GraphRAG guideline retrieval
8.  original MED-COPILOT final prompt
9.  LLM treatment-plan generation

The first full smoke test completed successfully on the Synthea note for
`Peter292...`, producing a final treatment plan.

The frozen source reproduction remains under:

`med_copilot/upstream/`

Do **not** make experimental changes directly inside
`med_copilot/upstream/`. It should remain the reference copy of the
released MED-COPILOT implementation.

------------------------------------------------------------------------

# Required Changes Before Running the Actual Experiment

## 2. Make Diagnosis an Explicit Model Output

### Problem

The current MED-COPILOT pipeline does not perform the task we ultimately
want to evaluate.

In the released pipeline, Stage 1 converts the input patient case into:

-   Subjective
-   Objective
-   **Assessment**

The generated Assessment is then passed into retrieval, question
generation, GraphRAG retrieval, and final plan generation.

For our experiment, diagnosis/assessment is itself a prediction target.
The pipeline therefore cannot generate an Assessment upstream and then
treat that Assessment as known evidence.

### Files to modify

**Primary:** - `scripts/run_med_copilot.py`

**Add a project-owned prompt file rather than editing the frozen
upstream prompt:** - recommended:
`med_copilot/prompts/diagnosis_plan.py`

**Reference only - do not edit:** -
`med_copilot/upstream/templates/SOA_P_TEMPLATE.py`

### Required implementation

Split the current workflow into:

``` text
patient evidence
      |
      v
Subjective + Objective representation
      |
      +-----------------------+
      |                       |
      v                       v
evidence retrieval       diagnosis reasoning
      |                       |
      +-----------+-----------+
                  |
                  v
       Diagnosis + Treatment Plan
```

At minimum, the final output schema should contain separate fields such
as:

``` json
{
  "diagnosis": "...",
  "differential_diagnoses": [],
  "plan": "..."
}
```

The exact diagnosis schema should be agreed on before batch experiments
begin.

### Acceptance test

A run must be possible in which the target diagnosis is absent from the
pipeline input and the final response contains an explicit diagnosis
plus treatment plan.

------------------------------------------------------------------------

## 3. Remove Assessment From Similar-Patient Retrieval

### Problem

The current runner reproduces MED-COPILOT's retrieval query:

``` text
Subjective + Objective + Assessment
```

That is appropriate for reproducing the released treatment-plan system,
but not for diagnosis prediction. If Assessment contains the model's
diagnosis before retrieval, the retrieval system is effectively being
told what disease to search for.

The smoke test demonstrates this path directly: the Stage 1 Assessment
was incorporated into `conditions`, and `conditions` became the query
for `HybridRetriever`.

### Files to modify

**Primary:** - `scripts/run_med_copilot.py`

**Upstream component used but preferably not edited:** -
`med_copilot/upstream/create_embeddings.py`

### Required implementation

Construct the retrieval query from only information legitimately
available before diagnosis prediction, initially:

``` text
Subjective + Objective
```

Then pass that query unchanged to:

-   `HybridRetriever.search(...)`
-   `CrossEncoderReranker.rerank(...)`

Do not change the retrieval algorithms themselves for the baseline
unless that becomes an explicit experimental condition.

### Acceptance test

Log the exact retrieval query and verify that it contains no generated
Assessment/diagnosis.

------------------------------------------------------------------------

## 4. Separate Baseline Fidelity From the New Experimental Task

### Problem

We now have two different things that must not be conflated:

1.  **Reproduced MED-COPILOT baseline:** intended to preserve the
    released pipeline.
2.  **Our diagnosis + plan task:** necessarily changes MED-COPILOT
    because diagnosis is no longer supplied through Assessment.

If `run_med_copilot.py` is continuously edited into the experimental
system, we lose the clean reference baseline.

### Files to modify/add

Keep:

-   `scripts/run_med_copilot.py` - frozen working reproduction after
    cleanup.

Add:

-   recommended: `scripts/run_diagnosis_plan.py`
-   recommended: `med_copilot/prompts/diagnosis_plan.py`

Later, if several experimental variants emerge, factor shared code only
after the variants are understood.

### Required implementation

Preserve a runnable version of the exact working pipeline before
introducing diagnosis-task changes.

The experimental runner should reuse the same retrieval models,
guideline database, and model configuration wherever those components
are intended to remain controlled.

### Acceptance test

Both commands should independently run:

``` bash
python scripts/run_med_copilot.py
python scripts/run_diagnosis_plan.py ...
```

The first should reproduce the MED-COPILOT treatment-plan path; the
second should implement the project's diagnosis + plan task.

------------------------------------------------------------------------

## 5. Fix the GraphRAG Return-Value Boundary

### Problem discovered by the successful smoke test

`run_local_search(...)` returned:

``` python
(response_text, context_data)
```

The smoke-test output shows the second object contains DataFrames for:

-   reports
-   entities
-   sources
-   other retrieved context

The current runner assigns the entire tuple to `key_info` and then
inserts it into the final prompt.

That means the final LLM may receive a Python string representation of
both the generated guideline answer **and** the raw GraphRAG context
object/DataFrames.

This needs to be made explicit before experimental runs. It can affect
prompt length, reproducibility, and what evidence the final model
actually sees.

### Files to modify

**Primary:** - `scripts/run_med_copilot.py` - future
`scripts/run_diagnosis_plan.py`

**Reference:** - `med_copilot/upstream/evaluater.py` -
`med_copilot/upstream/graphrag/cli/query.py`

### Required implementation

For the experimental pipeline, explicitly unpack:

``` python
guideline_text, graphrag_context = run_local_search(...)
```

Pass only the intended evidence object into the final prompt.

Most likely:

``` python
Key_info = guideline_text
```

Keep `graphrag_context` separately for provenance/debugging.

Because the released evaluator appears to pass the return value through
without unpacking it, preserve the current behavior in the frozen
reproduction if strict implementation fidelity is desired. Document the
difference.

### Acceptance test

Print or save the exact final prompt once. Confirm it contains the
intended guideline text and does not accidentally contain
pandas/DataFrame representations.

------------------------------------------------------------------------

## 6. Replace the Plan-Only Final Prompt With a Diagnosis + Plan Prompt

### Problem

The current final prompt is:

`EVALUATE_TEMPLATE_KEYINFO`

from:

`med_copilot/upstream/templates/SOA_P_TEMPLATE.py`

It is designed around treatment-plan generation. The successful smoke
test consequently returned only:

``` text
Plan:
...
```

### Files to add/modify

**Add:** - `med_copilot/prompts/diagnosis_plan.py`

**Modify:** - future `scripts/run_diagnosis_plan.py`

**Reference only:** - `med_copilot/upstream/templates/SOA_P_TEMPLATE.py`

### Required implementation

Create a project-owned prompt derived from the MED-COPILOT evidence
structure but requiring a machine-readable diagnosis + plan response.

The prompt should distinguish:

-   patient evidence
-   similar-patient evidence
-   guideline evidence
-   requested prediction

Use structured JSON output if supported by the selected model.

### Acceptance test

The same patient input should produce parseable fields for diagnosis and
plan, not free-form text requiring manual extraction.

------------------------------------------------------------------------

## 7. Define the Evidence Bundle Passed to the Final Model

### Problem

The experiment will likely include multiple variants that change
different MED-COPILOT components. If evidence is assembled ad hoc inside
each runner, variants will accidentally differ in more than the intended
intervention.

### Files to add/modify

Recommended project-owned module:

-   `med_copilot/evidence.py`

Modify:

-   future `scripts/run_diagnosis_plan.py`

### Required implementation

Represent the inputs to final reasoning explicitly, for example:

``` python
EvidenceBundle(
    patient_evidence=...,
    similar_patient=...,
    guideline_evidence=...,
    retrieval_metadata=...
)
```

This does **not** require a large framework. A dataclass or dictionary
is enough.

The important property is that experimental variants can consume the
same saved evidence bundle when the retrieval stage is supposed to be
held constant.

### Acceptance test

A single case's evidence can be serialized and inspected independently
of final generation.

------------------------------------------------------------------------

## 8. Make the Runner Accept Arbitrary Prepared Patient Files

### Problem

The current smoke-test runner hard-codes:

-   one absolute Synthea path
-   one specific patient
-   one encounter split
-   local preprocessing logic

That was appropriate for proving the pipeline works, but it cannot be
used for experiments or by teammates.

This item concerns only the **pipeline interface**, not Synthea
cleaning.

### Files to modify

-   future `scripts/run_diagnosis_plan.py`

Potentially add:

-   `med_copilot/io.py`

### Required implementation

Accept a prepared patient example through a command-line argument, for
example:

``` bash
python scripts/run_diagnosis_plan.py --input /path/to/prepared_case.txt
```

The runner should treat the prepared input as opaque clinical evidence.
Synthea-specific scrubbing/cleaning should remain outside this pipeline.

### Acceptance test

Two different prepared case files can be run without editing Python
source.

------------------------------------------------------------------------

## 9. Replace the Released Similar-Patient Corpus With the Project Corpus

### Problem

The current smoke test uses:

`med_copilot/upstream/soap_with_metadata.json`

This is the released 1,000-case MIMIC-style corpus. It is enough to
demonstrate that the MED-COPILOT retrieval machinery works, but it is
not the intended project database.

The current locally generated FAISS artifacts are also tied to those
1,000 records:

-   `med_copilot/upstream/embeddings.npy`
-   `med_copilot/upstream/index.faiss`

### Files to modify/add

**Do not overwrite the frozen upstream corpus.**

Add project-owned corpus/index locations, for example:

``` text
med_copilot/data/patient_corpus/
    patients.json
    embeddings.npy
    index.faiss
```

Modify:

-   future `scripts/run_diagnosis_plan.py`

Potentially wrap/reuse:

-   `med_copilot/upstream/create_embeddings.py`

### Required implementation

Make corpus/index paths explicit parameters instead of relying on the
upstream class defaults and current working directory.

When the team's intended patient database is ready, build the
embeddings/index from that corpus.

### Acceptance test

The runner logs the corpus path, record count, embedding file, and FAISS
index used for each run.

------------------------------------------------------------------------

## 10. Add Structured Experiment Output and Provenance

### Problem

The successful smoke test proves execution but its results exist
primarily as terminal output. That is insufficient for comparing
experimental variants.

### Files to modify/add

Modify:

-   future `scripts/run_diagnosis_plan.py`

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

## 11. Add Diagnosis-Specific and Plan-Specific Evaluation

### Problem

MED-COPILOT's existing evaluation code is centered on generated-plan
similarity metrics. The project has two distinct targets:

1.  diagnosis
2.  treatment plan

They should not be collapsed into one text-similarity score.

### Files to add

Recommended:

-   `evaluation/evaluate_diagnosis.py`
-   `evaluation/evaluate_plan.py`
-   optionally `evaluation/run_evaluation.py`

Reference existing MED-COPILOT evaluation logic rather than modifying
the frozen source.

### Required implementation

At minimum, keep diagnosis and plan metrics separate.

Diagnosis evaluation should operate on the structured diagnosis output.

Plan evaluation can initially reproduce MED-COPILOT's available text
metrics for comparability, while later adding clinically meaningful
evaluation if the project design calls for it.

Do not decide the final metric suite implicitly in the runner.

### Acceptance test

Given saved predictions and reference labels, evaluation can run without
making new LLM/retrieval calls.

------------------------------------------------------------------------

## 12. Add Experimental Configuration Instead of Hard-Coding Variants

### Problem

The current runner hard-codes important choices such as:

-   model (`gpt-4o-mini`)
-   retrieval `alpha=0.5`
-   hybrid top-k = 20
-   reranker top-k = 5
-   GraphRAG community level = 2
-   final prompt
-   evidence sources enabled

These become experimental variables or controlled constants.

### Files to add/modify

Recommended:

-   `configs/med_copilot_baseline.yaml`
-   later one config per experimental condition

Modify:

-   future `scripts/run_diagnosis_plan.py`

### Required implementation

Move experiment-defining values into configuration while keeping
infrastructure/path defaults simple.

This is especially important if the group tests ablations such as:

``` text
patient only
patient + similar patients
patient + guidelines
patient + similar patients + guidelines
agentic/revision variants
```

### Acceptance test

Two experimental conditions can be run by changing a config argument
rather than editing source.

------------------------------------------------------------------------

## 13. Reduce Debug Output Before Batch Runs

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

------------------------------------------------------------------------

# Reproducibility / Engineering Work Required Before Team-Wide Use

## 14. Freeze the Known-Working Environment

The end-to-end run is now successful, so this is the right time to
capture the environment rather than continuing to rely on partially
pinned upstream requirements.

Important known setup facts include:

-   Conda environment: `medcopilot`
-   Python 3.11.16
-   Apple Silicon environment
-   `openai==1.99.9` required for the currently working GraphRAG/fnllm
    path
-   upstream requirements allowed a newer incompatible OpenAI client
-   `textblob` was required but undeclared
-   `litellm` currently declares an OpenAI-version conflict even though
    the working path uses fnllm
-   BioClinicalBERT must be initialized before FAISS import in this
    environment to avoid a native segfault
-   `GRAPHRAG_API_KEY` and `OPENAI_API_KEY` are both required
-   API billing is separate from a ChatGPT subscription
-   the LanceDB vector-index warning is currently nonfatal
-   GraphRAG uses deprecated fnllm configuration but currently executes
    successfully

### Files to add

Recommended:

-   `docs/MED_COPILOT_SETUP.md`
-   `environment/medcopilot.yml` or an equivalent pinned environment
    file

### Required commands/checks before freezing

-   `pip check`
-   capture exact package versions
-   capture Conda environment
-   rerun one smoke test from the documented setup

Do not include API keys in any committed file.

------------------------------------------------------------------------

## 15. Document Upstream Provenance and Local Deviations

### Files to add

Recommended:

-   `med_copilot/README.md`

### Required contents

Record:

-   MED-COPILOT upstream commit:
    `c65ba255ba8e8f687a84c197b3a0c7c09574460d`
-   frozen source location: `med_copilot/upstream/`
-   released guideline artifact/revision used
-   locally regenerated 1,000-case BioClinicalBERT/FAISS index status
-   final generation call restored by project runner because it is
    commented out in the canonical upstream source
-   GraphRAG tuple handling decision
-   Apple-Silicon FAISS/BioClinicalBERT initialization workaround
-   any intentional differences between reproduction and experimental
    pipeline

------------------------------------------------------------------------

# Recommended Near-Term File Layout

This is intentionally small; do not build a large framework before it is
needed.

``` text
synthea-prep/
├── med_copilot/
│   ├── upstream/                    # frozen released MED-COPILOT
│   ├── data/
│   │   ├── guidelines/
│   │   └── patient_corpus/          # future project retrieval corpus
│   ├── prompts/
│   │   └── diagnosis_plan.py
│   ├── evidence.py
│   └── README.md
│
├── scripts/
│   ├── run_med_copilot.py           # preserved reproduction
│   └── run_diagnosis_plan.py        # project experimental task
│
├── configs/
│   └── med_copilot_baseline.yaml
│
├── evaluation/
│   ├── evaluate_diagnosis.py
│   ├── evaluate_plan.py
│   └── run_evaluation.py
│
├── results/
│   └── med_copilot/
│
├── environment/
│   └── medcopilot.yml
│
└── docs/
    └── MED_COPILOT_SETUP.md
```

------------------------------------------------------------------------

# Suggested Implementation Order

1.  Preserve the now-working `run_med_copilot.py` reproduction.
2.  Fix/document the GraphRAG return-value boundary.
3.  Create `run_diagnosis_plan.py`.
4.  Remove Assessment from pre-diagnosis reasoning/retrieval.
5.  Create the diagnosis + plan output prompt/schema.
6.  Make prepared patient input a CLI argument.
7.  Save structured evidence, predictions, and provenance.
8.  Add config-driven experimental conditions.
9.  Add separate diagnosis and plan evaluation.
10. Swap in the team's intended similar-patient corpus when ready.
11. Freeze and document the working environment before team-wide
    reproduction.

------------------------------------------------------------------------

# Items Explicitly Out of Scope for This Handoff

Handled by the Synthea/data-preparation workstream and intentionally not
specified here:

-   diagnosis scrubbing / leakage cleaning
-   construction of the longitudinal context window
-   extraction/preservation of Synthea ground-truth diagnosis and plan
-   other Synthea preprocessing decisions

The MED-COPILOT pipeline should consume the prepared patient
representation produced by that workstream rather than duplicating its
logic.
