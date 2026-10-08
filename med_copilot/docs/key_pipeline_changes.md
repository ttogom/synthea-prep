
# TODO to get our experiment configured correctly

## 1. Remove Assessment From the Input and Make Diagnosis an Explicit Output

### Problem

MED-COPILOT's initial LLM normalization generates Subjective, Objective, and Assessment (S/O/A). The generated Assessment is then used in downstream retrieval and reasoning, while the final LLM produces only a treatment plan.

For our experiment, **diagnosis/assessment must be a prediction target, not an input to downstream stages**. The final model must generate both a diagnosis and treatment plan.

### Files to add/modify

- **Modify:** `scripts/run_diagnosis_plan.py`
- **Add:** `med_copilot/prompts/diagnosis_plan.py`
- **Reference only:** `med_copilot/upstream/templates/SOA_P_TEMPLATE.py` (preserve frozen upstream)

### Required implementation

1. Modify initial normalization to generate **Subjective and Objective (S/O) only**, excluding Assessment.
2. Ensure downstream retrieval and reasoning use S/O without a pre-generated Assessment.
3. Replace the plan-only final prompt (`EVALUATE_TEMPLATE_KEYINFO`) with a project-owned prompt that generates **Diagnosis + Plan**, using patient evidence, similar-patient evidence, and guideline evidence.
4. Require a structured, machine-readable output, such as:

   ```json
   {
     "diagnosis": "...",
     "differential_diagnoses": [],
     "plan": "..."
   }

------------------------------------------------------------------------

## 2. Reconstruct and Verify the Complete MED-COPILOT Guideline Corpus

### Problem

The MED-COPILOT paper reports using **525 NICE guidelines and 118 WHO guidelines** (643 total) to construct its GraphRAG knowledge base.

Our audit of the released Hugging Face dataset `Cryo3978/medguidelines` identified:

- **221 identifiable NICE source documents**, but we have not established how many distinct guidelines these represent.
- No confirmed WHO guideline source corpus.
- Existing GraphRAG artifacts (Parquet tables and LanceDB indexes), but their completeness relative to the paper remains unverified.

The paper does not provide a manifest identifying the exact 643 guidelines used. Consequently, the released guideline corpus cannot currently be considered a faithful reproduction of the paper's dataset.

### Files to modify/add

**Existing guideline corpus:**
- `med_copilot/data/guidelines/input/`
- `med_copilot/data/guidelines/output/`
- `med_copilot/data/guidelines/settings.yaml`

**Add:**
- `med_copilot/data/guidelines/guideline_manifest.csv` — inventory of NICE and WHO guidelines, including identifiers, titles, versions, source URLs, and verification status.

**Reference:**
- Hugging Face: `Cryo3978/medguidelines`
- [NICE Guidance](https://www.nice.org.uk/guidance)
- [WHO Guidelines](https://www.who.int/publications/who-guidelines)

### Required implementation

1. **Audit NICE coverage:** Identify the distinct NICE guidelines represented in the released source documents and GraphRAG artifacts. Compare against the paper's reported 525 guidelines, accounting for multiple guidelines within individual documents.

2. **Recover missing NICE guidelines:** Identify and retrieve any missing guidelines, preferably matching the versions available when MED-COPILOT was developed.

3. **Recover WHO guidelines:** Attempt to obtain the original 118-guideline manifest from the authors or released artifacts. If unavailable, reconstruct a comparable corpus from official WHO publications and document the deviation.

4. **Create a verified guideline manifest:** Record each guideline's organization, identifier, title, publication/version date, source URL, local file path, and whether its inclusion in the original MED-COPILOT corpus is confirmed or reconstructed.

5. **Rebuild the GraphRAG index:** Reprocess the completed guideline corpus using the paper-described indexing methodology, including the appropriate embedding model. Preserve the original released artifacts separately for comparison.

### Acceptance test

- All **525 NICE and 118 WHO guidelines** are individually accounted for in the manifest.
- Each guideline maps to retrievable source material.
- The reconstructed GraphRAG index successfully loads and answers test queries.
- Any differences from the original MED-COPILOT corpus or indexing methodology are explicitly documented.

**Reproduction limitation:** If the original guideline identities or versions cannot be recovered, a corpus containing 643 guidelines is not sufficient to claim exact reproduction. The result must be identified as a documented reconstruction.

------------------------------------------------------------------------

## 3. Replace the Released Similar-Patient Corpus With the Project Corpus

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


# TODO before running pipeline to present results

## 1. Define the Evidence Bundle Passed to the Final Model

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

## 2. Add Experimental Configuration Instead of Hard-Coding Variants

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