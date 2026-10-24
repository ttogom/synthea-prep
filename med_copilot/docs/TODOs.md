# MED-COPILOT pipeline differences

## Intended differences

1\. **Target data and prediction task differ.** The paper evaluates MIMIC-IV ICU notes with **S/O/A supplied and P predicted**. Our experiment uses prepared Synthea **S/O to predict A/P**, including the diagnosis.  
*Verified:* paper §4.1; current input JSONL, runner loader, and project generation prompt.

2\. **Input normalization occurs outside this runner.** It directly accepts prepared `subjective` and `objective` strings. The released interactive pipeline first uses an LLM to convert free text into S/O/A. This is specifically a **released-demo difference**; the paper’s note-generation evaluation already starts with SOAP.  
*Verified:* runner `load_first_patient()` versus `upstream/evaluater.py`.

3\. **Guideline questions now explicitly support diagnosis.** Our prompt adds diagnostic criteria, distinguishing findings, confirmatory/exclusion investigations, and diagnoses framed as hypotheses. The released SOAP prompt concentrates on symptom management, medications, and treatment.  
*Verified:* project `diagnosis_plan.py` versus upstream `SOA_P_TEMPLATE.py`. Both retain at most four questions, forwarded together as one GraphRAG query.

4\. **Guideline-answer prompting is project-specific.** The active prompt requests question-focused diagnostic comparisons, investigations, and conditional management, and permits helpful general knowledge. It differs from the released default management-report prompt. The paper does not publish an exact equivalent prompt, so this is a **verified code adaptation**, not a fully measurable paper-prompt mismatch.  
*Verified:* configured prompt path and both prompt files.

## TODO pre-midterm

1\. **The similar-patient corpus is substantially smaller and differently composed.** The paper describes **36,000 cases: 18,000 MIMIC-IV and 18,000 Synthea**. The runner uses **1,000 released SOAP records**, representing **1,000 admissions from 453 subject IDs**, all with MIMIC-style discharge-note metadata. The project reference corpus is not integrated.  
*Verified:* every row of `soap_with_metadata.json`, its metadata, the selected corpus path, and the project corpus directory.

2\. **The guideline embedding model differs.** The paper specifies **BioClinicalBERT** for guideline TextUnits, entity descriptions, and community summaries. Our artifacts and query configuration use **OpenAI `text-embedding-3-small`**.  
*Verified:* settings, historical indexing logs, and all three LanceDB tables’ **1,536-dimensional vectors**.

3\. **The final generation backbone differs.** Our final call uses **GPT-4o-mini**. The paper evaluates **DeepSeek-Chat, GPT-4.1-mini, and Gemini 2.5**. Our prompt additionally requests Markdown Assessment and Plan sections because the prediction task changed.  
*Verified:* runner model selection and final call; project prompt; paper Table 1.

4\. **The study’s evaluation and ablations are not implemented in this runner.** It processes every record of the supplied 100-record JSONL sequentially. It does not run the paper’s 1,000-note evaluation, MedQA/MMLU experiments, generation metrics, QA accuracy, multiple backbones, standard-RAG baseline, or evidence-source ablations. Evaluation utilities exist upstream but are not invoked.  
*Verified:* complete runner flow, input count, upstream evaluation utilities, and paper §§4.1–4.4.

5\. **Interactive evidence inspection is absent.** This is a fixed script rather than the paper’s interactive system with evidence toggles, source inspection, colored saliency, and follow-up conversation. Saliency calculations still run and appear as text; their interactive presentation is absent. Selected reference output also omits its available patient/admission/note identifiers.  
*Verified:* runner outputs, upstream formatter and app, and paper §4.5.

6\. **Extend the script to process multiple patients in parallel.**  
Process all selected patient records with configurable concurrency. Respect API rate limits, retry transient failures, isolate failures by patient, and save each patient’s outputs separately. Reuse loaded models and indexes where safe, and report overall progress and completion status.

7\. **Define a reproducible run-output structure.**  
Define machine-readable run metadata and per-patient results that record:

- Run/patient identifiers, timestamps, code version, and input/corpus/index versions or hashes.
- Every model used, its pipeline role, and available model revision.
- Prompt versions and exact rendered prompts.
- Effective retrieval, GraphRAG, generation, and concurrency parameters.
- Generated questions, retrieved patient IDs/scores, guideline evidence and provenance, and final Assessment/Plan.
- Completion status, errors, retries, runtime, and token usage where available.

## TODO post-midterm

1\. **The study’s similarity-weight tuning is not reproduced.** The runner fixes `alpha=0.5`; the paper reports tuning its mixing weight on **100 validation cases**. No corresponding tuning procedure is wired or documented here. The paper does not disclose the selected value, so **0.5 is not a proven numerical mismatch**.  
*Verified:* paper §4.1; runner and released evaluator settings; repository search for tuning.

2\. **The searchable guideline corpus differs materially.** The paper reports **525 NICE + 118 WHO guidelines**. Our index contains **226 source documents**:

- **221 NICE NG-series page extracts**, covering NG1–NG250 with gaps, all named as first-part extracts.
- **Five additional guidelines:** CHEST/ATS ventilator liberation (2017); CHEST red-cell transfusion in critically ill adults (2025); ACC cardiogenic shock guidance (2025); ATS adult ARDS update (2024); Surviving Sepsis Campaign adult guidelines (2021).

**No standalone WHO-authored guideline is indexed.** Incidental WHO citations within NICE documents do not constitute the missing WHO corpus.  
*Verified:* source files, every indexed document’s title and text, document-to-chunk links, and the matching archive. The paper provides no manifest permitting exact version-by-version reconciliation.

## Needs further investigation

1\. **Patient matching is asymmetric.** **Resolved in the project runner:** `scripts/med_copilot_retrieval.py` now uses reference S/O for embeddings, keyword scoring, and cross-encoder reranking, with a separate validated S/O cache. The selected reference’s full SOAP remains the generation example. Upstream retains its original S/O/A implementation unchanged. Full-corpus integration and passage handling remain separate work.
*Verified:* project-owned adapter, runner wiring, direct matching/cache tests, and full-pipeline regression tests confirming A/P-independent selection and full-SOAP generation evidence.

2\. **Clinical keyword matching is a simpler implementation than the paper’s description establishes.** The used code averages extracted keyword embeddings and combines dense similarity with an **equally weighted substring-hit fraction**. There are no explicit clinical-importance weights or eligibility-constraint checks. This behavior is inherited from the release; the paper underspecifies its detailed scoring formula.  
*Verified:* paper §§1, 3.3; keyword extraction, embedding, and scoring implementations.

3\. **Guideline segmentation differs.** The paper describes recommendation/decision units respecting guideline hierarchy. The actual index uses **1,200-token windows with 100-token overlap**, producing **5,208 TextUnits**.  
*Verified:* configuration, historical build logs, chunking workflow, and actual TextUnit lengths.

4\. **Clinical graph normalization and relation structure are less explicit.** The paper describes medical NER, SNOMED-CT/ICD-10/ATC normalization, and typed clinical relations with qualifiers. The available implementation extracts names, types, descriptions, and weighted relationships, without structured terminology links or age/escalation/contraindication fields. Such information can still occur in prose.  
*Verified:* extraction prompt and workflow, configuration, and entity/relationship schemas.

5\. **Guideline provenance is coarser.** The paper specifies source, section title, character span, and inspectable source links. Our artifacts retain filenames and document/chunk/graph identifiers, but lack structured section/span metadata; all document metadata entries are null.  
*Verified:* actual document and TextUnit tables and their links.

**Publication provenance recovered (2026-10-22):** The [verified manifest](../data/guidelines/guideline_manifest.csv) maps all 226 document IDs to original publication titles, issuing organizations, bibliographic dates, and official source URLs. [Evidence and limitations](../data/guidelines/provenance/README.md) distinguish the 203 NICE extracts whose ISBNs match current PDFs, 17 whose ISBNs differ, and superseded NG51, whose overview no longer offers a PDF. The five journal publication editions are verified by DOI and NCBI/Crossref records. Exact historical NICE revision dates, original PDF snapshots, and recommendation-level section/span mappings remain unverified. The existing index metadata and retrieval behavior have not been changed by this provenance work.

6\. **Guideline retrieval/fusion differs from the published diagram.** Figure 8 shows vector-based and graph-based retrieval followed by weighted combination. Our execution uses **entity-description vector search to seed local graph context**, then selects linked communities, relationships, and TextUnits through context-building rules and budgets. There is no separate implementation of the diagram’s weighted retrieval-result fusion.  
*Verified:* Figure 8; GraphRAG API, entity mapping, and local-context builder.
