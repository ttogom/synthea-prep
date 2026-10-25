# synthea-prep

Generates a reproducible synthetic EHR dataset using [Synthea](https://github.com/synthetichealth/synthea) for a clinical-reasoning fine-tuning project.

```bash
# Generate the full cohort (10,000 patients, default seed)
./scripts/generate_synthea.sh 10000 20260916

# Summarize the run
python3 scripts/summarize_dataset.py data/pop10000-seed20260916/
```

See [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) for prerequisites, settings rationale, and reproducibility details.

## MED-COPILOT pipeline

Install Python 3.11 and [Git LFS](https://git-lfs.com/), then run these commands from the repository root:

```bash
git lfs install
git lfs pull

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4

export OPENAI_API_KEY="your-key"
export GRAPHRAG_API_KEY="$OPENAI_API_KEY"

python tests/test_med_copilot_helpers.py
python tests/test_med_copilot_retrieval.py
python tests/test_med_copilot_scoring.py
python tests/test_run_med_copilot.py
python scripts/run_med_copilot.py
# Include patients without presenting evidence as well:
python scripts/run_med_copilot.py --presenting-evidence-only false
```

[requirements.txt](requirements.txt) contains the Python dependencies and the `en_core_sci_md` model. Keep it updated as the pipeline changes. BioClinicalBERT, the cross-encoder, and BERTScore's default RoBERTa-large encoder download from Hugging Face on first use, so initial setup requires internet access. The unchanged upstream evaluator also attempts four NLTK resource downloads at import. The offline tests replace those download calls and pretrained BERTScore; real-tokenizer parity tests require locally provisioned NLTK data.

Configure `PATIENT_INPUT` and `PATIENT_LABELS` independently in [scripts/run_med_copilot.py](scripts/run_med_copilot.py). For now, their defaults are `experiments/input/test_sample_so.jsonl` and `experiments/input/test_sample_labels.jsonl`. Each input line has a nonempty, unique string `id` and prepared `subjective` and `objective` strings, used directly without an LLM normalization call. Labels are required and joined by exact `id`, independently of file order:

```json
{"id": "patient-id", "condition": "strep_throat", "plan": ["Treatment", "Follow-up"], "has_presenting_evidence": true}
```

Every input ID must have one label before filtering. Duplicate IDs, invalid condition/plan types, or a non-boolean evidence flag fail preflight before per-patient API calls. A label file may contain extra IDs for other input subsets. `bin` and other label metadata do not enter prompts. The configured sample contains 10 matched patients: the default evidence filter selects 7, and `--presenting-evidence-only false` includes all 10. To use the larger `patient_records_so.jsonl` input later, configure **both** paths and supply its matching label file.

`--presenting-evidence-only` defaults to `true` and selects patients whose joined label has `has_presenting_evidence: true`. Pass `--presenting-evidence-only false` to process every input patient. Input order is preserved; excluded patients have no result rows and do not become `not_processed`. An empty selected cohort fails before a run directory is created. The filter applies only to experiment patients, not the similar-patient corpus.

After generation, [scripts/med_copilot_scoring.py](scripts/med_copilot_scoring.py) invokes the unchanged upstream metric helper separately for each field. Condition scoring compares generated `assessment` to `condition.replace("_", " ")`; Plan scoring compares generated `plan` to `", ".join(label["plan"])`. Exactly `["none recorded"]` skips Plan scoring; an empty joined Plan is an unavailable reference. Each result's `evaluation.condition` and `evaluation.plan` contains six metric keys and an explicit status; unavailable/error scores are JSON `null`, not zero or `NaN`. Ordinary scoring errors preserve the generation and do not change its status or the run exit code.

Scores preserve upstream defaults and operate on one patient/field at a time. Thus BLEU is **singleton-corpus BLEU**; averaging saved BLEUs does not reproduce dataset corpus BLEU, and exact texts shorter than four tokens can score zero. BERTScore reloads its model for each helper call and truncates at its encoder's token limit; this implementation adds no cache, chunking, clinical normalization, or composite score. The run metadata records the label/helper hashes, selected cohort, installed package versions, and configured scoring defaults; it does not establish the model-weight revision used by the original study.

Reference matching uses S/O only for embeddings, keyword overlap, and cross-encoder reranking, through [scripts/med_copilot_retrieval.py](scripts/med_copilot_retrieval.py). The selected reference's complete SOAP is still supplied to final generation. S/O embeddings and the FAISS index are built on first use under `med_copilot/data/patient_retrieval_so/`, separate from the upstream S/O/A artifacts. A `matching.json` manifest binds the cache to the ordered S/O texts and encoder identity and checks artifact hashes; missing, stale, or corrupt caches are rebuilt. Upstream source and artifacts are left intact.

Each execution writes a new directory, never overwriting an existing one:

```text
experiments/output/<Pacific timestamp>__<input stem>__<branch>/   # e.g. 2026-10-17_153045PDT__test_sample_so__integrate-medcopilot
├── run.json        # run status, times, branch, commit, input and reference-resource fingerprints
└── results.jsonl   # one line per selected patient, including independent condition/plan evaluation
```

Run statuses are `running`, `completed`, `completed_with_errors`, and `failed_incomplete`; patient statuses are `completed`, `failed_incomplete`, and `not_processed`. The generated reply is saved exactly as returned in `assessment_plan_raw` and split at its Assessment and Plan headings (`## Assessment` or a standalone `Assessment:` label) into the `assessment` and `plan` fields; a section that is missing, empty, or repeated is saved as `null`. All three fields are `null` when no reply was generated. A failing patient does not stop the batch, and the command exits nonzero unless the run is `completed`. `experiments/output/` is gitignored. The tests mock external APIs; running the pipeline uses the configured OpenAI models. The runner locates guideline artifacts under `med_copilot/data/guidelines/` independently of the launch directory.

The [patient output schema](med_copilot/docs/output_schema.md) documents exact keyword replies, processed retrieval keywords, retained hybrid and cross-encoder IDs/scores, and per-stage provider response metadata. Captures survive subsequent failures and interruptions. They do not enter clinical prompts.
