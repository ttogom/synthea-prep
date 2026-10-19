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

export OPENAI_API_KEY="your-key"
export GRAPHRAG_API_KEY="$OPENAI_API_KEY"

python tests/test_med_copilot_helpers.py
python tests/test_run_med_copilot.py
python scripts/run_med_copilot.py
```

[requirements.txt](requirements.txt) contains the Python dependencies and the `en_core_sci_md` model. Keep it updated as the pipeline changes. BioClinicalBERT and the cross-encoder download from Hugging Face on the first run, so initial setup requires internet access.

The runner processes every record in `experiments/input/patient_records_so.jsonl`, configured by `PATIENT_INPUT` in [scripts/run_med_copilot.py](scripts/run_med_copilot.py). Each line is a JSON object with a nonempty, unique string `id` and prepared `subjective` and `objective` strings, used directly without an LLM normalization call. Optional evaluation labels belong in `experiments/input/patient_records_so_labels.jsonl`, joined by `id`; the runner does not read them.

Each execution writes a new directory, never overwriting an existing one:

```text
experiments/output/<Pacific timestamp>__<input stem>__<branch>/   # e.g. 2026-10-17_153045PDT__patient_records_so__integrate-medcopilot
├── run.json        # run status, times, branch, commit, input and reference-resource fingerprints
└── results.jsonl   # one line per patient: id, status, Assessment/Plan, reference, guideline evidence, exact prompts
```

Run statuses are `running`, `completed`, `completed_with_errors`, and `failed_incomplete`; patient statuses are `completed`, `failed_incomplete`, and `not_processed`. A failing patient does not stop the batch, and the command exits nonzero unless the run is `completed`. `experiments/output/` is gitignored. The tests mock external APIs; running the pipeline uses the configured OpenAI models. The runner locates guideline artifacts under `med_copilot/data/guidelines/` independently of the launch directory.
