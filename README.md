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

python tests/test_run_med_copilot.py
python scripts/run_med_copilot.py
```

[requirements.txt](requirements.txt) contains the Python dependencies and the `en_core_sci_md` model. Keep it updated as the pipeline changes. BioClinicalBERT and the cross-encoder download from Hugging Face on the first run, so initial setup requires internet access.

The runner uses the patient input configured by `NOTE` in [scripts/run_med_copilot.py](scripts/run_med_copilot.py) and saves results under `data_run_med_copilot/output/`. The tests mock external APIs; running the pipeline uses the configured OpenAI models. Run from the repository root so the guideline-table loader can find the artifacts under `med_copilot/data/guidelines/`.
