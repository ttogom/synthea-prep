# synthea-prep

Generates a reproducible synthetic EHR dataset using [Synthea](https://github.com/synthetichealth/synthea) for a clinical-reasoning fine-tuning project.

```bash
# Generate the full cohort (10,000 patients, default seed)
./scripts/generate_synthea.sh 10000 20260916

# Summarize the run
python3 scripts/summarize_dataset.py data/pop10000-seed20260916/
```

See [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) for prerequisites, settings rationale, and reproducibility details.