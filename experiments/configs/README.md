# MED-COPILOT runner configuration

This directory is the home for all configuration files associated with
`scripts/run_med_copilot.py`. Keep model registries and future runner/experiment
configuration files here.

- `experiment.yaml`: the main run configuration. It selects the models and sets
  the shared `temperature` and `max_retries` passed to every selected final model.
  `max_patients` limits the cohort after evidence filtering in input-file order:
  use `1` for one patient, a positive integer for the first N, or `null` for all.
- `condition_plan_models.yaml`: the model definitions (provider, model ID,
  credential-variable name, endpoint, output mode and optional upstream settings).
  Keep `selected_models`, `temperature`, `max_retries` and `max_patients`
  in `experiment.yaml`.

The current selection is `gpt_4o_mini` plus all three downloaded Ollama models:

| Alias | Installed model tag |
| --- | --- |
| `ollama_healthcare` | `Jayasimma/healthcare:latest` |
| `ollama_llama3` | `llama3:latest` |
| `ollama_gemma2_2b` | `gemma2:2b` |

All three use `http://localhost:11434/v1` through the existing OpenAI-compatible
adapter and request `json_schema` output. No credential variable is needed for
local Ollama; keep Ollama running before starting an experiment. The current
`max_patients: 1` limits the run to one eligible patient. GPT and the shared
stages still require OpenAI credentials. The unselected `llama_3_3_70b` registry
entry is an optional Groq enterprise model, not one of these downloaded models.

The runner always reads these two files, independently of the launch directory.
Edit `selected_models` in `experiment.yaml` to choose model aliases in execution
order. There are no command-line overrides for the config files or selection.
Before a run, the runner validates aliases and required API-key variables and
asks you to confirm the selected models with `y` or `yes`. Any other response,
EOF, or Ctrl+C cancels without experiment output or generation requests. Missing
or blank key values and unknown aliases produce an error naming the files to fix.
This checks key configuration; authentication and model access are verified only
by actual provider requests. The prompt omits temperature, max_retries and shared
key status, while `run.json` retains the effective settings and both YAML hashes.
Every future setting added to `experiment.yaml` must also be recorded under
`run.json.experiment` in the same change, with tests verifying complete coverage.

Input/output paths remain configured in the runner for now;
the main experiment configuration can be extended to hold them later.

API keys remain in the repository-root `.env` (see `.env.example`), referenced
by environment-variable name from the configuration files.
