# Adding a MED-COPILOT experiment variant

Each file to edit has its own section below. **Always** sections apply to every
new variant; the other sections state when an edit is needed. References to other
files point to their dedicated sections, not additional edits within this one.

A component prepares reusable inputs; a variant selects components and a final
prompt. Every selected final model receives that variant's exact rendered messages
and returns the existing three ranked condition/Plan pairs. S/O means the target
patient's prepared Subjective and Objective text.

Replace `<NEW_VARIANT>` with your variant ID. Use a valid Python identifier in
module paths; `scripts/med_copilot_<NEW_VARIANT>.py` is a placeholder, not a literal
filename. Preserve existing variants when adding another.

## Always: `scripts/med_copilot_variants.py`

**Edit `VARIANTS` at the top of the file.** Add one entry for `<NEW_VARIANT>`.
Choose the components, an existing prompt key, and the output bindings it needs.
For example, to reuse both existing retrieval components:

```python
"<NEW_VARIANT>": VariantSpec(
    description="S/O + existing patient and guideline evidence",
    components=("patient_retrieval", "guidelines"),
    prompt_key="patient_retrieval_guidelines",
    bindings={
        "example": ("patient_retrieval", "example"),
        "Key_info": ("guidelines", "Key_info"),
    },
),
```

- `components`: logical IDs from `COMPONENTS`; use `()` for no preparation.
- `prompt_key`: a key in the existing `VARIANT_TEMPLATES` lookup. If adding a
  prompt in its dedicated file section below, set this to that new key.
- `bindings`: `prompt_field: (component_id, declared_output_name)`. Bind every
  replacement field except `conditions`, which the runner supplies from target
  S/O. Never bind or overwrite `conditions`.

Identical component IDs and effective settings execute once per patient and are
shared by all consuming variants and final models. Different settings create
separate configured instances. Reuse a component ID only for the same behavior;
a replacement implementation gets its own ID.

**Only for a new component, also edit `COMPONENTS` in this file.** Declare its
outputs, allowed parameters, resource/credential requirements, and source path:

```python
"custom_context": ComponentSpec(
    description="Custom reference preparation",
    outputs=("example",),
    parameters={"limit": Parameter("int", default=3, minimum=1)},
    implementation="scripts/med_copilot_<NEW_VARIANT>.py",
),
```

For this custom-component option, use the following variant entry **instead of**
the combined-component example above:

```python
"<NEW_VARIANT>": VariantSpec(
    description="S/O + custom reference evidence",
    components=("custom_context",),
    prompt_key="patient_retrieval",
    bindings={"example": ("custom_context", "example")},
),
```

`Parameter` supports nonblank `str`, `bool`, `int`, and finite `number`, with
optional inclusive numeric bounds. Supply a literal `default` or a `config_field`
that names a shared experiment attribute. `resources` names inputs initialized by
the runner; built-in IDs are `patient_retrieval` and `guidelines`. `credentials`
contains required environment-variable names, never token values. `implementation`
is the source path fingerprinted for provenance; it does not auto-import code.

Keep `VARIANTS` immediately after `VariantSpec`, ahead of component declarations.
Keep the registry lightweight: do not import the runner, component implementations,
model clients, or retrieval/FAISS modules here.

## Always: `experiments/configs/experiment.yaml`

Add `<NEW_VARIANT>` to `selected_variants`, retaining the variants to compare:

```yaml
selected_variants:
  - llm_only
  - patient_retrieval
  - guidelines
  - patient_retrieval_guidelines
  - <NEW_VARIANT>

variant_settings: {}
```

To override a declared component parameter, replace the empty `variant_settings`
mapping with the appropriate nested values. For the combined-component example:

```yaml
variant_settings:
  <NEW_VARIANT>:
    patient_retrieval:
      top_k: 10
```

For the custom-component example, use this mapping instead:

```yaml
variant_settings:
  <NEW_VARIANT>:
    custom_context:
      limit: 5
```

Omitted parameters use registered shared-field/default values. Identical
component/settings combinations share one execution; the `top_k` override creates
a separate patient-retrieval instance when other variants use a different value.

Keep shared cohort, path, retrieval, and final-generation settings in their
existing sections. `selected_models` selects final models; an intermediate model
alias belongs in a parameter declared by its component. Paths resolve from the
repository root. API key values never belong in this YAML. Effective settings,
including defaults, are recorded automatically in `run.json.experiment`.

## Only for new processing behavior: `scripts/med_copilot_<NEW_VARIANT>.py`

Create this module only when existing components cannot supply the desired
behavior. Define the function the runner will call. For the `custom_context`
example, start with this interface and replace the explicit failure with the
actual algorithm:

```python
def prepare(s_o, *, settings, capture):
    """Return accepted reference text; retain intermediate work in capture."""
    raise NotImplementedError("Implement reference preparation using settings['limit']")
```

- Use prepared target S/O, validated settings, and explicitly supplied shared
  resources. Extend the signature when the implementation needs those resources.
- Reuse existing retrieval/search/model utilities. Apply `settings["limit"]` in
  this example; return a string for the declared `example` output.
- Write intermediate queries, rankings, raw replies, provider usage, and resource
  fingerprints into `capture` as work completes. Keep outputs JSON-serializable.
- Let ordinary failures and interruptions reach the runner's capture context.
  Restore working directories and any locally changed state in `finally` blocks.
- Return only accepted evidence. A valid empty search may return an explicit
  fallback; never silently substitute another implementation's results.
- Keep target labels and scoring out of preparation. Do not create final-model
  slots, mark the component completed, or perform final scoring here.

The wrapper validates published outputs and marks completion. One configured
component may conduct several searches/model calls internally. Keep those calls
sequential with the existing wrappers, which change process working directories
and use mutable capture callbacks. Intermediate model aliases must be resolved by
this implementation; merely registering an alias does not invoke it.

## Only for a different final prompt: `med_copilot/prompts/diagnosis_plan.py`

Append a new entry to `VARIANT_TEMPLATES` after its existing definition. For
example, reuse the combined layout and add one writing instruction:

```python
VARIANT_TEMPLATES["<NEW_VARIANT>"] = (
    VARIANT_TEMPLATES["patient_retrieval_guidelines"]
    + "\nSpell out abbreviations in condition names and plan actions.\n"
)
```

- Leave the four existing templates and `CONDITION_PLAN_RESPONSE_FORMAT` intact.
- Keep new prompts in this file. If writing a full template, escape literal JSON
  braces as `{{` and `}}`; ordinary `{field}` placeholders are filled by `.format()`.
- Use `conditions` for target S/O. Every other placeholder requires a binding
  declared in the registry section. Attribute/index lookups in placeholders are
  unsupported.
- Keep final instructions constant when comparing retrieval implementations.
  This file needs no edit when an existing prompt fits the new variant.

The example uses the same `example` and `Key_info` fields as the combined prompt.
Selecting its key and declaring bindings are registry-file edits, covered there.

## Only for a new component: `scripts/run_med_copilot.py`

Add the implementation import to the top import section, after any required
dependency initialization. Preserve encoder-before-FAISS ordering:

```python
from scripts.med_copilot_<NEW_VARIANT> import prepare as prepare_custom_context
```

Add a labeled block inside `process_patient()`, before the final variant loop:

```python
# Custom reference preparation, once per configured instance.
if experiment.needs("custom_context"):
    for instance_id, settings in experiment.component_instances("custom_context"):
        component = result["components"][instance_id]
        if (component["status"] != "not_processed"
                or not component_needed(experiment, instance_id, result)):
            continue
        with capture_component(
            component,
            error_metadata=assessment_models.error_metadata,
            credential_names=experiment.required_credentials,
        ) as capture:
            component["outputs"]["example"] = prepare_custom_context(
                s_o, settings=settings, capture=capture,
            )
```

- Initialize and pass any new shared resources in `main()`/`process_patient()`
  only when required. Record their fingerprints through the existing provenance
  helpers and component capture.
- Retain the status guard: setup failures already mark affected instances failed.
- Publish outputs only after successful preparation. The capture context validates
  declared strings, isolates ordinary errors, restores capture bindings, and
  propagates interruptions to existing run finalization.
- Leave the generic `prepare_variant_messages()`/final-model loop, scoring,
  saving, and RESULTS block unchanged. Existing-component or prompt-only variants
  require no runner edit.
- Add no command-line options without explicit user approval.

## Only when a required model is absent: `experiments/configs/condition_plan_models.yaml`

Add an alias under the existing `models` mapping; preserve its other entries:

```yaml
models:
  <MODEL_ALIAS>:
    type: chat
    model_provider: openai
    model: <PROVIDER_MODEL_ID>
    api_key_env: <KEY_ENV_NAME>
    output_mode: json_schema
```

Choose provider/model values for the intended backend and a supported output mode
for final generation. Use an existing alias when possible. Do not add literal
keys or per-run `temperature`/`max_retries` here. Registration does not add a model
to final generation; that selection remains in the experiment YAML. Components
using intermediate calls must resolve their alias through their own implementation.

## Only for additional credentials: `.env`

Add the environment variable declared by the selected model/component:

```dotenv
<KEY_ENV_NAME>=<SECRET_VALUE>
```

Replace both placeholders locally. Keep `.env` untracked; saved artifacts may
contain variable names, never resolved credentials. Existing exported variables
take precedence over `.env` values.

## When extending the registry inventory: `tests/test_med_copilot_variants.py`

Update the expected registered IDs and the subprocess import-count assertion,
which currently expect four variants. Cover the new component/variant declaration
and keep the lightweight-import and top-of-file placement checks.

## Only for a new implementation: `tests/test_med_copilot_<NEW_VARIANT>.py`

Add direct offline cases for accepted outputs, empty results, errors, partial
captures, and interruptions. Mock external model/API/retrieval dependencies and
use temporary files for filesystem effects.

## When adding parameters or dependencies: `tests/test_med_copilot_config.py`

Cover defaults, valid/invalid overrides, effective metadata, resource/credential
requirements, and sharing of identical settings versus separate configurations.

## When the variant matrix or pipeline changes: `tests/test_run_med_copilot.py`

Update affected matrix expectations and add full-pipeline coverage for the variant.
Verify exact prompts, once-per-patient sharing, final-model slots, saved artifacts,
independent failures, interruptions, and automatic reporting. Preserve existing
pipeline regressions.

## Only when shared helpers change: `tests/test_med_copilot_helpers.py`

Add direct normal, edge, failure, and filesystem cases for every added/changed
public helper. New variants using the existing contract need no helper changes.

## When startup resource requirements change: `tests/test_med_copilot_startup.py`

Verify that required encoders initialize once before FAISS and unused resources
are skipped, including initialization failures.

## When reporting behavior changes: `tests/test_collate_med_copilot_results.py`

Cover saved variant labels, missing/failed outputs, HTML/terminal summaries, paired
comparisons, and historical compatibility. Existing result contracts should need
no collator implementation changes.

## When configuration behavior changes: `experiments/configs/README.md`

Document added settings, their defaults, component/model roles, and sharing rules.

## When the output contract changes: `med_copilot/docs/output_schema.md`

Document new fields, statuses, provenance, and compatibility requirements. A new
variant using existing component/model slots does not itself change the schema.

## When user-facing behavior changes: `README.md`

Update the variant overview and relevant setup/run instructions; keep links to
this guide and the configuration/schema references accurate.

## Verify and run

Use `medcopilot` or an equivalent environment with pandas. Run the new/affected
suites and the complete offline suite; helper/runner changes also require their
direct commands:

```bash
python tests/test_med_copilot_helpers.py
python tests/test_run_med_copilot.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run_med_copilot.py
```

Review the displayed settings/cohort and confirm with `y` or `yes`. Inspect
`run.json` for effective settings/provenance, `results.jsonl` for component captures
and variant/model outputs, and the generated `results_summary.html`. Report live
provider/model-server behavior that offline checks did not verify. Historical
artifacts remain unchanged.
