"""A component prepares reusable prompt inputs; a variant selects inputs and a prompt.

Each component/configuration runs once per patient, shared across variants/models.
Importing these definitions loads no model runtime.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class VariantSpec:
    """Select components and bind their outputs into an existing prompt template."""

    description: str  # Label shown in run confirmation and results reports.
    components: tuple[str, ...]  # Required IDs from COMPONENTS; () needs no preparation.
    prompt_key: str  # Key in diagnosis_plan.VARIANT_TEMPLATES for the final model prompt.
    # Prompt field -> (component ID, output key). Target S/O fills `conditions` separately.
    bindings: dict[str, tuple[str, str]] = field(default_factory=dict)


# Keep VARIANTS here at the top, immediately after its required type definition.
VARIANTS = {
    "llm_only": VariantSpec(
        description="Patient S/O only",
        components=(),
        prompt_key="llm_only",
    ),
    "patient_retrieval": VariantSpec(
        description="S/O + similar-patient example",
        components=("patient_retrieval",),
        prompt_key="patient_retrieval",
        bindings={"example": ("patient_retrieval", "example")},
    ),
    "guidelines": VariantSpec(
        description="S/O + clinical guidelines",
        components=("guidelines",),
        prompt_key="guidelines",
        bindings={"Key_info": ("guidelines", "Key_info")},
    ),
    "patient_retrieval_guidelines": VariantSpec(
        description="S/O + both evidence sources",
        components=("patient_retrieval", "guidelines"),
        prompt_key="patient_retrieval_guidelines",
        bindings={"example": ("patient_retrieval", "example"), "Key_info": ("guidelines", "Key_info")},
    ),
    "agentic_patient_retrieval": VariantSpec(
        description="S/O + agent-selected similar-patient example",
        components=("agent_patients",),
        prompt_key="patient_retrieval",
        bindings={"example": ("agent_patients", "example")},
    ),
    "agentic_guidelines": VariantSpec(
        description="S/O + agent-selected clinical guidelines",
        components=("agent_guidelines",),
        prompt_key="guidelines",
        bindings={"Key_info": ("agent_guidelines", "Key_info")},
    ),
    "agentic_both": VariantSpec(
        description="S/O + agent-selected patient and guideline evidence",
        components=("agent_both",),
        prompt_key="patient_retrieval_guidelines",
        bindings={"example": ("agent_both", "example"), "Key_info": ("agent_both", "Key_info")},
    ),
}


@dataclass(frozen=True)
class Parameter:
    """A component parameter, optionally defaulting to a shared experiment field."""

    kind: str  # Allowed type: nonblank str, bool, int, or finite number (int/float).
    default: object = None  # Fallback when YAML omits this value and config_field is None.
    config_field: str | None = None  # ExperimentConfig attribute supplying the default instead.
    minimum: float | None = None  # Inclusive int/number lower bound; None means unbounded.
    maximum: float | None = None  # Inclusive int/number upper bound; None means unbounded.

    def validate(self, value, name):
        valid = {
            "str": isinstance(value, str) and bool(value.strip()) if isinstance(value, str) else False,
            "bool": type(value) is bool,
            "int": type(value) is int,
            "number": type(value) in (int, float),
        }
        if self.kind not in valid or not valid[self.kind]:
            raise ValueError(f"{name} must be {self.kind}")
        if self.kind in ("int", "number"):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if (not finite or self.minimum is not None and value < self.minimum
                    or self.maximum is not None and value > self.maximum):
                raise ValueError(f"{name} is outside its allowed finite range")
        return float(value) if self.kind == "number" else value


@dataclass(frozen=True)
class ComponentSpec:
    """One reusable preparation step and the string outputs it publishes."""

    description: str  # Human-readable step label saved in run metadata.
    outputs: tuple[str, ...]  # String output keys the step must publish for prompt bindings.
    parameters: dict[str, Parameter] = field(default_factory=dict)  # Allowed YAML overrides and defaults.
    resources: tuple[str, ...] = ()  # Shared resource IDs the runner must initialize for this step.
    credentials: tuple[str, ...] = ()  # Required environment-variable names, never token values.
    implementation: str = "scripts/run_med_copilot.py"  # Source file fingerprinted, not auto-imported.


# Shared parameter sets for the three agent components below.
AGENT_PARAMETERS = {
    # OpenAI chat model that drives the agent's tool calls (must support function calling).
    "agent_model": Parameter("str", default="gpt-4o-mini"),
    # Maximum searches (similar-patient plus guideline) before the agent must finish.
    "max_searches": Parameter("int", default=4, minimum=1, maximum=10),
    # Agent sampling temperature; 0 keeps runs as repeatable as the provider allows.
    "temperature": Parameter("number", default=0.0, minimum=0, maximum=2),
    # False applies the fixed pipeline's diagnosis-term screen to guideline questions.
    "allow_diagnosis_terms": Parameter("bool", default=False),
}
PATIENT_SEARCH_PARAMETERS = {
    "alpha": Parameter("number", config_field="patient_retrieval_alpha", minimum=0, maximum=1),
    "top_k": Parameter("int", config_field="patient_retrieval_top_k", minimum=1),
    # Reranked patients shown to the agent per search; the agent picks one for the prompt.
    "results_per_search": Parameter("int", default=3, minimum=1, maximum=10),
}
GUIDELINE_SEARCH_PARAMETERS = {
    "community_level": Parameter("int", config_field="guideline_community_level", minimum=0),
    "response_type": Parameter("str", config_field="guideline_response_type"),
    "verbose": Parameter("bool", config_field="guideline_verbose"),
}


COMPONENTS = {
    "patient_retrieval": ComponentSpec(
        description="Similar-patient retrieval and reranking",
        outputs=("example",),
        parameters={
            # Initial ranking: alpha * cosine similarity + (1-alpha) * keyword overlap.
            "alpha": Parameter("number", config_field="patient_retrieval_alpha", minimum=0, maximum=1),
            # Candidates sent to reranking; initial embedding search fetches up to 8x this.
            "top_k": Parameter("int", config_field="patient_retrieval_top_k", minimum=1),
            # Reranked cases retained; only the first enters the final prompt.
            "rerank_top_k": Parameter("int", config_field="patient_rerank_top_k", minimum=1),
        },
        resources=("patient_retrieval",), credentials=("OPENAI_API_KEY",),
    ),
    "guidelines": ComponentSpec(
        description="Clinical guideline questions and GraphRAG search",
        outputs=("Key_info",),
        parameters={
            # OpenAI model generating guideline-search questions from target S/O.
            "question_model": Parameter("str", config_field="guideline_question_model"),
            # GraphRAG community hierarchy cap: use indexed levels <= this value.
            "community_level": Parameter("int", config_field="guideline_community_level", minimum=0),
            # Free-text format requested for the intermediate guideline answer.
            "response_type": Parameter("str", config_field="guideline_response_type"),
            # Request extra GraphRAG diagnostics; runner prints are unaffected.
            "verbose": Parameter("bool", config_field="guideline_verbose"),
        },
        resources=("guidelines",), credentials=("OPENAI_API_KEY", "GRAPHRAG_API_KEY"),
    ),
    "agent_patients": ComponentSpec(
        description="Agent-driven similar-patient search",
        outputs=("example",),
        parameters={**AGENT_PARAMETERS, **PATIENT_SEARCH_PARAMETERS},
        resources=("patient_retrieval",), credentials=("OPENAI_API_KEY",),
        implementation="scripts/med_copilot_agent.py",
    ),
    "agent_guidelines": ComponentSpec(
        description="Agent-driven clinical guideline search",
        outputs=("Key_info",),
        parameters={**AGENT_PARAMETERS, **GUIDELINE_SEARCH_PARAMETERS},
        resources=("guidelines",), credentials=("OPENAI_API_KEY", "GRAPHRAG_API_KEY"),
        implementation="scripts/med_copilot_agent.py",
    ),
    "agent_both": ComponentSpec(
        description="Agent-driven patient and guideline search",
        outputs=("example", "Key_info"),
        parameters={**AGENT_PARAMETERS, **PATIENT_SEARCH_PARAMETERS, **GUIDELINE_SEARCH_PARAMETERS},
        resources=("patient_retrieval", "guidelines"), credentials=("OPENAI_API_KEY", "GRAPHRAG_API_KEY"),
        implementation="scripts/med_copilot_agent.py",
    ),
}
