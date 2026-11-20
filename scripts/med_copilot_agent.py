"""Agentic evidence retrieval for MED-COPILOT.

An LLM agent reads only the patient's S/O notes and decides what to search for. It has up to three
tools: similar-patient search, clinical-guideline search, and finish. The agent never writes the
final evidence: it only *selects* (by reference id) which retrieved patient and which raw guideline
answers go into the prompt, so the output stays comparable with the fixed pipeline.

This module has no project imports so it can be tested offline with a fake client. The runner builds
the tool callables (wrapping HybridRetriever/CrossEncoderReranker and GraphRAG run_local_search).
Exceptions raised by tools or by the client propagate on purpose: an infrastructure failure must fail
the component, not silently produce an empty evidence set.
"""
import json

PATIENT_TOOL = "search_similar_patients"
GUIDELINE_TOOL = "search_guidelines"
FINISH_TOOL = "finish"
_ARG_NAME = {PATIENT_TOOL: "query", GUIDELINE_TOOL: "question"}
_TRACE_OUTPUT_LIMIT = 1500      # characters kept in the saved trace per tool call


class RetrievalAgentError(RuntimeError):
    """An unfinished agent run, including its partial trace for the saved capture."""

    def __init__(self, message, outcome):
        super().__init__(message)
        self.outcome = outcome


def _tool_specs(use_patients, use_guidelines, guideline_byte_budget=None):
    specs = []
    if use_patients:
        specs.append({"type": "function", "function": {
            "name": PATIENT_TOOL,
            "description": "Search past patients similar to a description of the presenting symptoms, "
                           "findings and risk factors. Returns reference ids like [P1] with their notes.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"]}}})
    if use_guidelines:
        specs.append({"type": "function", "function": {
            "name": GUIDELINE_TOOL,
            "description": "Ask ONE focused, self-contained clinical-guideline question. GraphRAG sees only "
                           "this question, not the patient notes or previous answers. Include relevant "
                           "target-patient findings and measurement values; spell out follow-up references. "
                           "Do not attribute reference-patient facts to the target patient. "
                           "Returns a reference id like [G1] and the answer.",
            "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                           "required": ["question"]}}})
    props = {}
    if use_patients:
        props["patient_ref"] = {"type": "string", "minLength": 1,
                                "description": "ID of the single closest patient to the original S/O, "
                                               "chosen across ALL patient searches, e.g. P2"}
    if use_guidelines:
        props["guideline_refs"] = {"type": "array", "minItems": 1, "uniqueItems": True,
                                   "items": {"type": "string", "minLength": 1},
                                   "description": "IDs of the most relevant complete answers across ALL "
                                                  "guideline searches that fit the guideline byte budget, "
                                                  "e.g. [\"G1\", \"G2\"]"}
        if guideline_byte_budget is not None:
            props["guideline_refs"]["description"] += f" Maximum {guideline_byte_budget} UTF-8 bytes including separators."
    specs.append({"type": "function", "function": {
        "name": FINISH_TOOL, "description": "Stop searching and return the evidence to use.",
        "parameters": {"type": "object", "properties": props, "required": list(props),
                       "additionalProperties": False}}})
    return specs


def build_system_prompt(max_searches, use_patients, use_guidelines, allow_diagnosis_terms=False,
                        guideline_byte_budget=None):
    parts = ["You gather supporting evidence for a clinician. You only see the patient's subjective and "
             "objective notes; you do not know the diagnosis or plan."]
    if use_patients:
        parts.append("Use search_similar_patients to find comparable past patients; reword the query to "
                     "emphasise the most discriminating symptoms and findings. Select the single most "
                     "similar patient to the original S/O across ALL searches, not just the latest results.")
    if use_guidelines:
        parts.append("Use search_guidelines with focused questions, one topic per question. Select the most "
                     "relevant answers across ALL earlier and later searches; selected "
                     "answers are passed in full to the final clinician model, without summarization.")
        parts.append("Each guideline search is independent: GraphRAG receives only your question, not the "
                     "patient notes or conversation history. Make every question self-contained with the "
                     "relevant target-patient findings and measurement values from the original S/O. "
                     "Restate what follow-up references mean; do not rely on phrases such as 'this patient' "
                     "or 'that finding' without the actual facts. Never present reference-patient facts "
                     "as facts about the target patient.")
        if guideline_byte_budget is not None:
            parts.append(f"Selected guideline content must total at most {guideline_byte_budget} UTF-8 bytes. "
                         "Tool replies report each answer's evidence_bytes; add 2 bytes between selected answers "
                         "for the blank-line separator. Metadata labels are not evidence. Omit lower-priority "
                         "answers to fit; never shorten an answer or omit all guideline evidence.")
    if not allow_diagnosis_terms:
        parts.append("Describe symptoms, findings and risk factors in your searches; do not name specific diagnoses.")
    parts.append(f"You may make at most {max_searches} searches in total. When you have enough, call finish "
                 "with the references worth keeping. You must call finish to complete the task; a text "
                 "reply does not finish it. Call exactly ONE tool per response and read its result before "
                 "deciding the next call. Search each enabled source once before repeating searches. Finish must select "
                 "one patient when patient search is enabled and at least one guideline when guideline "
                 "search is enabled. Choose only evidence that is relevant to this patient.")
    return " ".join(parts)


def _usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    return {"prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None)}


def _assistant_dict(msg):
    d = {"role": "assistant", "content": msg.content if msg.tool_calls else msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in msg.tool_calls]
    return d


def _finish_arguments(args, use_patients, use_guidelines):
    allowed = ({"patient_ref"} if use_patients else set()) | (
        {"guideline_refs"} if use_guidelines else set())
    if args is None or set(args) - allowed:
        raise ValueError("Finish accepts only reference fields for the enabled tools.")
    ref = args.get("patient_ref")
    refs = args.get("guideline_refs", [])
    if use_patients and (not isinstance(ref, str) or not ref.strip()):
        raise ValueError("patient_ref must select one nonempty returned patient ID.")
    if (not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs)
            or len(set(refs)) != len(refs) or use_guidelines and not refs):
        raise ValueError("guideline_refs must select at least one distinct, nonempty returned guideline ID.")
    return ref, refs


def run_retrieval_agent(client, s_o, tools, *, model, max_searches=4, temperature=0.0,
                        use_patients=True, use_guidelines=True, allow_diagnosis_terms=False,
                        on_response=None, validate_selection=None, on_diagnostic=None,
                        guideline_byte_budget=None, validate_guideline_question=None):
    """Run the agent loop.

    tools: {"search_similar_patients": f(query)->str, "search_guidelines": f(question)->str}; each
           returns the text shown to the agent and registers its own reference ids ([P1], [G1], ...).
    Returns {"patient_ref", "guideline_refs", "searches", "finished", "stop_reason", "trace", "usage"}
    only after a valid finish call. Exhausting the turn limit raises RetrievalAgentError.
    on_response persists each provider response before any parsing or tool execution.
    validate_selection(args, searches_remaining) checks the caller's evidence registry;
    args=None checks availability only. ValueError is correctable; other exceptions are not.
    on_diagnostic(event, outcome) persists errors/recovery without changing the legacy trace.
    validate_guideline_question(question) runs before spending a search; ValueError
    shares two search-phase corrections. Finish has one separate correction after
    mandatory searches. Allowances persist across successful calls within this invocation.
    All replies, including invalid ones, count toward max_searches + 4.
    """
    if not (use_patients or use_guidelines):
        raise ValueError("agent needs at least one search tool")
    if type(max_searches) is not int or max_searches < 1:
        raise ValueError("max_searches must be a positive integer")
    if guideline_byte_budget is not None and (type(guideline_byte_budget) is not int or guideline_byte_budget < 1):
        raise ValueError("guideline_byte_budget must be a positive integer")
    enabled = {n for n, on in ((PATIENT_TOOL, use_patients), (GUIDELINE_TOOL, use_guidelines)) if on}
    if max_searches < len(enabled):
        raise ValueError("max_searches must allow at least one search of every enabled source")
    missing = enabled - set(tools)
    if missing:
        raise ValueError(f"missing tool callables: {sorted(missing)}")
    specs = _tool_specs(use_patients, use_guidelines, guideline_byte_budget)
    messages = [{"role": "system", "content": build_system_prompt(max_searches, use_patients, use_guidelines,
                                                                  allow_diagnosis_terms, guideline_byte_budget)},
                {"role": "user", "content": s_o}]
    out = {"patient_ref": None, "guideline_refs": [], "searches": 0, "finished": False,
           "stop_reason": None, "trace": [], "usage": []}
    searched, pending_errors = set(), []
    corrections_used, correction_limits = {"search": 0, "finish": 0}, {"search": 2, "finish": 1}
    cap = max_searches + 4

    def diagnose(code, message, substage, action, error_type="ValueError", error_phase=None):
        if on_diagnostic is not None:
            on_diagnostic({"turn": turn, "searches": out["searches"], "substage": substage,
                           "code": code, "error_type": error_type, "message": message,
                           "action": action, "correction_used": any(corrections_used.values()),
                           "phase": error_phase or phase, "corrections_used": dict(corrections_used),
                           "response_id": getattr(resp, "id", None),
                           "calls": [{"tool": c.function.name, "call_id": c.id,
                                      "arguments": c.function.arguments} for c in calls]}, out)

    for turn in range(1, cap + 1):
        required = next((n for n in (PATIENT_TOOL, GUIDELINE_TOOL) if n in enabled - searched), None)
        forced = FINISH_TOOL if out["searches"] >= max_searches else required
        choice = {"type": "function", "function": {"name": forced}} if forced else "required"
        phase = pending_errors[-1][1] if pending_errors else "search"
        if forced == FINISH_TOOL:
            phase = "finish"
        resp, calls = None, []
        try:
            resp = client.chat.completions.create(model=model, messages=messages, tools=specs,
                                                  tool_choice=choice, temperature=temperature, parallel_tool_calls=False)
        except Exception as exc:
            out["stop_reason"] = "api_error"
            diagnose("api_error", str(exc), "agent_request", "exit", type(exc).__name__)
            raise
        if on_response is not None:
            on_response(resp)
        out["usage"].append(_usage(resp))
        try:
            response_choice = resp.choices[0]
            msg = response_choice.message
            candidate_calls = [] if msg.tool_calls is None else msg.tool_calls
            # Only replay structurally valid messages. Bad JSON inside an argument
            # string is correctable, but missing/duplicate call IDs cannot be replayed safely.
            if (msg.content is not None and not isinstance(msg.content, str)
                    or not isinstance(candidate_calls, list)
                    or any(not isinstance(c.id, str) or not c.id
                           or not isinstance(c.function.name, str)
                           or not isinstance(c.function.arguments, str) for c in candidate_calls)
                    or len({c.id for c in candidate_calls}) != len(candidate_calls)):
                raise ValueError("Malformed agent response or tool-call structure")
            calls = candidate_calls
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            out["stop_reason"] = "malformed_response"
            diagnose("malformed_response", str(exc), "response_validation", "exit", type(exc).__name__)
            raise
        if getattr(msg, "refusal", None) or getattr(response_choice, "finish_reason", None) == "content_filter":
            out["stop_reason"] = "refusal"
            message = getattr(msg, "refusal", None) or "Agent response blocked by content filtering"
            diagnose("refusal", message, "response_validation", "exit", "RetrievalAgentError")
            raise RetrievalAgentError(message, out)
        messages.append(_assistant_dict(msg))
        if any(c.function.name in _ARG_NAME for c in calls):
            phase = "search"  # Search errors never consume the reserved finish correction.
        elif calls and all(c.function.name == FINISH_TOOL for c in calls):
            phase = "search" if enabled - searched else "finish"
        code, substage = "invalid_response", "response_validation"
        try:
            if getattr(response_choice, "finish_reason", None) == "length":
                code = "truncated_response"
                raise ValueError("Response was truncated. Return one complete, concise tool call.")
            if len(calls) != 1:
                code = "multiple_tool_calls" if calls else "missing_tool_call"
                raise ValueError("Call exactly ONE tool, then read its result before the next call."
                                 if calls else "A text reply does not complete selection. Call one enabled tool.")
            call = calls[0]
            name, substage, code = call.function.name, "argument_validation", "invalid_arguments"
            try:
                args = json.loads(call.function.arguments)
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = None
            if name == FINISH_TOOL:
                code, substage = "invalid_finish", "finish_validation"
                if enabled - searched:
                    raise ValueError("Search every enabled source and read its results before finish.")
                if validate_selection is not None:
                    validate_selection(args, max_searches - out["searches"])
                ref, refs = _finish_arguments(args, use_patients, use_guidelines)
            elif name not in enabled:
                code = "unknown_tool"
                raise ValueError(f"Unknown or disabled tool: {name}. Call an enabled tool.")
            elif args is None or not isinstance(args.get(_ARG_NAME[name]), str) or not args[_ARG_NAME[name]].strip():
                raise ValueError(f"{name} needs a non-empty '{_ARG_NAME[name]}' string.")
            elif out["searches"] >= max_searches:
                code = "search_budget_exhausted"
                raise ValueError("Search budget exhausted. Call finish with returned references.")
            elif required is not None and name != required:
                code = "required_source"
                raise ValueError(f"Call {required} first so every required source fits in the search budget.")
            elif name == GUIDELINE_TOOL and validate_guideline_question is not None:
                code, substage = "invalid_guideline_question", "query_validation"
                validate_guideline_question(args["question"].strip())
        except ValueError as exc:
            retry = corrections_used[phase] < correction_limits[phase] and turn < cap
            if not retry:
                out["stop_reason"] = "turn_cap" if turn == cap else "validation_error"
            diagnose(code, str(exc), substage, "retry" if retry else "exit", type(exc).__name__)
            if not retry:
                raise RetrievalAgentError(str(exc), out) from exc
            corrections_used[phase] += 1
            pending_errors.append((code, phase))
            feedback = (f"Error: {exc} Retry this {phase}-phase decision. Corrections remaining: "
                        f"search={correction_limits['search'] - corrections_used['search']}, "
                        f"finish={correction_limits['finish'] - corrections_used['finish']}.")
            if calls:
                messages.extend({"role": "tool", "tool_call_id": c.id, "content": feedback} for c in calls)
            else:
                messages.append({"role": "user", "content": feedback})
            continue
        except Exception as exc:
            out["stop_reason"] = "evidence_error"
            diagnose("evidence_error", str(exc), substage, "exit", type(exc).__name__)
            raise

        if name == FINISH_TOOL:
            out["patient_ref"], out["guideline_refs"] = ref, refs
            out["finished"], out["stop_reason"], reply = True, "finish", "ok"
        else:
            try:
                out["searches"] += 1
                text = args[_ARG_NAME[name]].strip()
                # The agent selects the full evidence used in the final prompt.
                # Truncation would hide candidates or part of a guideline answer.
                reply = str(tools[name](text))   # tool errors propagate
                out["trace"].append({"search": out["searches"], "tool": name, "input": text,
                                     "output": reply[:_TRACE_OUTPUT_LIMIT]})
                searched.add(name)
            except Exception as exc:
                out["stop_reason"] = "tool_error"
                diagnose("tool_error", str(exc), "tool_execution", "exit", type(exc).__name__)
                raise
            if out["searches"] == max_searches and validate_selection is not None:
                try:
                    validate_selection(None, 0)
                except Exception as exc:
                    out["stop_reason"] = "evidence_error"
                    diagnose("evidence_error", str(exc), "evidence_availability", "exit", type(exc).__name__)
                    raise
        messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
        for code, error_phase in pending_errors:
            diagnose(code, "Agent correction accepted", "response_validation", "recovered", error_phase=error_phase)
        pending_errors.clear()
        if out["finished"]:
            break
    else:
        out["stop_reason"] = "turn_cap"
        diagnose("turn_cap", "Agent response cap exhausted", "response_validation", "exit", "RetrievalAgentError")
        raise RetrievalAgentError("Retrieval agent did not call a valid finish before the turn limit", out)
    return out
