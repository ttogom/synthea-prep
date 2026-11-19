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


def _tool_specs(use_patients, use_guidelines):
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
            "description": "Ask ONE focused clinical-guideline question about the patient's symptoms, "
                           "findings or risk factors. Returns a reference id like [G1] and the answer.",
            "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                           "required": ["question"]}}})
    props = {}
    if use_patients:
        props["patient_ref"] = {"type": "string", "description": "id of the single most relevant patient, e.g. P2"}
    if use_guidelines:
        props["guideline_refs"] = {"type": "array", "items": {"type": "string"},
                                   "description": "ids of guideline answers worth keeping, e.g. [\"G1\"]"}
    specs.append({"type": "function", "function": {
        "name": FINISH_TOOL, "description": "Stop searching and return the evidence to use.",
        "parameters": {"type": "object", "properties": props, "required": []}}})
    return specs


def build_system_prompt(max_searches, use_patients, use_guidelines, allow_diagnosis_terms=False):
    parts = ["You gather supporting evidence for a clinician. You only see the patient's subjective and "
             "objective notes; you do not know the diagnosis or plan."]
    if use_patients:
        parts.append("Use search_similar_patients to find comparable past patients; reword the query to "
                     "emphasise the most discriminating symptoms and findings.")
    if use_guidelines:
        parts.append("Use search_guidelines with focused questions, one topic per question.")
    if not allow_diagnosis_terms:
        parts.append("Describe symptoms, findings and risk factors in your searches; do not name specific diagnoses.")
    parts.append(f"You may make at most {max_searches} searches in total. When you have enough, call finish "
                 "with the references worth keeping. You must call finish to complete the task; a text "
                 "reply does not finish it. Choose only evidence that is relevant to this patient.")
    return " ".join(parts)


def _usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    return {"prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None)}


def _assistant_dict(msg):
    d = {"role": "assistant", "content": msg.content}
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
    if "patient_ref" in args and (not isinstance(ref, str) or not ref.strip()):
        raise ValueError("patient_ref must be a nonempty reference ID, or omitted.")
    if (not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs)
            or len(set(refs)) != len(refs)):
        raise ValueError("guideline_refs must be a list of distinct, nonempty reference IDs.")
    return ref, refs


def run_retrieval_agent(client, s_o, tools, *, model, max_searches=4, temperature=0.0,
                        use_patients=True, use_guidelines=True, allow_diagnosis_terms=False):
    """Run the agent loop.

    tools: {"search_similar_patients": f(query)->str, "search_guidelines": f(question)->str}; each
           returns the text shown to the agent and registers its own reference ids ([P1], [G1], ...).
    Returns {"patient_ref", "guideline_refs", "searches", "finished", "stop_reason", "trace", "usage"}
    only after a valid finish call. Exhausting the turn limit raises RetrievalAgentError.
    """
    if not (use_patients or use_guidelines):
        raise ValueError("agent needs at least one search tool")
    if type(max_searches) is not int or max_searches < 1:
        raise ValueError("max_searches must be a positive integer")
    enabled = {n for n, on in ((PATIENT_TOOL, use_patients), (GUIDELINE_TOOL, use_guidelines)) if on}
    missing = enabled - set(tools)
    if missing:
        raise ValueError(f"missing tool callables: {sorted(missing)}")
    specs = _tool_specs(use_patients, use_guidelines)
    messages = [{"role": "system", "content": build_system_prompt(max_searches, use_patients, use_guidelines,
                                                                  allow_diagnosis_terms)},
                {"role": "user", "content": s_o}]
    out = {"patient_ref": None, "guideline_refs": [], "searches": 0, "finished": False,
           "stop_reason": None, "trace": [], "usage": []}
    finish_requested = False
    for _ in range(2 * max_searches + 3):               # hard cap on turns
        force_finish = finish_requested or out["searches"] >= max_searches
        choice = {"type": "function", "function": {"name": FINISH_TOOL}} if force_finish else "required"
        resp = client.chat.completions.create(model=model, messages=messages, tools=specs,
                                              tool_choice=choice, temperature=temperature)
        out["usage"].append(_usage(resp))
        msg = resp.choices[0].message
        messages.append(_assistant_dict(msg))
        calls = msg.tool_calls or []
        if not calls:
            finish_requested = True
            messages.append({"role": "user", "content": "Call finish with the references you selected. "
                             "A text reply does not complete evidence selection."})
            continue
        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = None
            if name == FINISH_TOOL:
                finish_requested = True
                try:
                    ref, refs = _finish_arguments(args, use_patients, use_guidelines)
                except ValueError as exc:
                    reply = str(exc)
                else:
                    out["patient_ref"], out["guideline_refs"] = ref, refs
                    out["finished"], out["stop_reason"], reply = True, "finish", "ok"
            elif name not in enabled:
                reply = f"Unknown or disabled tool: {name}"
            elif args is None or not isinstance(args.get(_ARG_NAME[name]), str) or not args[_ARG_NAME[name]].strip():
                reply = f"{name} needs a non-empty '{_ARG_NAME[name]}' string."
            elif out["searches"] >= max_searches:
                reply = "Search budget exhausted. Call finish."
            else:
                out["searches"] += 1
                text = args[_ARG_NAME[name]].strip()
                # The agent selects the full evidence used in the final prompt.
                # Truncation would hide candidates or part of a guideline answer.
                reply = str(tools[name](text))   # tool errors propagate
                out["trace"].append({"search": out["searches"], "tool": name, "input": text,
                                     "output": reply[:_TRACE_OUTPUT_LIMIT]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
            if out["finished"]:
                break
        if out["finished"]:
            break
    else:
        out["stop_reason"] = "turn_cap"
        raise RetrievalAgentError("Retrieval agent did not call a valid finish before the turn limit", out)
    return out
