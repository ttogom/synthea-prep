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
_TOOL_OUTPUT_LIMIT = 3500       # characters shown back to the agent per tool call
_TRACE_OUTPUT_LIMIT = 1500      # characters kept in the saved trace per tool call


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
                 "with the references worth keeping. Choose only evidence that is relevant to this patient.")
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


def run_retrieval_agent(client, s_o, tools, *, model, max_searches=4, temperature=0.0,
                        use_patients=True, use_guidelines=True, allow_diagnosis_terms=False):
    """Run the agent loop.

    tools: {"search_similar_patients": f(query)->str, "search_guidelines": f(question)->str}; each
           returns the text shown to the agent and registers its own reference ids ([P1], [G1], ...).
    Returns {"patient_ref", "guideline_refs", "searches", "finished", "stop_reason", "trace", "usage"}.
    """
    if not (use_patients or use_guidelines):
        raise ValueError("agent needs at least one search tool")
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
    for _ in range(2 * max_searches + 3):               # hard cap on turns
        force_finish = out["searches"] >= max_searches
        choice = {"type": "function", "function": {"name": FINISH_TOOL}} if force_finish else "auto"
        resp = client.chat.completions.create(model=model, messages=messages, tools=specs,
                                              tool_choice=choice, temperature=temperature)
        out["usage"].append(_usage(resp))
        msg = resp.choices[0].message
        messages.append(_assistant_dict(msg))
        calls = msg.tool_calls or []
        if not calls:
            out["stop_reason"] = "no_tool_call"
            break
        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args = None
            if name == FINISH_TOOL:
                if args is None:
                    reply = "Invalid arguments for finish."
                else:
                    ref = args.get("patient_ref")
                    refs = args.get("guideline_refs") or []
                    out["patient_ref"] = ref if isinstance(ref, str) else None
                    out["guideline_refs"] = [r for r in refs if isinstance(r, str)]
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
                reply = str(tools[name](text))[:_TOOL_OUTPUT_LIMIT]   # tool errors propagate
                out["trace"].append({"search": out["searches"], "tool": name, "input": text,
                                     "output": reply[:_TRACE_OUTPUT_LIMIT]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
            if out["finished"]:
                break
        if out["finished"]:
            break
    else:
        out["stop_reason"] = "turn_cap"
    return out