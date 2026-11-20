"""Offline tests for scripts/med_copilot_agent.py (no network): python tests/test_med_copilot_agent.py"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import httpx
from openai import OpenAI

# Direct script execution must resolve the project before importing its agent.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.med_copilot_agent import RetrievalAgentError, run_retrieval_agent


class _Fn:
    def __init__(self, name, args): self.name, self.arguments = name, args if isinstance(args, str) else json.dumps(args)
class _Call:
    def __init__(self, i, name, args): self.id, self.function = f"c{i}", _Fn(name, args)
class _Msg:
    def __init__(self, calls=None, content=None): self.tool_calls, self.content = calls, content
class _Choice:
    def __init__(self, msg): self.message = msg
class _Resp:
    def __init__(self, msg): self.choices, self.usage = [_Choice(msg)], None
class FakeClient:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = type("C", (), {"completions": self})()
    def create(self, **kw):
        self.requests.append(deepcopy(kw))
        return _Resp(self.script.pop(0))

def tools():
    seen = {"p": 0, "g": 0}
    def patients(q): seen["p"] += 1; return f"[P{seen['p']}] note for {q}"
    def guidelines(q): seen["g"] += 1; return f"[G{seen['g']}] answer for {q}"
    return {"search_similar_patients": patients, "search_guidelines": guidelines}

def msg(*calls): return _Msg([_Call(i, n, a) for i, (n, a) in enumerate(calls)])

def test_happy_path():
    c = FakeClient([msg(("search_similar_patients", {"query": "chest pain"})),
                    msg(("search_guidelines", {"question": "chest pain workup"})),
                    msg(("finish", {"patient_ref": "P1", "guideline_refs": ["G1"]}))])
    r = run_retrieval_agent(c, "S/O text", tools(), model="m", max_searches=4)
    assert r["finished"] and r["stop_reason"] == "finish" and r["searches"] == 2
    assert r["patient_ref"] == "P1" and r["guideline_refs"] == ["G1"]
    assert [t["tool"] for t in r["trace"]] == ["search_similar_patients", "search_guidelines"]
    assert c.requests[0]["messages"][1] == {"role": "user", "content": "S/O text"}
    assert c.requests[0]["tool_choice"] == "required"

def test_budget_forces_finish():
    c = FakeClient([msg(("search_similar_patients", {"query": "a"})),
                    msg(("finish", {"patient_ref": "P1"}))])
    r = run_retrieval_agent(c, "x", tools(), model="m", max_searches=1)
    assert c.requests[1]["tool_choice"]["function"]["name"] == "finish"
    assert r["finished"] and r["searches"] == 1

def test_extra_search_beyond_budget_is_refused():
    c = FakeClient([msg(("search_similar_patients", {"query": "a"})),
                    msg(("search_similar_patients", {"query": "b"})),
                    msg(("finish", {}))])
    r = run_retrieval_agent(c, "x", tools(), model="m", max_searches=1)
    assert r["searches"] == 1
    assert c.requests[1]["tool_choice"]["function"]["name"] == "finish"
    assert any("budget" in m["content"] for m in c.requests[2]["messages"] if m["role"] == "tool")

def test_bad_arguments_and_disabled_tool():
    c = FakeClient([msg(("search_similar_patients", "{not json")),
                    msg(("search_guidelines", {"question": "q"})),
                    msg(("finish", {}))])
    r = run_retrieval_agent(c, "x", tools(), model="m", use_guidelines=False)
    assert r["searches"] == 0 and r["finished"] and r["patient_ref"] is None

def test_no_tool_call_requires_an_explicit_finish():
    c = FakeClient([msg(("search_similar_patients", {"query": "a"})),
                    _Msg(None, "done"), msg(("finish", {"patient_ref": "P1"}))])
    r = run_retrieval_agent(c, "x", tools(), model="m")
    assert r["finished"] and r["stop_reason"] == "finish" and r["patient_ref"] == "P1"
    assert len(c.requests) == 3 and r["searches"] == 1
    assert c.requests[2]["tool_choice"]["function"]["name"] == "finish"
    assert "does not complete" in c.requests[2]["messages"][-1]["content"]


def test_turn_cap_fails_and_preserves_partial_trace():
    c = FakeClient([msg(("search_similar_patients", {"query": "a"})),
                    *[_Msg(None, "done") for _ in range(4)]])
    try:
        run_retrieval_agent(c, "x", tools(), model="m", max_searches=1)
    except RetrievalAgentError as exc:
        assert not exc.outcome["finished"] and exc.outcome["stop_reason"] == "turn_cap"
        assert exc.outcome["searches"] == 1 and len(exc.outcome["trace"]) == 1
        assert len(exc.outcome["usage"]) == len(c.requests) == 5
        assert all(r["tool_choice"]["function"]["name"] == "finish" for r in c.requests[1:])
    else:
        raise AssertionError("an unfinished agent was accepted")


def test_all_patient_candidates_reach_the_agent_without_truncation():
    reply = "\n\n".join(f"[P{i}] " + str(i) * 4000 for i in range(1, 4))
    t = tools()
    t["search_similar_patients"] = lambda query: reply
    c = FakeClient([msg(("search_similar_patients", {"query": "a"})),
                    msg(("finish", {"patient_ref": "P3"}))])
    r = run_retrieval_agent(c, "x", t, model="m", use_guidelines=False)
    assert c.requests[1]["messages"][-1]["content"] == reply
    assert r["patient_ref"] == "P3"
    assert len(r["trace"][0]["output"]) == 1500  # saved preview remains bounded


def test_sdk_requests_single_calls_and_supplies_full_evidence_before_finish():
    patient_reply = "\n\n".join(f"[P{i}] " + str(i) * 4000 for i in range(1, 4))
    cases = [
        (True, False, "search_similar_patients", "query", patient_reply, {"patient_ref": "P3"}),
        (True, True, "search_similar_patients", "query", patient_reply, {"patient_ref": "P3"}),
        (False, True, "search_guidelines", "question", "[G1] " + "evidence " * 1000,
         {"guideline_refs": ["G1"]}),
    ]
    for patients, guidelines, tool, argument, evidence, selection in cases:
        for budget in (1, 4):
            requests, events = [], []

            def handle(request):
                body = json.loads(request.content)
                requests.append(body)
                assert body["parallel_tool_calls"] is False
                if len(requests) == 1:
                    assert events == []
                    assert body["tool_choice"] == "required"
                    assert not any(item["role"] == "tool" for item in body["messages"])
                    events.append("request_search")
                    name, args, call_id = tool, {argument: "cough"}, "search-call"
                else:
                    assert len(requests) == 2
                    assert events == ["request_search", "search_result"]
                    assert body["messages"][-1] == {
                        "role": "tool", "tool_call_id": "search-call", "content": evidence}
                    assert body["messages"][-2]["tool_calls"][0]["function"]["name"] == tool
                    expected_choice = ({"type": "function", "function": {"name": "finish"}}
                                       if budget == 1 else "required")
                    assert body["tool_choice"] == expected_choice
                    # Generate the selection only after checking its complete evidence in this request.
                    events.append("request_finish")
                    name, args, call_id = "finish", selection, "finish-call"
                return httpx.Response(200, json={
                    "id": f"offline-{len(requests)}", "object": "chat.completion", "created": 0,
                    "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": None, "tool_calls": [{
                            "id": call_id, "type": "function", "function": {
                                "name": name, "arguments": json.dumps(args)}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                })

            def search(query):
                assert query == "cough"
                assert events == ["request_search"]
                events.append("search_result")
                return evidence

            available_tools = tools()
            available_tools[tool] = search
            with OpenAI(api_key="offline", base_url="https://offline.invalid/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
                outcome = run_retrieval_agent(
                    client, "Subjective: cough\nObjective: recorded findings", available_tools,
                    model="gpt-4o-mini", use_patients=patients, use_guidelines=guidelines,
                    max_searches=budget,
                )
            assert events == ["request_search", "search_result", "request_finish"]
            assert outcome["finished"] and outcome["searches"] == 1
            assert outcome["patient_ref"] == selection.get("patient_ref")
            assert outcome["guideline_refs"] == selection.get("guideline_refs", [])


def test_four_patient_searches_supply_each_result_before_next_decision_and_finish():
    for guidelines in (False, True):
        c = FakeClient([
            *[msg(("search_similar_patients", {"query": f"query-{i}"})) for i in range(4)],
            msg(("finish", {"patient_ref": "P12"})),
        ])
        replies = []

        def search(query):
            i = len(replies)
            assert query == f"query-{i}"
            assert len(c.requests) == i + 1
            if i:
                assert c.requests[-1]["messages"][-1]["content"] == replies[-1]
            reply = "\n\n".join(f"[P{3 * i + j}] {query} candidate {j}" for j in range(1, 4))
            replies.append(reply)
            return reply

        t = tools()
        t["search_similar_patients"] = search
        outcome = run_retrieval_agent(c, "S/O", t, model="m", use_guidelines=guidelines)
        assert outcome["finished"] and outcome["patient_ref"] == "P12" and outcome["searches"] == 4
        assert len(c.requests) == 5
        assert all(request["parallel_tool_calls"] is False for request in c.requests)
        assert c.requests[-1]["tool_choice"]["function"]["name"] == "finish"
        for i, reply in enumerate(replies):
            assert c.requests[i + 1]["messages"][-1] == {
                "role": "tool", "tool_call_id": "c0", "content": reply}


def test_full_guideline_answer_reaches_the_agent():
    reply = "answer" * 2000
    t = tools()
    t["search_guidelines"] = lambda query: reply
    c = FakeClient([msg(("search_guidelines", {"question": "q"})),
                    msg(("finish", {"guideline_refs": ["G1"]}))])
    r = run_retrieval_agent(c, "x", t, model="m", use_patients=False)
    assert c.requests[1]["messages"][-1]["content"] == reply
    assert r["guideline_refs"] == ["G1"]
    assert len(r["trace"][0]["output"]) == 1500


def test_invalid_finish_arguments_are_rejected_until_corrected():
    invalid = ["{not json", "[]", {"patient_ref": 3}, {"patient_ref": ""},
               {"guideline_refs": "G1"}, {"guideline_refs": [1]},
               {"guideline_refs": ["G1", "G1"]}, {"guideline_refs": [""]},
               {"guideline_refs": None}, {"unexpected": "P1"}]
    for args in invalid:
        c = FakeClient([msg(("finish", args)), msg(("finish", {}))])
        r = run_retrieval_agent(c, "x", tools(), model="m")
        assert len(c.requests) == 2 and r["finished"]
        assert c.requests[1]["messages"][-1]["content"] != "ok"
        assert c.requests[1]["tool_choice"]["function"]["name"] == "finish"


def test_finish_cannot_select_a_disabled_source():
    for patient_enabled, guideline_enabled, args in [
            (True, False, {"guideline_refs": ["G1"]}),
            (False, True, {"patient_ref": "P1"})]:
        c = FakeClient([msg(("finish", args)), msg(("finish", {}))])
        r = run_retrieval_agent(c, "x", tools(), model="m", use_patients=patient_enabled,
                                use_guidelines=guideline_enabled)
        assert len(c.requests) == 2 and r["finished"]
        assert r["patient_ref"] is None and r["guideline_refs"] == []


def test_invalid_search_budget_fails_before_any_request():
    for budget in [0, -1, True, 1.5, None]:
        c = FakeClient([])
        try:
            run_retrieval_agent(c, "x", tools(), model="m", max_searches=budget)
        except ValueError:
            assert c.requests == []
        else:
            raise AssertionError("invalid search budget was accepted")

def test_tool_errors_propagate():
    def boom(q): raise RuntimeError("quota")
    t = tools(); t["search_similar_patients"] = boom
    c = FakeClient([msg(("search_similar_patients", {"query": "a"}))])
    try:
        run_retrieval_agent(c, "x", t, model="m")
    except RuntimeError:
        return
    raise AssertionError("tool exception was swallowed")


def test_response_callback_runs_before_tools_and_includes_the_finish_response():
    events = []
    c = FakeClient([msg(("search_similar_patients", {"query": "a"})), msg(("finish", {}))])
    t = tools()
    t["search_similar_patients"] = lambda query: events.append("tool") or "[P1] note"
    run_retrieval_agent(c, "x", t, model="m", use_guidelines=False,
                        on_response=lambda response: events.append("response"))
    assert events == ["response", "tool", "response"]


def test_response_callback_saves_malformed_replies_before_parsing():
    c = FakeClient([_Msg()])
    received = []
    original = c.create

    def create(**kwargs):
        response = original(**kwargs)
        response.choices = []
        return response

    c.chat.completions.create = create
    try:
        run_retrieval_agent(c, "x", tools(), model="m", on_response=received.append)
    except IndexError:
        assert len(received) == len(c.requests) == 1
    else:
        raise AssertionError("malformed response was accepted")


def test_usage_persistence_failure_prevents_further_paid_work():
    c = FakeClient([msg(("search_similar_patients", {"query": "a"}))])
    executed = []
    t = tools()
    t["search_similar_patients"] = executed.append

    def save(response):
        raise OSError("disk full")

    try:
        run_retrieval_agent(c, "x", t, model="m", on_response=save)
    except OSError:
        assert executed == [] and len(c.requests) == 1
    else:
        raise AssertionError("usage persistence failure was ignored")

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
