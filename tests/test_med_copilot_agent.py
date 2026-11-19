"""Offline tests for scripts/med_copilot_agent.py (no network): python tests/test_med_copilot_agent.py"""
from copy import deepcopy
import json
from pathlib import Path
import sys

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
    c = FakeClient([msg(("search_similar_patients", {"query": "a"}), ("search_similar_patients", {"query": "b"})),
                    msg(("finish", {}))])
    r = run_retrieval_agent(c, "x", tools(), model="m", max_searches=1)
    assert r["searches"] == 1
    assert any("budget" in m["content"] for m in c.requests[1]["messages"] if m["role"] == "tool")

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







if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
