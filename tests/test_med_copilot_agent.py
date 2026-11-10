"""Offline tests for scripts/med_copilot_agent.py (no network): python tests/test_med_copilot_agent.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.med_copilot_agent import run_retrieval_agent


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
        self.requests.append(kw)
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
    assert c.requests[0]["tool_choice"] == "auto"

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

def test_no_tool_call_stops():
    r = run_retrieval_agent(FakeClient([_Msg(None, "done")]), "x", tools(), model="m")
    assert r["stop_reason"] == "no_tool_call" and not r["finished"]

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