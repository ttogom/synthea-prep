"""Offline agent regressions: python tests/test_med_copilot_agent.py."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
from openai import OpenAI

# Direct script execution must resolve the project before importing its agent.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.med_copilot_agent import RetrievalAgentError, run_retrieval_agent


P, G, F = "search_similar_patients", "search_guidelines", "finish"


def msg(*calls, content=None, refusal=None):
    return SimpleNamespace(content=content, refusal=refusal, tool_calls=[
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(
            name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
        for i, (name, args) in enumerate(calls)])


class FakeClient:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if hasattr(reply, "choices"):
            return reply
        return SimpleNamespace(choices=[SimpleNamespace(message=reply, finish_reason="tool_calls")],
                               usage=None, id=f"r{len(self.requests)}")


def tools():
    seen = {P: 0, G: 0}

    def search(name, prefix, query):
        seen[name] += 1
        return f"[{prefix}{seen[name]}] evidence for {query}"

    return {P: lambda q: search(P, "P", q), G: lambda q: search(G, "G", q)}


def run(client, available=None, **kwargs):
    options = {"model": "m", "use_guidelines": False, **kwargs}
    return run_retrieval_agent(client, "Subjective: cough\nObjective: O2 98%", available or tools(), **options)


def fail(client, code, available=None, **kwargs):
    events = []
    try:
        run(client, available, on_diagnostic=lambda e, o: events.append(deepcopy(e)), **kwargs)
    except RetrievalAgentError as exc:
        assert not exc.outcome["finished"]
        assert events[-1]["code"] == code and events[-1]["action"] == "exit"
        return exc.outcome, events
    raise AssertionError("invalid agent output was accepted")


def test_happy_paths_and_required_finish_schema_for_all_variants():
    for patients, guidelines in [(True, False), (False, True), (True, True)]:
        script, selection = [], {}
        if patients:
            script.append(msg((P, {"query": "cough"})))
            selection["patient_ref"] = "P1"
        if guidelines:
            script.append(msg((G, {"question": "Which findings?"})))
            selection["guideline_refs"] = ["G1"]
        client = FakeClient([*script, msg((F, selection))])
        out = run(client, use_patients=patients, use_guidelines=guidelines)
        assert out == {"patient_ref": selection.get("patient_ref"),
                       "guideline_refs": selection.get("guideline_refs", []),
                       "searches": len(script), "finished": True, "stop_reason": "finish",
                       "trace": out["trace"], "usage": [None] * (len(script) + 1)}
        assert len(out["trace"]) == len(script)
        assert all(set(entry) == {"search", "tool", "input", "output"} for entry in out["trace"])
        specs = {s["function"]["name"]: s["function"]["parameters"] for s in client.requests[0]["tools"]}
        assert set(specs[F]["required"]) == set(selection)
        assert specs[F]["additionalProperties"] is False
        if guidelines:
            assert specs[F]["properties"]["guideline_refs"]["minItems"] == 1
        prompt = client.requests[0]["messages"][0]["content"]
        assert "ALL searches" in prompt if patients else "ALL" in prompt
        if guidelines:
            assert "earlier and later" in prompt and "passed in full" in prompt
        assert all(r["parallel_tool_calls"] is False for r in client.requests)


def test_empty_or_premature_finish_can_recover_without_locking_out_search():
    for bad in [{}, {"patient_ref": "P1"}]:
        client = FakeClient([msg((F, bad)), msg((P, {"query": "cough"})), msg((F, {"patient_ref": "P1"}))])
        events = []
        out = run(client, on_diagnostic=lambda e, o: events.append(deepcopy(e)))
        assert out["finished"] and out["searches"] == 1
        assert client.requests[1]["tool_choice"]["function"]["name"] == P
        assert [e["action"] for e in events] == ["retry", "recovered"]
        assert events[0]["substage"] == "finish_validation" and events[0]["searches"] == 0


def test_repeated_empty_finish_fails_without_search_or_fallback():
    c = FakeClient([msg((F, {})), msg((F, {})), msg((P, {"query": "never"}))])
    out, events = fail(c, "invalid_finish")
    assert len(c.requests) == 2 and out["searches"] == 0 and out["trace"] == []
    assert [e["action"] for e in events] == ["retry", "exit"]


def test_multiple_calls_execute_none_and_reply_to_every_id_before_retry():
    for batch in [((P, {"query": "unseen"}), (F, {"patient_ref": "P3"})),
                  ((F, {}), (P, {"query": "unseen"})),
                  ((P, {"query": "a"}), (P, {"query": "b"}))]:
        executed, events = [], []
        reply = "\n\n".join(f"[P{i}] " + str(i) * 4000 for i in range(1, 4))
        c = FakeClient([msg(*batch), msg((P, {"query": "real"})), msg((F, {"patient_ref": "P3"}))])

        def search(query):
            assert query == "real" and len(c.requests) == 2
            executed.append(query)
            return reply

        out = run(c, {P: search}, on_diagnostic=lambda e, o: events.append(deepcopy(e)))
        feedback = c.requests[1]["messages"][-2:]
        assert [m["tool_call_id"] for m in feedback] == ["c0", "c1"]
        assert all(m["role"] == "tool" and "exactly ONE" in m["content"] for m in feedback)
        assert c.requests[2]["messages"][-1]["content"] == reply
        assert executed == ["real"] and out["searches"] == 1 and out["patient_ref"] == "P3"
        assert len(events[0]["calls"]) == 2 and events[0]["code"] == "multiple_tool_calls"


def test_repeated_multicall_fails_and_executes_no_tools():
    batch = msg((P, {"query": "a"}), (F, {"patient_ref": "P1"}))
    c = FakeClient([batch, batch])
    out, _ = fail(c, "multiple_tool_calls", {P: lambda q: (_ for _ in ()).throw(AssertionError("executed"))})
    assert out["searches"] == 0 and len(c.requests) == 2


def test_correctable_argument_categories_each_get_one_successful_correction():
    invalid = [msg((P, raw)) for raw in ["{bad", "[]", "null", "", {}, {"query": " "},
                                        {"query": None}, {"query": 7}, {"query": []}]]
    invalid += [msg((name, {})) for name in [G, "invented_tool"]]
    invalid += [msg(content="done"), msg(content=""), msg(content=None)]
    for bad in invalid:
        c = FakeClient([bad, msg((P, {"query": "cough"})), msg((F, {"patient_ref": "P1"}))])
        events = []
        out = run(c, on_diagnostic=lambda e, o: events.append(deepcopy(e)))
        assert out["searches"] == 1 and len(c.requests) == 3
        assert [e["action"] for e in events] == ["retry", "recovered"]
        assert "only correction" in c.requests[1]["messages"][-1]["content"]
        assert all(m.get("content") is not None for m in c.requests[1]["messages"]
                   if m["role"] == "assistant" and not m.get("tool_calls"))


def test_finish_argument_types_missing_empty_duplicate_and_disabled_fields():
    bad = ["{bad", "[]", {}, {"patient_ref": None}, {"patient_ref": 2}, {"patient_ref": " "},
           {"patient_ref": "P1"}, {"guideline_refs": ["G1"]},
           *[{"patient_ref": "P1", "guideline_refs": refs} for refs in
             [None, "G1", [], [1], [""], [" "], ["G1", "G1"]]],
           {"patient_ref": "P1", "guideline_refs": ["G1"], "extra": 1}]
    for args in bad:
        c = FakeClient([msg((P, {"query": "cough"})), msg((G, {"question": "findings?"})),
                        msg((F, args)), msg((F, {"patient_ref": "P1", "guideline_refs": ["G1"]}))])
        out = run(c, use_guidelines=True)
        assert out["finished"] and out["searches"] == 2
        assert "Error:" in c.requests[-1]["messages"][-1]["content"]
    for patients, guidelines, search, args, good in [
            (True, False, (P, {"query": "cough"}), {"patient_ref": "P1", "guideline_refs": ["G1"]}, {"patient_ref": "P1"}),
            (False, True, (G, {"question": "findings?"}), {"patient_ref": "P1", "guideline_refs": ["G1"]}, {"guideline_refs": ["G1"]})]:
        c = FakeClient([msg(search), msg((F, args)), msg((F, good))])
        assert run(c, use_patients=patients, use_guidelines=guidelines)["finished"]


def test_one_shared_correction_not_one_per_error_or_reset_after_success():
    for script in [
            [msg(content="done"), msg((P, {"query": " "}))],
            [msg((P, "{bad")), msg((P, {"query": "cough"})), msg((F, {}))],
            [msg((P, {"query": "cough"})), msg((F, {})), msg((F, {}))]]:
        c = FakeClient(script)
        out, events = fail(c, "invalid_finish" if len(script) == 3 else "invalid_arguments")
        assert len(c.requests) == len(script)
        assert events[0]["action"] == "retry" and events[-1]["correction_used"] is True
        assert sum(e["action"] == "retry" for e in events) == 1


def test_mandatory_source_reserved_before_repeated_searches():
    c = FakeClient([msg((P, {"query": "first"})), msg((P, {"query": "too soon"})),
                    msg((G, {"question": "findings?"})), msg((F, {"patient_ref": "P1", "guideline_refs": ["G1"]}))])
    out = run(c, use_guidelines=True, max_searches=2)
    assert out["searches"] == 2 and len(out["trace"]) == 2
    assert c.requests[1]["tool_choice"]["function"]["name"] == G
    assert c.requests[-1]["tool_choice"]["function"]["name"] == F
    assert "search_similar_patients first" not in c.requests[2]["messages"][-1]["content"]


def test_full_budget_and_correction_stay_within_original_response_cap():
    for budget in [1, 4]:
        c = FakeClient([*[msg((P, {"query": str(i)})) for i in range(budget)],
                        msg((P, {"query": "over budget"})), msg((F, {"patient_ref": "P1"}))])
        out = run(c, max_searches=budget)
        assert out["searches"] == budget and len(out["trace"]) == budget
        assert len(c.requests) == budget + 2 <= 2 * budget + 3
        assert c.requests[-2]["tool_choice"]["function"]["name"] == F
        assert "budget exhausted" in c.requests[-1]["messages"][-1]["content"]


def test_all_prior_full_evidence_remains_visible_and_selectable():
    answers = ["\n\n".join(f"[P{3 * i + j}] " + str(j) * 4000 for j in range(1, 4)) for i in range(4)]
    replies = iter(answers)
    c = FakeClient([*[msg((P, {"query": str(i)})) for i in range(4)], msg((F, {"patient_ref": "P3"}))])
    out = run(c, {P: lambda q: next(replies)})
    final = [m["content"] for m in c.requests[-1]["messages"] if m["role"] == "tool"]
    assert final == answers and out["patient_ref"] == "P3"
    assert all(len(t["output"]) == 1500 for t in out["trace"])
    answer = "[G1] " + "full answer " * 2000
    c = FakeClient([msg((G, {"question": "findings?"})), msg((F, {"guideline_refs": ["G1"]}))])
    assert run(c, {G: lambda q: answer}, use_patients=False, use_guidelines=True)["finished"]
    assert c.requests[1]["messages"][-1]["content"] == answer


def test_invalid_configuration_fails_before_paid_work():
    for options in [{"max_searches": n} for n in [0, -1, True, 1.5, None]] + [
            {"max_searches": 1, "use_guidelines": True}, {"use_patients": False},
            {"use_guidelines": True}]:
        c = FakeClient([])
        try:
            run(c, {P: lambda q: "unused"}, **options)
        except ValueError:
            assert c.requests == []
        else:
            raise AssertionError("bad configuration accepted")


def test_response_usage_saved_before_parsing_tools_and_finish():
    events = []
    c = FakeClient([msg((P, {"query": "cough"})), msg((F, {"patient_ref": "P1"}))])
    run(c, {P: lambda q: events.append("tool") or "[P1] full"},
        on_response=lambda r: events.append("response"))
    assert events == ["response", "tool", "response"]
    c = FakeClient([SimpleNamespace(choices=[], usage=None)])
    seen = []
    try:
        run(c, on_response=seen.append)
    except IndexError:
        assert len(seen) == len(c.requests) == 1
    else:
        raise AssertionError("malformed response accepted")


def test_infrastructure_errors_keep_original_exception_without_agent_retries():
    for stage in ["api", "tool", "usage", "diagnostic"]:
        error = OSError("disk or service failure")
        seen = []

        def boom(*args):
            raise error

        c = FakeClient([error if stage == "api" else msg(content="invalid") if stage == "diagnostic"
                        else msg((P, {"query": "cough"}))])
        try:
            run(c, {P: boom if stage == "tool" else lambda q: seen.append(q)},
                on_response=boom if stage == "usage" else None,
                on_diagnostic=boom if stage == "diagnostic" else lambda e, o: seen.append(e))
        except OSError as exc:
            assert exc is error and len(c.requests) == 1
            if stage in ["usage", "diagnostic"]:
                assert not seen
        else:
            raise AssertionError("infrastructure exception swallowed")


def test_refusal_and_content_filter_exit_without_correction():
    for reason, refusal in [("stop", "Cannot comply"), ("content_filter", None)]:
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason,
            message=msg(refusal=refusal))], usage=None)
        c = FakeClient([response])
        fail(c, "refusal")
        assert len(c.requests) == 1


def test_truncated_safe_response_gets_one_correction_without_executing_call():
    for bad in [msg((P, '{"query": "cut')), msg((F, {"patient_ref": "P1"})), msg(content="cut")]:
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=bad)], usage=None)
        c = FakeClient([response, msg((P, {"query": "cough"})), msg((F, {"patient_ref": "P1"}))])
        out = run(c)
        assert out["searches"] == 1 and "truncated" in c.requests[1]["messages"][-1]["content"]


def test_malformed_structures_are_not_replayed_or_retried():
    bad_id = msg((P, {})); bad_id.tool_calls[0].id = None
    duplicate = msg((P, {}), (F, {})); duplicate.tool_calls[1].id = "c0"
    bad_args = msg((P, {})); bad_args.tool_calls[0].function.arguments = None
    for bad in [bad_id, duplicate, bad_args, SimpleNamespace(content=[], tool_calls=[]),
                SimpleNamespace(content=None, tool_calls={}), SimpleNamespace(content=None, tool_calls=False),
                SimpleNamespace(content=None, tool_calls=[None])]:
        c, events = FakeClient([bad]), []
        try:
            run(c, on_diagnostic=lambda e, o: events.append(e))
        except (ValueError, AttributeError):
            assert events[0]["code"] == "malformed_response" and events[0]["calls"] == []
            assert len(c.requests) == 1
        else:
            raise AssertionError("unsafe message accepted")


def test_sdk_multicall_rejection_and_full_evidence_protocol_for_all_variants():
    for patients, guidelines in [(True, False), (False, True), (True, True)]:
        requests, executed = [], []
        source = [(P, "query", "[P1] " + "patient " * 2000)] if patients else []
        if guidelines:
            source.append((G, "question", "[G1] " + "guideline " * 2000))
        selection = {**({"patient_ref": "P1"} if patients else {}),
                     **({"guideline_refs": ["G1"]} if guidelines else {})}

        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            assert body["parallel_tool_calls"] is False
            if len(requests) == 1:
                assert not executed
                calls = [("bad-search", source[0][0], {source[0][1]: "speculative"}),
                         ("bad-finish", F, selection)]
            else:
                if len(requests) == 2:
                    assert not executed
                    assert [m["tool_call_id"] for m in body["messages"][-2:]] == ["bad-search", "bad-finish"]
                for name, arg, answer in source[:len(executed)]:
                    assert any(m["role"] == "tool" and m["content"] == answer for m in body["messages"])
                index = len(requests) - 2
                if index < len(source):
                    name, arg, answer = source[index]
                    calls = [(f"search-{index}", name, {arg: "cough"})]
                else:
                    calls = [("finish", F, selection)]
            return httpx.Response(200, json={"id": f"offline-{len(requests)}", "object": "chat.completion",
                "created": 0, "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    for cid, name, args in calls]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

        def search(name, answer, query):
            assert query == "cough"
            executed.append(name)
            return answer

        available = {name: lambda q, n=name, a=answer: search(n, a, q) for name, arg, answer in source}
        with OpenAI(api_key="offline", base_url="https://offline.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            out = run(client, available, use_patients=patients, use_guidelines=guidelines,
                      max_searches=len(source))
        assert out["finished"] and out["searches"] == len(source)
        assert len(requests) == len(source) + 2 and len(out["usage"]) == len(requests)
        assert out["usage"][0] == {"prompt_tokens": 10, "completion_tokens": 2}


def test_guideline_budget_is_in_instructions_and_schema_without_changing_protocol():
    for budget in [4117, 8637]:
        c = FakeClient([msg((G, {"question": "findings?"})), msg((F, {"guideline_refs": ["G1"]}))])
        assert run(c, use_patients=False, use_guidelines=True, guideline_byte_budget=budget)["finished"]
        request = c.requests[0]
        prompt = request["messages"][0]["content"]
        description = request["tools"][-1]["function"]["parameters"]["properties"]["guideline_refs"]["description"]
        assert str(budget) in prompt and str(budget) in description
        assert "UTF-8 bytes" in prompt and "add 2 bytes" in prompt and "evidence_bytes" in prompt
        assert "ALL earlier and later searches" in prompt and "without summarization" in prompt
        assert "Keep ALL" not in prompt and "IDs of ALL answers" not in description
        assert request["parallel_tool_calls"] is False and len(c.requests) == 2
    c = FakeClient([msg((P, {"query": "cough"})), msg((F, {"patient_ref": "P1"}))])
    assert run(c)["finished"]
    assert "evidence_bytes" not in c.requests[0]["messages"][0]["content"]
    for bad in [0, -1, True, "100", 1.5]:
        c = FakeClient([])
        try:
            run(c, guideline_byte_budget=bad)
        except ValueError as exc:
            assert "guideline_byte_budget" in str(exc) and not c.requests
        else:
            raise AssertionError("invalid byte budget accepted")


def test_guideline_context_contract_is_in_system_and_tool_instructions_only_when_enabled():
    for patients in (False, True):
        script = [msg((P, {"query": "cough"}))] if patients else []
        selection = {"guideline_refs": ["G1"], **({"patient_ref": "P1"} if patients else {})}
        c = FakeClient([*script, msg((G, {"question": "For cough and O2 98%, which tests?"})),
                        msg((F, selection))])
        run(c, use_patients=patients, use_guidelines=True, guideline_byte_budget=4117)
        request = c.requests[0]
        prompt = request["messages"][0]["content"]
        description = next(t["function"]["description"] for t in request["tools"] if t["function"]["name"] == G)
        for text in (prompt, description):
            assert "self-contained" in text and "measurement values" in text
            assert "follow-up references" in text and "reference-patient facts" in text
        assert "only your question, not the patient notes or conversation history" in prompt
        assert "only this question, not the patient notes or previous answers" in description
        assert "original S/O" in prompt and "4117" in prompt
    c = FakeClient([msg((P, {"query": "cough"})), msg((F, {"patient_ref": "P1"}))])
    run(c)
    assert "GraphRAG" not in c.requests[0]["messages"][0]["content"]


def test_question_validation_retries_before_spending_search_and_preserves_usage_and_query():
    question, events, diagnostics = "For cough and O₂ 98%, which investigations?", [], []
    script = [msg((G, {"question": "bad question"})), msg((G, {"question": "  " + question + "  "})),
              msg((F, {"guideline_refs": ["G1"]}))]
    c = FakeClient([SimpleNamespace(id=f"r{i}", choices=[SimpleNamespace(message=m)],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2)) for i, m in enumerate(script)])

    def validate(text):
        events.append(("validate", text))
        if text == "bad question":
            raise ValueError("Rewrite using recorded target findings and measurement values.")

    def search(text):
        assert text == question and len(c.requests) == 2
        events.append(("tool", text))
        return "[G1] complete answer"

    def diagnostic(event, out):
        diagnostics.append((deepcopy(event), deepcopy(out)))
        events.append((event["action"], event["searches"]))

    out = run(c, {G: search}, use_patients=False, use_guidelines=True, max_searches=1,
              validate_guideline_question=validate, on_diagnostic=diagnostic,
              on_response=lambda r: events.append(("response", r.id)))
    assert events == [("response", "r0"), ("validate", "bad question"), ("retry", 0),
                      ("response", "r1"), ("validate", question), ("tool", question),
                      ("recovered", 1), ("response", "r2")]
    assert out["searches"] == 1 and out["trace"][0]["input"] == question
    assert out["usage"] == [{"prompt_tokens": 10, "completion_tokens": 2}] * 3
    assert diagnostics[0][0]["code"] == "invalid_guideline_question"
    assert diagnostics[0][0]["substage"] == "query_validation" and diagnostics[0][1]["trace"] == []
    feedback = c.requests[1]["messages"][-1]
    assert feedback["role"] == "tool" and feedback["tool_call_id"] == "c0" and "Rewrite" in feedback["content"]
    assert c.requests[1]["tool_choice"]["function"]["name"] == G
    assert c.requests[2]["tool_choice"]["function"]["name"] == F
    assert len(c.requests) == 3 <= 2 * 1 + 3


def test_question_validator_runs_only_after_call_structure_and_tool_arguments_are_valid():
    invalid = [msg((G, args)) for args in ["{bad", "[]", {}, {"question": 1}, {"question": " "}]]
    invalid += [msg(("disabled", {})), msg((G, {"question": "not executed"}), (F, {}))]
    for bad in invalid:
        seen = []
        c = FakeClient([bad, msg((G, {"question": "valid"})), msg((F, {"guideline_refs": ["G1"]}))])
        out = run(c, use_patients=False, use_guidelines=True, validate_guideline_question=seen.append)
        assert out["finished"] and seen == ["valid"]
    seen = []
    c = FakeClient([msg((P, {"query": "cough"})), msg((F, {"patient_ref": "P1"}))])
    run(c, validate_guideline_question=seen.append)
    assert seen == []
    c = FakeClient([msg((G, {"question": "valid"})), msg((G, {"question": "beyond budget"})),
                    msg((F, {"guideline_refs": ["G1"]}))])
    run(c, use_patients=False, use_guidelines=True, max_searches=1, validate_guideline_question=seen.append)
    assert seen == ["valid"]
    seen = []
    c = FakeClient([msg((G, {"question": "before required patient search"})),
                    msg((P, {"query": "cough"})), msg((G, {"question": "valid"})),
                    msg((F, {"patient_ref": "P1", "guideline_refs": ["G1"]}))])
    run(c, use_guidelines=True, validate_guideline_question=seen.append)
    assert seen == ["valid"]


def test_question_errors_share_one_correction_with_each_other_and_other_error_types():
    def validate(text):
        if text == "bad":
            raise ValueError("Rewrite the question.")

    cases = [
        ([msg((G, {"question": "bad"})), msg((G, {"question": "bad"}))], "invalid_guideline_question", 0),
        ([msg(content="done"), msg((G, {"question": "bad"}))], "invalid_guideline_question", 0),
        ([msg((G, {"question": "bad"})), msg((G, {"question": "good"})), msg((F, {}))], "invalid_finish", 1),
    ]
    for script, code, searches in cases:
        c = FakeClient(script)
        out, events = fail(c, code, use_patients=False, use_guidelines=True, validate_guideline_question=validate)
        assert len(c.requests) == len(script) and out["searches"] == searches
        assert sum(e["action"] == "retry" for e in events) == 1
        assert events[-1]["correction_used"] is True


def test_question_validator_infrastructure_failures_and_tool_value_errors_are_not_retried():
    for where, error in [("validator", OSError("resource failure")), ("validator", KeyboardInterrupt()),
                         ("tool", ValueError("GraphRAG failed"))]:
        seen = []

        def boom(text):
            raise error

        c = FakeClient([msg((G, {"question": "valid"}))])
        try:
            run(c, {G: boom if where == "tool" else seen.append}, use_patients=False, use_guidelines=True,
                validate_guideline_question=boom if where == "validator" else None,
                on_diagnostic=lambda e, o: seen.append(deepcopy(e)))
        except BaseException as exc:
            assert exc is error and len(c.requests) == 1
            if not isinstance(error, KeyboardInterrupt):
                assert seen[-1]["action"] == "exit"
                assert seen[-1]["substage"] == ("tool_execution" if where == "tool" else "query_validation")
        else:
            raise AssertionError("Infrastructure failure was retried or swallowed")


def test_sdk_question_correction_has_valid_tool_feedback_and_no_speculative_retrieval():
    requests, executed, responses = [], [], []
    question = "For cough and O2 98%, which investigations?"

    def validate(text):
        if text != question:
            raise ValueError("Use recorded findings and values in a self-contained question.")

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["parallel_tool_calls"] is False
        if len(requests) == 1:
            name, args = G, {"question": "bad"}
        elif len(requests) == 2:
            assert executed == []
            assert body["tool_choice"]["function"]["name"] == G
            assert body["messages"][-1]["tool_call_id"] == "call-1"
            assert "self-contained" in body["messages"][-1]["content"]
            assert "O2 98%" in body["messages"][1]["content"]
            name, args = G, {"question": question}
        else:
            assert len(requests) == 3 and executed == [question]
            assert body["messages"][-1] == {"role": "tool", "tool_call_id": "call-2", "content": "[G1] full answer"}
            name, args = F, {"guideline_refs": ["G1"]}
        return httpx.Response(200, json={"id": f"response-{len(requests)}", "object": "chat.completion",
            "created": 0, "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call-{len(requests)}", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    with OpenAI(api_key="offline", base_url="https://offline.invalid/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as c:
        out = run(c, {G: lambda q: executed.append(q) or "[G1] full answer"},
                  use_patients=False, use_guidelines=True, max_searches=1,
                  validate_guideline_question=validate, on_response=lambda r: responses.append(r.id))
    assert out["searches"] == 1 and len(out["usage"]) == 3
    assert responses == ["response-1", "response-2", "response-3"]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"Passed {len(tests)} tests")
