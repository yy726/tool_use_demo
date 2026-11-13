from planner.trace import extract_answer, parse_assistant_text, parse_tool_content, trace_from_messages

from conftest import assistant_calls


def test_parse_assistant_text_with_parallel_calls_and_thinking():
    text = (
        "<think>\nneed two lookups\n</think>\n\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "A", "date": "2024-01-01"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "B", "date": "2024-01-01"}}\n</tool_call><|im_end|>'
    )
    message = parse_assistant_text(text)
    assert message["content"] == ""
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["get_weather", "get_weather"]
    assert message["tool_calls"][1]["function"]["arguments"]["city"] == "B"


def test_parse_assistant_text_keeps_malformed_calls_as_invalid():
    message = parse_assistant_text('<tool_call>\n{"name": "calculator", "arguments": {"expression": 1+}}\n</tool_call>')
    assert message["tool_calls"][0]["function"]["name"] is None
    assert "tool_calls" not in parse_assistant_text("Answer: 42<|im_end|>")
    # a tool call written inside the thinking block is not a call
    assert "tool_calls" not in parse_assistant_text('<think><tool_call>{"name": "x", "arguments": {}}</tool_call></think>Answer: hi')


def test_extract_answer():
    assert extract_answer("The capital is X.\nAnswer: Paris") == "Paris"
    assert extract_answer("Answer: no\nwait.\nANSWER: yes") == "yes"
    assert extract_answer("<think>hmm</think>Just some text<|im_end|>") == "Just some text"


def test_parse_tool_content_handles_json_and_python_reprs():
    assert parse_tool_content('{"result": 3}') == ({"result": 3}, None)
    assert parse_tool_content("{'error': 'Tool x not found.'}")[1] == "Tool x not found."  # TRL's str(dict)
    assert parse_tool_content('{"error": "bad date"}')[1] == "bad date"
    assert parse_tool_content("plain text") == ("plain text", None)


def test_trace_pairs_results_with_calls_in_order():
    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "ignored"},
        assistant_calls(("calculator", {"expression": "1+1"}), ("calculator", '{"expression": "2+2"}')),
        {"role": "tool", "name": "calculator", "content": '{"result": 2}'},
        {"role": "tool", "name": "calculator", "content": '{"error": "boom"}'},
        assistant_calls(("get_weather", "not json")),
        {"role": "tool", "name": "get_weather", "content": '{"error": "tool arguments must be a JSON object"}'},
        {"role": "assistant", "content": "Answer: 2"},
    ]
    trace = trace_from_messages(messages)
    assert [len(r) for r in trace.rounds] == [2, 1]
    first, second = trace.rounds[0]
    assert first.result == {"result": 2} and first.error is None and first.round == 0
    assert second.args == {"expression": "2+2"} and second.error == "boom"
    assert trace.rounds[1][0].args is None and trace.rounds[1][0].round == 1
    assert trace.final_answer == "2" and trace.has_answer_marker and trace.pending_calls == 0


def test_calls_requested_after_the_round_budget_are_not_counted():
    messages = [
        assistant_calls(("calculator", {"expression": "1+1"})),
        {"role": "tool", "name": "calculator", "content": '{"result": 2}'},
        assistant_calls(("calculator", {"expression": "2+2"})),  # never executed
    ]
    trace = trace_from_messages(messages)
    assert len(trace.calls) == 1 and trace.pending_calls == 1
    assert trace.final_message is None and trace.final_answer is None


def test_raw_tool_call_text_in_content_is_recognised():
    messages = [
        {"role": "assistant", "content": '<tool_call>\n{"name": "calculator", "arguments": {"expression": "3*3"}}\n</tool_call>'},
        {"role": "tool", "content": '{"result": 9}'},
        {"role": "assistant", "content": "Answer: 9"},
    ]
    trace = trace_from_messages(messages)
    assert trace.calls[0].name == "calculator" and trace.calls[0].result == {"result": 9}
