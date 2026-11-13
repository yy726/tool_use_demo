import json

import pytest

from planner.data_public import convert_all, convert_hermes, convert_toolace, parse_bracket_calls, parse_toolace_tools

TOOLACE_SYSTEM = (
    "You are an expert in composing functions. You are given a question and a set of possible functions. \n"
    "Here is a list of functions in JSON format that you can invoke:\n"
    '[{"name": "Market Trends API", "description": "Get market trends.", "parameters": {"type": "dict", "properties": '
    '{"trend_type": {"description": "Trend type.", "type": "string"}, "filters": {"type": "dict", "properties": {"n": {"type": "integer"}}}}, '
    '"required": ["trend_type"]}, "required": null}, '
    '{"name": "Ratings", "description": "Get ratings.", "parameters": {"type": "dict", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}, "required": null}]. \n'
    "Should you decide to return the function call(s). \nPut it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n\nNO other text MUST be included. \n"
)


def toolace_row(*turns):
    return {"system": TOOLACE_SYSTEM, "conversations": [{"from": role, "value": value} for role, value in turns]}


def test_parse_toolace_tools_normalises_schema():
    tools = parse_toolace_tools(TOOLACE_SYSTEM)
    assert [t["function"]["name"] for t in tools] == ["Market Trends API", "Ratings"]
    params = tools[0]["function"]["parameters"]
    assert params["type"] == "object" and params["properties"]["filters"]["type"] == "object"
    assert "required" not in tools[0]["function"]  # the stray top-level `"required": null` is dropped
    with pytest.raises(ValueError):
        parse_toolace_tools("no tools here")


@pytest.mark.parametrize(
    "text, expected",
    [
        ('[Market Trends API(trend_type="MARKET_INDEXES", country="us")]', [("Market Trends API", {"trend_type": "MARKET_INDEXES", "country": "us"})]),
        ('[Ratings(symbol="SPY"), Ratings(symbol="QQQ")]', [("Ratings", {"symbol": "SPY"}), ("Ratings", {"symbol": "QQQ"})]),
        ("[GetCompetitions()]", [("GetCompetitions", {})]),
        ('[f(items=["a, b", "c)"], opts={"k": [1, 2]}, flag=true, n=-3.5, note="it\'s (fine)")]', [("f", {"items": ["a, b", "c)"], "opts": {"k": [1, 2]}, "flag": True, "n": -3.5, "note": "it's (fine)"})]),
    ],
)
def test_parse_bracket_calls(text, expected):
    calls = parse_bracket_calls(text)
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in calls] == expected


@pytest.mark.parametrize("text", ["plain text", "[]", "[f(1, 2)]", "[f(x=undefined_name)]", '[f(x="unterminated)]', "[not a call]"])
def test_parse_bracket_calls_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_bracket_calls(text)


def test_convert_toolace_multi_turn():
    row = toolace_row(
        ("user", "Ratings for SPY and QQQ?"),
        ("assistant", '[Ratings(symbol="SPY"), Ratings(symbol="QQQ")]'),
        ("tool", '[{"name": "Ratings", "results": {"rating": "A"}}, {"name": "Ratings", "results": {"rating": "B"}}]'),
        ("assistant", "SPY is rated A and QQQ is rated B."),
        ("user", "What do ratings mean?"),
        ("assistant", "They summarise analyst opinion."),
    )
    example = convert_toolace(row)
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["user", "assistant", "tool", "tool", "assistant", "user", "assistant"]
    assert example["messages"][2] == {"role": "tool", "name": "Ratings", "content": '{"rating": "A"}'}
    assert example["source"] == "toolace" and len(example["tools"]) == 2
    json.dumps(example)


def test_convert_toolace_drops_inconsistent_rows():
    undeclared = toolace_row(("user", "hi"), ("assistant", '[Unknown Tool(x=1)]'))
    empty = toolace_row(("user", "hi"), ("assistant", ""))
    orphan_tool = toolace_row(("user", "hi"), ("tool", "[]"), ("assistant", "ok"))
    ends_with_user = toolace_row(("user", "hi"), ("assistant", "hello"), ("user", "bye"))
    good = toolace_row(("user", "hi"), ("assistant", "I cannot help with that using the available functions."))
    examples, dropped = convert_all([undeclared, empty, orphan_tool, ends_with_user, good], convert_toolace)
    assert dropped == 4 and len(examples) == 1


HERMES_TOOLS = json.dumps([{"type": "function", "function": {"name": "get_camera_live_feed", "description": "Live feed.", "parameters": {"type": "object", "properties": {"camera_id": {"type": "string"}}, "required": ["camera_id"]}}}])


def test_convert_hermes():
    row = {
        "tools": HERMES_TOOLS,
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model. <tools>...</tools>"},
            {"from": "human", "value": "Show the front door camera."},
            {"from": "gpt", "value": '<tool_call>\n{"name": "get_camera_live_feed", "arguments": {"camera_id": "front_door"}}\n</tool_call>\n'},
            {"from": "tool", "value": '<tool_response>\n{"name": "get_camera_live_feed", "content": {"url": "https://x"}}\n</tool_response>\n'},
            {"from": "gpt", "value": "Here is the live feed."},
        ],
    }
    example = convert_hermes(row)
    assert [m["role"] for m in example["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert example["messages"][1]["tool_calls"][0]["function"] == {"name": "get_camera_live_feed", "arguments": {"camera_id": "front_door"}}
    assert example["messages"][1]["content"] == ""
    python_style = dict(row, conversations=row["conversations"][:2] + [{"from": "gpt", "value": "<tool_call>\n{'name': 'get_camera_live_feed', 'arguments': {'camera_id': 'a'}}\n</tool_call>"}])
    assert convert_hermes(python_style)["messages"][-1]["tool_calls"][0]["function"]["arguments"] == {"camera_id": "a"}
    stops_after_tool = dict(row, conversations=row["conversations"][:4])
    assert [m["role"] for m in convert_hermes(stops_after_tool)["messages"]] == ["user", "assistant"]
    garbage = dict(row, conversations=row["conversations"][:2] + [{"from": "gpt", "value": "<tool_call>\nget_camera_live_feed(front)\n</tool_call>"}])
    assert convert_all([garbage], convert_hermes) == ([], 1)


def test_tool_names_with_parentheses_and_apostrophes():
    names = ["User Feed (Video Posts) V2", "Get Agent's Active Listings", "Get Agent"]
    calls = parse_bracket_calls('[User Feed (Video Posts) V2(username="sunny"), Get Agent\'s Active Listings(zuid="Z1"), Get Agent(id=3)]', names)
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in calls] == [
        ("User Feed (Video Posts) V2", {"username": "sunny"}),
        ("Get Agent's Active Listings", {"zuid": "Z1"}),
        ("Get Agent", {"id": 3}),
    ]
