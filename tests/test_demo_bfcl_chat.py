import json
import os
import random
from types import SimpleNamespace

import pytest

from planner import bfcl
from planner import tasks as T
from planner.chat import EMPTY_THINK, serving_template
from planner.demo import OpenAIBackend, answer, format_expected, format_markdown, to_openai_messages
from planner.tools import MAX_TOOL_ROUNDS, TOOL_SCHEMAS

from conftest import assistant_calls


# ------------------------------------------------------------------- demo
class FakeClient:
    """Stands in for `openai.OpenAI`: replays scripted chat completions and records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="served-model")]))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        calls = [SimpleNamespace(id=f"x{i}", function=SimpleNamespace(name=n, arguments=a)) for i, (n, a) in enumerate(reply.get("calls", []))]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply.get("content"), tool_calls=calls or None))])


@pytest.fixture()
def weather_task(world):
    return T._finish(T.make_weather_lookup(random.Random(4), world, "eval"), "weather_lookup", "eval", 0)


def test_demo_loop_against_openai_compatible_endpoint(weather_task):
    args = weather_task["optimal_plan"][0][0]["args"]
    gold = weather_task["aspects"][0]["answers"][0]
    client = FakeClient([{"calls": [("get_weather", json.dumps(args))]}, {"content": f"<think>ok</think>Answer: {gold}"}])
    backend = OpenAIBackend("http://unused", client=client)
    assert backend.model == "served-model"
    result = answer(weather_task, backend)
    assert result["score"]["good"] and result["rounds"] == 1 and result["cost"] == 0.25
    first, second = client.requests
    assert first["tools"] == TOOL_SCHEMAS and first["stream"] is False
    # second request carries the tool result in OpenAI wire format
    assistant, tool = second["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["arguments"] == json.dumps(args)
    assert tool["role"] == "tool" and tool["tool_call_id"] == assistant["tool_calls"][0]["id"] and "temp_c" in tool["content"]
    text = format_markdown(result, title="grpo")
    assert "### grpo" in text and "**Round 1**" in text and "good" in text and "get_weather" in text
    assert "get_weather" in format_expected(weather_task) and gold in format_expected(weather_task)


def test_demo_handles_servers_that_return_raw_tool_call_text(weather_task):
    args = json.dumps(weather_task["optimal_plan"][0][0]["args"])
    client = FakeClient([{"content": '<tool_call>\n{"name": "get_weather", "arguments": %s}\n</tool_call>' % args}, {"content": "Answer: 0"}])
    result = answer(weather_task["query"], OpenAIBackend("http://unused", client=client, model="m"))
    assert result["rounds"] == 1 and "score" not in result  # free-text query: no gold labels, no score


def test_demo_stops_after_the_round_budget(weather_task):
    args = json.dumps(weather_task["optimal_plan"][0][0]["args"])
    client = FakeClient([{"calls": [("get_weather", args)]}] * (MAX_TOOL_ROUNDS + 1))
    result = answer(weather_task, OpenAIBackend("http://unused", client=client, model="m"))
    assert len(client.requests) == MAX_TOOL_ROUNDS + 1 and result["final_message"] is None
    assert "Stopped" in format_markdown(result) and "no_final_answer" in result["score"]["tags"]


def test_to_openai_messages_assigns_matching_ids():
    messages = [
        {"role": "user", "content": "q"},
        assistant_calls(("calculator", {"expression": "1+1"}), ("calculator", {"expression": "2+2"})),
        {"role": "tool", "name": "calculator", "content": '{"result": 2}'},
        {"role": "tool", "name": "calculator", "content": '{"result": 4}'},
        {"role": "assistant", "content": "Answer: 2"},
    ]
    wire = to_openai_messages(messages)
    ids = [c["id"] for c in wire[1]["tool_calls"]]
    assert [m["tool_call_id"] for m in wire[2:4]] == ids and len(set(ids)) == 2
    assert all(isinstance(c["function"]["arguments"], str) for c in wire[1]["tool_calls"])


def test_shipped_examples_are_well_formed():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "examples.json")
    if not os.path.exists(path):
        pytest.skip("demo/examples.json not generated yet")
    with open(path, encoding="utf-8") as f:
        examples = json.load(f)
    assert len(examples) == 5
    for example in examples:
        task = example["task"]
        assert example["label"] and task["query"] and task["aspects"]
        assert 1 <= len(task["optimal_plan"]) <= MAX_TOOL_ROUNDS


# ------------------------------------------------------------------- bfcl
def write_score(root, model, group, category, correct, total):
    directory = os.path.join(root, model.replace("/", "_"), group)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"BFCL_v4_{category}_score.json"), "w") as f:
        f.write(json.dumps({"accuracy": correct / total, "correct_count": correct, "total_count": total}) + "\n")
        f.write(json.dumps({"id": "x", "valid": False}) + "\n")


def test_bfcl_score_reading_and_grouping(tmp_path):
    root, model = str(tmp_path), bfcl.NOTHINK_ID
    for category, correct, total in (("simple_python", 360, 400), ("simple_java", 60, 100), ("simple_javascript", 30, 50), ("multiple", 180, 200), ("parallel", 160, 200), ("parallel_multiple", 150, 200)):
        write_score(root, model, "non_live", category, correct, total)
    write_score(root, model, "live", "live_simple", 200, 258)
    write_score(root, model, "live", "live_multiple", 800, 1053)
    write_score(root, model, "non_live", "irrelevance", 216, 240)
    write_score(root, model, "live", "live_irrelevance", 441, 882)
    scores = bfcl.read_scores(root, model)
    assert scores["simple_java"] == {"accuracy": 0.6, "correct": 60, "total": 100}
    summary = bfcl.summarise(scores)
    simple = (0.9 + 0.6 + 0.6) / 3
    assert summary["groups"]["non_live"] == pytest.approx((simple + 0.9 + 0.8 + 0.75) / 4)
    assert summary["groups"]["live"] == pytest.approx(1000 / 1311)  # weighted by entries
    assert summary["groups"]["hallucination"] == pytest.approx((0.9 + 0.5) / 2)
    assert summary["groups"]["multi_turn"] is None  # not run
    assert summary["bfcl_core"] == pytest.approx(sum(v for v in summary["groups"].values() if v is not None) / 3)
    table = bfcl.format_table({"base": summary, "sft": summary})
    assert "n/a" in table and "BFCL-core" in table
    assert bfcl.read_scores(root, "other/model") == {}


def test_bfcl_categories_exist_in_the_installed_package():
    mapping = pytest.importorskip("bfcl_eval.constants.category_mapping")
    assert set(bfcl.ALL_CATEGORIES) <= set(mapping.ALL_SCORING_CATEGORIES)
    assert set(bfcl.SMOKE_CATEGORIES) <= set(bfcl.ALL_CATEGORIES)
    assert mapping.VERSION_PREFIX == "BFCL_v4"


def test_bfcl_nothink_handler_only_adds_the_empty_think_block():
    pytest.importorskip("bfcl_eval")
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    except Exception as exc:  # bfcl-eval's own heavy dependencies are missing
        pytest.skip(f"bfcl-eval handlers not importable here: {exc}")
    model_id = bfcl.register_nothink("Qwen/Qwen3-1.7B")
    config = MODEL_CONFIG_MAPPING[model_id]
    assert config.model_name == "Qwen/Qwen3-1.7B" and config.is_fc_model
    handler_cls = config.model_handler
    assert issubclass(handler_cls, QwenFCHandler)
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    functions = [{"name": "add", "description": "Add.", "parameters": {"type": "dict", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}]
    handler = object.__new__(handler_cls)  # prompt formatting needs no server or tokenizer
    stock = QwenFCHandler._format_prompt(handler, messages, functions)
    assert handler._format_prompt(messages, functions) == stock + EMPTY_THINK
    assert stock.endswith("<|im_start|>assistant\n")


# ------------------------------------------------------------------- chat
def test_serving_template_strips_training_only_markers():
    training = "{%- generation %}\nBODY\n{%- endgeneration %}\n{%- if enable_thinking is defined and enable_thinking is false %}X{%- endif %}"
    serving = serving_template(training)
    assert "generation" not in serving and "BODY" in serving
    assert "enable_thinking is not defined or enable_thinking is false" in serving


def test_bfcl_commands_and_server_wait(tmp_path):
    import http.server
    import threading

    generate, evaluate = bfcl.bfcl_commands("/venv/bin/python", bfcl.NOTHINK_ID, ["simple_python", "parallel"], "/r", "/s", "/models/sft", 8)
    assert generate[:4] == ["/venv/bin/python", "-m", "planner.bfcl", "generate"]
    assert "--skip-server-setup" in generate and generate[generate.index("--test-category") + 1] == "simple_python,parallel"
    assert generate[-2:] == ["--local-model-path", "/models/sft"] and evaluate[evaluate.index("--score-dir") + 1] == "/s"
    assert "--local-model-path" not in bfcl.bfcl_commands("py", "m", ["x"], "/r", "/s", None, 1)[0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("localhost", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        bfcl.wait_for_server(server.server_address[1], timeout=10, poll=0.1)
    finally:
        server.shutdown()
    with pytest.raises(TimeoutError):
        bfcl.wait_for_server(1, timeout=0.3, poll=0.1)
    dead = SimpleNamespace(poll=lambda: 3, returncode=3)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        bfcl.wait_for_server(1, process=dead, timeout=5, poll=0.1)
    bfcl.stop_server(None)
    bfcl.stop_server(dead)
