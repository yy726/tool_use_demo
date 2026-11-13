import json
import random

import pytest

from planner import tasks as T
from planner.bench import evaluate, format_report, run_benchmark, run_rollouts, text_backend
from planner.rewards import REWARD_MODES, ToolLedger, make_reward_fn, zero_variance_fraction
from planner.tools import MAX_TOOL_ROUNDS

from conftest import run


@pytest.fixture()
def weather_tasks(world):
    rng = random.Random(7)
    return [T._finish(T.make_weather_lookup(rng, world, "eval"), "weather_lookup", "eval", i) for i in range(3)]


def good(task):
    calls = [(c["tool"], c["args"]) for c in task["optimal_plan"][0]]
    return run(calls, final="Answer: " + task["aspects"][0]["answers"][0])


def test_reward_fn_follows_trl_signature_and_logs(tmp_path, weather_tasks):
    task = weather_tasks[0]
    completions = [good(task), run(final="Answer: 12"), good(task) + []]
    columns = [json.dumps(task)] * 3  # TRL passes dataset columns as lists of raw values
    logged = {}
    ledger = ToolLedger(str(tmp_path / "ledger.json"))
    fn = make_reward_fn("graded_cost", ledger=ledger)
    rewards = fn(prompts=[None] * 3, completions=completions, task=columns, log_metric=lambda k, v: logged.__setitem__(k, v), trainer_state=None)
    assert rewards == [1.0, 0.0, 1.0] and fn.__name__ == "reward_graded_cost"
    assert logged["planner/good_rate"] == pytest.approx(2 / 3)
    assert logged["calls/get_weather"] == pytest.approx(2 / 3)
    assert logged["planner/zero_variance_groups"] == 0.0
    assert logged["qps/total_get_weather"] == 2 and logged["qps/total_rollouts"] == 3
    # the ledger survives a restart
    again = ToolLedger(str(tmp_path / "ledger.json"))
    assert again.calls["get_weather"] == 2 and again.rollouts == 3 and again.total_cost == 0.5


def test_reward_modes(weather_tasks):
    task = weather_tasks[0]
    sloppy = run([(c["tool"], c["args"]) for c in task["optimal_plan"][0]] * 2, final="Answer: " + task["aspects"][0]["answers"][0])
    values = {mode: make_reward_fn(mode)(completions=[sloppy], task=[task])[0] for mode in REWARD_MODES}
    assert values["binary"] == 1.0 and values["graded"] == 1.0
    assert values["graded_cost"] == pytest.approx(1.0 - 0.3 * 0.25 / 3)
    with pytest.raises(ValueError):
        make_reward_fn("llm_judge")


def test_zero_variance_fraction():
    assert zero_variance_fraction(["a", "a", "b", "b"], [1.0, 1.0, 0.0, 1.0]) == 0.5
    assert zero_variance_fraction(["a", "b"], [1.0, 0.0]) == 0.0  # groups of one carry no information


class ScriptedBackend:
    """Replays one scripted reply per turn for each conversation, keyed by the user query."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.batch_sizes = []

    def __call__(self, batch):
        self.batch_sizes.append(len(batch))
        replies = []
        for messages in batch:
            query = next(m["content"] for m in messages if m["role"] == "user")
            turn = sum(m["role"] == "assistant" for m in messages)
            replies.append(self.scripts[query][turn])
        return replies


def test_rollout_loop_executes_tools_and_stops(weather_tasks):
    ok, lazy, runaway = weather_tasks
    call = lambda t: {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": c["tool"], "arguments": c["args"]}} for c in t["optimal_plan"][0]]}
    scripts = {
        ok["query"]: [call(ok), {"role": "assistant", "content": "Answer: " + ok["aspects"][0]["answers"][0]}],
        lazy["query"]: [{"role": "assistant", "content": "Answer: 3"}],
        runaway["query"]: [call(runaway)] * (MAX_TOOL_ROUNDS + 1),
    }
    backend = ScriptedBackend(scripts)
    report = run_benchmark(weather_tasks, backend)
    assert backend.batch_sizes == [3, 2, 1, 1]  # finished conversations leave the batch
    records = {r["task_id"]: r for r in report["records"]}
    assert records[ok["task_id"]]["score"]["good"]
    assert records[lazy["task_id"]]["score"]["tags"] == ["specialized_tool_not_called", "no_tool_called"]
    run_rec = records[runaway["task_id"]]
    assert run_rec["score"]["n_rounds"] == MAX_TOOL_ROUNDS and not run_rec["score"]["has_final_answer"]
    assert sum(m["role"] == "tool" for m in run_rec["completion"]) == MAX_TOOL_ROUNDS  # 4th request not executed
    assert report["summary"]["good_rate"] == pytest.approx(1 / 3)
    assert report["per_family"]["weather_lookup"]["n"] == 3
    assert report["failure_tag_counts"]["no_final_answer"] == 1
    assert "| good rate |" in format_report({"base": report, "sft": report})


def test_text_backend_parses_raw_generations(weather_tasks):
    task = weather_tasks[0]
    args = json.dumps(task["optimal_plan"][0][0]["args"])
    texts = iter(['<think>\n\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": %s}\n</tool_call><|im_end|>' % args, "Answer: %s<|im_end|>" % task["aspects"][0]["answers"][0]])
    backend = text_backend(render=lambda messages: "PROMPT", generate_text=lambda prompts: [next(texts) for _ in prompts])
    completion = run_rollouts([T.build_prompt(task)], backend)[0]
    assert [m["role"] for m in completion] == ["assistant", "tool", "assistant"]
    assert evaluate([task], [completion])["summary"]["good_rate"] == 1.0


def test_malformed_call_is_answered_with_an_error(weather_tasks):
    task = weather_tasks[0]
    texts = iter(["<tool_call>\n{broken json\n</tool_call>", "Answer: 1"])
    backend = text_backend(lambda m: "", lambda prompts: [next(texts) for _ in prompts])
    completion = run_rollouts([T.build_prompt(task)], backend)[0]
    assert "error" in completion[1]["content"]
    assert evaluate([task], [completion])["summary"]["invalid_call_rate"] == 1.0


def test_hf_generate_fn_retries_oom_in_halves():
    torch = pytest.importorskip("torch")
    from planner.unsloth_utils import hf_generate_fn

    class Enc(dict):
        def to(self, device):
            return self

    class Tok:
        padding_side, pad_token_id, eos_token, pad_token = "right", 0, "</s>", "<pad>"

        def __call__(self, prompts, **kw):
            return Enc(input_ids=torch.zeros(len(prompts), 2, dtype=torch.long))

        def batch_decode(self, out, skip_special_tokens=False):
            return ["ok"] * out.shape[0]

    class Model:
        device = "cpu"
        sizes = []

        def generate(self, input_ids, **kw):
            self.sizes.append(input_ids.shape[0])
            if input_ids.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("simulated")
            return torch.zeros(input_ids.shape[0], 3, dtype=torch.long)

    model, tok = Model(), Tok()
    out = hf_generate_fn(model, tok, max_new_tokens=1, batch_size=8)([f"p{i}" for i in range(8)])
    assert out == ["ok"] * 8
    assert model.sizes == [8, 4, 2, 2, 4, 2, 2]  # 8 -> 4+4 -> 2+2 each, until it fits
    assert tok.padding_side == "right"
