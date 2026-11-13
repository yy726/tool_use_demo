import json
import random

import pytest

from planner import tasks as T
from planner.config import get_config
from planner.expert import build_expert_traces, expert_rollout
from planner.scoring import score_messages
from planner.setup_env import prepare
from planner.tools import TOOL_COSTS, execute_tool


def test_kb_items_filters_unusable_rows(corpus, rows):
    bad_answer = dict(rows[0], id="x1", answer="an answer that is not in any passage")
    long_answer = dict(rows[0], id="x2", answer="one two three four five six seven")
    one_gold = dict(rows[0], id="x3", supporting_facts={"title": ["Silver Lantern"], "sent_id": [0]})
    items = T.kb_items(list(rows) + [bad_answer, long_answer, one_gold], corpus)
    assert [i["id"] for i in items] == [r["id"] for r in rows]
    assert {i["answer_type"] for i in items} == {"text", "yesno"}


def test_synthetic_tasks_are_deterministic_and_executable(world):
    for family, maker in T._SYNTHETIC.items():
        a = [maker(random.Random(11), world, "train") for _ in range(1)][0]
        b = maker(random.Random(11), world, "train")
        assert a == b, family
        rng = random.Random(5)
        for _ in range(40):
            task = T._finish(maker(rng, world, "eval"), family, "eval", 0)
            json.dumps(task)  # serialisable
            assert task["optimal_cost"] == sum(TOOL_COSTS[c["tool"]] for r in task["optimal_plan"] for c in r)
            for calls in task["optimal_plan"]:
                for c in calls:
                    assert "error" not in execute_tool(c["tool"], c["args"]), (family, c)
            if family != "no_tool":
                assert len(task["optimal_plan"]) <= 3 and all(len(r) <= 4 for r in task["optimal_plan"])


def test_train_and_eval_tasks_use_disjoint_entities(world):
    def entities(split):
        names = set()
        rng = random.Random(0)
        for maker in (T.make_weather_lookup, T.make_election_lookup, T.make_fanout_calc, T.make_tool_bridge):
            for _ in range(60):
                for calls in maker(rng, world, split)["optimal_plan"]:
                    for c in calls:
                        names |= {str(v) for k, v in c["args"].items() if k in ("city", "region")}
        return names

    assert not entities("train") & entities("eval")


def test_build_split_and_pools(world, corpus, rows):
    items = T.kb_items(rows, corpus)
    pools = T.split_kb_pools(items * 3, items)
    assert len(pools["sft"]) == 4 and len(pools["grpo"]) == 8 and len(pools["eval"]) == 4
    tasks = T.build_split("eval", 2, world, items, seed=1)
    counts = T.family_counts(tasks)
    assert counts["weather_lookup"] == counts["election_lookup"] == counts["fanout_calc"] == counts["tool_bridge"] == counts["no_tool"] == 2
    assert counts["kb_bridge"] == 2 and counts["kb_compare"] == 2
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    assert tasks == T.build_split("eval", 2, world, items, seed=1)
    # kb questions are not reused across families
    used = [t["meta"]["hotpot_id"] for t in tasks if "hotpot_id" in t["meta"]]
    assert len(used) == len(set(used))


def test_expert_solves_every_family(world, corpus, rows):
    items = T.kb_items(rows, corpus)
    tasks = T.build_split("sft", 3, world, items * 4, seed=3)
    solved = set()
    for task in tasks:
        rollout = expert_rollout(task)
        if rollout is not None:
            score = score_messages(task, rollout)
            assert score.good, (task["family"], task["query"], score)
            if score.score == 4:
                solved.add(task["family"])
    assert solved >= {"kb_bridge", "kb_compare", "weather_lookup", "election_lookup", "fanout_calc", "tool_bridge", "chain_calc", "dual_bridge"}


def test_expert_bridge_searches_with_what_it_read(world, corpus, rows):
    item = next(i for i in T.kb_items(rows, corpus) if i["id"] == "q-bridge")
    task = T._finish(T.make_kb_task(item), "kb_bridge", "sft", 0)
    rollout = expert_rollout(task)
    queries = [c["function"]["arguments"]["query"] for m in rollout if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    assert queries[0] == task["query"]  # starts from the question, not from the gold titles
    assert rollout[-1]["content"] == "Answer: Marlow Bay"


def test_build_expert_traces_keeps_only_perfect_rollouts(world, corpus, rows):
    items = T.kb_items(rows, corpus)
    tasks = T.build_split("sft", 3, world, items, seed=3)
    traces = build_expert_traces(tasks, direct_answer_fn=lambda queries: ["Sure: " + q for q in queries])
    by_id = {t["task_id"]: t for t in tasks}
    assert traces
    for trace in traces:
        assert [m["role"] for m in trace["messages"][:2]] == ["system", "user"]
        assert trace["tools"] and score_messages(by_id[trace["task_id"]], trace["messages"][2:]).score == 4
    assert {t["family"] for t in traces} >= {"weather_lookup", "no_tool", "tool_bridge"}
    without = build_expert_traces(tasks)
    assert "no_tool" not in {t["family"] for t in without}
    refused = build_expert_traces(tasks, direct_answer_fn=lambda queries: ["<tool_call>{}</tool_call>" for _ in queries])
    assert "no_tool" not in {t["family"] for t in refused}


def test_prepare_caches_environment(tmp_path, monkeypatch, rows):
    import planner.setup_env as setup_env

    calls = []

    def fake_rows(n_train, n_val):
        calls.append((n_train, n_val))
        return list(rows) * 2, list(rows)

    monkeypatch.setattr(setup_env, "load_hotpot_rows", fake_rows)
    config = get_config("smoke", n_eval_per_family=2, n_grpo_per_family=2, n_sft_per_family=2)
    first = prepare(config, str(tmp_path), verbose=False)
    second = prepare(config, str(tmp_path), verbose=False)
    assert calls == [(config.hotpot_train_rows, config.hotpot_val_rows)]  # second call used the cache
    changed = prepare(get_config("smoke", n_eval_per_family=2, n_grpo_per_family=2, n_sft_per_family=2, eval_mix={"weather_lookup": 2.0}), str(tmp_path), verbose=False)
    assert len(calls) == 2 and changed.tasks["eval"] != first.tasks["eval"]  # other task settings: the cache is rebuilt
    assert first.tasks == second.tasks and len(second.corpus) == len(first.corpus)
    assert set(first.tasks) == {"sft", "grpo", "eval"}


def test_grpo_rows_carry_prompt_and_task(world):
    tasks = [T._finish(T.make_tool_bridge(random.Random(i), world, "train"), "tool_bridge", "grpo", i) for i in range(5)]
    rows = T.grpo_rows(tasks, seed=1)
    assert rows == T.grpo_rows(tasks, seed=1) and len(rows) == 5
    assert {json.loads(r["task"])["task_id"] for r in rows} == {t["task_id"] for t in tasks}
    for row in rows:
        task = json.loads(row["task"])
        assert [m["role"] for m in row["prompt"]] == ["system", "user"] and row["prompt"][1]["content"] == task["query"]


def test_family_mix_shifts_a_split_towards_multi_step(world, corpus, rows):
    assert T.family_sizes(100, {"kb_bridge": 1.5, "no_tool": 0.5})["kb_bridge"] == 150
    assert T.family_sizes(100, {"no_tool": 0.5})["no_tool"] == 50 and T.family_sizes(100)["chain_calc"] == 100
    with pytest.raises(ValueError):
        T.family_sizes(10, {"not_a_family": 2})
    tasks = T.build_split("eval", 4, world, T.kb_items(rows, corpus) * 3, seed=1, mix={"weather_lookup": 0.25, "chain_calc": 2.0})
    counts = T.family_counts(tasks)
    assert counts["weather_lookup"] == 1 and counts["chain_calc"] == 8 and counts["tool_bridge"] == 4
    config = get_config("a100")
    sizes = T.family_sizes(config.n_eval_per_family, config.eval_mix)
    assert sum(sizes.values()) == 800 and sum(T.family_sizes(config.n_grpo_per_family, config.grpo_mix).values()) == 4000
    multi = sum(sizes[f] for f in T.MULTI_STEP_FAMILIES) + sizes["composite"] // 2
    assert multi / 800 >= 0.7  # the benchmark is mostly multi-step


def test_three_step_and_dual_bridge_tasks(world, corpus, rows):
    import random

    rng = random.Random(5)
    for _ in range(20):
        task = T._finish(T.make_chain_calc(rng, world, "eval"), "chain_calc", "eval", 0)
        plan = task["optimal_plan"]
        assert T.required_rounds(task) == 3 and [c["tool"] for c in plan[0]] == ["get_election"] * 2
        assert [c["tool"] for c in plan[1]] == ["get_weather"] * 2 and plan[2][0]["tool"] == "calculator"
        assert score_messages(task, expert_rollout(task)).score == 4
        # stopping after the first two rounds is a multi-hop failure, not a good answer
        first_round_only = expert_rollout(task)[:3] + [{"role": "assistant", "content": "Answer: 123456"}]
        score = score_messages(task, first_round_only)
        assert not score.good and "multi_hop_incomplete" in score.tags
    bridge = next(i for i in T.kb_items(rows, corpus) if i["type"] == "bridge")
    dual = T._finish(T.make_dual_bridge(rng, world, "eval", bridge), "dual_bridge", "eval", 0)
    assert T.required_rounds(dual) == 2 and len(dual["aspects"]) == 2 and all(a["multi_hop"] for a in dual["aspects"])
    assert {c["tool"] for c in dual["optimal_plan"][1]} == {"knowledge_search", "get_weather"}
