"""GRPO reward functions built on the two-axis score.

    g = (min(relevance, factuality) - 1) / 3                     in {0, 1/3, 2/3, 1}
    R = g - 0.3 * min(1, excess_cost / 3) - 0.2 * invalid_fraction      ("graded_cost", default)

`excess_cost` is the cost above the task's cheapest correct plan, so necessary
calls are free and only redundant ones are penalised. No LLM judge is involved.

The functions follow TRL's reward signature: they receive `completions` (one
list of messages per rollout, including the tool messages) and every extra
dataset column as a keyword argument - here `task`, a JSON string.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from .scoring import Score, score_messages
from .tools import TOOL_COSTS

REWARD_MODES = ("binary", "graded", "graded_cost")
COST_WEIGHT = 0.3
COST_SCALE = 3.0
INVALID_WEIGHT = 0.2


def reward_from_score(score: Score, mode: str = "graded_cost") -> float:
    if mode == "binary":
        return 1.0 if score.good else 0.0
    graded = (score.score - 1) / 3.0
    if mode == "graded":
        return graded
    if mode == "graded_cost":
        return graded - COST_WEIGHT * min(1.0, score.excess_cost / COST_SCALE) - INVALID_WEIGHT * score.invalid_fraction
    raise ValueError(f"unknown reward mode {mode!r}; choose from {REWARD_MODES}")


class ToolLedger:
    """Running count of tool calls made during RL - the 'QPS bill' of the run.

    Persisted as JSON so the totals survive a Colab disconnect.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.calls: Counter = Counter()
        self.rollouts = 0
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.calls.update(data.get("calls", {}))
            self.rollouts = int(data.get("rollouts", 0))

    def add(self, score: Score) -> None:
        self.rollouts += 1
        self.calls.update(score.calls_by_tool)

    @property
    def total_cost(self) -> float:
        return sum(TOOL_COSTS.get(tool, 0.25) * n for tool, n in self.calls.items())

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"calls": dict(self.calls), "rollouts": self.rollouts}, f)
        os.replace(tmp, self.path)


def zero_variance_fraction(task_ids: list[str], rewards: list[float]) -> float:
    """Share of prompt groups whose rollouts all got the same reward (no learning signal)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for task_id, reward in zip(task_ids, rewards):
        groups[task_id].append(reward)
    multi = [g for g in groups.values() if len(g) > 1]
    if not multi:
        return 0.0
    return sum(max(g) - min(g) < 1e-9 for g in multi) / len(multi)


def make_reward_fn(mode: str = "graded_cost", ledger: ToolLedger | None = None, print_every: int = 0):
    """Build a TRL-compatible reward function for the given mode.

    Besides returning the rewards it logs the two axes, the good rate, cost and
    per-tool call counts through TRL's `log_metric`, and updates the ledger.
    `print_every` > 0 prints one sample rollout every that many calls.
    """
    if mode not in REWARD_MODES:
        raise ValueError(f"unknown reward mode {mode!r}; choose from {REWARD_MODES}")
    state = {"calls": 0}

    def reward_fn(prompts=None, completions=None, task=None, **kwargs):
        tasks = [json.loads(t) if isinstance(t, str) else t for t in task]
        scores = [score_messages(t, c) for t, c in zip(tasks, completions)]
        rewards = [reward_from_score(s, mode) for s in scores]

        if ledger is not None:
            for s in scores:
                ledger.add(s)
            ledger.save()

        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and scores:
            n = len(scores)
            metrics = {
                "planner/good_rate": sum(s.good for s in scores) / n,
                "planner/relevance": sum(s.relevance for s in scores) / n,
                "planner/factuality": sum(s.factuality for s in scores) / n,
                "planner/cost": sum(s.cost for s in scores) / n,
                "planner/excess_cost": sum(s.excess_cost for s in scores) / n,
                "planner/invalid_fraction": sum(s.invalid_fraction for s in scores) / n,
                "planner/no_final_answer": sum(not s.has_final_answer for s in scores) / n,
                "planner/over_budget": sum(s.over_budget for s in scores) / n,
                "planner/rounds": sum(s.n_rounds for s in scores) / n,
                "planner/zero_variance_groups": zero_variance_fraction([t["task_id"] for t in tasks], rewards),
            }
            for tool in TOOL_COSTS:
                metrics[f"calls/{tool}"] = sum(s.calls_by_tool.get(tool, 0) for s in scores) / n
            if ledger is not None:
                metrics["qps/total_rollouts"] = float(ledger.rollouts)
                metrics["qps/total_cost"] = float(ledger.total_cost)
                for tool in TOOL_COSTS:
                    metrics[f"qps/total_{tool}"] = float(ledger.calls.get(tool, 0))
            for name, value in metrics.items():
                log_metric(name, float(value))

        if print_every and state["calls"] % print_every == 0 and completions:
            print("*" * 20, f"\n[{tasks[0]['family']}] {tasks[0]['query']}")
            for message in completions[0]:
                print(f"[{message['role']}]", message.get("tool_calls") or message.get("content"))
            print(f"-> relevance={scores[0].relevance} factuality={scores[0].factuality} cost={scores[0].cost} reward={rewards[0]:.3f} tags={scores[0].tags}")
        state["calls"] += 1
        return rewards

    reward_fn.__name__ = f"reward_{mode}"
    return reward_fn
