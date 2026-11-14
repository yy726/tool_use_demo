"""PlannerQA-Bench: run the planner on held-out tasks and aggregate the scores.

The loop is the same one the trainer and the demo use: up to `MAX_TOOL_ROUNDS`
rounds of tool calls, then one final turn that must not call tools. It is
batched by round so a vLLM / batched-generate backend stays efficient.

A backend is any callable `generate(list_of_message_lists) -> list_of_assistant_messages`.
`text_backend` adapts a raw-text generator (chat template in, text out).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Sequence

from .scoring import TAGS, Score, score_messages
from .tasks import FAMILIES, build_prompt, required_rounds
from .tools import MAX_CALLS_PER_ROUND, MAX_TOOL_ROUNDS, TOOL_COSTS, execute_tool
from .trace import parse_assistant_text

Backend = Callable[[list[list[dict]]], list[dict]]
FANOUT_FAMILIES = ("kb_compare", "fanout_calc", "composite", "chain_calc", "dual_bridge")
# Hard stop on calls executed in one round, so a runaway turn cannot flood the tools.
EXECUTION_CAP = 2 * MAX_CALLS_PER_ROUND


def text_backend(render: Callable[[list[dict]], str], generate_text: Callable[[list[str]], list[str]]) -> Backend:
    """Wrap `render(messages) -> prompt` and `generate_text(prompts) -> texts` as a backend."""

    def backend(batch: list[list[dict]]) -> list[dict]:
        return [parse_assistant_text(text) for text in generate_text([render(m) for m in batch])]

    return backend


def run_rollouts(
    prompts: Sequence[list[dict]],
    backend: Backend,
    max_rounds: int = MAX_TOOL_ROUNDS,
    execute: Callable[[object, object], str] = execute_tool,
) -> list[list[dict]]:
    """Run the tool loop for a batch of prompts; return the new messages of each rollout.

    The returned lists have the same shape as TRL's `completions`:
    assistant(tool_calls) -> tool ... -> assistant(final).
    """
    conversations = [list(p) for p in prompts]
    completions: list[list[dict]] = [[] for _ in prompts]
    active = list(range(len(prompts)))
    for round_index in range(max_rounds + 1):
        if not active:
            break
        replies = backend([conversations[i] for i in active])
        still_active = []
        for i, reply in zip(active, replies):
            conversations[i].append(reply)
            completions[i].append(reply)
            calls = reply.get("tool_calls") or []
            if not calls or round_index == max_rounds:
                continue  # final answer, or out of rounds (calls stay unexecuted)
            for call in calls[:EXECUTION_CAP]:
                fn = call["function"]
                message = {"role": "tool", "name": fn.get("name"), "content": execute(fn.get("name"), fn.get("arguments"))}
                conversations[i].append(message)
                completions[i].append(message)
            still_active.append(i)
        active = still_active
    return completions


def run_tasks(tasks: Sequence[dict], backend: Backend, batch_size: int = 64, progress: bool = False) -> list[list[dict]]:
    completions: list[list[dict]] = []
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        completions.extend(run_rollouts([build_prompt(t) for t in chunk], backend))
        if progress:
            print(f"  bench: {min(start + batch_size, len(tasks))}/{len(tasks)}")
    return completions


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _summarise(scores: list[Score]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "good_rate": _mean(s.good for s in scores),
        "bad_rate": _mean(not s.good for s in scores),
        "relevance": _mean(s.relevance for s in scores),
        "groundedness": _mean(s.groundedness for s in scores),
        "factuality": _mean(s.factuality for s in scores),
        "score": _mean(s.score for s in scores),
        "cost": _mean(s.cost for s in scores),
        "excess_cost": _mean(s.excess_cost for s in scores),
        "search_calls": _mean(s.n_search for s in scores),
        "tool_calls": _mean(s.n_calls for s in scores),
        "rounds": _mean(s.n_rounds for s in scores),
        "invalid_call_rate": sum(s.n_invalid for s in scores) / max(1, sum(s.n_calls for s in scores)),
        "final_answer_rate": _mean(s.has_final_answer for s in scores),
        "over_budget_rate": _mean(s.over_budget for s in scores),
    }


def evaluate(tasks: Sequence[dict], completions: Sequence[list[dict]], keep_records: bool = True) -> dict:
    """Score rollouts and aggregate: overall, per family, failure tags, tool usage."""
    scores = [score_messages(t, c) for t, c in zip(tasks, completions)]
    by_family: dict[str, list[Score]] = defaultdict(list)
    for t, s in zip(tasks, scores):
        by_family[t["family"]].append(s)

    bad = [s for s in scores if not s.good]
    tag_counts = Counter(tag for s in bad for tag in s.tags)
    fanout = [s for t, s in zip(tasks, scores) if t["family"] in FANOUT_FAMILIES]
    calls = Counter()
    for s in scores:
        calls.update(s.calls_by_tool)

    by_rounds: dict[int, list[Score]] = defaultdict(list)
    for t, s in zip(tasks, scores):
        by_rounds[required_rounds(t)].append(s)
    multi_step = [s for r, group in by_rounds.items() if r >= 2 for s in group]

    report = {
        "summary": _summarise(scores),
        # Tasks whose cheapest correct plan needs two or three dependent tool rounds: the headline number.
        "multi_step": _summarise(multi_step),
        "per_required_rounds": {str(r): _summarise(by_rounds[r]) for r in sorted(by_rounds)},
        "per_family": {f: _summarise(by_family[f]) for f in FAMILIES if by_family.get(f)},
        # Share of all tasks that failed with each tag (a rollout can carry several).
        "failure_tags": {tag: tag_counts.get(tag, 0) / max(1, len(scores)) for tag in TAGS},
        "failure_tag_counts": {tag: tag_counts.get(tag, 0) for tag in TAGS},
        "parallel_rate_on_fanout": _mean(s.parallel for s in fanout),
        "calls_per_task_by_tool": {tool: calls.get(tool, 0) / max(1, len(scores)) for tool in list(TOOL_COSTS) + ["unknown"]},
    }
    if keep_records:
        report["records"] = [
            {"task_id": t["task_id"], "family": t["family"], "query": t["query"], "score": s.to_dict(), "completion": c}
            for t, s, c in zip(tasks, scores, completions)
        ]
    return report


def run_benchmark(tasks: Sequence[dict], backend: Backend, batch_size: int = 64, progress: bool = False, keep_records: bool = True) -> dict:
    return evaluate(tasks, run_tasks(tasks, backend, batch_size=batch_size, progress=progress), keep_records=keep_records)


def format_report(reports: dict[str, dict]) -> str:
    """Markdown tables comparing checkpoints, e.g. {"base": r0, "sft": r1, "grpo": r2}."""
    names = list(reports)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [
        ("good rate", "good_rate", "{:.1%}"),
        ("relevance (1-4)", "relevance", "{:.2f}"),
        ("groundedness (1-4)", "groundedness", "{:.2f}"),
        ("factuality (1-4)", "factuality", "{:.2f}"),
        ("tool cost / query", "cost", "{:.2f}"),
        ("excess cost / query", "excess_cost", "{:.2f}"),
        ("knowledge_search calls / query", "search_calls", "{:.2f}"),
        ("tool rounds / query", "rounds", "{:.2f}"),
        ("invalid call rate", "invalid_call_rate", "{:.1%}"),
        ("final answer rate", "final_answer_rate", "{:.1%}"),
    ]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt.format(reports[n]["summary"].get(key, 0.0)) for n in names) + " |")
    if all("multi_step" in reports[n] for n in names):
        cells = [f"{reports[n]['multi_step'].get('good_rate', float('nan')):.1%}" for n in names]
        lines.insert(2, "| **multi-step good rate** (tasks needing 2-3 tool rounds) | " + " | ".join(cells) + " |")
    lines.append("| parallel rate on fan-out tasks | " + " | ".join(f"{reports[n]['parallel_rate_on_fanout']:.1%}" for n in names) + " |")

    lines += ["", "Good rate by task family:", "", "| family | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for family in FAMILIES:
        if any(family in reports[n]["per_family"] for n in names):
            cells = [f"{reports[n]['per_family'].get(family, {}).get('good_rate', float('nan')):.1%}" for n in names]
            lines.append(f"| {family} | " + " | ".join(cells) + " |")

    if all("per_required_rounds" in reports[n] for n in names):
        lines += ["", "Good rate by tool rounds the task needs:", "", "| rounds needed (tasks) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for r in sorted({r for n in names for r in reports[n]["per_required_rounds"]}):
            groups = [reports[n]["per_required_rounds"].get(r, {}) for n in names]
            lines.append(f"| {r} ({max(g.get('n', 0) for g in groups)}) | " + " | ".join(f"{g.get('good_rate', float('nan')):.1%}" for g in groups) + " |")

    lines += ["", "Failure tags (share of all tasks):", "", "| tag | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for tag in TAGS:
        if any(reports[n]["failure_tags"].get(tag, 0) > 0 for n in names):
            lines.append(f"| {tag} | " + " | ".join(f"{reports[n]['failure_tags'].get(tag, 0):.1%}" for n in names) + " |")
    return "\n".join(lines)
