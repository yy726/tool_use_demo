"""Pick the five preset demo queries from the held-out tasks and write demo/examples.json.

Run once after changing the environment config:
    python scripts/make_demo_examples.py --cache-dir outputs/env_a100

Selection is rule-based and deterministic (shortest qualifying query per type),
using the real BM25 index, so the examples are guaranteed to be solvable and to
need the behaviour they are labelled with.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.config import get_config  # noqa: E402
from planner.expert import expert_rollout  # noqa: E402
from planner.scoring import score_messages  # noqa: E402
from planner.setup_env import prepare  # noqa: E402
from planner.trace import trace_from_messages  # noqa: E402


def expert_trace(task):
    rollout = expert_rollout(task)
    if rollout is None or score_messages(task, rollout).score != 4:
        return None
    return trace_from_messages(rollout)


def gold_hits(env, task, query):
    gold = {i["doc_id"] for a in task["aspects"] for i in a["evidence"] if i["type"] == "doc"}
    return len(gold & {p.doc_id for p in env.corpus.search(query)})


def pick(tasks, family, predicate):
    candidates = [t for t in tasks if t["family"] == family and predicate(t)]
    if not candidates:
        raise SystemExit(f"no candidate for {family}")
    return min(candidates, key=lambda t: (len(t["query"]), t["task_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="outputs/env_a100")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "examples.json"))
    args = parser.parse_args()
    env = prepare(get_config("a100"), args.cache_dir)
    tasks = env.tasks["eval"]

    def two_sequential_searches(task):
        trace = expert_trace(task)
        return trace is not None and len(trace.rounds) == 2 and 8 <= len(task["query"].split()) <= 22

    def parallel_searches(task):
        trace = expert_trace(task)
        answer = task["aspects"][0]["answers"][0]
        return trace is not None and len(trace.rounds) == 1 and len(trace.rounds[0]) == 2 and answer.isascii() and answer.replace(" ", "").isalnum() and answer.lower() in task["query"].lower()

    def needs_decomposition_and_rewrite(task):
        # Searching with the whole two-part ask finds nothing useful; the plan must split it,
        # then follow up with an entity it read about (a bridge question).
        trace = expert_trace(task)
        return task["aspects"][0]["multi_hop"] and trace is not None and len(trace.rounds) == 2 and gold_hits(env, task, task["query"]) == 0

    examples = [
        {
            "label": "Bridge / multi-hop",
            "why": "The second search depends on an entity found by the first one.",
            "task": pick(tasks, "kb_bridge", two_sequential_searches),
        },
        {
            "label": "Decomposition + fan-out + synthesis",
            "why": "Two entities must be looked up in parallel and then compared.",
            "task": pick(tasks, "kb_compare", parallel_searches),
        },
        {
            "label": "Multi-date fan-out + calculation",
            "why": "Three weather lookups in one round, then the calculator combines them.",
            "task": pick(tasks, "fanout_calc", lambda t: t["aspects"][0]["name"] == "calc:temp_avg"),
        },
        {
            "label": "Cross-tool bridge + tool selection",
            "why": "The election result supplies the capital and the date for the weather call.",
            "task": pick(tasks, "tool_bridge", lambda t: True),
        },
        {
            "label": "Multi-part ask: fan-out + query reformulation",
            "why": "Searching with the whole question retrieves nothing useful; it has to be split, and the knowledge part needs a follow-up query.",
            "task": pick(tasks, "composite", needs_decomposition_and_rewrite),
        },
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for i, example in enumerate(examples, 1):
        print(f"{i}. [{example['label']}] {example['task']['query']}")
        print("   answers:", [a["answers"][0] for a in example["task"]["aspects"]])


if __name__ == "__main__":
    main()
