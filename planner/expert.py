"""Scripted expert that produces in-domain SFT demonstrations.

The expert actually runs the tools, so every demonstration is a real rollout.
Only rollouts that score 4 on both axes are kept - the simplest form of
rejection sampling. The expert is deliberately plain (it searches with the
question, then with an entity it saw in the results); GRPO is expected to
improve on it.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .scoring import score_messages
from .tasks import build_prompt
from .tools import TOOL_SCHEMAS, execute_tool
from .trace import clean_text, parse_tool_content


def _assistant_calls(calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": c["tool"], "arguments": dict(c["args"])}} for c in calls],
    }


def _run_round(calls: list[dict], out: list[dict]) -> list:
    """Append the assistant turn and its tool results; return the parsed results."""
    out.append(_assistant_calls(calls))
    results = []
    for c in calls:
        content = execute_tool(c["tool"], c["args"])
        out.append({"role": "tool", "name": c["tool"], "content": content})
        results.append(parse_tool_content(content)[0])
    return results


def _search(query: str) -> dict:
    return {"tool": "knowledge_search", "args": {"query": query}}


def _kb_rounds(aspect: dict, question: str, first_round_extra: list[dict], out: list[dict]) -> bool:
    """Search for the two gold passages of a HotpotQA aspect the way a careful user would.

    Returns False when this plain strategy does not find them (no demonstration).
    """
    docs = [i for i in aspect["evidence"] if i["type"] == "doc"]
    titles = [d["title"] for d in docs]
    gold = {d["doc_id"]: d["title"] for d in docs}
    in_question = [t for t in titles if t.lower() in question.lower()]

    if not aspect.get("multi_hop") and len(in_question) == len(titles):
        # Comparison: both entities are named in the question -> fan out in one round.
        _run_round([_search(t) for t in titles] + first_round_extra, out)
        return True

    # Otherwise start from the question itself.
    results = _run_round([_search(question)] + first_round_extra, out)[0]
    found = {p["doc_id"]: p for p in results if isinstance(p, dict) and p.get("doc_id") in gold} if isinstance(results, list) else {}
    if len(found) == len(gold):
        return True
    if not found:
        return False
    # Second hop: the missing entity must be visible in what we just read (or in the question).
    missing_title = next(t for d, t in gold.items() if d not in found)
    seen_text = " ".join(f"{p['title']} {p['text']}" for p in found.values()).lower()
    if missing_title.lower() not in seen_text and missing_title.lower() not in question.lower():
        return False
    _run_round([_search(missing_title)], out)
    return True


def expert_rollout(task: dict) -> list[dict] | None:
    """One demonstration rollout (assistant/tool messages), or None if the expert has no plan."""
    out: list[dict] = []
    aspects = task["aspects"]
    if not aspects:
        return None  # no_tool answers are free text: see `build_expert_traces`
    kb = [a for a in aspects if any(i["type"] == "doc" for i in a["evidence"])]
    if kb:
        # First-round calls that do not belong to the knowledge aspect (composite tasks).
        extra = [c for c in task["optimal_plan"][0] if c["tool"] != "knowledge_search"]
        question = task["query"] if len(aspects) == 1 else _kb_question(task)
        if question is None or not _kb_rounds(kb[0], question, extra, out):
            return None
    else:
        for calls in task["optimal_plan"]:
            _run_round(calls, out)
    out.append({"role": "assistant", "content": "Answer: " + "; ".join(a["answers"][0] for a in aspects)})
    return out


def _kb_question(task: dict) -> str | None:
    """The knowledge sub-question of a composite task (stored at generation time)."""
    return task.get("meta", {}).get("kb_question")


def build_expert_traces(
    tasks: Sequence[dict],
    direct_answer_fn: Callable[[list[str]], list[str]] | None = None,
) -> list[dict]:
    """Demonstrations for SFT: `{"messages": [...], "tools": [...], "task_id", "family"}`.

    `direct_answer_fn(queries) -> answers` supplies the reply for no_tool tasks
    (in the notebook: the base model answering without tools). Without it,
    no_tool tasks are skipped.
    """
    traces = []
    no_tool = [t for t in tasks if not t["aspects"]]
    for task in tasks:
        if not task["aspects"]:
            continue
        rollout = expert_rollout(task)
        if rollout is None or score_messages(task, rollout).score != 4:
            continue
        traces.append({"task_id": task["task_id"], "family": task["family"], "messages": build_prompt(task) + rollout, "tools": TOOL_SCHEMAS})
    if no_tool and direct_answer_fn is not None:
        answers = direct_answer_fn([t["query"] for t in no_tool])
        for task, answer in zip(no_tool, answers):
            if "<tool_call>" in (answer or ""):
                continue
            answer = clean_text(answer)
            if not answer:
                continue
            rollout = [{"role": "assistant", "content": answer}]
            traces.append({"task_id": task["task_id"], "family": task["family"], "messages": build_prompt(task) + rollout, "tools": TOOL_SCHEMAS})
    return traces
