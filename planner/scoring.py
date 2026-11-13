"""Two-axis scoring of a planner rollout, plus failure tags.

This is a rule-based stand-in for an LLM autograder. Each axis is scored 1-4 and
the final score is the minimum of the two; 3 and 4 count as a good answer.

Relevance   - did the plan go after every part of what the user asked, without
              off-topic tool calls?
Factuality  - is the final answer correct, and is it backed by evidence that the
              tools actually returned?

Evidence is read only from tool results (written by Python), never from what
the model says it found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .textnorm import answer_matches, numbers_close
from .tools import MAX_CALLS_PER_ROUND, MAX_SEARCH_CALLS, MAX_TOOL_ROUNDS, TOOL_COSTS, UNKNOWN_TOOL_COST
from .trace import Call, Trace, trace_from_messages
from .world import canon

SPECIALISED_TOOLS = ("get_weather", "get_election")
TAGS = (
    "specialized_tool_not_called",
    "wrong_tool",
    "wrong_arguments",
    "constraint_missed",
    "not_decomposed",
    "query_rewrite_failed",
    "multi_hop_incomplete",
    "synthesis_error",
    "no_tool_called",
    "no_final_answer",
    "over_budget",
    "unneeded_tool_call",
)


@dataclass
class Score:
    relevance: int
    factuality: int
    score: int
    good: bool
    cost: float
    excess_cost: float
    n_calls: int
    n_rounds: int
    n_search: int
    n_invalid: int
    invalid_fraction: float
    over_budget: bool
    has_final_answer: bool
    parallel: bool
    tags: list[str] = field(default_factory=list)
    calls_by_tool: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _canon_args(args: dict | None) -> dict:
    out = {}
    for key, value in (args or {}).items():
        if key == "year":
            try:
                out[key] = int(str(value).strip())
            except ValueError:
                out[key] = canon(value)
        else:
            out[key] = canon(value)
    return out


def _ok(call: Call) -> bool:
    return call.executed and call.error is None


def _retrieved_doc_ids(call: Call) -> set[str]:
    if call.name != "knowledge_search" or not _ok(call) or not isinstance(call.result, list):
        return set()
    return {p["doc_id"] for p in call.result if isinstance(p, dict) and "doc_id" in p}


def _calc_value(call: Call) -> float | None:
    if call.name != "calculator" or not _ok(call) or not isinstance(call.result, dict):
        return None
    value = call.result.get("result")
    return float(value) if isinstance(value, (int, float)) else None


def _satisfies(call: Call, item: dict) -> bool:
    """Does this executed call provide this evidence item?"""
    if item["type"] == "doc":
        return item["doc_id"] in _retrieved_doc_ids(call)
    if item["type"] == "call":
        return call.name == item["tool"] and _ok(call) and _canon_args(call.args) == _canon_args(item["args"])
    if item["type"] == "calc":
        value = _calc_value(call)
        return value is not None and numbers_close(value, float(item["value"]))
    return False


def score_trace(task: dict, trace: Trace) -> Score:
    calls = trace.calls
    aspects = task["aspects"]
    n_calls = len(calls)
    n_search = sum(c.name == "knowledge_search" for c in calls)
    n_invalid = sum(not _ok(c) for c in calls)
    cost = round(sum(TOOL_COSTS.get(c.name, UNKNOWN_TOOL_COST) for c in calls), 4)
    over_budget = (
        len(trace.rounds) > MAX_TOOL_ROUNDS
        or any(len(r) > MAX_CALLS_PER_ROUND for r in trace.rounds)
        or n_search > MAX_SEARCH_CALLS
    )
    final = trace.final_answer
    has_final = bool(final and final.strip())
    calls_by_tool: dict[str, int] = {}
    for c in calls:
        key = c.name if c.name in TOOL_COSTS else "unknown"
        calls_by_tool[key] = calls_by_tool.get(key, 0) + 1

    # Per-aspect evidence bookkeeping.
    task_has_calc = any(i["type"] == "calc" for a in aspects for i in a["evidence"])
    satisfied = [[any(_satisfies(c, item) for c in calls) for item in a["evidence"]] for a in aspects]
    hit = [any(s) for s in satisfied]
    complete = [all(s) for s in satisfied]
    correct = [has_final and answer_matches(final, a["answers"], a["answer_type"]) for a in aspects]

    def on_topic(call: Call) -> bool:
        if call.name == "calculator" and task_has_calc and _ok(call):
            return True
        return any(_satisfies(call, item) for a in aspects for item in a["evidence"])

    n_off_topic = sum(not on_topic(c) for c in calls)

    # ---- relevance
    if not aspects:
        relevance = 4 if n_calls == 0 else 2
    elif all(hit):
        relevance = 4 if n_off_topic == 0 else (3 if n_off_topic * 2 <= n_calls else 2)
    elif any(hit):
        relevance = 2
    else:
        relevance = 1

    # ---- factuality
    if not has_final:
        factuality = 1
    elif not aspects:
        factuality = 4
    elif all(correct):
        factuality = 4 if all(complete) else (3 if all(hit) else 2)
    elif any(correct):
        factuality = 2
    else:
        factuality = 1

    score = 1 if over_budget else min(relevance, factuality)
    result = Score(
        relevance=relevance,
        factuality=factuality,
        score=score,
        good=score >= 3,
        cost=cost,
        excess_cost=round(max(0.0, cost - float(task.get("optimal_cost", 0.0))), 4),
        n_calls=n_calls,
        n_rounds=len(trace.rounds),
        n_search=n_search,
        n_invalid=n_invalid,
        invalid_fraction=(n_invalid / n_calls) if n_calls else 0.0,
        over_budget=over_budget,
        has_final_answer=has_final,
        parallel=any(len(r) >= 2 for r in trace.rounds),
        calls_by_tool=calls_by_tool,
    )
    if not result.good:
        result.tags = _failure_tags(task, trace, satisfied, hit, complete, correct, result)
    return result


def _failure_tags(task, trace, satisfied, hit, complete, correct, result: Score) -> list[str]:
    """Why a bad rollout was bad. A rollout can carry several tags."""
    tags: set[str] = set()
    calls = trace.calls
    aspects = task["aspects"]
    if not result.has_final_answer:
        tags.add("no_final_answer")
    if result.over_budget:
        tags.add("over_budget")
    if not aspects:
        if calls:
            tags.add("unneeded_tool_call")
        return [t for t in TAGS if t in tags]

    needed_tools = {i["tool"] if i["type"] == "call" else ("knowledge_search" if i["type"] == "doc" else "calculator") for a in aspects for i in a["evidence"]}
    if not calls:
        tags.add("no_tool_called")
        if needed_tools & set(SPECIALISED_TOOLS):
            tags.add("specialized_tool_not_called")
    if any(c.name not in needed_tools for c in calls):
        tags.add("wrong_tool")
    if any(c.executed and c.error is not None and c.name in TOOL_COSTS for c in calls):
        tags.add("wrong_arguments")

    n_search = result.n_search
    for aspect, sat in zip(aspects, satisfied):
        items = aspect["evidence"]
        missing = [i for i, s in zip(items, sat) if not s]
        if not missing:
            continue
        for item in missing:
            if item["type"] == "call":
                same_tool = [c for c in calls if c.name == item["tool"]]
                if not same_tool:
                    if item["tool"] in SPECIALISED_TOOLS and calls:
                        tags.add("specialized_tool_not_called")
                    continue
                tags.add("wrong_arguments")
                want = _canon_args(item["args"])
                constraints = set(aspect.get("constraint_args", [])) & set(want)
                entity_keys = set(want) - constraints
                for c in same_tool:
                    got = _canon_args(c.args)
                    if all(got.get(k) == want[k] for k in entity_keys) and any(got.get(k) != want[k] for k in constraints):
                        tags.add("constraint_missed")
            elif item["type"] == "calc" and any(c.name == "calculator" for c in calls):
                tags.add("wrong_arguments")
        doc_items = [i for i in items if i["type"] == "doc"]
        if doc_items:
            got = sum(s for i, s in zip(items, sat) if i["type"] == "doc")
            if got < len(doc_items) and n_search > 0:
                if aspect.get("multi_hop") and got >= 1:
                    tags.add("multi_hop_incomplete")
                elif got == 0 or n_search >= len(doc_items):
                    tags.add("query_rewrite_failed")
                else:
                    tags.add("not_decomposed")
        call_items = [i for i in items if i["type"] == "call"]
        if call_items and any(sat) and not all(s for i, s in zip(items, sat) if i["type"] == "call"):
            tags.add("multi_hop_incomplete" if aspect.get("multi_hop") else "not_decomposed")

    if len(aspects) > 1 and any(hit) and not all(hit):
        tags.add("not_decomposed")
    if result.has_final_answer and all(complete) and not all(correct):
        tags.add("synthesis_error")
    if not tags and result.has_final_answer and not all(correct):
        tags.add("synthesis_error")
    return [t for t in TAGS if t in tags]


def score_messages(task: dict, messages: list[dict]) -> Score:
    """Score a rollout given as chat messages (TRL `completions`, bench or demo output)."""
    return score_trace(task, trace_from_messages(messages))
