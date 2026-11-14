"""The two-axis score and failure tags on hand-built rollouts."""

import random

import pytest

from planner import tasks as T
from planner.rewards import reward_from_score
from planner.scoring import score_messages
from planner.textnorm import answer_matches

from conftest import run


@pytest.fixture(scope="module")
def kb(corpus, rows):
    items = {i["id"]: i for i in T.kb_items(rows, corpus)}
    assert set(items) == {"q-bridge", "q-compare", "q-bridge-2", "q-compare-2"}
    return items


def finish(task, family):
    return T._finish(task, family, "eval", 0)


@pytest.fixture()
def bridge(kb):
    return finish(T.make_kb_task(kb["q-bridge"]), "kb_bridge")


@pytest.fixture()
def compare(kb):
    return finish(T.make_kb_task(kb["q-compare"]), "kb_compare")


@pytest.fixture()
def weather(world):
    return finish(T.make_weather_lookup(random.Random(1), world, "eval"), "weather_lookup")


@pytest.fixture()
def tool_bridge(world):
    return finish(T.make_tool_bridge(random.Random(2), world, "eval"), "tool_bridge")


@pytest.fixture()
def fanout(world):
    return finish(T.make_fanout_calc(random.Random(5), world, "eval"), "fanout_calc")


@pytest.fixture()
def composite(world, kb):
    return finish(T.make_composite(random.Random(3), world, "eval", kb["q-compare-2"]), "composite")


def plan_rounds(task):
    return [[(c["tool"], c["args"]) for c in calls] for calls in task["optimal_plan"]]


def gold_answer(task):
    return "Answer: " + "; ".join(a["answers"][0] for a in task["aspects"])


def search(query):
    return ("knowledge_search", {"query": query})


# ------------------------------------------------------------ perfect plans
@pytest.mark.parametrize("name", ["bridge", "compare", "weather", "tool_bridge", "fanout", "composite"])
def test_reference_plan_scores_four_with_no_excess_cost(name, request):
    task = request.getfixturevalue(name)
    s = score_messages(task, run(*plan_rounds(task), final=gold_answer(task)))
    assert (s.relevance, s.factuality, s.score, s.good) == (4, 4, 4, True)
    assert s.cost == task["optimal_cost"] and s.excess_cost == 0 and s.tags == []
    assert reward_from_score(s, "graded_cost") == 1.0


# -------------------------------------------- the four axis-separating cases
def test_bridge_first_hop_only_is_relevant_but_not_factual(bridge):
    s = score_messages(bridge, run([search("Silver Lantern 1987 drama")], final="Answer: Paris"))
    assert s.relevance == 4 and s.factuality == 1 and not s.good
    assert "multi_hop_incomplete" in s.tags


def test_composite_answering_one_part_scores_two_on_both_axes(composite):
    tool_calls = [c for c in plan_rounds(composite)[0] if c[0] != "knowledge_search"]
    s = score_messages(composite, run(tool_calls, final="Answer: " + composite["aspects"][1]["answers"][0]))
    assert (s.relevance, s.factuality) == (2, 2)
    assert "not_decomposed" in s.tags


def test_guessing_without_tools_is_not_grounded(compare):
    s = score_messages(compare, run(final="Answer: yes"))
    assert (s.relevance, s.factuality, s.score) == (1, 2, 1)
    assert "no_tool_called" in s.tags
    assert reward_from_score(s, "graded") == 0.0


def test_correct_answer_with_an_off_topic_call_is_still_good(weather, world):
    extra = ("get_election", {"region": world.region_list("eval")[0].name, "year": 2024})
    s = score_messages(weather, run(plan_rounds(weather)[0] + [extra], final=gold_answer(weather)))
    assert (s.relevance, s.factuality, s.good) == (3, 4, True)
    assert s.excess_cost == 0.25
    assert reward_from_score(s, "graded_cost") == pytest.approx(2 / 3 - 0.3 * 0.25 / 3)


# ---------------------------------------------------------------- other paths
def test_mostly_off_topic_calls_drop_relevance_to_two(weather):
    junk = [search("Granite Peak"), search("River Lark")]
    s = score_messages(weather, run(plan_rounds(weather)[0] + junk, final=gold_answer(weather)))
    assert s.relevance == 2 and not s.good and "wrong_tool" in s.tags


def test_correct_with_partial_evidence_is_factuality_three(compare):
    s = score_messages(compare, run([search("Tobias Quill sea stories")], final="Answer: yes"))
    assert (s.relevance, s.factuality, s.good) == (4, 3, True)


def test_complete_evidence_wrong_answer_is_a_synthesis_error(compare):
    s = score_messages(compare, run(*plan_rounds(compare), final="Answer: no"))
    assert (s.relevance, s.factuality) == (4, 1) and s.tags == ["synthesis_error"]


def test_no_final_answer(bridge):
    s = score_messages(bridge, run(*plan_rounds(bridge)))
    assert s.factuality == 1 and not s.has_final_answer and "no_final_answer" in s.tags
    still_calling = run(*plan_rounds(bridge)) + [run([search("x")])[0]]  # asks for a 3rd round, never answers
    assert "no_final_answer" in score_messages(bridge, still_calling).tags


def test_search_that_misses_is_a_query_rewrite_failure(bridge):
    s = score_messages(bridge, run([search("Copper Kettle Inn Sunday roast")], final="Answer: unknown"))
    assert s.relevance == 1 and "query_rewrite_failed" in s.tags


def test_comparison_with_one_search_is_not_decomposed(compare):
    s = score_messages(compare, run([search("Tobias Quill sea stories")], final="Answer: no"))
    assert "not_decomposed" in s.tags


def test_specialised_tool_not_called(weather):
    s = score_messages(weather, run([search(weather["query"])], final="Answer: 20"))
    assert s.relevance == 1 and {"specialized_tool_not_called", "wrong_tool"} <= set(s.tags)


def test_wrong_date_is_a_missed_constraint(weather):
    args = dict(weather["optimal_plan"][0][0]["args"])
    args["date"] = "2024-01-01" if args["date"] != "2024-01-01" else "2024-01-02"
    s = score_messages(weather, run([("get_weather", args)], final="Answer: 1"))
    assert {"wrong_arguments", "constraint_missed"} <= set(s.tags)


def test_wrong_entity_is_wrong_arguments_but_not_a_missed_constraint(weather, world):
    args = dict(weather["optimal_plan"][0][0]["args"])
    args["city"] = next(c for c in world.city_list("eval") if c != args["city"])
    s = score_messages(weather, run([("get_weather", args)], final="Answer: 1"))
    assert "wrong_arguments" in s.tags and "constraint_missed" not in s.tags


def test_invalid_calls_are_counted_and_penalised(weather):
    bad = ("get_weather", {"city": weather["optimal_plan"][0][0]["args"]["city"], "date": "yesterday"})
    s = score_messages(weather, run([bad], plan_rounds(weather)[0], final=gold_answer(weather)))
    assert s.n_invalid == 1 and s.invalid_fraction == 0.5 and s.relevance == 3
    assert reward_from_score(s, "graded_cost") == pytest.approx(2 / 3 - 0.3 * 0.25 / 3 - 0.2 * 0.5)


def test_tool_bridge_second_hop_missing(tool_bridge):
    s = score_messages(tool_bridge, run(plan_rounds(tool_bridge)[0], final="Answer: sunny"))
    assert s.relevance == 4 and "multi_hop_incomplete" in s.tags and "specialized_tool_not_called" in s.tags


def test_fanout_without_calculator_is_good_but_not_perfect(fanout):
    s = score_messages(fanout, run(plan_rounds(fanout)[0], final=gold_answer(fanout)))
    assert (s.relevance, s.factuality, s.good) == (4, 3, True) and s.parallel


def test_over_budget_scores_one(compare):
    five = [search(f"Tobias Quill {i}") for i in range(3)] + [search("Henrietta Moss"), search("Henrietta Moss novelist")]
    s = score_messages(compare, run(five, final="Answer: yes"))
    assert s.over_budget and s.score == 1 and "over_budget" in s.tags
    many_searches = run([search("Tobias Quill")], [search("Henrietta Moss")], [search("a novelist"), search("sea stories"), search("lake country")], final="Answer: yes")
    assert score_messages(compare, many_searches).over_budget


def test_duplicate_calls_cost_extra(weather):
    call = plan_rounds(weather)[0][0]
    s = score_messages(weather, run([call, call], final=gold_answer(weather)))
    assert s.relevance == 4 and s.excess_cost == 0.25


def test_no_tool_family(world):
    task = finish(T.make_no_tool(random.Random(0), world, "eval"), "no_tool")
    assert score_messages(task, run(final="Here is a short poem.")).score == 4
    called = score_messages(task, run([search("poem")], final="Here is a short poem."))
    assert called.relevance == 2 and called.tags == ["unneeded_tool_call"]
    assert score_messages(task, run(final="")).factuality == 1


@pytest.mark.parametrize(
    "prediction, answers, kind, expected",
    [
        ("yes, both were", ["yes"], "yesno", True),
        ("They were not; no.", ["no"], "yesno", False),
        ("Henry J. Kaiser", ["Henry J. Kaiser"], "text", True),
        ("the answer is henry j kaiser.", ["Henry J. Kaiser"], "text", True),
        ("Kaiserslautern", ["Kaiser"], "text", False),
        ("Rob Parissi", ['Robert "Rob" Parissi'], "text", True),
        ("Rob", ['Robert "Rob" Parissi'], "text", False),
        ("film director", ["film director, film producer, writer"], "text", False),
        ("about 18.44 degrees", ["18.4"], "number", True),
        ("19.6", ["18.4"], "number", False),
        ("1,235 items", ["1235"], "number", True),
        ("0 mm", ["0.0"], "number", True),
        ("", ["x"], "text", False),
    ],
)
def test_answer_matching(prediction, answers, kind, expected):
    assert answer_matches(prediction, answers, kind) is expected


# ------------------------------------------------------------- groundedness
def test_groundedness_checks_the_answer_against_tool_results_not_against_gold(weather, bridge):
    plan = plan_rounds(weather)
    assert score_messages(weather, run(*plan, final=gold_answer(weather))).groundedness == 4
    # A number or a name that no tool returned is not grounded, even though the right tool was called.
    made_up = score_messages(weather, run(*plan, final="Answer: 987654 according to Zorblax"))
    assert made_up.groundedness == 1 and made_up.relevance == 4
    # An answer from memory has nothing to stand on.
    assert score_messages(weather, run(final=gold_answer(weather))).groundedness == 1
    assert score_messages(weather, run(*plan, final="")).groundedness == 1
    # Grounded but wrong: the text comes from the retrieved passages, it is just not the answer.
    wrong = score_messages(bridge, run(*plan_rounds(bridge), final="Answer: Silver Lantern"))
    assert wrong.groundedness == 4 and wrong.factuality == 1 and not wrong.good
    # Pasting a passage is not an answer.
    pasted = score_messages(bridge, run(*plan_rounds(bridge), final="Answer: " + "Silver Lantern director born " * 15))
    assert pasted.groundedness <= 2


def test_reward_uses_relevance_and_groundedness_and_leaves_factuality_out(bridge, weather):
    from planner.rewards import make_reward_fn

    grounded_but_wrong = run(*plan_rounds(bridge), final="Answer: Silver Lantern")
    s = score_messages(bridge, grounded_but_wrong)
    assert s.axis_min() == 4 and s.axis_min(("relevance", "factuality")) == 1
    assert reward_from_score(s, "graded") == 1.0  # the default reward cannot see that this is wrong ...
    assert reward_from_score(s, "graded", axes=("relevance", "factuality")) == 0.0  # ... factuality can
    logged = {}
    rewards = make_reward_fn("graded")(completions=[grounded_but_wrong], task=[bridge], log_metric=logged.__setitem__)
    assert rewards == [1.0] and logged["planner/groundedness"] == 4 and logged["planner/factuality"] == 1
    assert logged["planner/reward_axes_good_rate"] == 1.0 and logged["planner/good_rate"] == 0.0
    # From memory: correct, but not grounded and nothing retrieved -> no reward.
    assert reward_from_score(score_messages(weather, run(final=gold_answer(weather))), "graded") == 0.0
    with pytest.raises(ValueError):
        make_reward_fn("graded", axes=("vibes",))
