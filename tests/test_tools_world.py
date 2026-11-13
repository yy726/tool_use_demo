import json

import pytest

from planner import tools
from planner.corpus import Corpus, truncate_words
from planner.world import ELECTION_YEARS, World, canon


def test_world_is_deterministic_and_splits_are_disjoint():
    a, b = World(seed=1), World(seed=1)
    assert [r.name for r in a.region_list("train")] == [r.name for r in b.region_list("train")]
    assert len(a.region_list("train")) == 60 and len(a.region_list("eval")) == 30
    assert not set(a.city_list("train")) & set(a.city_list("eval"))
    city = a.city_list("train")[0]
    assert a.weather(city, "2024-05-01") == b.weather(city, "2024-05-01")


def test_region_and_city_names_do_not_collide(world):
    assert not {canon(r.name) for r in world.regions.values()} & set(world.city_names)


def test_weather_record_and_errors(world):
    city = world.city_list("eval")[0]
    record = world.weather(city.upper(), "2025-07-04")
    assert record["city"] == city and record["condition"] and isinstance(record["temp_c"], float)
    assert (record["precip_mm"] > 0) == (record["condition"] in ("rainy", "stormy", "snowy"))
    with pytest.raises(ValueError, match="unknown city"):
        world.weather("Atlantis", "2025-07-04")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        world.weather(city, "July 4, 2025")
    with pytest.raises(ValueError, match="supported range"):
        world.weather(city, "1999-01-01")


def test_election_record_links_to_weather(world):
    region = world.region_list("train")[0]
    record = world.election(region.name, "2024")
    assert record["capital"] == region.capital
    assert record["winner"] != record["runner_up"] and record["winner_party"] != record["runner_up_party"]
    assert record["winner_vote_share_pct"] > record["runner_up_vote_share_pct"]
    assert record["winner_vote_share_pct"] + record["runner_up_vote_share_pct"] <= 100
    # cross-tool hop: the capital on election day has a weather record
    assert world.weather(record["capital"], record["election_date"])["city"] == region.capital
    with pytest.raises(ValueError, match="no election in 2023"):
        world.election(region.name, 2023)
    with pytest.raises(ValueError, match="unknown region"):
        world.election("Nowhere", 2024)
    assert set(ELECTION_YEARS) == {2016, 2020, 2024}


@pytest.mark.parametrize(
    "expression, expected",
    [("2 + 3 * 4", 14), ("(21.5 + 18.0 + 19.4) / 3", 19.633333), ("abs(12.3 - 17.9)", 5.6), ("1,234 + 1", 1235), ("2 ^ 10", 1024), ("round(7 / 3, 2)", 2.33), ("max(1, 2)", 2)],
)
def test_calculator(expression, expected):
    assert json.loads(tools.calculator(expression))["result"] == pytest.approx(expected)


@pytest.mark.parametrize("expression", ["__import__('os').system('ls')", "open('x')", "1 / 0", "", "2 ** 9999", "x + 1", "'a' * 3"])
def test_calculator_rejects_unsafe_or_invalid(expression):
    with pytest.raises(ValueError):
        tools.calculator(expression)


def test_knowledge_search_returns_top_passages():
    hits = json.loads(tools.knowledge_search("Odette Varnier film director"))
    assert 1 <= len(hits) <= 3
    assert hits[0]["title"] == "Odette Varnier" and set(hits[0]) == {"doc_id", "title", "text"}
    with pytest.raises(ValueError):
        tools.knowledge_search("   ")


def test_execute_tool_turns_failures_into_error_messages(world):
    city = world.city_list("train")[0]
    assert "temp_c" in json.loads(tools.execute_tool("get_weather", {"city": city, "date": "2024-03-01"}))
    assert "temp_c" in json.loads(tools.execute_tool("get_weather", json.dumps({"city": city, "date": "2024-03-01"})))
    assert "not found" in json.loads(tools.execute_tool("get_stock_price", {}))["error"]
    assert "error" in json.loads(tools.execute_tool("get_weather", {"city": city}))  # missing argument
    assert "error" in json.loads(tools.execute_tool("get_weather", {"city": city, "date": "tomorrow"}))
    assert "error" in json.loads(tools.execute_tool(None, "not json"))
    assert "error" in json.loads(tools.execute_tool("calculator", ["1+1"]))


def test_schemas_match_what_transformers_derives_from_the_functions():
    utils = pytest.importorskip("transformers.utils")
    assert [utils.get_json_schema(fn) for fn in tools.TOOLS] == tools.TOOL_SCHEMAS
    assert [s["function"]["name"] for s in tools.TOOL_SCHEMAS] == list(tools.TOOL_COSTS)


def test_corpus_dedupes_saves_and_loads(tmp_path, corpus):
    n = len(corpus)
    again = corpus.add("Odette Varnier", "different text")
    assert len(corpus) == n and "born in Marlow Bay" in again.text
    path = str(tmp_path / "corpus.jsonl")
    corpus.save(path)
    loaded = Corpus.load(path)
    assert [p.doc_id for p in loaded.passages] == [p.doc_id for p in corpus.passages]
    assert loaded.search("Glass Orchard opera")[0].title == "Glass Orchard"
    assert truncate_words("a b c d", 2) == "a b ..."
    assert Corpus().search("anything") == []
