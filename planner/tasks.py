"""Task generators for the eight PlannerQA task families.

A task is a JSON-serialisable dict:

    task_id, family, split, query,
    aspects:       one entry per thing the user asked for, each with gold answers and the
                   evidence a correct plan must retrieve,
    optimal_cost:  cost of the cheapest correct plan (for the efficiency penalty),
    optimal_plan:  that plan as rounds of tool calls (for the scripted expert and the demo).

Evidence items are one of
    {"type": "doc",  "doc_id": ..., "title": ...}     a gold passage must be retrieved
    {"type": "call", "tool": ..., "args": {...}}      a specialised tool must be called with these args
    {"type": "calc", "value": ...}                    the calculator must have produced this value
"""

from __future__ import annotations

import datetime as _dt
import random
from typing import Callable

from .corpus import Corpus, truncate_words
from .textnorm import normalize_text
from .tools import TOOL_COSTS
from .world import ELECTION_YEARS, World

FAMILIES = (
    "kb_bridge",
    "kb_compare",
    "weather_lookup",
    "election_lookup",
    "fanout_calc",
    "tool_bridge",
    "composite",
    "no_tool",
)

SYSTEM_PROMPT = """You are a planner that answers questions by calling tools.

Rules:
- Pick the most specific tool: get_weather for weather, get_election for election results, calculator for arithmetic, knowledge_search for other factual knowledge. Do not call a tool when none is needed.
- Break a multi-part question into its parts. Independent calls can go in the same turn, in parallel.
- Use the results of earlier calls to form later calls. Write specific search queries, and rewrite a query that returned nothing useful.
- You have at most 3 tool-calling turns. knowledge_search is expensive: use as few searches as you can.
- When you have enough information, stop calling tools and end your reply with one line: Answer: <short answer>"""

_WEATHER_DATE_START = _dt.date(2024, 1, 1)
_WEATHER_DATE_DAYS = 730


def build_prompt(task: dict) -> list[dict]:
    """The [system, user] messages the planner sees for a task."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["query"]}]


def call(tool: str, **args) -> dict:
    return {"tool": tool, "args": args}


def plan_cost(plan: list[list[dict]]) -> float:
    return round(sum(TOOL_COSTS[c["tool"]] for r in plan for c in r), 4)


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _rand_date(rng: random.Random, margin: int = 0) -> _dt.date:
    """A date with weather records; `margin` leaves room for follow-up dates."""
    return _WEATHER_DATE_START + _dt.timedelta(days=rng.randrange(_WEATHER_DATE_DAYS - margin))


def _say_date(rng: random.Random, day: _dt.date) -> str:
    """Either ISO or a natural phrasing, so the model has to normalise the argument."""
    if rng.random() < 0.5:
        return day.isoformat()
    return f"{day.strftime('%B')} {day.day}, {day.year}"


def _pick(rng: random.Random, templates: list[str], eval_templates: list[str], split: str) -> str:
    """Eval tasks also draw from a few phrasings never seen in training."""
    pool = templates + eval_templates if split == "eval" else templates
    return rng.choice(pool)


# ------------------------------------------------------------ weather lookup
_WEATHER_ATTRS = {
    "temp_c": (
        "number",
        ["What was the temperature in {city} on {date}?", "How warm was it in {city} on {date}, in Celsius?"],
        ["Tell me the temperature recorded in {city} on {date}."],
    ),
    "condition": (
        "text",
        ["What was the weather like in {city} on {date}?", "Was it sunny, rainy or something else in {city} on {date}?"],
        ["Describe the weather condition in {city} on {date}."],
    ),
    "precip_mm": (
        "number",
        ["How much precipitation fell in {city} on {date}, in millimetres?", "How many millimetres of precipitation did {city} get on {date}?"],
        ["What was the precipitation total for {city} on {date}?"],
    ),
}
_WEATHER_DISTRACTORS = [
    "The {year} election in {region} is long over, but I have a different question. {question}",
    "People in {region} were still arguing about who won the vote. Anyway: {question}",
]


def _weather_part(rng: random.Random, world: World, split: str, distract: bool = True) -> dict:
    region = rng.choice(world.region_list(split))
    city = rng.choice(region.cities)
    day = _rand_date(rng)
    attr = rng.choice(list(_WEATHER_ATTRS))
    answer_type, templates, eval_templates = _WEATHER_ATTRS[attr]
    record = world.weather(city, day.isoformat())
    question = _pick(rng, templates, eval_templates, split).format(city=city, date=_say_date(rng, day))
    if distract and rng.random() < 0.3:
        question = rng.choice(_WEATHER_DISTRACTORS).format(year=rng.choice(ELECTION_YEARS), region=region.name, question=question)
    step = call("get_weather", city=city, date=day.isoformat())
    return {
        "question": question,
        "aspect": {
            "name": f"weather:{attr}",
            "answers": [str(record[attr])],
            "answer_type": answer_type,
            "evidence": [{"type": "call", **step}],
            "multi_hop": False,
            "constraint_args": ["date"],
        },
        "plan": [[step]],
    }


# ----------------------------------------------------------- election lookup
_ELECTION_ATTRS = {
    "winner": ("text", ["Who won the {year} election in {region}?", "Which candidate won in {region} in {year}?"], ["Name the winner of {region}'s {year} election."]),
    "winner_party": ("text", ["Which party won the {year} election in {region}?"], ["The winner of the {year} election in {region} belonged to which party?"]),
    "winner_vote_share_pct": (
        "number",
        ["What share of the vote did the winner get in the {year} election in {region}?", "What was the winning vote share in {region} in {year}, in percent?"],
        ["In {region}'s {year} election, what percentage of the vote went to the winner?"],
    ),
    "runner_up": ("text", ["Who came second in the {year} election in {region}?", "Who was the runner-up in {region}'s {year} election?"], []),
    "turnout_pct": ("number", ["What was the turnout in the {year} election in {region}, in percent?"], ["How high was voter turnout in {region} in {year}?"]),
}
_ELECTION_DISTRACTORS = [
    "It was pouring in {capital} that week, but never mind the weather. {question}",
    "Forget the forecast for {capital} for a moment. {question}",
]


def _election_part(rng: random.Random, world: World, split: str, distract: bool = True) -> dict:
    region = rng.choice(world.region_list(split))
    year = rng.choice(ELECTION_YEARS)
    attr = rng.choice(list(_ELECTION_ATTRS))
    answer_type, templates, eval_templates = _ELECTION_ATTRS[attr]
    record = world.election(region.name, year)
    question = _pick(rng, templates, eval_templates, split).format(region=region.name, year=year)
    if distract and rng.random() < 0.3:
        question = rng.choice(_ELECTION_DISTRACTORS).format(capital=region.capital, question=question)
    step = call("get_election", region=region.name, year=year)
    return {
        "question": question,
        "aspect": {
            "name": f"election:{attr}",
            "answers": [str(record[attr])],
            "answer_type": answer_type,
            "evidence": [{"type": "call", **step}],
            "multi_hop": False,
            "constraint_args": ["year"],
        },
        "plan": [[step]],
    }


def _single_aspect_task(part: dict) -> dict:
    return {"query": part["question"], "aspects": [part["aspect"]], "optimal_plan": part["plan"]}


def make_weather_lookup(rng: random.Random, world: World, split: str) -> dict:
    return _single_aspect_task(_weather_part(rng, world, split))


def make_election_lookup(rng: random.Random, world: World, split: str) -> dict:
    return _single_aspect_task(_election_part(rng, world, split))


# --------------------------------------------------------------- fanout_calc
def make_fanout_calc(rng: random.Random, world: World, split: str) -> dict:
    """Parallel calls to one specialised tool, then a calculation over the results."""
    kind = rng.choice(["temp_diff", "temp_avg", "precip_total", "share_diff"])
    regions = world.region_list(split)
    if kind == "temp_diff":
        c1, c2 = rng.sample(world.city_list(split), 2)
        day = _rand_date(rng)
        t1, t2 = (world.weather(c, day.isoformat())["temp_c"] for c in (c1, c2))
        if t2 > t1:
            c1, c2, t1, t2 = c2, c1, t2, t1
        steps = [call("get_weather", city=c, date=day.isoformat()) for c in (c1, c2)]
        expression, value = f"{t1} - {t2}", round(t1 - t2, 2)
        query = _pick(
            rng,
            ["How many degrees Celsius warmer was {a} than {b} on {date}?", "What was the temperature difference between {a} and {b} on {date}?"],
            ["On {date}, by how many degrees did {a}'s temperature exceed {b}'s?"],
            split,
        ).format(a=c1, b=c2, date=_say_date(rng, day))
    elif kind == "temp_avg":
        city = rng.choice(world.city_list(split))
        start = _rand_date(rng, margin=20)
        step = rng.choice((1, 2, 7))
        days = [start + _dt.timedelta(days=i * step) for i in range(3)]
        temps = [world.weather(city, d.isoformat())["temp_c"] for d in days]
        steps = [call("get_weather", city=city, date=d.isoformat()) for d in days]
        expression, value = f"({temps[0]} + {temps[1]} + {temps[2]}) / 3", round(sum(temps) / 3, 2)
        query = _pick(
            rng,
            ["What was the average temperature in {city} across {d1}, {d2} and {d3}?", "Average the temperatures in {city} on {d1}, {d2} and {d3}."],
            ["Take {city} on {d1}, {d2} and {d3}: what was the mean temperature over those three days?"],
            split,
        ).format(city=city, d1=days[0].isoformat(), d2=days[1].isoformat(), d3=days[2].isoformat())
    elif kind == "precip_total":
        city = rng.choice(world.city_list(split))
        d1 = _rand_date(rng, margin=20)
        days = [d1, d1 + _dt.timedelta(days=rng.choice((1, 3, 7)))]
        vals = [world.weather(city, d.isoformat())["precip_mm"] for d in days]
        steps = [call("get_weather", city=city, date=d.isoformat()) for d in days]
        expression, value = f"{vals[0]} + {vals[1]}", round(sum(vals), 2)
        query = _pick(
            rng,
            ["What was the total precipitation in {city} over {d1} and {d2}, in millimetres?", "Add up the precipitation in {city} on {d1} and {d2}."],
            ["How many millimetres of precipitation did {city} receive in total on {d1} and {d2}?"],
            split,
        ).format(city=city, d1=_say_date(rng, days[0]), d2=_say_date(rng, days[1]))
    else:
        r1, r2 = rng.sample(regions, 2)
        year = rng.choice(ELECTION_YEARS)
        s1, s2 = (world.election(r.name, year)["winner_vote_share_pct"] for r in (r1, r2))
        if s2 > s1:
            r1, r2, s1, s2 = r2, r1, s2, s1
        steps = [call("get_election", region=r.name, year=year) for r in (r1, r2)]
        expression, value = f"{s1} - {s2}", round(s1 - s2, 2)
        query = _pick(
            rng,
            ["By how many percentage points did the winner's vote share in {a} exceed the winner's vote share in {b} in the {year} elections?"],
            ["Compare the {year} elections in {a} and {b}: how many points higher was the winning vote share in {a}?"],
            split,
        ).format(a=r1.name, b=r2.name, year=year)
    aspect = {
        "name": f"calc:{kind}",
        "answers": [str(value)],
        "answer_type": "number",
        "evidence": [{"type": "call", **s} for s in steps] + [{"type": "calc", "value": value}],
        "multi_hop": False,
        "constraint_args": ["date", "year"],
    }
    return {"query": query, "aspects": [aspect], "optimal_plan": [steps, [call("calculator", expression=expression)]]}


# --------------------------------------------------------------- tool_bridge
def make_tool_bridge(rng: random.Random, world: World, split: str) -> dict:
    """Cross-tool multi-hop: the election result names the capital and the date for the weather call."""
    region = rng.choice(world.region_list(split))
    year = rng.choice(ELECTION_YEARS)
    election = world.election(region.name, year)
    weather = world.weather(election["capital"], election["election_date"])
    attr, answer_type, templates, eval_templates = rng.choice(
        [
            (
                "temp_c",
                "number",
                ["What was the temperature in the capital of {region} on the day of its {year} election?", "How warm was it in {region}'s capital on the day the {year} election was held?"],
                ["On the date of the {year} vote in {region}, what temperature was recorded in its capital city?"],
            ),
            (
                "condition",
                "text",
                ["What was the weather like in the capital of {region} on the day of its {year} election?"],
                ["Describe the weather in {region}'s capital on the day of the {year} election."],
            ),
            (
                "precip_mm",
                "number",
                ["How much precipitation fell in the capital of {region} on the day of its {year} election?"],
                ["On {region}'s {year} election day, how many millimetres of precipitation did its capital get?"],
            ),
        ]
    )
    hop1 = call("get_election", region=region.name, year=year)
    hop2 = call("get_weather", city=election["capital"], date=election["election_date"])
    aspect = {
        "name": f"bridge:{attr}",
        "answers": [str(weather[attr])],
        "answer_type": answer_type,
        "evidence": [{"type": "call", **hop1}, {"type": "call", **hop2}],
        "multi_hop": True,
        "constraint_args": ["year", "date"],
    }
    return {
        "query": _pick(rng, templates, eval_templates, split).format(region=region.name, year=year),
        "aspects": [aspect],
        "optimal_plan": [[hop1], [hop2]],
    }


# ------------------------------------------------------------------- no_tool
_NO_TOOL_TEMPLATES = [
    "Write a {length} poem about {topic} in a {tone} tone.",
    "Give me three synonyms for the word '{word}' and use one of them in a sentence about {topic}.",
    "Say hello and wish me a good {moment} in a {tone} way.",
    "Suggest a name for a {pet} that likes {topic}.",
    "Rewrite this sentence so it sounds more {tone}: 'hey, can u send me the {thing} asap?'",
    "Give me one tip for staying focused while {activity} in the {moment}.",
    "Write a one-sentence thank-you note for a gift of {thing}, in a {tone} tone.",
    "Come up with a {tone} title for a short story about a {pet} and {topic}.",
    "Write a {length} toast for a friend who loves {activity}.",
    "Brainstorm two ideas for a {tone} birthday card that mentions {thing}.",
]
_NO_TOOL_EVAL_TEMPLATES = [
    "Come up with a {tone} slogan for a shop that sells {thing}.",
    "Explain in one sentence why people enjoy {activity}, using the word '{word}'.",
]
_NO_TOOL_FILL = {
    "train": {
        "topic": ["autumn leaves", "the ocean", "old libraries", "morning coffee", "city lights", "mountains", "rain on a window", "a quiet garden", "paper boats", "a long walk"],
        "word": ["happy", "quick", "bright", "calm", "brave", "tiny", "smart", "strange", "warm", "quiet"],
        "moment": ["morning", "evening", "afternoon", "weekend"],
        "pet": ["cat", "dog", "parrot", "hamster", "pony"],
        "thing": ["report", "books", "tea", "flowers", "a scarf", "notes", "a mug", "postcards"],
        "activity": ["studying", "working from home", "reading", "writing", "running", "knitting"],
        "tone": ["playful", "formal", "cheerful", "dry", "warm", "dramatic"],
        "length": ["two-line", "four-line", "very short"],
    },
    "eval": {
        "topic": ["winter snow", "a lighthouse", "train journeys", "fresh bread", "starry skies", "a wooden bridge", "street markets"],
        "word": ["gentle", "loud", "ancient", "clever", "bold", "narrow", "eager"],
        "moment": ["night", "day", "holiday"],
        "pet": ["rabbit", "turtle", "goldfish", "ferret"],
        "thing": ["candles", "a notebook", "chocolate", "a map", "pencils", "a kite"],
        "activity": ["cooking", "hiking", "painting", "gardening", "sailing"],
        "tone": ["witty", "polite", "upbeat", "solemn", "friendly"],
        "length": ["three-line", "one-line", "brief"],
    },
}


def make_no_tool(rng: random.Random, world: World, split: str) -> dict:
    fill = _NO_TOOL_FILL["eval" if split == "eval" else "train"]
    template = _pick(rng, _NO_TOOL_TEMPLATES, _NO_TOOL_EVAL_TEMPLATES, split)
    return {"query": template.format(**{k: rng.choice(v) for k, v in fill.items()}), "aspects": [], "optimal_plan": []}


# ------------------------------------------------------------------ HotpotQA
def kb_items(rows: list[dict], corpus: Corpus, check_retrievable: bool = True) -> list[dict]:
    """HotpotQA rows that make well-posed tasks against this corpus.

    Kept only if there are exactly two gold paragraphs, the answer is short and
    visible in the (truncated) gold passages the tool returns, and each gold
    passage can be retrieved at all (searching its title returns it).
    """
    items = []
    for row in rows:
        titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
        if len(titles) != 2 or any(t not in corpus.by_title for t in titles):
            continue
        answer = row["answer"].strip()
        if not answer or len(answer.split()) > 6 or len(row["question"].split()) > 60:
            continue
        norm = normalize_text(answer)
        yesno = norm in ("yes", "no")
        if not yesno:
            visible = " ".join(normalize_text(f"{t} {truncate_words(corpus.by_title[t].text)}") for t in titles)
            if not norm or f" {norm} " not in f" {visible} ":
                continue
        if check_retrievable and any(corpus.by_title[t] not in corpus.search(t) for t in titles):
            continue
        if row["type"] == "bridge":
            # Reference plan in hop order: first the passage the question itself leads to.
            first_hits = corpus.search(row["question"])
            if corpus.by_title[titles[1]] in first_hits and corpus.by_title[titles[0]] not in first_hits:
                titles.reverse()
        items.append(
            {
                "id": row["id"],
                "question": row["question"].strip(),
                "answer": answer,
                "answer_type": "yesno" if yesno else "text",
                "type": row["type"],
                "titles": titles,
                "doc_ids": [corpus.by_title[t].doc_id for t in titles],
            }
        )
    return items


def _kb_part(item: dict) -> dict:
    searches = [call("knowledge_search", query=t) for t in item["titles"]]
    bridge = item["type"] == "bridge"
    return {
        "question": item["question"],
        "aspect": {
            "name": f"kb:{item['type']}",
            "answers": [item["answer"]],
            "answer_type": item["answer_type"],
            "evidence": [{"type": "doc", "doc_id": d, "title": t} for d, t in zip(item["doc_ids"], item["titles"])],
            "multi_hop": bridge,
            "constraint_args": [],
        },
        # Titles as queries are a reference plan, not the only correct one.
        "plan": [[s] for s in searches] if bridge else [searches],
        "hotpot_id": item["id"],
    }


def make_kb_task(item: dict) -> dict:
    part = _kb_part(item)
    return {"query": part["question"], "aspects": [part["aspect"]], "optimal_plan": part["plan"], "meta": {"hotpot_id": item["id"]}}


_COMPOSITE_JOIN = [
    "{kb} Also, {tool_lc}",
    "Two questions. First: {tool} Second: {kb}",
    "{tool} And separately: {kb}",
]
_COMPOSITE_JOIN_EVAL = ["I need two things answered. {kb} In addition, {tool_lc}"]


def make_composite(rng: random.Random, world: World, split: str, item: dict) -> dict:
    """A two-part ask: one part needs a specialised tool, the other needs the knowledge base."""
    tool_part = (_weather_part if rng.random() < 0.5 else _election_part)(rng, world, split, distract=False)
    kb_part = _kb_part(item)
    template = _pick(rng, _COMPOSITE_JOIN, _COMPOSITE_JOIN_EVAL, split)
    tool_q = tool_part["question"]
    query = template.format(kb=kb_part["question"], tool=tool_q, tool_lc=_lower_first(tool_q))
    # The specialised call goes in parallel with the first search.
    plan = [kb_part["plan"][0] + tool_part["plan"][0]] + kb_part["plan"][1:]
    return {
        "query": query,
        "aspects": [kb_part["aspect"], tool_part["aspect"]],
        "optimal_plan": plan,
        "meta": {"hotpot_id": item["id"], "kb_question": kb_part["question"]},
    }


# ----------------------------------------------------------------- assembly
_SYNTHETIC: dict[str, Callable[[random.Random, World, str], dict]] = {
    "weather_lookup": make_weather_lookup,
    "election_lookup": make_election_lookup,
    "fanout_calc": make_fanout_calc,
    "tool_bridge": make_tool_bridge,
    "no_tool": make_no_tool,
}


def _finish(task: dict, family: str, split: str, index: int) -> dict:
    task.update(
        task_id=f"{family}-{split}-{index:05d}",
        family=family,
        split=split,
        optimal_cost=plan_cost(task["optimal_plan"]),
    )
    task.setdefault("meta", {})
    return task


def build_split(
    split: str,
    n_per_family: int,
    world: World,
    kb_pool: list[dict],
    seed: int,
) -> list[dict]:
    """Build `n_per_family` tasks of each family for one split ("sft", "grpo" or "eval").

    `kb_pool` are the HotpotQA items reserved for this split. They are divided
    between kb_bridge, kb_compare and composite without overlap. If the pool is
    too small for a family, that family simply gets fewer tasks.
    """
    world_split = "eval" if split == "eval" else "train"
    tasks: list[dict] = []
    seen: set[str] = set()

    for family, maker in _SYNTHETIC.items():
        rng = random.Random(f"{seed}|{split}|{family}")
        made, attempts = 0, 0
        while made < n_per_family and attempts < n_per_family * 50:
            attempts += 1
            task = maker(rng, world, world_split)
            if task["query"] in seen:
                continue
            seen.add(task["query"])
            tasks.append(_finish(task, family, split, made))
            made += 1

    rng = random.Random(f"{seed}|{split}|kb")
    pool = list(kb_pool)
    rng.shuffle(pool)
    bridge = [i for i in pool if i["type"] == "bridge"]
    compare = [i for i in pool if i["type"] == "comparison"]
    take_compare, compare = compare[:n_per_family], compare[n_per_family:]
    take_bridge, bridge = bridge[:n_per_family], bridge[n_per_family:]
    for i, item in enumerate(take_bridge):
        tasks.append(_finish(make_kb_task(item), "kb_bridge", split, i))
    for i, item in enumerate(take_compare):
        tasks.append(_finish(make_kb_task(item), "kb_compare", split, i))
    # Composite: half comparison (one parallel round), half bridge (two rounds), as available.
    half = n_per_family // 2
    comp_items = compare[:half] + bridge[: n_per_family - min(half, len(compare))]
    rng_c = random.Random(f"{seed}|{split}|composite")
    for i, item in enumerate(comp_items[:n_per_family]):
        tasks.append(_finish(make_composite(rng_c, world, world_split, item), "composite", split, i))
    return tasks


def split_kb_pools(train_items: list[dict], val_items: list[dict]) -> dict[str, list[dict]]:
    """Disjoint HotpotQA pools: SFT demonstrations (1/3 of train), GRPO training (2/3), held-out evaluation."""
    sft = train_items[0::3]
    grpo = [item for i, item in enumerate(train_items) if i % 3 != 0]
    return {"sft": sft, "grpo": grpo, "eval": list(val_items)}


def family_counts(tasks: list[dict]) -> dict[str, int]:
    counts = {f: 0 for f in FAMILIES}
    for t in tasks:
        counts[t["family"]] += 1
    return counts


def grpo_rows(tasks: list[dict], seed: int) -> list[dict]:
    """Rows for TRL's GRPOTrainer: `prompt` is the chat prompt, `task` (JSON) reaches the reward function."""
    import json

    rows = [{"prompt": build_prompt(t), "task": json.dumps(t, ensure_ascii=False)} for t in tasks]
    random.Random(seed).shuffle(rows)
    return rows
