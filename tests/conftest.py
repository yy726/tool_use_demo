import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner import tools  # noqa: E402
from planner.corpus import Corpus  # noqa: E402
from planner.world import World  # noqa: E402


def hotpot_row(qid, question, answer, qtype, gold, distractors):
    """A row in HotpotQA's "distractor" format. `gold`/`distractors` are (title, text) pairs."""
    paragraphs = list(gold) + list(distractors)
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": qtype,
        "level": "hard",
        "supporting_facts": {"title": [t for t, _ in gold], "sent_id": [0] * len(gold)},
        "context": {"title": [t for t, _ in paragraphs], "sentences": [[text] for _, text in paragraphs]},
    }


DISTRACTORS = [
    ("Granite Peak", "Granite Peak is a mountain in the northern range, popular with climbers."),
    ("River Lark", "The River Lark flows through farmland and joins the sea near a small harbour."),
    ("Copper Kettle Inn", "The Copper Kettle Inn is a historic tavern known for its Sunday roast."),
    ("Saffron Walden Fair", "The fair is an annual market with craft stalls and folk music."),
]

ROWS = [
    hotpot_row(
        "q-bridge",
        "Which city is the birthplace of the director of the film Silver Lantern?",
        "Marlow Bay",
        "bridge",
        [
            ("Silver Lantern", "Silver Lantern is a 1987 drama film directed by Odette Varnier and filmed on location."),
            ("Odette Varnier", "Odette Varnier is a film director who was born in Marlow Bay and trained as a painter."),
        ],
        DISTRACTORS,
    ),
    hotpot_row(
        "q-compare",
        "Were Tobias Quill and Henrietta Moss both novelists?",
        "yes",
        "comparison",
        [
            ("Tobias Quill", "Tobias Quill was a novelist best known for his sea stories."),
            ("Henrietta Moss", "Henrietta Moss was a novelist and essayist from the lake country."),
        ],
        DISTRACTORS,
    ),
    hotpot_row(
        "q-bridge-2",
        "What instrument did the composer of the opera Glass Orchard play?",
        "cello",
        "bridge",
        [
            ("Glass Orchard", "Glass Orchard is an opera in two acts composed by Lucan Ferro."),
            ("Lucan Ferro", "Lucan Ferro was a composer who played the cello in his youth."),
        ],
        DISTRACTORS,
    ),
    hotpot_row(
        "q-compare-2",
        "Which was founded first, Halden Press or Wren & Sons?",
        "Wren & Sons",
        "comparison",
        [
            ("Halden Press", "Halden Press is a publishing house founded in 1921."),
            ("Wren & Sons", "Wren & Sons is a bookbinding company founded in 1874."),
        ],
        DISTRACTORS,
    ),
]


@pytest.fixture(scope="session")
def rows():
    return ROWS


@pytest.fixture(scope="session")
def world():
    return World(seed=3407)


@pytest.fixture(scope="session")
def corpus():
    corpus = Corpus()
    for row in ROWS:
        corpus.add_hotpot_row(row)
    return corpus


@pytest.fixture(autouse=True)
def configured_tools(world, corpus):
    tools.configure(world=world, corpus=corpus)
    yield


# ---- helpers to hand-build rollouts -------------------------------------
def assistant_calls(*calls):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}} for name, args in calls],
    }


def run(*rounds, final=None):
    """Build a rollout by actually executing the given rounds of (tool, args) calls."""
    messages = []
    for calls in rounds:
        messages.append(assistant_calls(*calls))
        for name, args in calls:
            messages.append({"role": "tool", "name": name, "content": tools.execute_tool(name, args)})
    if final is not None:
        messages.append({"role": "assistant", "content": final})
    return messages
