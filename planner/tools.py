"""The four planner tools.

They are plain, stateless Python functions with type hints and Google-style
docstrings, which is the form TRL's `GRPOTrainer(tools=[...])` expects (it
derives the JSON schema from the signature and docstring). The same functions
back the benchmark loop and the demo.

Every tool returns a JSON string. Bad arguments raise an exception: the caller
(TRL, `bench.py` or `demo.py`) turns it into an `{"error": ...}` tool message
that the model can see and react to.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re

from .corpus import TOP_K, Corpus
from .world import World

# Relative cost of one call. knowledge_search stands in for the expensive,
# QPS-limited web retrieval tool; the others are cheap specialised lookups.
TOOL_COSTS = {
    "knowledge_search": 1.0,
    "calculator": 0.25,
    "get_weather": 0.25,
    "get_election": 0.25,
}
UNKNOWN_TOOL_COST = 0.25

# Per-query budget, shared by the trainer, the benchmark and the demo.
MAX_TOOL_ROUNDS = 3
MAX_CALLS_PER_ROUND = 4
MAX_SEARCH_CALLS = 4

_STATE: dict = {"world": None, "corpus": None}


def configure(world: World | None = None, corpus: Corpus | None = None) -> None:
    """Bind the tools to a world and a corpus (call once before using the tools)."""
    if world is not None:
        _STATE["world"] = world
    if corpus is not None:
        _STATE["corpus"] = corpus


def _require(name: str):
    value = _STATE[name]
    if value is None:
        raise RuntimeError(f"tools are not configured: call planner.tools.configure({name}=...) first")
    return value


# --------------------------------------------------------------------- tools
def knowledge_search(query: str) -> str:
    """
    Search the knowledge base for encyclopedic facts about real people, places, works and events.

    Args:
        query: A specific search query, for example the name of an entity plus the fact you need.

    Returns:
        A JSON list of the top passages, each with a doc_id, a title and the passage text.
    """
    corpus: Corpus = _require("corpus")
    return json.dumps([p.public() for p in corpus.search(query, k=TOP_K)], ensure_ascii=False)


def calculator(expression: str) -> str:
    """
    Evaluate an arithmetic expression. Use it for any calculation instead of mental math.

    Args:
        expression: An arithmetic expression such as "(21.5 + 18.0 + 19.4) / 3" or "abs(12.3 - 7.9)".

    Returns:
        A JSON object with the numeric result.
    """
    return json.dumps({"result": safe_eval(expression)})


def get_weather(city: str, date: str) -> str:
    """
    Get the recorded weather for a city on a given date.

    Args:
        city: The name of the city.
        date: The date in YYYY-MM-DD format.

    Returns:
        A JSON object with the temperature in Celsius, the weather condition and the precipitation in millimetres.
    """
    world: World = _require("world")
    return json.dumps(world.weather(city, date))


def get_election(region: str, year: int) -> str:
    """
    Get the result of a regional election.

    Args:
        region: The name of the region.
        year: The election year, for example 2024.

    Returns:
        A JSON object with the winner, parties, vote shares, turnout, election date and the region's capital city.
    """
    world: World = _require("world")
    return json.dumps(world.election(region, year))


TOOLS = [knowledge_search, calculator, get_weather, get_election]
TOOL_REGISTRY = {fn.__name__: fn for fn in TOOLS}

# OpenAI-style schemas for clients that cannot introspect Python functions
# (the OpenAI-compatible demo client, SFT rendering outside TRL). A unit test
# checks they match what transformers derives from the functions above.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "Search the knowledge base for encyclopedic facts about real people, places, works and events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific search query, for example the name of an entity plus the fact you need.",
                    }
                },
                "required": ["query"],
            },
            "return": {
                "type": "string",
                "description": "A JSON list of the top passages, each with a doc_id, a title and the passage text.",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression. Use it for any calculation instead of mental math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": 'An arithmetic expression such as "(21.5 + 18.0 + 19.4) / 3" or "abs(12.3 - 7.9)".',
                    }
                },
                "required": ["expression"],
            },
            "return": {"type": "string", "description": "A JSON object with the numeric result."},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the recorded weather for a city on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city."},
                    "date": {"type": "string", "description": "The date in YYYY-MM-DD format."},
                },
                "required": ["city", "date"],
            },
            "return": {
                "type": "string",
                "description": "A JSON object with the temperature in Celsius, the weather condition and the precipitation in millimetres.",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_election",
            "description": "Get the result of a regional election.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "The name of the region."},
                    "year": {"type": "integer", "description": "The election year, for example 2024."},
                },
                "required": ["region", "year"],
            },
            "return": {
                "type": "string",
                "description": "A JSON object with the winner, parties, vote shares, turnout, election date and the region's capital city.",
            },
        },
    },
]


def execute_tool(name: object, arguments: object) -> str:
    """Run one tool call and always return a string: the result, or `{"error": ...}`.

    Mirrors what TRL's tool loop does (unknown tool or exception -> error message
    sent back to the model instead of crashing).
    """
    try:
        if name not in TOOL_REGISTRY:
            raise ValueError(f"Tool {name} not found.")
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        return TOOL_REGISTRY[name](**arguments)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a tool message
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------- calculator
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


def safe_eval(expression: object) -> float | int:
    """Evaluate arithmetic without `eval`: numbers, + - * / // % **, parentheses and a few functions."""
    text = str(expression).strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)  # 1,234,567 -> 1234567 (keeps "min(1, 2)")
    if not text:
        raise ValueError("expression must be a non-empty string")
    if len(text) > 200:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid arithmetic expression: {text!r}") from exc

    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1e6):
                raise ValueError("exponent is too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression: {text!r}")

    try:
        value = ev(tree)
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result is not finite")
        value = round(value, 6)
        if value == int(value) and abs(value) < 1e15:
            value = int(value)
    return value
