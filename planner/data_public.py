"""Converters from public function-calling datasets to chat messages with `tool_calls`.

These supply the general function-calling half of the SFT mix (arbitrary
schemas, parallel calls, multi-turn, declining to call), which is what BFCL
measures. Both converters return `{"messages": [...], "tools": [...], "source": ...}`
or raise `ValueError` for rows that do not parse cleanly (those are dropped).

- NousResearch/hermes-function-calling-v1 ("func_calling"): same conversion as the
  Unsloth Qwen3 tool-calling notebook.
- Team-ACE/ToolACE: tool calls are written as `[Name(arg=value), ...]` and the tool
  list sits inside the system prompt; both are rewritten to JSON.
"""

from __future__ import annotations

import ast
import json
import random
import re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_TOOLACE_MARKER = "Here is a list of functions in JSON format that you can invoke"


def _tool_call(name: str, arguments: dict) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _check(messages: list[dict], tools: list[dict]) -> None:
    """Reject conversations that would teach the wrong thing."""
    if not tools:
        raise ValueError("no tools")
    if not messages or messages[0]["role"] != "user":
        raise ValueError("conversation must start with a user turn")
    if messages[-1]["role"] != "assistant":
        raise ValueError("conversation must end with an assistant turn")
    names = {t["function"]["name"] for t in tools}
    previous = None
    for m in messages:
        if m["role"] == "tool" and previous not in ("assistant_call", "tool"):
            raise ValueError("tool result without a preceding tool call")
        if m["role"] == "assistant":
            for call in m.get("tool_calls", []):
                if call["function"]["name"] not in names:
                    raise ValueError(f"call to undeclared tool {call['function']['name']!r}")
            previous = "assistant_call" if m.get("tool_calls") else "assistant"
        else:
            previous = m["role"]


# -------------------------------------------------------------------- Hermes
def _loads_lenient(text: str):
    """JSON, or a Python literal (some rows write the call with single quotes)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(text.replace("\\n", "\n"))
    except (ValueError, SyntaxError) as exc:
        raise ValueError("tool call is neither JSON nor a Python literal") from exc
    json.dumps(value)  # must be JSON-serialisable
    return value


def convert_hermes(row: dict) -> dict:
    messages = []
    for turn in row["conversations"]:
        role, text = turn["from"], turn["value"]
        if role == "system":
            continue  # the chat template writes its own tool instructions
        if role == "human":
            messages.append({"role": "user", "content": text})
        elif role == "gpt":
            calls = [_loads_lenient(c) for c in TOOL_CALL_RE.findall(text)]
            message = {"role": "assistant", "content": TOOL_CALL_RE.sub("", text).strip()}
            if calls:
                if any(not isinstance(c, dict) or not isinstance(c.get("arguments", {}), dict) or "name" not in c for c in calls):
                    raise ValueError("malformed tool call")
                message["tool_calls"] = [_tool_call(c["name"], c.get("arguments", {})) for c in calls]
            messages.append(message)
        elif role == "tool":
            for response in TOOL_RESPONSE_RE.findall(text):
                messages.append({"role": "tool", "content": response})
        else:
            raise ValueError(f"unknown role {role!r}")
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()  # some rows stop right after the tool results
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    _check(messages, tools)
    return {"messages": messages, "tools": tools, "source": "hermes"}


# ------------------------------------------------------------------- ToolACE
def _fix_schema(node):
    """ToolACE writes JSON-schema objects as "dict"; normalise recursively."""
    if isinstance(node, dict):
        out = {k: _fix_schema(v) for k, v in node.items()}
        if out.get("type") == "dict":
            out["type"] = "object"
        return out
    if isinstance(node, list):
        return [_fix_schema(v) for v in node]
    return node


def parse_toolace_tools(system: str) -> list[dict]:
    start = system.find(_TOOLACE_MARKER)
    if start < 0:
        raise ValueError("tool list marker not found")
    start = system.find("[", start)
    try:
        raw, _ = json.JSONDecoder().raw_decode(system[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("tool list is not JSON") from exc
    tools = []
    for fn in raw:
        if not isinstance(fn, dict) or "name" not in fn:
            raise ValueError("malformed tool definition")
        parameters = _fix_schema(fn.get("parameters") or {"type": "object", "properties": {}})
        tools.append({"type": "function", "function": {"name": fn["name"], "description": fn.get("description", ""), "parameters": parameters}})
    return tools


def _split_top_level(text: str) -> list[str]:
    """Split `A(x=1), B(y=[1, 2])` on the commas between calls."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    if quote or depth != 0:
        raise ValueError("unbalanced call list")
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


_JSON_NAMES = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}


def _literal(node: ast.AST):
    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    return ast.literal_eval(node)


def parse_bracket_calls(text: str, tool_names: list[str] | None = None) -> list[dict]:
    """Parse ToolACE's `[Name with spaces(arg="v", n=1), Other()]` into tool calls.

    Tool names can themselves contain parentheses or apostrophes
    ("User Feed (Video Posts) V2", "Get Agent's Listings"), so known names are
    swapped for placeholders before the call list is split and parsed.
    """
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("not a bracketed call list")
    body = text[1:-1]
    placeholders = {}
    for i, name in enumerate(sorted(tool_names or [], key=len, reverse=True)):
        token = f"__tool{i}__"
        if name + "(" in body:
            body = body.replace(name + "(", token + "(")
            placeholders[token] = name
    calls = []
    for part in _split_top_level(body):
        open_paren = part.find("(")
        if open_paren <= 0 or not part.endswith(")"):
            raise ValueError(f"not a call: {part[:40]!r}")
        name = part[:open_paren].strip()
        name = placeholders.get(name, name)
        try:
            node = ast.parse(f"f({part[open_paren + 1 : -1]})", mode="eval").body
            if node.args:
                raise ValueError("positional arguments are not supported")
            arguments = {kw.arg: _literal(kw.value) for kw in node.keywords}
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"cannot parse arguments of {name!r}") from exc
        if None in arguments:
            raise ValueError("**kwargs are not supported")
        calls.append(_tool_call(name, arguments))
    if not calls:
        raise ValueError("empty call list")
    return calls


def convert_toolace(row: dict) -> dict:
    tools = parse_toolace_tools(row["system"])
    names = [t["function"]["name"] for t in tools]
    messages = []
    for turn in row["conversations"]:
        role, text = turn["from"], turn["value"]
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "assistant":
            stripped = text.strip()
            if stripped.startswith("[") and stripped.endswith("]") and "(" in stripped:
                messages.append({"role": "assistant", "content": "", "tool_calls": parse_bracket_calls(stripped, names)})
            else:
                messages.append({"role": "assistant", "content": stripped})
        elif role == "tool":
            try:
                results = json.loads(text)
            except json.JSONDecodeError:
                results = None
            if isinstance(results, list) and results and all(isinstance(r, dict) and "name" in r for r in results):
                for r in results:
                    messages.append({"role": "tool", "name": r["name"], "content": json.dumps(r.get("results"), ensure_ascii=False)})
            else:
                messages.append({"role": "tool", "content": text})
        else:
            raise ValueError(f"unknown role {role!r}")
    if any(m["role"] == "assistant" and not m.get("tool_calls") and not m["content"] for m in messages):
        raise ValueError("empty assistant turn")
    _check(messages, tools)
    return {"messages": messages, "tools": tools, "source": "toolace"}


# ---------------------------------------------------------------- assembling
def convert_all(rows, converter) -> tuple[list[dict], int]:
    """Convert rows, dropping the ones that fail; return (examples, number dropped)."""
    out, dropped = [], 0
    for row in rows:
        try:
            out.append(converter(row))
        except (ValueError, KeyError, TypeError):
            dropped += 1
    return out, dropped


def load_public_examples(n_toolace: int, n_hermes: int, seed: int = 3407) -> list[dict]:
    """Download, convert and subsample the public function-calling data."""
    from datasets import load_dataset

    rng = random.Random(seed)
    examples = []
    for name, config, converter, n in (
        ("Team-ACE/ToolACE", None, convert_toolace, n_toolace),
        ("NousResearch/hermes-function-calling-v1", "func_calling", convert_hermes, n_hermes),
    ):
        if n <= 0:
            continue
        converted, dropped = convert_all(load_dataset(name, config, split="train"), converter)
        rng.shuffle(converted)
        print(f"{name}: {len(converted)} usable rows ({dropped} dropped), taking {min(n, len(converted))}")
        examples.extend(converted[:n])
    return examples


def render_examples(examples: list[dict], chat_format, max_tokens: int) -> tuple[list[str], int]:
    """Render examples to training text, dropping ones that fail or exceed `max_tokens`."""
    texts, dropped = [], 0
    tokenizer = chat_format.tokenizer
    for ex in examples:
        try:
            text = chat_format.render(ex["messages"], tools=ex["tools"])
        except Exception:  # noqa: BLE001 - template errors on odd rows
            dropped += 1
            continue
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_tokens:
            dropped += 1
            continue
        texts.append(text)
    return texts, dropped
