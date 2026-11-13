"""Turn a rollout (a list of chat messages) into a structured trace.

The same message format comes out of three places: TRL's GRPO tool loop
(`completions`), our benchmark loop and the demo client. A rollout looks like

    assistant(tool_calls=[...]) -> tool, tool -> assistant(tool_calls) -> tool -> assistant("Answer: ...")

Scoring only trusts `role == "tool"` messages for evidence, because those are
written by Python, never by the model.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
SPECIAL_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>")
ANSWER_RE = re.compile(r"answer\s*:", re.IGNORECASE)


@dataclass
class Call:
    name: str | None
    args: dict | None  # None when the arguments were not a JSON object
    round: int
    result: Any = None  # parsed tool output
    error: str | None = None  # set when the tool returned {"error": ...}
    executed: bool = False


@dataclass
class Trace:
    rounds: list[list[Call]] = field(default_factory=list)
    final_message: str | None = None  # last assistant turn without tool calls
    pending_calls: int = 0  # calls the model asked for after the round budget ran out

    @property
    def calls(self) -> list[Call]:
        return [c for r in self.rounds for c in r]

    @property
    def final_answer(self) -> str | None:
        return None if self.final_message is None else extract_answer(self.final_message)

    @property
    def has_answer_marker(self) -> bool:
        return self.final_message is not None and ANSWER_RE.search(self.final_message) is not None


def clean_text(text: str | None) -> str:
    """Drop thinking blocks and chat special tokens."""
    return SPECIAL_RE.sub("", THINK_RE.sub("", text or "")).strip()


def extract_answer(text: str) -> str:
    """Text after the last `Answer:` marker, or the whole message if there is none."""
    text = clean_text(text)
    matches = list(ANSWER_RE.finditer(text))
    return text[matches[-1].end() :].strip() if matches else text


def parse_arguments(raw: Any) -> dict | None:
    """Tool arguments as a dict, accepting a dict or a JSON string; None if malformed."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return {} if raw is None else None


def parse_assistant_text(text: str) -> dict:
    """Parse raw generated text (Hermes/Qwen `<tool_call>` format) into an assistant message.

    A `<tool_call>` block that is not valid JSON is kept as a call with
    `name=None`, so it is executed as an error and counted as an invalid call
    instead of silently disappearing.
    """
    text = SPECIAL_RE.sub("", text or "")
    visible = THINK_RE.sub("", text)
    tool_calls = []
    for raw in TOOL_CALL_RE.findall(visible):
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("tool call is not an object")
            name, arguments = obj.get("name"), obj.get("arguments", {})
        except (json.JSONDecodeError, ValueError):
            name, arguments = None, raw
        tool_calls.append({"type": "function", "function": {"name": name, "arguments": arguments}})
    message = {"role": "assistant", "content": TOOL_CALL_RE.sub("", visible).strip()}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def parse_tool_content(content: Any) -> tuple[Any, str | None]:
    """Parse a tool message's content; return (value, error message or None)."""
    value = content
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            try:  # TRL writes a failed call as str({"error": ...})
                value = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                value = content
    if isinstance(value, dict) and "error" in value:
        return value, str(value["error"])
    return value, None


def _message_tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    out = []
    for call in calls:
        fn = call.get("function", call) if isinstance(call, dict) else {}
        out.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    return out


def trace_from_messages(messages: list[dict]) -> Trace:
    """Build a Trace from chat messages (system/user messages are ignored)."""
    trace = Trace()
    current: list[Call] | None = None
    slot = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            calls = _message_tool_calls(message)
            # Some stacks leave tool calls as raw text in `content`.
            if not calls and "<tool_call>" in (message.get("content") or ""):
                calls = _message_tool_calls(parse_assistant_text(message["content"]))
            if calls:
                current = [Call(name=c["name"], args=parse_arguments(c["arguments"]), round=len(trace.rounds)) for c in calls]
                trace.rounds.append(current)
                slot = 0
                trace.final_message = None
            else:
                current = None
                trace.final_message = clean_text(message.get("content"))
        elif role == "tool" and current is not None and slot < len(current):
            call = current[slot]
            slot += 1
            call.result, call.error = parse_tool_content(message.get("content"))
            call.executed = True
    # A last round with no tool results was never executed (round budget ran out).
    if trace.rounds and not any(c.executed for c in trace.rounds[-1]):
        trace.pending_calls = len(trace.rounds.pop())
        trace.final_message = None
    return trace
