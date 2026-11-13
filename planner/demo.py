"""Demo client: run the planner against any OpenAI-compatible endpoint.

The model is served elsewhere (Unsloth `unsloth run`, llama.cpp `llama-server
--jinja`, or vLLM). This client owns the tool loop: it sends the four tool
schemas, executes the calls the model asks for against the local mock world and
knowledge base, feeds the results back, and stops after at most three tool
rounds. That split is required because hosted chat UIs can only run their own
built-in tools, not ours.

CLI:
    python -m planner.demo --base-url http://localhost:8000/v1 --api-key sk-... --example 1
    python -m planner.demo --base-url ... --query "Who won the 2024 election in Norland?"
"""

from __future__ import annotations

import argparse
import json
import os

from .bench import run_rollouts
from .scoring import score_messages
from .tasks import build_prompt
from .tools import MAX_TOOL_ROUNDS, TOOL_COSTS, TOOL_SCHEMAS
from .trace import clean_text, parse_arguments, parse_assistant_text, trace_from_messages

EXAMPLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "examples.json")


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Our message dicts -> OpenAI wire format (call ids, JSON-string arguments)."""
    out, pending, counter = [], [], 0
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                fn = call["function"]
                call_id = call.get("id") or f"call_{counter}"
                counter += 1
                arguments = fn.get("arguments")
                calls.append({"id": call_id, "type": "function", "function": {"name": fn.get("name") or "", "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
            pending = [c["id"] for c in calls]
            out.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
        elif role == "tool":
            call_id = pending.pop(0) if pending else f"call_{counter}"
            out.append({"role": "tool", "tool_call_id": call_id, "content": str(message.get("content", ""))})
        else:
            out.append({"role": role, "content": message.get("content") or ""})
    return out


class OpenAIBackend:
    """Backend for `bench.run_rollouts` that talks to an OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str | None = None, temperature: float = 0.0, max_tokens: int = 512, client=None, extra_body: dict | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.client = client
        self.model = model or self._first_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body  # e.g. {"chat_template_kwargs": {"enable_thinking": False}} for vLLM

    def _first_model(self) -> str:
        return self.client.models.list().data[0].id

    def _one(self, messages: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=to_openai_messages(messages),
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,  # required behind a Cloudflare quick tunnel
            **({"extra_body": self.extra_body} if self.extra_body else {}),
        )
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls and "<tool_call>" in content:
            return parse_assistant_text(content)  # server did not parse the calls for us
        reply = {"role": "assistant", "content": clean_text(content)}
        if tool_calls:
            reply["tool_calls"] = [
                {"type": "function", "function": {"name": c.function.name, "arguments": parse_arguments(c.function.arguments) if parse_arguments(c.function.arguments) is not None else c.function.arguments}}
                for c in tool_calls
            ]
        return reply

    def __call__(self, batch: list[list[dict]]) -> list[dict]:
        return [self._one(messages) for messages in batch]


def answer(query_or_task, backend) -> dict:
    """Run one query (a string, or a task dict with gold labels) and describe what happened."""
    task = query_or_task if isinstance(query_or_task, dict) else {"query": query_or_task}
    completion = run_rollouts([build_prompt(task)], backend)[0]
    trace = trace_from_messages(completion)
    result = {
        "query": task["query"],
        "completion": completion,
        "final_message": trace.final_message,
        "cost": sum(TOOL_COSTS.get(c.name, 0.25) for c in trace.calls),
        "rounds": len(trace.rounds),
    }
    if "aspects" in task:
        result["score"] = score_messages(task, completion).to_dict()
    return result


def _short(value, limit: int = 600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " ..."


def format_markdown(result: dict, title: str | None = None) -> str:
    """Readable trace: each tool round, the tool results, the final answer, and the score if known."""
    lines = [f"### {title}"] if title else []
    trace = trace_from_messages(result["completion"])
    for i, calls in enumerate(trace.rounds, 1):
        lines.append(f"**Round {i}** ({len(calls)} call{'s' if len(calls) != 1 else ''}{', parallel' if len(calls) > 1 else ''})")
        for call in calls:
            lines.append(f"- `{call.name}({_short(call.args, 200)})`")
            lines.append(f"  - {'error: ' + call.error if call.error else _short(call.result)}")
    if trace.pending_calls:
        lines.append(f"**Stopped:** the model asked for {trace.pending_calls} more call(s) after {MAX_TOOL_ROUNDS} tool rounds.")
    lines.append(f"**Final:** {trace.final_message if trace.final_message else '(no final answer)'}")
    lines.append(f"_Tool cost: {result['cost']:.2f} in {result['rounds']} round(s)_")
    score = result.get("score")
    if score:
        verdict = "good" if score["good"] else "bad"
        lines.append(f"**Score:** {verdict} - relevance {score['relevance']}/4, factuality {score['factuality']}/4" + (f", tags: {', '.join(score['tags'])}" if score["tags"] else ""))
    return "\n\n".join(lines)


def format_expected(task: dict) -> str:
    plan = " -> ".join("[" + ", ".join(f"{c['tool']}({_short(c['args'], 80)})" for c in calls) + "]" for calls in task.get("optimal_plan", []))
    answers = "; ".join(a["answers"][0] for a in task.get("aspects", []))
    return f"**Reference plan:** {plan or '(no tools)'}\n\n**Reference answer:** {answers or '(free text)'}"


def load_examples(path: str = EXAMPLES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_ui(backends: dict, examples: list[dict] | None = None):
    """Gradio app: pick a preset hard query or type one, compare the served checkpoints side by side."""
    import gradio as gr

    examples = load_examples() if examples is None else examples
    by_label = {f"{i + 1}. {e['label']}": e for i, e in enumerate(examples)}
    names = list(backends)

    def run(example_label: str, query: str, selected: list[str]):
        example = by_label.get(example_label)
        task = example["task"] if example and query.strip() == example["task"]["query"] else None
        target = task or query.strip()
        if not target:
            return "Type a question or pick an example."
        sections = [format_expected(task)] if task else []
        for name in selected or names:
            try:
                sections.append(format_markdown(answer(target, backends[name]), title=name))
            except Exception as exc:  # noqa: BLE001 - show endpoint errors in the UI
                sections.append(f"### {name}\n\nRequest failed: {exc}")
        return "\n\n---\n\n".join(sections)

    with gr.Blocks(title="Tool-calling planner demo") as ui:
        gr.Markdown(
            "# Tool-calling planner\n"
            "The planner may use four tools - `knowledge_search`, `calculator`, `get_weather`, `get_election` - "
            f"for at most {MAX_TOOL_ROUNDS} rounds, then answers. Weather and election data are a fictional world; "
            "`knowledge_search` runs over a local Wikipedia-paragraph index."
        )
        picker = gr.Dropdown(choices=list(by_label), label="Preset hard queries", value=None)
        query = gr.Textbox(label="Question", lines=3)
        models = gr.CheckboxGroup(choices=names, value=names, label="Models")
        button = gr.Button("Run", variant="primary")
        output = gr.Markdown()
        picker.change(lambda label: by_label[label]["task"]["query"] if label in by_label else "", picker, query)
        button.click(run, [picker, query, models], output)
    return ui


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Query a served planner model with the local tools.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("PLANNER_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=None, help="served model name (default: the first one the server lists)")
    parser.add_argument("--query", default=None)
    parser.add_argument("--example", type=int, default=None, help="1-based index into demo/examples.json")
    parser.add_argument("--hotpot-train", type=int, default=None, help="override the corpus size (rows of HotpotQA train)")
    parser.add_argument("--hotpot-val", type=int, default=None)
    parser.add_argument("--corpus", default=None, help="path to a saved corpus.jsonl (skips the HotpotQA download)")
    args = parser.parse_args(argv)

    from .setup_env import setup_environment

    setup_environment(corpus_path=args.corpus, n_train=args.hotpot_train, n_val=args.hotpot_val)
    backend = OpenAIBackend(args.base_url, args.api_key, args.model)
    if args.example is not None:
        task = load_examples()[args.example - 1]["task"]
        print(format_expected(task), "\n")
        target = task
    elif args.query:
        target = args.query
    else:
        parser.error("give --query or --example")
    print(format_markdown(answer(target, backend), title=backend.model))


if __name__ == "__main__":
    main()
