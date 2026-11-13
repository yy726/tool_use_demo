"""Teacher demonstrations by rejection sampling.

The scripted expert only solves the tasks its fixed strategy happens to solve
(about 70% of the knowledge-base multi-hop families), so SFT on its rollouts
is capped there. Here a stronger model attempts the tasks the expert failed:
it runs the same tool loop against the same environment, each attempt is
scored by `scoring.py`, and only perfect rollouts (4 on both axes) are kept.
Failed tasks are retried up to `samples` times at a non-zero temperature.

The teacher is any OpenAI-compatible endpoint. `generate_teacher_traces` can
start one with vLLM, or use an endpoint you already run.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

from .bench import run_rollouts
from .scoring import score_messages
from .tasks import build_prompt
from .tools import TOOL_SCHEMAS
from .trace import extract_answer

# Families where the scripted expert leaves tasks unsolved (it is perfect on the purely synthetic ones).
TEACHER_FAMILIES = ("kb_bridge", "kb_compare", "composite", "dual_bridge")


def teacher_targets(tasks: Sequence[dict], expert_traces: Sequence[dict], families: Sequence[str] = TEACHER_FAMILIES) -> list[dict]:
    """Tasks of the teacher families that have no perfect expert demonstration."""
    solved = {t["task_id"] for t in expert_traces}
    return [t for t in tasks if t["family"] in families and t["task_id"] not in solved]


def parallel_backend(backend: Callable[[list[list[dict]]], list[dict]], workers: int = 32) -> Callable[[list[list[dict]]], list[dict]]:
    """Send the conversations of a batch concurrently. A failed request ends that rollout (it is then rejected)."""

    def one(messages: list[dict]) -> dict:
        try:
            return backend([messages])[0]
        except Exception:  # noqa: BLE001 - timeouts, context overflow, malformed replies
            return {"role": "assistant", "content": ""}

    def call(batch: list[list[dict]]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(one, batch))

    return call


def as_demonstration(completion: list[dict]) -> list[dict]:
    """A rollout in the shape of the expert's demonstrations: bare tool calls, then `Answer: ...`.

    The teacher's side remarks and any reasoning are dropped, so the student
    learns the plan and the short answer format, not the teacher's prose.
    """
    out = []
    for message in completion:
        if message["role"] == "tool":
            out.append({"role": "tool", "name": message.get("name"), "content": message["content"]})
        elif message.get("tool_calls"):
            calls = [{"type": "function", "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}} for c in message["tool_calls"]]
            out.append({"role": "assistant", "content": "", "tool_calls": calls})
        else:
            out.append({"role": "assistant", "content": "Answer: " + extract_answer(message.get("content") or "")})
    return out


def collect_teacher_traces(tasks: Sequence[dict], backend, samples: int = 4, batch_size: int = 64, verbose: bool = True) -> tuple[list[dict], dict]:
    """Rejection sampling: up to `samples` attempts per task, keep the first rollout that scores 4.

    Returns `(traces, stats)`; traces have the format of `expert.build_expert_traces`
    plus `"source": "teacher"`.
    """
    pending = list(tasks)
    traces: list[dict] = []
    attempts = 0
    for attempt in range(samples):
        if not pending:
            break
        still_failing = []
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            completions = run_rollouts([build_prompt(t) for t in chunk], backend)
            attempts += len(chunk)
            for task, completion in zip(chunk, completions):
                demo = as_demonstration(completion)
                if score_messages(task, completion).score == 4 and score_messages(task, demo).score == 4:
                    traces.append({"task_id": task["task_id"], "family": task["family"], "source": "teacher", "messages": build_prompt(task) + demo, "tools": TOOL_SCHEMAS})
                else:
                    still_failing.append(task)
        pending = still_failing
        if verbose:
            print(f"  teacher attempt {attempt + 1}/{samples}: {len(traces)}/{len(tasks)} tasks have a perfect rollout")
    totals, accepted = Counter(t["family"] for t in tasks), Counter(t["family"] for t in traces)
    stats = {
        "tasks": len(tasks),
        "accepted": len(traces),
        "rollouts": attempts,
        "acceptance_per_family": {f: f"{accepted[f]}/{totals[f]}" for f in totals},
    }
    return traces, stats


def generate_teacher_traces(
    tasks: Sequence[dict],
    model: str,
    samples: int = 4,
    endpoint: tuple[str, str] | None = None,
    temperature: float = 0.7,
    workers: int = 32,
    port: int = 8100,
    log_path: str = "/content/teacher_server.log",
    gpu_memory_utilization: float = 0.9,
) -> tuple[list[dict], dict]:
    """Serve the teacher (unless `endpoint=(base_url, api_key)` is given), sample, and shut the server down.

    The local server is vLLM from `bfcl.vllm_python()` (this environment's, or
    its own virtualenv), with Qwen3 thinking switched off: the demonstrations
    must be plans the non-thinking student can imitate.
    """
    from . import bfcl
    from .demo import OpenAIBackend
    from .serve import start_vllm, wait_ready

    server = None
    try:
        if endpoint:
            base_url, api_key = endpoint
            backend = OpenAIBackend(base_url, api_key, model=model, temperature=temperature, max_tokens=512)
        else:
            python = bfcl.vllm_python()
            if python is None:
                raise RuntimeError("no GPU large enough to serve the teacher; pass endpoint=(base_url, api_key) instead")
            server = wait_ready(start_vllm("teacher", model, port, log_path, gpu_memory_utilization=gpu_memory_utilization, python=python))
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
            backend = OpenAIBackend(server.base_url, server.api_key, model="teacher", temperature=temperature, max_tokens=512, extra_body=extra)
        return collect_teacher_traces(tasks, parallel_backend(backend, workers), samples=samples)
    finally:
        if server is not None:
            server.stop()
