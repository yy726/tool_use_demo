"""Run the official BFCL pipeline (`bfcl-eval`) and collect its scores.

Generation and scoring are BFCL's own code. The only addition is one registered
model id, `planner/qwen3-nothink-FC`: BFCL's Qwen function-calling handler with
the empty `<think></think>` block prefilled, i.e. Qwen3 in non-thinking mode.
BFCL's stock Qwen3 entries leave thinking on, which would compare a thinking
base model against fine-tuned models that were trained not to think.

bfcl-eval pins old versions of numpy and friends, so it lives in its own
virtualenv. `run_bfcl(...)` (called from the notebooks) creates that virtualenv,
serves the model with vLLM (this environment's, or one installed into a second
virtualenv when the notebook's environment has none), and runs
`python -m planner.bfcl generate|evaluate ...` inside the virtualenv - the
`bfcl` CLI with the extra model id registered.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

NOTHINK_ID = "planner/qwen3-nothink-FC"
EMPTY_THINK = "<think>\n\n</think>\n\n"

# Offline, key-free categories. The agentic group (web search, memory) is left out.
GROUPS = {
    "non_live": ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple"],
    "live": ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"],
    "hallucination": ["irrelevance", "live_irrelevance"],
    "multi_turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
}
ALL_CATEGORIES = [c for cats in GROUPS.values() for c in cats]
SMOKE_CATEGORIES = ["simple_python", "parallel", "irrelevance"]


def register_nothink(base_model: str = "Qwen/Qwen3-1.7B") -> str:
    """Register the non-thinking Qwen3 handler with bfcl-eval; return its model id."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    from .bfcl_handler import QwenFCNoThinkHandler

    template = MODEL_CONFIG_MAPPING["Qwen/Qwen3-1.7B-FC"]
    MODEL_CONFIG_MAPPING[NOTHINK_ID] = dataclasses.replace(
        template,
        model_name=base_model,
        display_name=f"{base_model} (FC, no-think)",
        url=f"https://huggingface.co/{base_model}",
        model_handler=QwenFCNoThinkHandler,
    )
    return NOTHINK_ID


def main(argv: list[str] | None = None) -> None:
    """`python -m planner.bfcl <bfcl args>`: the bfcl CLI plus the no-think model id."""
    register_nothink(os.environ.get("PLANNER_BFCL_BASE_MODEL", "Qwen/Qwen3-1.7B"))
    from bfcl_eval.__main__ import cli

    sys.argv = ["bfcl"] + list(sys.argv[1:] if argv is None else argv)
    cli()


# ----------------------------------------------------------------- reading
def read_scores(score_dir: str, model_id: str) -> dict[str, dict]:
    """Per-category `{"accuracy", "correct", "total"}` from BFCL's score files."""
    model_dir = os.path.join(score_dir, model_id.replace("/", "_"))
    scores = {}
    for path in glob.glob(os.path.join(model_dir, "**", "BFCL_v4_*_score.json"), recursive=True):
        category = os.path.basename(path)[len("BFCL_v4_") : -len("_score.json")]
        with open(path, encoding="utf-8") as f:
            header = json.loads(f.readline())
        scores[category] = {"accuracy": header["accuracy"], "correct": header["correct_count"], "total": header["total_count"]}
    return scores


def summarise(scores: dict[str, dict]) -> dict:
    """Group scores following BFCL V4's published aggregation, plus their mean (BFCL-core).

    - non_live: unweighted mean of simple / multiple / parallel / parallel_multiple,
      where simple is the unweighted mean of its Python, Java and JavaScript parts.
    - live: mean weighted by the number of test entries.
    - hallucination: unweighted mean of the two irrelevance categories.
    - multi_turn: unweighted mean of its four categories.

    BFCL-core is the mean of the available groups. It leaves out the agentic
    group (40% of the official overall score), so it is not the leaderboard number.
    """

    def acc(cat):
        return scores[cat]["accuracy"] if cat in scores else None

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    simple = mean(acc(c) for c in ("simple_python", "simple_java", "simple_javascript"))
    live = [scores[c] for c in GROUPS["live"] if c in scores]
    groups = {
        "non_live": mean([simple, acc("multiple"), acc("parallel"), acc("parallel_multiple")]),
        "live": (sum(s["correct"] for s in live) / max(1, sum(s["total"] for s in live))) if live else None,
        "hallucination": mean(acc(c) for c in GROUPS["hallucination"]),
        "multi_turn": mean(acc(c) for c in GROUPS["multi_turn"]),
    }
    return {
        "groups": groups,
        "bfcl_core": mean(groups.values()),
        "categories": {c: s["accuracy"] for c, s in sorted(scores.items())},
        "n_entries": sum(s["total"] for s in scores.values()),
    }


# ----------------------------------------------------------------- running
BFCL_EVAL_VERSION = "2026.3.23"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_venv(venv_dir: str = "/content/bfcl_venv") -> str:
    """Create (once) a separate virtualenv with bfcl-eval and return its python.

    bfcl-eval pins old versions of numpy and other packages, so installing it
    next to the Unsloth / TRL training stack would break that stack.
    """
    python = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, ".bfcl_ready")
    if os.path.exists(marker):
        return python
    packages = [f"bfcl-eval=={BFCL_EVAL_VERSION}", "soundfile"]
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", venv_dir], check=True)
        subprocess.run(["uv", "pip", "install", "--python", python, "-q"] + packages, check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        subprocess.run([python, "-m", "pip", "install", "-q"] + packages, check=True)
    open(marker, "w").close()
    return python


VLLM_VERSION = "0.19.1"  # the version Unsloth's GRPO notebook installs
MIN_VLLM_GPU_MIB = 20000  # a T4 (15 GB) is too small for the BFCL server


def gpu_memory_mib() -> int:
    """Total memory of GPU 0 in MiB, or 0 without a GPU (asked from nvidia-smi: no torch import needed)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0


def vllm_python(venv_dir: str = "/content/vllm_venv") -> str | None:
    """A python that can run the vLLM server, or None on a GPU that is too small.

    Uses this environment's vLLM when it has one (02_grpo). Otherwise vLLM goes
    into its own virtualenv: it pins its own torch and numpy, and installing it
    over Colab's would change the environment the model is trained in.
    """
    import importlib.util

    if importlib.util.find_spec("vllm") is not None:
        return sys.executable
    if gpu_memory_mib() < MIN_VLLM_GPU_MIB:
        return None
    python = os.path.join(venv_dir, "bin", "python")
    marker = os.path.join(venv_dir, ".vllm_ready")
    if not os.path.exists(marker):
        print(f"Installing vLLM {VLLM_VERSION} into {venv_dir} (once per runtime, a few minutes) ...")
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", venv_dir], check=True)
            subprocess.run(["uv", "pip", "install", "--python", python, "-q", f"vllm=={VLLM_VERSION}"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            subprocess.run([python, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"], check=True)
        open(marker, "w").close()
    return python


def wait_for_server(port: int, process=None, timeout: float = 900.0, poll: float = 5.0) -> None:
    """Block until the OpenAI-compatible server answers /v1/models."""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"model server exited with code {process.returncode} before becoming ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll)
    raise TimeoutError(f"model server on port {port} was not ready after {timeout:.0f}s")


def start_vllm(model: str, port: int, gpu_memory_utilization: float, log_path: str, python: str | None = None):
    """Serve `model` (a local directory or a Hub id) with vLLM, run by `python` (default: this environment)."""
    command = [
        python or sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
    ]  # fmt: skip
    print("$", " ".join(command), f"> {log_path}")
    log = open(log_path, "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def stop_server(process) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def bfcl_commands(python: str, model_id: str, categories: list[str], result_dir: str, score_dir: str, local_model_path: str | None, num_threads: int) -> tuple[list[str], list[str]]:
    base = [python, "-m", "planner.bfcl"]
    generate = base + [
        "generate",
        "--model", model_id,
        "--test-category", ",".join(categories),
        "--backend", "vllm",
        "--skip-server-setup",
        "--num-threads", str(num_threads),
        "--result-dir", result_dir,
    ]  # fmt: skip
    if local_model_path:
        generate += ["--local-model-path", local_model_path]
    evaluate = base + ["evaluate", "--model", model_id, "--test-category", ",".join(categories), "--result-dir", result_dir, "--score-dir", score_dir]
    return generate, evaluate


def run_bfcl(
    work_dir: str,
    model_id: str = NOTHINK_ID,
    base_model: str = "Qwen/Qwen3-1.7B",
    local_model_path: str | None = None,
    categories: list[str] | None = None,
    gpu_memory_utilization: float = 0.85,
    num_threads: int = 64,
    port: int = 1053,
    venv_dir: str = "/content/bfcl_venv",
    start_server: bool = True,
    server_python: str | None = None,
) -> dict:
    """Generate + evaluate one checkpoint with BFCL; return `summarise(...)` of its scores.

    The model is served by vLLM (run by `server_python`, default this
    environment); BFCL itself runs from its own virtualenv and talks to that server. `work_dir` holds the checkpoint's
    `result/` and `score/` folders. BFCL skips test entries that already have a
    result, so calling this again after a disconnect continues where it stopped.

    `local_model_path` is a merged fine-tuned model; without it the base model
    is pulled from the Hub. Set `start_server=False` to use a server you started.
    """
    categories = categories or ALL_CATEGORIES
    result_dir, score_dir = os.path.join(work_dir, "result"), os.path.join(work_dir, "score")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)
    python = ensure_venv(venv_dir)
    env = dict(os.environ, PLANNER_BFCL_BASE_MODEL=base_model, LOCAL_SERVER_PORT=str(port))
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    generate, evaluate = bfcl_commands(python, model_id, categories, result_dir, score_dir, local_model_path, num_threads)
    server, log_path = None, os.path.join(work_dir, "vllm_server.log")
    try:
        if start_server:
            # BFCL asks the server for the model under the same name it was started with.
            server = start_vllm(local_model_path or base_model, port, gpu_memory_utilization, log_path, server_python)
            wait_for_server(port, server)
        for command in (generate, evaluate):
            print("$", " ".join(command))
            subprocess.run(command, check=True, env=env)
    except Exception:
        if server is not None and os.path.exists(log_path):  # the reason is usually at the end of the server log
            with open(log_path, errors="replace") as f:
                print(f"--- last lines of {log_path} ---\n" + "".join(f.readlines()[-40:]))
        raise
    finally:
        stop_server(server)
    return summarise(read_scores(score_dir, model_id))


def run_stage(cache, name: str, config, persist: str, local_model_path: str | None = None, enabled: bool = True, **kwargs) -> dict | None:
    """Cached BFCL run for one checkpoint ("base", "sft", "grpo"). Returns None when skipped."""
    if cache.has(f"bfcl_{name}"):
        return cache.load(f"bfcl_{name}")
    if not enabled:
        print(f"BFCL ({name}) skipped: RUN_BFCL is off")
        return None
    if name != "base" and not local_model_path:
        print(f"BFCL ({name}) skipped: merged weights for this checkpoint were not found")
        return None
    if "server_python" not in kwargs and kwargs.get("start_server", True):
        kwargs["server_python"] = vllm_python()
        if kwargs["server_python"] is None:
            print(f"BFCL ({name}) skipped: the vLLM server needs a GPU with at least {MIN_VLLM_GPU_MIB // 1000} GB (not a T4)")
            return None
    try:
        result = run_bfcl(
            work_dir=os.path.join(persist, "bfcl", name),
            model_id=config.bfcl_model_id,
            base_model=config.base_model_hf,
            local_model_path=local_model_path,
            categories=config.bfcl_categories or None,
            **kwargs,
        )
    except (subprocess.SubprocessError, RuntimeError, TimeoutError, OSError) as exc:
        # BFCL is evaluation only: report the failure and let training go on. Nothing is cached, so a re-run retries.
        print(f"BFCL ({name}) FAILED and was skipped: {type(exc).__name__}: {exc}")
        return None
    return cache.save(f"bfcl_{name}", result)


def format_table(results: dict[str, dict]) -> str:
    """Markdown table of group scores for several checkpoints."""
    names = list(results)
    lines = ["| BFCL (official checker) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def cell(value):
        return "n/a" if value is None else f"{100 * value:.1f}"

    for label, key in (("Non-Live AST", "non_live"), ("Live AST", "live"), ("Hallucination (irrelevance)", "hallucination"), ("Multi-Turn", "multi_turn")):
        lines.append(f"| {label} | " + " | ".join(cell(results[n]["groups"].get(key)) for n in names) + " |")
    lines.append("| **BFCL-core** (mean of the groups above) | " + " | ".join(cell(results[n]["bfcl_core"]) for n in names) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
