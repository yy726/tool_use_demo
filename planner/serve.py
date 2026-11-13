"""Start OpenAI-compatible model servers for the demo and wait until they answer.

Two ways to host a checkpoint:

- `start_unsloth`: `unsloth run --model <repo>-GGUF:<quant>` (Unsloth Studio's
  llama.cpp-based server). It prints an API key (`sk-unsloth-...`) on start-up,
  which is read from the log.
- `start_vllm`: vLLM serving the merged 16-bit weights, with Hermes-style tool
  call parsing switched on.

Either way the demo client (`planner.demo`) runs the tool loop itself; the
server only has to produce the model's turns.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

UNSLOTH_KEY_RE = re.compile(r"sk-unsloth-[A-Za-z0-9_\-]+")


@dataclass
class Endpoint:
    name: str
    base_url: str  # includes /v1
    api_key: str
    process: object = None
    log_path: str | None = None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def find_unsloth_key(log_text: str) -> str | None:
    """The API key `unsloth run` prints to its console, if it is there yet."""
    matches = UNSLOTH_KEY_RE.findall(log_text)
    return matches[-1] if matches else None


def models_ready(base_url: str, api_key: str = "EMPTY", timeout: float = 5.0) -> bool:
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200 and bool(json.loads(response.read() or b"{}").get("data"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _read(path: str) -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def wait_ready(endpoint: Endpoint, timeout: float = 900.0, poll: float = 3.0, key_from_log: bool = False) -> Endpoint:
    """Wait for the server; with `key_from_log`, first pick the API key out of the server log."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if endpoint.process is not None and endpoint.process.poll() is not None:
            raise RuntimeError(f"{endpoint.name}: server exited with code {endpoint.process.returncode}. Log tail:\n{_read(endpoint.log_path)[-2000:]}")
        if key_from_log and endpoint.api_key in ("", "EMPTY"):
            key = find_unsloth_key(_read(endpoint.log_path))
            if key:
                endpoint.api_key = key
        if (not key_from_log or endpoint.api_key not in ("", "EMPTY")) and models_ready(endpoint.base_url, endpoint.api_key):
            return endpoint
        time.sleep(poll)
    raise TimeoutError(f"{endpoint.name}: not ready after {timeout:.0f}s. Log tail:\n{_read(endpoint.log_path)[-2000:]}")


def start_unsloth(name: str, gguf_repo: str, port: int, log_path: str, variant: str = "Q8_0", extra_args: tuple[str, ...] = ()) -> Endpoint:
    """`unsloth run --model <repo>:<variant> -p <port> -y` in the background."""
    command = ["unsloth", "run", "--model", f"{gguf_repo}:{variant}", "-p", str(port), "-y", "--disable-tools", *extra_args]
    print("$", " ".join(command), f"> {log_path}")
    process = subprocess.Popen(command, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    return Endpoint(name=name, base_url=f"http://127.0.0.1:{port}/v1", api_key="", process=process, log_path=log_path)


def start_vllm(name: str, model: str, port: int, log_path: str, gpu_memory_utilization: float = 0.28, max_model_len: int = 8192) -> Endpoint:
    """vLLM chat server for merged weights (a Hub repo id or a local directory)."""
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", name,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]  # fmt: skip
    print("$", " ".join(command), f"> {log_path}")
    process = subprocess.Popen(command, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    return Endpoint(name=name, base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY", process=process, log_path=log_path)
