import http.server
import json
import threading
from types import SimpleNamespace

import pytest

from planner.serve import Endpoint, find_unsloth_key, models_ready, wait_ready


def test_find_unsloth_key():
    log = "Starting...\nEndpoint: http://127.0.0.1:8888\nAPI key: sk-unsloth-AbC_123-xyz\nReady."
    assert find_unsloth_key(log) == "sk-unsloth-AbC_123-xyz"
    assert find_unsloth_key("no key yet") is None


@pytest.fixture()
def server():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            ok = self.headers.get("Authorization") == "Bearer sk-unsloth-secret" and self.path == "/v1/models"
            self.send_response(200 if ok else 401)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "m"}]} if ok else {}).encode())

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", seen
    httpd.shutdown()


def test_wait_ready_reads_the_key_from_the_log(tmp_path, server):
    base_url, seen = server
    log = tmp_path / "server.log"
    log.write_text("booting\n")
    endpoint = Endpoint(name="grpo", base_url=base_url, api_key="", log_path=str(log))
    assert not models_ready(base_url, "wrong")
    threading.Timer(0.3, lambda: log.write_text("booting\nkey: sk-unsloth-secret\n")).start()
    ready = wait_ready(endpoint, timeout=10, poll=0.1, key_from_log=True)
    assert ready.api_key == "sk-unsloth-secret" and seen[-1] == "Bearer sk-unsloth-secret"


def test_wait_ready_reports_a_dead_server(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("CUDA out of memory")
    dead = Endpoint(name="sft", base_url="http://127.0.0.1:1/v1", api_key="EMPTY", process=SimpleNamespace(poll=lambda: 1, returncode=1), log_path=str(log))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        wait_ready(dead, timeout=5, poll=0.1)
    with pytest.raises(TimeoutError):
        wait_ready(Endpoint(name="x", base_url="http://127.0.0.1:1/v1", api_key="EMPTY", log_path=str(log)), timeout=0.3, poll=0.1)


def test_first_existing_repo_prefers_the_full_run_and_falls_back_to_smoke():
    from planner.config import get_config
    from planner.serve import first_existing_repo

    candidates = [get_config(mode, "qwen3-1.7b").repo("me", "grpo", "GGUF") for mode in ("a100", "smoke")]
    assert candidates == ["me/qwen3-1.7b-planner-grpo-GGUF", "me/qwen3-1.7b-planner-smoke-grpo-GGUF"]
    assert first_existing_repo(candidates, exists=lambda repo: True) == candidates[0]
    assert first_existing_repo(candidates, exists=lambda repo: "smoke" in repo) == candidates[1]
    with pytest.raises(FileNotFoundError, match="none of these"):
        first_existing_repo(candidates, exists=lambda repo: False)


def test_first_existing_repo_prefers_the_first_candidate_that_exists():
    from planner.serve import first_existing_repo

    hub = {"me/planner-smoke-grpo-GGUF"}
    assert first_existing_repo(["me/planner-grpo-GGUF", "me/planner-smoke-grpo-GGUF"], exists=hub.__contains__) == "me/planner-smoke-grpo-GGUF"
    with pytest.raises(FileNotFoundError, match="planner-sft"):
        first_existing_repo(["me/planner-sft"], exists=hub.__contains__)


def test_corpus_save_creates_the_cache_directory(tmp_path):
    from planner.corpus import Corpus

    corpus = Corpus()
    corpus.add("Paris", "Paris is the capital of France.")
    path = tmp_path / "not" / "there" / "corpus.jsonl"
    corpus.save(str(path))  # 03_demo failed here: /content/planner_env did not exist
    assert len(Corpus.load(str(path))) == 1
