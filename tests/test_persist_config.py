import os

import pytest

from planner.config import MODELS, get_config
from planner.persist import StageCache, atomic_write_json, latest_checkpoint, model_card, persist_dir, push_with_retry, read_json, read_jsonl, wandb_init, write_jsonl


def test_persist_dir_falls_back_to_local_outside_colab(tmp_path):
    path = persist_dir("exp", use_drive=True, local_root=str(tmp_path))
    assert path == str(tmp_path / "exp") and os.path.isdir(path)


def test_atomic_json_and_jsonl_roundtrip(tmp_path):
    path = str(tmp_path / "a.json")
    atomic_write_json(path, {"x": [1, 2], "s": "é"})
    assert read_json(path) == {"x": [1, 2], "s": "é"} and not os.path.exists(path + ".tmp")
    rows = [{"i": i} for i in range(3)]
    write_jsonl(str(tmp_path / "r.jsonl"), rows)
    assert read_jsonl(str(tmp_path / "r.jsonl")) == rows


def test_stage_cache_skips_finished_stages(tmp_path):
    cache = StageCache(str(tmp_path))
    runs = []
    work = lambda: runs.append(1) or {"good_rate": 0.5}
    assert cache.get_or_run("bench_base", work) == {"good_rate": 0.5}
    assert StageCache(str(tmp_path)).get_or_run("bench_base", work) == {"good_rate": 0.5}  # survives a "restart"
    assert runs == [1] and cache.done() == ["bench_base"]
    assert StageCache(str(tmp_path), enabled=False).get_or_run("bench_base", work) and runs == [1, 1]


def test_stage_cache_is_invalidated_by_a_config_change(tmp_path, capsys):
    assert StageCache(str(tmp_path)).check_config({"lr": 1e-4, "model": "a"})
    same = StageCache(str(tmp_path))
    same.save("stage", 1)
    assert same.check_config({"lr": 1e-4, "model": "a"}) and same.has("stage")
    changed = StageCache(str(tmp_path))
    assert not changed.check_config({"lr": 5e-5, "model": "a"})
    assert not changed.has("stage") and "lr" in capsys.readouterr().out


def test_latest_checkpoint_ignores_incomplete_ones(tmp_path):
    assert latest_checkpoint(str(tmp_path / "missing")) is None
    for step, complete in ((25, True), (100, True), (125, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-final").mkdir()
    assert latest_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-100")


def test_wandb_is_disabled_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_init(str(tmp_path), "project", "run") == (None, "none")


def test_push_with_retry(capsys):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("reset")

    assert push_with_retry(flaky, "adapter", retries=3, wait=0) and len(attempts) == 2
    assert not push_with_retry(lambda: 1 / 0, "merged", retries=2, wait=0)


def test_model_card_has_front_matter():
    card = model_card("T", "Qwen/Qwen3-1.7B", "desc", {"Results": "table"})
    assert card.startswith("---\nbase_model: Qwen/Qwen3-1.7B") and "## Results" in card


def test_config_presets():
    full, smoke = get_config("a100"), get_config("smoke")
    assert full.grpo_batch % full.grpo_num_generations == 0 and smoke.grpo_batch % smoke.grpo_num_generations == 0
    assert smoke.grpo_steps < full.grpo_steps and smoke.exp_name != full.exp_name
    assert full.repo("me", "grpo", "GGUF") == "me/qwen3-1.7b-planner-grpo-GGUF" and full.repo("me", "sft") == "me/qwen3-1.7b-planner-sft"
    big = get_config("a100", model_key="qwen3-4b-instruct", grpo_steps=10)
    assert big.model_name == MODELS["qwen3-4b-instruct"]["model_name"] and big.grpo_steps == 10
    with pytest.raises(ValueError):
        get_config("huge")
    with pytest.raises(ValueError):
        get_config("a100", model_key="gpt")
