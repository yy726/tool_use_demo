"""Run configuration: one dataclass, two presets ("smoke" and "a100"), two base models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

MODELS = {
    # Hybrid-thinking Qwen3: thinking is switched off with an empty <think></think> block.
    "qwen3-1.7b": {
        "model_name": "unsloth/Qwen3-1.7B",
        "base_model_hf": "Qwen/Qwen3-1.7B",
        "bfcl_model_id": "planner/qwen3-nothink-FC",
        "repo_stem": "qwen3-1.7b-planner",
    },
    # Non-thinking instruct model used by the Unsloth notebooks; BFCL supports it directly.
    "qwen3-4b-instruct": {
        "model_name": "unsloth/Qwen3-4B-Instruct-2507",
        "base_model_hf": "Qwen/Qwen3-4B-Instruct-2507",
        "bfcl_model_id": "Qwen/Qwen3-4B-Instruct-2507-FC",
        "repo_stem": "qwen3-4b-planner",
    },
}


@dataclass
class Config:
    mode: str = "a100"
    exp_name: str = "planner_v1"
    seed: int = 3407

    # model
    model_key: str = "qwen3-1.7b"
    model_name: str = "unsloth/Qwen3-1.7B"
    base_model_hf: str = "Qwen/Qwen3-1.7B"
    bfcl_model_id: str = "planner/qwen3-nothink-FC"
    repo_stem: str = "qwen3-1.7b-planner"
    max_seq_length: int = 4096
    lora_rank: int = 32
    enable_thinking: bool = False

    # environment and data
    hotpot_train_rows: int = 8000
    hotpot_val_rows: int = 1500
    n_eval_per_family: int = 100
    n_grpo_per_family: int = 500
    n_sft_per_family: int = 250
    n_toolace: int = 10000
    n_hermes: int = 1900  # all usable rows (about 1.1k convert cleanly)

    # SFT
    sft_lr: float = 1e-4
    sft_epochs: float = 2.0
    sft_max_steps: int = -1
    sft_batch: int = 16
    sft_accum: int = 2
    sft_max_tokens: int = 4096
    sft_save_steps: int = 100

    # GRPO
    reward_mode: str = "graded_cost"
    grpo_lr: float = 5e-6
    grpo_steps: int = 400
    grpo_num_generations: int = 8
    grpo_batch: int = 8
    grpo_accum: int = 4
    grpo_max_completion: int = 2048  # counts every turn AND the tool results (4 searches ~ 1.6k tokens)
    grpo_save_steps: int = 25

    # evaluation
    bench_max_new_tokens: int = 384
    bench_batch: int = 64
    bfcl_categories: list[str] = field(default_factory=list)  # empty = all offline categories

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> dict:
        """The settings that cached environment / evaluation results depend on."""
        keys = ("mode", "seed", "model_name", "enable_thinking", "hotpot_train_rows", "hotpot_val_rows", "n_eval_per_family", "n_grpo_per_family", "n_sft_per_family", "bench_max_new_tokens", "bfcl_categories")
        return {k: getattr(self, k) for k in keys}

    def repo(self, hf_user: str, stage: str, kind: str = "") -> str:
        """Hub repo name, e.g. repo("me", "grpo", "GGUF") -> "me/qwen3-1.7b-planner-grpo-GGUF"."""
        return f"{hf_user}/{self.repo_stem}-{stage}" + (f"-{kind}" if kind else "")


_SMOKE = dict(
    hotpot_train_rows=600,
    hotpot_val_rows=300,
    n_eval_per_family=6,
    n_grpo_per_family=16,
    n_sft_per_family=12,
    n_toolace=150,
    n_hermes=60,
    sft_max_steps=12,
    sft_save_steps=6,
    sft_batch=4,
    grpo_steps=4,
    grpo_num_generations=4,
    grpo_batch=4,
    grpo_accum=1,
    grpo_save_steps=2,
    bench_batch=16,
    bfcl_categories=["simple_python", "parallel", "irrelevance"],
)


def get_config(mode: str = "a100", model_key: str = "qwen3-1.7b", **overrides) -> Config:
    """Build a config. "smoke" shrinks everything so the whole pipeline runs in minutes."""
    if mode not in ("smoke", "a100"):
        raise ValueError("mode must be 'smoke' or 'a100'")
    if model_key not in MODELS:
        raise ValueError(f"model_key must be one of {sorted(MODELS)}")
    config = Config(mode=mode, model_key=model_key, exp_name=f"planner_{model_key}", **MODELS[model_key])
    if mode == "smoke":
        config = replace(config, exp_name=config.exp_name + "_smoke", repo_stem=config.repo_stem + "-smoke", **_SMOKE)
    return replace(config, **overrides)
