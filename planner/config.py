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


# 800 evaluation tasks, about three quarters of which need two or three dependent tool rounds.
_EVAL_MIX = {"kb_bridge": 1.5, "kb_compare": 0.5, "weather_lookup": 0.25, "election_lookup": 0.25, "no_tool": 0.5}
# 4,000 GRPO tasks: rollouts go where the policy still fails, not where every rollout already gets full reward.
_GRPO_MIX = {
    "kb_bridge": 2.25, "composite": 1.5, "dual_bridge": 1.5, "chain_calc": 1.0, "tool_bridge": 0.75,
    "fanout_calc": 0.75, "kb_compare": 0.75, "weather_lookup": 0.375, "election_lookup": 0.375, "no_tool": 0.75,
}  # fmt: skip
# More demonstrations for the families a scripted expert cannot fully solve (the teacher fills in the rest).
_SFT_MIX = {"kb_bridge": 2.0, "composite": 1.6, "dual_bridge": 1.6, "weather_lookup": 0.6, "election_lookup": 0.6}


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
    # Tasks per family = base size x the family's multiplier in the mix (1.0 when not listed). The mixes lean
    # towards multi-step families: single-call families are near their ceiling after SFT and tell us little.
    n_eval_per_family: int = 100
    n_grpo_per_family: int = 400
    n_sft_per_family: int = 250
    eval_mix: dict = field(default_factory=lambda: dict(_EVAL_MIX))
    grpo_mix: dict = field(default_factory=lambda: dict(_GRPO_MIX))
    sft_mix: dict = field(default_factory=lambda: dict(_SFT_MIX))
    task_version: int = 2  # bump when task generation changes: cached environments are then rebuilt
    n_toolace: int = 10000
    n_hermes: int = 1900  # all usable rows (about 1.1k convert cleanly)

    # teacher demonstrations (rejection sampling) for the tasks the scripted expert cannot solve
    teacher_model: str = "Qwen/Qwen3-14B"  # ~30 GB in 16-bit: fits an A100-40GB while nothing else is loaded
    teacher_samples: int = 4

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
        keys = ("mode", "seed", "model_name", "enable_thinking", "hotpot_train_rows", "hotpot_val_rows", "n_eval_per_family", "n_grpo_per_family", "n_sft_per_family", "eval_mix", "grpo_mix", "sft_mix", "task_version", "bench_max_new_tokens", "bfcl_categories")
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
    eval_mix={},
    grpo_mix={},
    sft_mix={},
    n_toolace=150,
    n_hermes=60,
    teacher_model="Qwen/Qwen3-4B",
    teacher_samples=2,
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
    # The task version is part of the experiment name: results for a changed benchmark get a new Drive folder
    # instead of mixing with checkpoints and scores from the old one.
    config = Config(mode=mode, model_key=model_key, **MODELS[model_key])
    config = replace(config, exp_name=f"planner_{model_key}_v{config.task_version}")
    if mode == "smoke":
        config = replace(config, exp_name=config.exp_name + "_smoke", repo_stem=config.repo_stem + "-smoke", **_SMOKE)
    return replace(config, **overrides)
