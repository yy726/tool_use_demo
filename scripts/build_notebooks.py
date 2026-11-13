"""Generate the three Colab notebooks from the cell definitions below.

    python scripts/build_notebooks.py

The notebooks are thin: all logic lives in the `planner` package, which the
first cells clone from GitHub. Editing cells here (rather than in the .ipynb)
keeps the three notebooks' shared setup identical.
"""

from __future__ import annotations

import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/yy726/tool_use_demo.git"
REPO_BRANCH = "claude/inspiring-planck-c4a3hh"


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# --------------------------------------------------------------- shared cells
# Same install cell as Unsloth's "Qwen3 (4B) GRPO Tool Use" notebook, plus the retriever and W&B.
INSTALL = r'''
%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
!pip install --upgrade -qqq uv
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except: _numpy = "numpy"; _pil = "pillow"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    if is_t4:
        # Keep the machine's torch: upgrading it can pull a CUDA build the driver cannot load
        try:
            import importlib.metadata as _md
            _torch, _tv = (f'{_p}=={_md.version(_p).split("+")[0]}' for _p in ('torch', 'torchvision'))
        except Exception:
            _torch, _tv = 'torch', 'torchvision'
        !uv pip install -qqq {_numpy} {_pil} {_torch} {_tv} bitsandbytes xformers unsloth
    else:
        !uv pip install -qqq --upgrade vllm==0.19.1 {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq triton
    try:
        import importlib.metadata as _md; _torch_v = tuple(int(_p) for _p in _md.version("torch").split("+")[0].split(".")[:2])
    except Exception:
        _torch_v = ()
    _torchao = "torchao>=0.16.0" if _torch_v >= (2, 10) else "torchao>=0.16.0,<0.18.0"
    !uv pip install -qqq --no-deps --upgrade "{_torchao}"
!uv pip install transformers==5.15.1 "datasets>=4.7.0,<5.0.0"
!uv pip install --no-deps trl==1.13.0
!uv pip install -qqq --no-deps --force-reinstall "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo" "unsloth @ git+https://github.com/unslothai/unsloth"
!uv pip install -qqq bm25s==0.3.16 wandb
'''

CLONE = f'''
import os, subprocess, sys

REPO_URL = "{REPO_URL}"
REPO_BRANCH = "{REPO_BRANCH}"

if os.path.isdir("planner"):                      # already inside a checkout
    REPO_DIR = os.getcwd()
elif os.path.isdir("../planner"):
    REPO_DIR = os.path.abspath("..")
else:
    REPO_DIR = "/content/tool_use_demo"
    if not os.path.isdir(REPO_DIR):
        url = REPO_URL
        try:                                      # private repo: add a GITHUB_TOKEN secret
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
            if token: url = url.replace("https://", f"https://{{token}}@")
        except Exception:
            pass
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, url, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path: sys.path.insert(0, REPO_DIR)
print("repo:", REPO_DIR)
'''

SETUP = '''
import unsloth  # import before transformers / trl so its patches apply
import json, random
from planner.config import get_config
from planner.persist import StageCache, persist_dir, read_secrets
from planner.setup_env import prepare

cfg = get_config("smoke" if SMOKE else "a100", MODEL_KEY, enable_thinking=ENABLE_THINKING, **OVERRIDES)

print("secrets:", read_secrets())                 # HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN from Colab Secrets
HF_USER = None
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login, whoami
    login(token=os.environ["HF_TOKEN"])
    HF_USER = whoami()["name"]

PERSIST = persist_dir(cfg.exp_name, USE_DRIVE)    # Google Drive: survives a disconnect
LOCAL = f"/content/work/{cfg.exp_name}"            # local disk: large merged weights
os.makedirs(LOCAL, exist_ok=True)
CACHE = StageCache(PERSIST, enabled=RESUME)
CACHE.check_config(cfg.fingerprint())
print("persist dir:", PERSIST, "| finished stages:", CACHE.done())

env = prepare(cfg, os.path.join(PERSIST, "env"))  # world + knowledge base + sft / grpo / eval tasks
'''

BENCH_HELPERS = '''
from unsloth import FastLanguageModel
from planner.bench import format_report, run_benchmark, text_backend
from planner.chat import setup_chat_format
from planner.tasks import build_prompt
from planner.unsloth_utils import has_vllm, hf_generate_fn, load_lora_weights, load_model, vllm_generate_fn

MAX_NEW = cfg.bench_max_new_tokens * (4 if cfg.enable_thinking else 1)

def bench(name, lora_dir=None):
    """PlannerQA-Bench for the current weights (or a saved LoRA when the vLLM engine is on). Cached."""
    def run():
        if FAST:
            generate = vllm_generate_fn(model, tokenizer, lora_dir, MAX_NEW)
        else:
            FastLanguageModel.for_inference(model)
            generate = hf_generate_fn(model, tokenizer, MAX_NEW, cfg.bench_batch)
        return run_benchmark(env.tasks["eval"], text_backend(fmt.render_prompt, generate), batch_size=cfg.bench_batch, progress=True)
    return CACHE.get_or_run(f"bench_{name}", run)
'''


# ------------------------------------------------------------------ 01_sft
def sft_notebook():
    cells = [
        md(
            """
# 01 - Supervised fine-tuning of a tool-calling planner

This notebook fine-tunes **Qwen3-1.7B** (or Qwen3-4B-Instruct) into a *planner*: given a question it calls tools -
`knowledge_search`, `calculator`, `get_weather`, `get_election` - for at most three rounds, several calls per round
when they are independent, and then answers.

What it does, in order:

1. **BFCL** on the base model (general function calling, official `bfcl-eval` pipeline).
2. **PlannerQA-Bench** on the base model (our own held-out planning benchmark, scored on relevance and factuality).
3. Builds the SFT mix: public function-calling data (ToolACE + Hermes) and in-domain demonstrations from a scripted expert.
4. **SFT** with Unsloth + TRL `SFTTrainer`, checkpointing to Google Drive.
5. PlannerQA-Bench again, then pushes the adapter, merged weights and GGUF to the Hugging Face Hub.

`02_grpo.ipynb` continues from the adapter this notebook saves.

**Nothing is lost on a disconnect.** Checkpoints, data, and finished evaluation stages live on Drive: after a
disconnect just *Run all* again. Finished stages are skipped and training resumes from the last checkpoint.

**Before you start:** add `HF_TOKEN` (write access) and `WANDB_API_KEY` to Colab *Secrets* (key icon in the left
bar), and pick an **A100** or **L4** runtime. Run once with `SMOKE = True` (a few minutes, tiny data) to check the
whole pipeline end to end, then set `SMOKE = False`. On a T4 the BFCL stage is skipped, because it needs vLLM.
"""
        ),
        md("## 1. Configuration"),
        code(
            '''
SMOKE = True                 # True: tiny end-to-end check (minutes). False: the full run.
MODEL_KEY = "qwen3-1.7b"     # or "qwen3-4b-instruct" (the model of the Unsloth notebooks; ~2.5x slower)
ENABLE_THINKING = False      # Qwen3-1.7B only: let the planner think before calling tools
USE_DRIVE = True             # checkpoints and caches on Google Drive
RESUME = True                # reuse finished stages and checkpoints found on Drive
RUN_BFCL = True              # official BFCL evaluation (needs vLLM: A100 / L4)
WANDB_PROJECT = "tool-use-planner"
PUSH_TO_HUB = True           # push adapter + merged model (+ GGUF) to your Hugging Face account
HF_PRIVATE = False
PUSH_GGUF = True             # needed for Unsloth Studio / llama.cpp serving in 03_demo.ipynb
OVERRIDES = {}               # e.g. {"sft_epochs": 1, "n_toolace": 5000}; see planner/config.py
'''
        ),
        md("## 2. Install and set up\nThe install cell is the one from Unsloth's *Qwen3 (4B) GRPO Tool Use* notebook, so both notebooks share one software stack."),
        code(INSTALL),
        code(CLONE),
        code(SETUP),
        md(
            """
## 3. BFCL on the base model

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) measures *general* function calling: unseen function schemas,
parallel calls, refusing to call irrelevant functions, multi-turn tool use. It is evaluation only - nothing is trained
on it. We run the offline categories (about 4,400 test entries) with BFCL's own generation and checker code, and report
the four official groups plus their mean, **BFCL-core**. The agentic group (web search needs a paid key) is not run, so
BFCL-core is not the leaderboard's overall score.

BFCL pins old package versions, so it runs from its own virtualenv and talks to a vLLM server started here. This is why
it runs *before* the training model is loaded. For Qwen3-1.7B the model is evaluated in non-thinking mode at every
stage, so the three checkpoints are compared like for like.
"""
        ),
        code(
            '''
from planner import bfcl

BFCL = {"base": bfcl.run_stage(CACHE, "base", cfg, PERSIST, enabled=RUN_BFCL)}
if BFCL["base"]: print(bfcl.format_table({"base": BFCL["base"]}))
'''
        ),
        md("## 4. Load the model\nUnsloth loads the base model in 16-bit and adds a LoRA (rank 32, all projection layers)."),
        code(
            BENCH_HELPERS
            + '''
FAST = False                                       # SFT notebook: plain transformers generation
model, tokenizer = load_model(cfg, fast_inference=False)
fmt = setup_chat_format(tokenizer, enable_thinking=cfg.enable_thinking)
print(fmt.render_prompt(build_prompt(env.tasks["eval"][0]))[-700:])
'''
        ),
        md(
            """
A note on the chat template. TRL's GRPO tool loop needs a template that renders earlier turns identically when tool
results are appended, so it always writes the `<think>...</think>` block of an assistant turn (empty when the model does
not think). `setup_chat_format` uses that same template here, so SFT text, benchmark prompts and GRPO rollouts all show
the model one format.

## 5. PlannerQA-Bench on the base model

800 held-out tasks, 100 per family: `kb_bridge` (multi-hop), `kb_compare` (fan-out), `weather_lookup`,
`election_lookup`, `fanout_calc`, `tool_bridge` (cross-tool multi-hop), `composite` (two-part asks), `no_tool`.
Each rollout gets two scores from 1 to 4 and the final score is their minimum (3 and 4 are "good"):

- **relevance** - did the plan go after every part of the question, without off-topic calls?
- **factuality** - is the answer correct and backed by what the tools returned?
"""
        ),
        code(
            '''
BENCH = {"base": bench("base")}
print(format_report(BENCH))
'''
        ),
        md(
            """
## 6. Build the SFT data

| source | role | share |
|---|---|---|
| ToolACE + Hermes function calling | general function calling: arbitrary schemas, parallel calls, declining to call | ~85% |
| scripted expert in the mock environment | our tool set, the three-round protocol, the final `Answer:` line | ~15% |

The expert really runs the tools and only rollouts with a perfect score are kept. It is deliberately plain - it
searches with the question, then with an entity it read about - so GRPO has room to improve on it.
"""
        ),
        code(
            '''
from planner.data_public import load_public_examples, render_examples
from planner.expert import build_expert_traces
from planner.persist import read_jsonl, write_jsonl

SFT_DATA = os.path.join(PERSIST, "sft_train.jsonl")
if RESUME and os.path.exists(SFT_DATA):
    texts = [row["text"] for row in read_jsonl(SFT_DATA)]      # identical data and order after a restart
else:
    FastLanguageModel.for_inference(model)
    plain = hf_generate_fn(model, tokenizer, max_new_tokens=128, batch_size=cfg.bench_batch)
    def direct_answers(queries):                               # no_tool tasks: the base model's own tool-free reply
        return plain([fmt.render_prompt([{"role": "user", "content": q}], tools=None) for q in queries])
    traces = build_expert_traces(env.tasks["sft"], direct_answer_fn=direct_answers)
    public = load_public_examples(cfg.n_toolace, cfg.n_hermes, cfg.seed)
    examples = public + traces
    random.Random(cfg.seed).shuffle(examples)
    texts, dropped = render_examples(examples, fmt, cfg.sft_max_tokens)
    write_jsonl(SFT_DATA, [{"text": t} for t in texts])
    print(f"{len(public)} public + {len(traces)} in-domain examples -> {len(texts)} kept, {dropped} dropped (too long or unrenderable)")
print(len(texts), "training examples")
print(texts[0][-900:])
'''
        ),
        md(
            """
## 7. Train

Loss is computed on assistant turns only (`train_on_responses_only`), so tool results and prompts are not learned as
outputs. A checkpoint goes to Drive every `sft_save_steps` steps; if one is already there, training resumes from it.
"""
        ),
        code(
            '''
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from planner.persist import latest_checkpoint, log_summary, wandb_init
from planner.unsloth_utils import filter_kwargs, save_adapter, tokenizer_kwarg

SFT_DIR = os.path.join(PERSIST, "sft")             # trainer checkpoints
SFT_LORA = os.path.join(PERSIST, "sft_lora")       # final adapter (02_grpo.ipynb starts from this)
os.makedirs(SFT_DIR, exist_ok=True)
wandb_run, REPORT_TO = wandb_init(SFT_DIR, WANDB_PROJECT, f"sft-{cfg.repo_stem}", cfg.to_dict())

if RESUME and os.path.exists(os.path.join(SFT_LORA, "adapter_model.safetensors")):
    load_lora_weights(model, SFT_LORA)
    print("SFT already finished: adapter loaded from Drive")
else:
    FastLanguageModel.for_training(model)
    args = SFTConfig(**filter_kwargs(SFTConfig, dict(
        dataset_text_field = "text",
        max_length = cfg.sft_max_tokens,
        per_device_train_batch_size = cfg.sft_batch,
        gradient_accumulation_steps = cfg.sft_accum,
        num_train_epochs = cfg.sft_epochs,
        max_steps = cfg.sft_max_steps,              # -1: train for num_train_epochs
        learning_rate = cfg.sft_lr,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.03,
        weight_decay = 0.001,
        optim = "adamw_8bit",
        logging_steps = 5,
        seed = cfg.seed,
        output_dir = SFT_DIR,
        save_strategy = "steps",
        save_steps = cfg.sft_save_steps,
        save_total_limit = 2,
        report_to = REPORT_TO,
        run_name = f"sft-{cfg.repo_stem}",
    )))
    trainer = SFTTrainer(model = model, train_dataset = Dataset.from_dict({"text": texts}), args = args, **tokenizer_kwarg(SFTTrainer, tokenizer))
    trainer = train_on_responses_only(trainer, instruction_part = "<|im_start|>user\\n", response_part = "<|im_start|>assistant\\n")
    checkpoint = latest_checkpoint(SFT_DIR) if RESUME else None
    print("resume from checkpoint:", checkpoint)
    trainer.train(resume_from_checkpoint = checkpoint)
    save_adapter(model, tokenizer, fmt, SFT_LORA)
'''
        ),
        md("## 8. Evaluate the SFT model"),
        code(
            '''
BENCH["sft"] = bench("sft")
print(format_report(BENCH))
log_summary(wandb_run, "bench_base", BENCH["base"]["summary"])
log_summary(wandb_run, "bench_sft", BENCH["sft"]["summary"])
'''
        ),
        md(
            """
## 9. Save and push

Three repositories per checkpoint: the LoRA adapter, the merged 16-bit model (for vLLM, transformers and BFCL) and a
GGUF build (for Unsloth Studio / llama.cpp). The merged weights also stay on local disk for the BFCL run that opens
`02_grpo.ipynb`; after a disconnect that notebook downloads them from the Hub again.
"""
        ),
        code(
            '''
from planner.persist import model_card
from planner.unsloth_utils import export_model

SFT_MERGED = os.path.join(LOCAL, "sft_merged")
card = model_card(
    f"{cfg.repo_stem}-sft",
    cfg.base_model_hf,
    "Tool-calling planner, supervised fine-tuning stage. It plans with `knowledge_search`, `calculator`, `get_weather` and "
    "`get_election` for at most three tool rounds and ends with an `Answer:` line. Code: " + REPO_URL,
    {"PlannerQA-Bench (held-out)": format_report(BENCH), "BFCL": bfcl.format_table({k: v for k, v in BFCL.items() if v}) if BFCL["base"] else "not run"},
)
status = export_model(model, tokenizer, fmt, cfg, "sft", HF_USER, SFT_LORA, SFT_MERGED, card, push = PUSH_TO_HUB, private = HF_PRIVATE, gguf = PUSH_GGUF)
print(status)
if wandb_run: wandb_run.finish()
print("Done. Next: 02_grpo.ipynb (it evaluates this checkpoint on BFCL first).")
'''
        ),
    ]
    return cells


# ----------------------------------------------------------------- 02_grpo
def grpo_notebook():
    cells = [
        md(
            """
# 02 - GRPO on a mock tool environment

Continues from the SFT adapter of `01_sft.ipynb`. The planner now learns from its own rollouts: for each question it
runs up to three rounds of real tool calls against a local environment, and a rule-based reward scores the outcome.
Training uses TRL's `GRPOTrainer` with native multi-turn tool calling, as in Unsloth's *Qwen3 (4B) GRPO Tool Use*
notebook: tool results are masked out of the loss, and a tool error is sent back to the model instead of crashing.

In order:

1. **BFCL** on the SFT checkpoint.
2. Load the model with Unsloth's vLLM engine and the SFT adapter.
3. **GRPO**, checkpointing to Google Drive.
4. **PlannerQA-Bench** on the GRPO checkpoint; push adapter, merged weights and GGUF.
5. **BFCL** on the GRPO checkpoint, and a final base / SFT / GRPO comparison.

Use the same configuration as in `01_sft.ipynb` (same `SMOKE`, `MODEL_KEY`), so both notebooks share one Drive folder.
After a disconnect: *Run all* again.
"""
        ),
        md("## 1. Configuration"),
        code(
            '''
SMOKE = True                 # must match 01_sft.ipynb
MODEL_KEY = "qwen3-1.7b"     # must match 01_sft.ipynb
ENABLE_THINKING = False
REWARD_MODE = "graded_cost"  # "graded_cost" (default) | "graded" | "binary"
USE_DRIVE = True
RESUME = True
RUN_BFCL = True
WANDB_PROJECT = "tool-use-planner"
PUSH_TO_HUB = True
HF_PRIVATE = False
PUSH_GGUF = True
OVERRIDES = {}               # e.g. {"grpo_steps": 200}; see planner/config.py
'''
        ),
        md("## 2. Install and set up"),
        code(INSTALL),
        code(CLONE),
        code("OVERRIDES = dict(OVERRIDES, reward_mode=REWARD_MODE)\n" + SETUP.strip("\n")),
        md(
            """
## 3. BFCL on the SFT checkpoint

Run now, while the GPU is free. Base-model scores come from the cache written by `01_sft.ipynb`.
"""
        ),
        code(
            '''
from planner import bfcl
from planner.unsloth_utils import resolve_merged

SFT_LORA = os.path.join(PERSIST, "sft_lora")
SFT_MERGED = os.path.join(LOCAL, "sft_merged")
STAGE = f"grpo_{cfg.reward_mode}"                  # reward ablations get their own folders and cache entries
GRPO_DIR = os.path.join(PERSIST, STAGE)
GRPO_LORA = os.path.join(PERSIST, f"{STAGE}_lora")
GRPO_MERGED = os.path.join(LOCAL, f"{STAGE}_merged")
HUB_STAGE = "grpo" if cfg.reward_mode == "graded_cost" else STAGE.replace("_", "-")
assert os.path.exists(os.path.join(SFT_LORA, "adapter_model.safetensors")), "Run 01_sft.ipynb first (same SMOKE / MODEL_KEY)."

BFCL = {"base": bfcl.run_stage(CACHE, "base", cfg, PERSIST, enabled=RUN_BFCL)}
merged = resolve_merged(cfg, "sft", SFT_MERGED, HF_USER) if RUN_BFCL and not CACHE.has("bfcl_sft") else None
BFCL["sft"] = bfcl.run_stage(CACHE, "sft", cfg, PERSIST, local_model_path=merged, enabled=RUN_BFCL)
# A GRPO checkpoint that is finished but not scored yet (see the last section) is scored now, while the GPU is free.
if CACHE.has(f"bfcl_{STAGE}"):
    BFCL["grpo"] = CACHE.load(f"bfcl_{STAGE}")
elif RUN_BFCL and os.path.exists(os.path.join(GRPO_LORA, "adapter_model.safetensors")):
    BFCL["grpo"] = bfcl.run_stage(CACHE, STAGE, cfg, PERSIST, local_model_path=resolve_merged(cfg, HUB_STAGE, GRPO_MERGED, HF_USER), enabled=True)
shown = {k: v for k, v in BFCL.items() if v}
if shown: print(bfcl.format_table(shown))
'''
        ),
        md(
            """
## 4. Load the model and the SFT adapter

`fast_inference=True` gives Unsloth's vLLM engine, which generates the rollouts with the LoRA being trained.
"""
        ),
        code(
            BENCH_HELPERS
            + '''
FAST = has_vllm()                                  # no vLLM on a T4: rollouts fall back to transformers (slow)
model, tokenizer = load_model(cfg, fast_inference=FAST, gpu_memory_utilization=0.6)
fmt = setup_chat_format(tokenizer, enable_thinking=cfg.enable_thinking)
load_lora_weights(model, SFT_LORA)

BENCH = {"base": CACHE.load("bench_base") if CACHE.has("bench_base") else None, "sft": bench("sft", SFT_LORA)}
BENCH = {k: v for k, v in BENCH.items() if v}
print(format_report(BENCH))
'''
        ),
        md(
            """
## 5. The environment, the tools and the reward

The four tools are plain Python functions with type hints and docstrings; TRL derives their schemas and runs them
inside the rollout loop. `knowledge_search` is BM25 over a local Wikipedia-paragraph index and stands in for the
expensive retrieval tool (cost 1.0 per call; the others cost 0.25).

The reward needs no LLM judge. Every training task carries gold labels: the answer, and the evidence a correct plan
must retrieve. Evidence is read only from tool messages, which Python writes, so the model cannot fake it.

```
g = (min(relevance, factuality) - 1) / 3                       # 0, 1/3, 2/3 or 1
R = g - 0.3 * min(1, excess_cost / 3) - 0.2 * invalid_fraction   # "graded_cost"
```

`excess_cost` is the cost above the task's cheapest correct plan: necessary calls are free, redundant ones are not.
Taking the minimum of the two axes means guessing an answer without evidence earns at most 1/3, and retrieving
evidence without answering earns 0.
"""
        ),
        code(
            '''
from planner.rewards import ToolLedger, make_reward_fn
from planner.scoring import score_messages
from planner.tasks import grpo_rows
from planner.tools import MAX_TOOL_ROUNDS, TOOLS, execute_tool

task = next(t for t in env.tasks["grpo"] if t["family"] == "tool_bridge")
print("Q:", task["query"])
for calls in task["optimal_plan"]:
    for c in calls: print(" ", c["tool"], c["args"], "->", execute_tool(c["tool"], c["args"])[:160])
print("gold answer:", task["aspects"][0]["answers"], "| cheapest plan costs", task["optimal_cost"])
'''
        ),
        md(
            """
## 6. GRPO

Each optimizer step uses `grpo_batch * grpo_accum / grpo_num_generations` prompts with `grpo_num_generations` rollouts
each. Rollouts stop when the model answers without a tool call, after three tool rounds, or when
`max_completion_length` (which counts tool results too) is used up. No KL term (`beta = 0`).

Logged to W&B every step: the reward, good rate, both axes, tool cost, the share of groups whose rollouts all got the
same reward (those carry no learning signal), TRL's `tools/call_frequency` and `tools/failure_frequency`, and a running
count of calls per tool under `qps/` - an estimate of what this run would cost against real tools.
"""
        ),
        code(
            '''
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from planner.persist import latest_checkpoint, log_summary, wandb_init
from planner.unsloth_utils import filter_kwargs, save_adapter

os.makedirs(GRPO_DIR, exist_ok=True)
wandb_run, REPORT_TO = wandb_init(GRPO_DIR, WANDB_PROJECT, f"{STAGE}-{cfg.repo_stem}", cfg.to_dict())
ledger = ToolLedger(os.path.join(GRPO_DIR, "tool_ledger.json"))

if RESUME and os.path.exists(os.path.join(GRPO_LORA, "adapter_model.safetensors")):
    load_lora_weights(model, GRPO_LORA)
    print("GRPO already finished: adapter loaded from Drive")
else:
    FastLanguageModel.for_training(model)
    dataset = Dataset.from_list(grpo_rows(env.tasks["grpo"], cfg.seed))
    extra = {"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}} if fmt.is_thinking_model else {}
    args = GRPOConfig(**filter_kwargs(GRPOConfig, dict(
        temperature = 1.0,
        learning_rate = cfg.grpo_lr,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        beta = 0.0,
        logging_steps = 1,
        per_device_train_batch_size = cfg.grpo_batch,
        gradient_accumulation_steps = cfg.grpo_accum,
        num_generations = cfg.grpo_num_generations,
        max_completion_length = cfg.grpo_max_completion,
        max_tool_calling_iterations = MAX_TOOL_ROUNDS,
        max_steps = cfg.grpo_steps,
        save_steps = cfg.grpo_save_steps,
        save_total_limit = 2,
        seed = cfg.seed,
        report_to = REPORT_TO,
        run_name = f"{STAGE}-{cfg.repo_stem}",
        output_dir = GRPO_DIR,
        **extra,
    )))
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [make_reward_fn(cfg.reward_mode, ledger = ledger, print_every = 20)],
        tools = TOOLS,
        args = args,
        train_dataset = dataset,
    )
    checkpoint = latest_checkpoint(GRPO_DIR) if RESUME else None
    print("resume from checkpoint:", checkpoint)
    trainer.train(resume_from_checkpoint = checkpoint)
    save_adapter(model, tokenizer, fmt, GRPO_LORA)
print("tool calls made during RL:", dict(ledger.calls), "| rollouts:", ledger.rollouts, "| total cost:", round(ledger.total_cost, 1))
'''
        ),
        md("## 7. Evaluate the GRPO model"),
        code(
            '''
BENCH["grpo"] = bench(STAGE, GRPO_LORA)
print(format_report(BENCH))
for name, report in BENCH.items(): log_summary(wandb_run, f"bench_{name}", report["summary"])
'''
        ),
        md("## 8. Save and push"),
        code(
            '''
from planner.persist import model_card
from planner.unsloth_utils import export_model

card = model_card(
    f"{cfg.repo_stem}-{HUB_STAGE}",
    cfg.base_model_hf,
    f"Tool-calling planner after SFT and GRPO (reward: `{cfg.reward_mode}`). It plans with `knowledge_search`, `calculator`, "
    "`get_weather` and `get_election` for at most three tool rounds and ends with an `Answer:` line. Code: " + REPO_URL,
    {"PlannerQA-Bench (held-out)": format_report(BENCH), "BFCL": bfcl.format_table({k: v for k, v in BFCL.items() if v}) if any(BFCL.values()) else "not run"},
)
status = export_model(model, tokenizer, fmt, cfg, HUB_STAGE, HF_USER, GRPO_LORA, GRPO_MERGED, card, push = PUSH_TO_HUB, private = HF_PRIVATE, gguf = PUSH_GGUF)
print(status)
'''
        ),
        md(
            """
## 9. BFCL on the GRPO checkpoint, and the final comparison

BFCL needs most of the GPU for its own vLLM server. The cell first tries to release the training model. If the GPU is
still busy it says so: then choose *Runtime > Restart session* and *Run all*. Every finished stage is skipped, and the
BFCL cell in section 3 scores the GRPO checkpoint.
"""
        ),
        code(
            '''
from planner.unsloth_utils import gpu_free_fraction, release_gpu

if RUN_BFCL and not BFCL.get("grpo") and has_vllm():
    try: del trainer
    except NameError: pass
    del model
    free = release_gpu()
    if free >= 0.7:
        BFCL["grpo"] = bfcl.run_stage(CACHE, STAGE, cfg, PERSIST, local_model_path=GRPO_MERGED, enabled=True, gpu_memory_utilization=round(0.9 * free, 2))
    else:
        print(f"Only {free:.0%} of the GPU is free. Restart the session and Run all: section 3 will score the GRPO checkpoint.")

report = "## PlannerQA-Bench (800 held-out tasks)\\n\\n" + format_report(BENCH)
shown = {k: v for k, v in BFCL.items() if v}
if shown:
    report += "\\n\\n## BFCL (official checker, offline categories)\\n\\n" + bfcl.format_table(shown)
    for name, result in shown.items(): log_summary(wandb_run, f"bfcl_{name}", dict(result["groups"], bfcl_core=result["bfcl_core"]))
report += f"\\n\\n## Tool calls made during RL\\n\\n{dict(ledger.calls)} over {ledger.rollouts} rollouts (cost {ledger.total_cost:.1f})\\n"
with open(os.path.join(PERSIST, f"report_{STAGE}.md"), "w") as f: f.write(report)
if wandb_run: wandb_run.finish()

from IPython.display import Markdown, display
display(Markdown(report))
'''
        ),
        md(
            """
### Reading the result

- **SFT** is expected to move BFCL and the simpler task families: it teaches formats, schemas and the tool inventory
  with dense per-token supervision.
- **GRPO** is expected to move the hard families (`kb_bridge`, `tool_bridge`, `composite`) and the tool cost, because it
  trains on the states the model itself reaches and optimises the outcome directly. On BFCL it should hold steady: the
  GRPO environment has a fixed tool set, so BFCL here is a check that general function calling was not lost.
- The failure-tag table shows which error types each stage removed.

These are expectations, not guarantees: compare them with the tables above. To try a different reward, change
`REWARD_MODE` and run again - SFT results and the environment are reused.
"""
        ),
    ]
    return cells


# ----------------------------------------------------------------- 03_demo
def demo_notebook():
    cells = [
        md(
            """
# 03 - Demo: try the planner

Serves the trained checkpoints behind OpenAI-compatible endpoints and opens a small web app where reviewers can run the
five preset hard queries, or their own, and compare checkpoints side by side.

How the pieces fit:

- **Model server.** `SERVER = "unsloth"` runs Unsloth Studio's `unsloth run` on the GGUF builds pushed by the training
  notebooks. `SERVER = "vllm"` serves the merged weights with vLLM instead. `SERVER = "external"` uses endpoints you
  already run somewhere (for example Unsloth Desktop on your own machine).
- **Tool loop.** Unsloth's chat window can only run its own built-in tools, not our four. So this notebook's client
  sends our tool schemas with each request, executes the calls the model asks for against the local mock world and
  knowledge base, and returns the results - at most three rounds.

A T4 is enough for `SERVER = "unsloth"`; `SERVER = "vllm"` needs an L4 or A100.
"""
        ),
        code(
            '''
HF_USER = ""                 # your Hugging Face user name ("" = take it from the HF_TOKEN secret)
MODEL_KEY = "qwen3-1.7b"     # must match the training notebooks
SMOKE = False                # True to serve the "-smoke" repositories from a SMOKE training run
STAGES = ["sft", "grpo"]     # trained checkpoints to serve
INCLUDE_BASE = True          # also serve the untuned base model for comparison
SERVER = "unsloth"           # "unsloth" | "vllm" | "external"
EXTERNAL = {}                # for SERVER = "external": {"grpo": ("https://host/v1", "sk-unsloth-...")}
GGUF_VARIANT = "Q8_0"
SHARE = True                 # public Gradio link for reviewers (valid while this runtime is alive)
'''
        ),
        md("## 1. Install and set up"),
        code(
            '''
!pip install -q openai gradio bm25s==0.3.16 "datasets>=4.7.0,<5.0.0" huggingface_hub
if SERVER == "unsloth":
    !curl -fsSL https://unsloth.ai/install.sh | sh
elif SERVER == "vllm":
    !pip install -q vllm
'''
        ),
        code(CLONE),
        code(
            '''
from planner.config import get_config
from planner.persist import read_secrets
from planner.setup_env import setup_environment

cfg = get_config("smoke" if SMOKE else "a100", MODEL_KEY)
read_secrets(("HF_TOKEN",))
if not HF_USER and os.environ.get("HF_TOKEN"):
    from huggingface_hub import whoami
    HF_USER = whoami(token=os.environ["HF_TOKEN"])["name"]
assert HF_USER or SERVER == "external", "Set HF_USER (or add an HF_TOKEN secret)."

# Rebuild the same world and knowledge base the models were trained against (about a minute).
world, corpus = setup_environment(cache_dir="/content/planner_env", n_train=cfg.hotpot_train_rows, n_val=cfg.hotpot_val_rows, seed=cfg.seed)
'''
        ),
        md(
            """
## 2. Start the model servers

Each checkpoint gets its own port. With `unsloth run`, the API key it prints on start-up is read from its log.
"""
        ),
        code(
            '''
import shutil
from planner.demo import OpenAIBackend
from planner.serve import Endpoint, start_unsloth, start_vllm, wait_ready

os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.path.expanduser("~/.unsloth/bin") + os.pathsep + os.environ["PATH"]
names = (["base"] if INCLUDE_BASE else []) + STAGES
endpoints = {}
if SERVER == "external":
    endpoints = {name: Endpoint(name, url, key) for name, (url, key) in EXTERNAL.items()}
else:
    for i, name in enumerate(names):
        port, log = 8001 + i, f"/content/server_{name}.log"
        try:
            if SERVER == "unsloth":
                assert shutil.which("unsloth"), "the `unsloth` command is not on PATH: check the install cell's output"
                repo = f"unsloth/{cfg.base_model_hf.split('/')[-1]}-GGUF" if name == "base" else cfg.repo(HF_USER, name, "GGUF")
                endpoints[name] = wait_ready(start_unsloth(name, repo, port, log, GGUF_VARIANT), key_from_log=True)
            else:
                repo = cfg.base_model_hf if name == "base" else cfg.repo(HF_USER, name)
                endpoints[name] = wait_ready(start_vllm(name, repo, port, log, gpu_memory_utilization=round(0.85 / len(names), 2)))
            print(f"{name}: ready at {endpoints[name].base_url}")
        except Exception as error:
            print(f"{name}: could not start ({error})")

backends = {name: OpenAIBackend(e.base_url, e.api_key) for name, e in endpoints.items()}
assert backends, "No endpoint is up. Check /content/server_*.log, or try SERVER = 'vllm'."
print({name: b.model for name, b in backends.items()})
'''
        ),
        md(
            """
## 3. Five hard queries

| # | type | what a good plan does |
|---|---|---|
| 1 | bridge / multi-hop | search, then search again with an entity found in the first result |
| 2 | decomposition + fan-out + synthesis | look up two entities in parallel, then compare |
| 3 | multi-date fan-out + calculation | three weather lookups in one round, then the calculator |
| 4 | cross-tool bridge + tool selection | `get_election` gives the capital and the date for `get_weather` |
| 5 | multi-part ask + query reformulation | split the question; the knowledge part needs a follow-up query |

They come from the held-out set, so each has a reference plan and gold answer, and the output shows the relevance and
factuality scores along with failure tags.
"""
        ),
        code(
            '''
from IPython.display import Markdown, display
from planner.demo import answer, format_expected, format_markdown, load_examples

examples = load_examples()
for i, example in enumerate(examples, 1):
    task = example["task"]
    display(Markdown(f"## {i}. {example['label']}\\n\\n> {task['query']}\\n\\n{example['why']}\\n\\n{format_expected(task)}"))
    for name, backend in backends.items():
        try:
            display(Markdown(format_markdown(answer(task, backend), title=name)))
        except Exception as error:
            print(f"{name}: request failed ({error})")
'''
        ),
        md(
            """
## 4. Web app for reviewers

Pick a preset or type any question. With `SHARE = True` Gradio prints a public `gradio.live` link that works while this
runtime is running. Weather and elections belong to a fictional world (regions such as the ones in the presets), so
ask about those; `knowledge_search` covers the Wikipedia paragraphs of the HotpotQA questions used to build the index.
"""
        ),
        code(
            '''
from planner.demo import build_ui

ui = build_ui(backends, examples)
ui.launch(share=SHARE, debug=False)
'''
        ),
        md(
            """
## Hosting it yourself with Unsloth Desktop / Studio

On your own machine, with [Unsloth](https://unsloth.ai/docs/get-started/unsloth-studio) installed:

```bash
unsloth run --model <hf-user>/qwen3-1.7b-planner-grpo-GGUF:Q8_0 -p 8888 --disable-tools
```

It prints the endpoint URL and an API key. Then, from a clone of this repository:

```bash
python -m planner.demo --base-url http://127.0.0.1:8888/v1 --api-key sk-unsloth-... --example 4
```

`unsloth studio --secure` publishes the endpoint on a Cloudflare HTTPS URL for remote reviewers; use that URL as
`--base-url` (the client already sends `stream: false`, which the tunnel requires). You can also load the GGUF in the
Unsloth chat window to talk to the model directly, but its tool calls will not be executed there: only the client in
this repository runs the four planner tools.

When you are done, stop the servers:
"""
        ),
        code(
            '''
for endpoint in endpoints.values(): endpoint.stop()
'''
        ),
    ]
    return cells


def write(name: str, cells) -> None:
    notebook = new_notebook(cells=cells)
    notebook.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for i, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{i:02d}"  # stable ids -> stable diffs
    nbformat.validate(notebook)
    path = os.path.join(ROOT, "notebooks", name)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
    print("wrote", path, f"({len(cells)} cells)")


if __name__ == "__main__":
    write("01_sft.ipynb", sft_notebook())
    write("02_grpo.ipynb", grpo_notebook())
    write("03_demo.ipynb", demo_notebook())
