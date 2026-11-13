# Tool-calling planner: SFT + GRPO on Colab

A minimal end-to-end example of training a **planner** that answers a question by calling tools: one or several
calls per round, up to three rounds, then a final answer. It is trained with supervised fine-tuning and then with
GRPO on a local mock environment, evaluated on BFCL and on a purpose-built planning benchmark, pushed to the Hugging
Face Hub and served for a small demo app.

| notebook | what it does | runtime (Qwen3-1.7B, A100, estimate) |
|---|---|---|
| [`notebooks/01_sft.ipynb`](notebooks/01_sft.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/tool_use_demo/blob/main/notebooks/01_sft.ipynb) | base-model evaluation, SFT data mix, SFT, push | ~2-3 h |
| [`notebooks/02_grpo.ipynb`](notebooks/02_grpo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/tool_use_demo/blob/main/notebooks/02_grpo.ipynb) | GRPO with real multi-round tool rollouts, evaluation, push | ~3-6 h |
| [`notebooks/03_demo.ipynb`](notebooks/03_demo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yy726/tool_use_demo/blob/main/notebooks/03_demo.ipynb) | serve the checkpoints, five preset hard queries, web app | minutes |

The run times are estimates and have not been measured. Use the `SMOKE = True` switch at the top of each notebook
first: it runs the whole pipeline on tiny data in minutes.

## Quick start

1. In Colab, add the secrets `HF_TOKEN` (write access) and `WANDB_API_KEY` (key icon in the left bar). If this
   repository is private, also add `GITHUB_TOKEN`.
2. Open `01_sft.ipynb` with its *Open in Colab* button on an A100 or L4 runtime, leave `SMOKE = True`, and *Run all*.
3. Do the same with `02_grpo.ipynb`. When both pass, set `SMOKE = False` in both and run them again.
4. Open `03_demo.ipynb` to try the models.

**After a disconnect, just *Run all* again.** Checkpoints, the built environment, the training data and every
finished evaluation stage are on Google Drive under `MyDrive/tool_use_demo/<experiment>/`. Finished stages are
skipped, trainers resume from their last checkpoint, and the W&B run continues under the same id.

## The task

Four tools, executed locally and deterministically:

| tool | backed by | cost |
|---|---|---|
| `knowledge_search(query)` | BM25 over ~83k Wikipedia paragraphs (from HotpotQA) | 1.0 |
| `calculator(expression)` | safe arithmetic evaluator | 0.25 |
| `get_weather(city, date)` | synthetic database | 0.25 |
| `get_election(region, year)` | synthetic database | 0.25 |

`knowledge_search` stands in for an expensive, rate-limited retrieval tool. Regions, cities and candidates are
fictional, so those facts can only come from the specialised tools; `get_weather` and `get_election` act as negatives
for each other.

A rollout is at most **three rounds of tool calls** followed by a turn with no tool call that ends in `Answer: ...`
(a stand-in for the downstream summarizer). The same loop runs in the trainer, the benchmark and the demo.

## Two benchmarks

**BFCL** (official `bfcl-eval` generation and checker) measures general function calling: unseen schemas, parallel
calls, irrelevance, multi-turn. It is evaluation only. We run the offline categories (~4,400 entries) and report the
four official groups and their mean, *BFCL-core*. The agentic group is not run, so BFCL-core is not the leaderboard's
overall score.

**PlannerQA-Bench** (this repository) measures planning with our tools. 800 held-out tasks, 100 per family:

| family | tests | cheapest correct plan |
|---|---|---|
| `kb_bridge` | multi-hop, query rewriting | search, then search with an entity from the first result |
| `kb_compare` | decomposition, fan-out, synthesis | two searches in one round |
| `weather_lookup` / `election_lookup` | tool selection, arguments, explicit date / year | one specialised call |
| `fanout_calc` | fan-out over cities or dates, then synthesis | parallel lookups, then `calculator` |
| `tool_bridge` | cross-tool multi-hop | `get_election`, then `get_weather(capital, election date)` |
| `composite` | two-part asks | specialised call and search in parallel, maybe one more search |
| `no_tool` | restraint | answer directly |

Each rollout is scored on two axes from 1 to 4 and the final score is the minimum; 3 and 4 count as good.

- **Relevance**: did the plan go after every part of the question, without off-topic calls?
- **Factuality**: is the answer correct and backed by evidence the tools returned?

Evidence is read only from tool messages, which Python writes. Bad rollouts are tagged with failure types
(`specialized_tool_not_called`, `wrong_tool`, `wrong_arguments`, `constraint_missed`, `not_decomposed`,
`query_rewrite_failed`, `multi_hop_incomplete`, `synthesis_error`, ...), so the report shows which errors each
training stage removed. The scoring is a rule-based approximation on synthetic, gold-labelled tasks; it is not an LLM
autograder.

For reference, the scripted expert used for SFT demonstrations scores 92% overall on the held-out set: 69% on
`kb_bridge`, 89% on `kb_compare`, 78% on `composite`, 100% elsewhere.

## Training

**SFT** (`01_sft.ipynb`): about 10k ToolACE and 1.1k Hermes function-calling conversations for general function
calling, plus about 1.9k in-domain demonstrations from the scripted expert (only rollouts with a perfect score are
kept). Unsloth + TRL `SFTTrainer`, LoRA rank 32, loss on assistant turns only.

**GRPO** (`02_grpo.ipynb`): TRL `GRPOTrainer` with native multi-round tool calling, as in Unsloth's
*Qwen3 (4B) GRPO Tool Use* notebook. 4,000 training tasks, 8 rollouts per prompt, tool results masked out of the loss.
The reward is rule-based:

```
g = (min(relevance, factuality) - 1) / 3
R = g - 0.3 * min(1, excess_cost / 3) - 0.2 * invalid_fraction
```

`excess_cost` is the cost above the task's cheapest correct plan. `REWARD_MODE` switches between `graded_cost`
(default), `graded` and `binary`. Every step logs the good rate, both axes, tool cost, the share of zero-variance
groups, and a running count of tool calls (the "QPS bill" of the run) to W&B.

The split of responsibilities: SFT teaches formats, schemas and the tool inventory with dense supervision and gets the
model to a starting success rate where GRPO has signal; GRPO then trains on the states the model itself reaches and
optimises the outcome and the cost directly. BFCL is expected to move mostly in the SFT stage; in the GRPO stage it
is a check that general function calling was not lost. These are expectations to compare against your results.

## Serving and demo

Each checkpoint is pushed as three repositories: `<user>/qwen3-1.7b-planner-{sft,grpo}-lora` (adapter),
`...-{sft,grpo}` (merged 16-bit) and `...-{sft,grpo}-GGUF`.

`03_demo.ipynb` serves them with Unsloth (`unsloth run` on the GGUF builds) or vLLM and opens a Gradio app with five
preset hard queries (bridge, decomposition and fan-out, multi-date fan-out, cross-tool bridge, multi-part ask with
query reformulation). The model server only produces the model's turns: the client in `planner/demo.py` executes the
four tools, because a hosted chat window can only run its own built-in tools. To use an endpoint you host yourself:

```bash
unsloth run --model <hf-user>/qwen3-1.7b-planner-grpo-GGUF:Q8_0 -p 8888 --disable-tools
```

```bash
python -m planner.demo --base-url http://127.0.0.1:8888/v1 --api-key sk-unsloth-... --example 4
```

## Repository layout

```
planner/
  tools.py, world.py, corpus.py   the four tools, the synthetic world, the BM25 knowledge base
  tasks.py                        the eight task families, gold labels, cheapest plans
  trace.py, scoring.py            rollout parsing; relevance / factuality scores and failure tags
  rewards.py                      GRPO reward functions and the tool-call ledger
  bench.py                        PlannerQA-Bench loop and report
  expert.py, data_public.py       SFT data: scripted expert, ToolACE / Hermes converters
  chat.py                         one chat template for SFT, GRPO, benchmark and serving
  bfcl.py, bfcl_handler.py        official BFCL in its own virtualenv; non-thinking Qwen3 handler
  persist.py, setup_env.py        Drive persistence, stage cache, environment build / cache
  unsloth_utils.py, serve.py      GPU helpers for the notebooks; model servers for the demo
  demo.py                         OpenAI-compatible tool-loop client, Gradio app, CLI
scripts/build_notebooks.py        generates the notebooks (edit cells there)
scripts/make_demo_examples.py     picks the five preset queries -> demo/examples.json
tests/                            unit tests (CPU only)
```

## Tests

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests -q
```

The tests cover the tools, task generation, rollout parsing, both score axes and every failure tag, the rewards, the
benchmark and demo loops, the data converters, persistence and the BFCL wrapper. Tests that need `trl`, `bfcl-eval`
or network access to download the Qwen3 tokenizer skip themselves when those are missing.

## What has and has not been verified

Verified on CPU: all of the above tests; a full-size build of the environment (83k passages, 2,000 / 4,000 / 800
tasks, about one minute); conversion and rendering of the real ToolACE and Hermes data with the real Qwen3 tokenizer;
the benchmark, reward and demo loops driven by the scripted expert; registration of the non-thinking BFCL handler with
the real `bfcl-eval` package.

A Colab A100 run of `01_sft.ipynb` got as far as the first benchmark and exposed the `bm25s` / JAX problem below and
an install line that failed silently; both are fixed, but the fixed notebook has not been re-run on Colab yet.

Not verified, because it needs a GPU and Colab: the training runs themselves and any resulting scores, checkpoint
resume inside the Unsloth-patched trainers, the vLLM server started for BFCL, GGUF export, and `unsloth run` on Colab.
The `SMOKE` mode exists to check exactly these quickly. Known fallbacks: set `RUN_BFCL = False` to skip BFCL, use
`SERVER = "vllm"` in the demo notebook, and set `PUSH_GGUF = False` to skip the GGUF build.

Two details worth knowing:

- `bfcl-eval` pins old versions of numpy and other packages, so it runs from a separate virtualenv and talks to a
  vLLM server. In `01_sft.ipynb` that server also has its own virtualenv, so Colab's torch and numpy are left alone;
  `02_grpo.ipynb` needs vLLM in the notebook's own environment (Unsloth's `fast_inference` engine) and installs it there.
- `bm25s` switches to JAX when JAX is installed, and JAX pre-allocates 75% of the GPU (about 30 GB of an A100-40GB,
  invisible to PyTorch). The first `knowledge_search` call therefore made the next generation step run out of memory.
  `planner/corpus.py` imports `bm25s` with JAX hidden and selects with numpy.
- Everything lives on the `main` branch. Colab cannot resolve notebook paths on branches with a `/` in the name
  (saving back to GitHub then creates a stray folder named after the branch), so the notebooks clone `main`.
- TRL's tool loop only accepts chat templates it recognises, and Unsloth's tokenizers ship modified Qwen3 templates.
  `planner/chat.py` resets the tokenizer to TRL's canonical template and uses TRL's training variant (which always
  writes the `<think>` block of an assistant turn) for SFT text and benchmark prompts, so every stage sees one format.

## Possible extensions

- **Rejection sampling with a teacher** (see `code/rejection_sampling` in the RLHF book): seed synthetic queries from
  the failure tags, have a stronger model produce several trajectories each, keep those that `scoring.py` rates 4,
  and add them to the SFT mix. The scripted expert with its perfect-score filter is the simplest form of this.
- More environments in the GRPO stage through TRL's `environment_factory`, including stateful ones.
- A Hugging Face Space for a demo that stays online without a Colab runtime.

## References

- Unsloth notebooks: *Qwen3 (4B) Tool Calling* and *Qwen3 (4B) GRPO Tool Use*; Unsloth tool-calling guide.
- ToolRL (arXiv 2504.13958), Nemotron-Research-Tool-N1 (2505.00024), Search-R1 (2503.09516), PROVE (2606.03892),
  CM2 (2602.12268), TMax (2606.23321), OpenThoughts-Agent (2606.24855); RLHF book, chapter 13. Notes on these are
  in `arxiv_docs/`.
- Data: HotpotQA, Team-ACE/ToolACE, NousResearch/hermes-function-calling-v1. Benchmark: Berkeley Function-Calling
  Leaderboard (BFCL V4).
