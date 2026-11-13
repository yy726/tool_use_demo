# Design decisions and presentation Q&A

Tool-calling planner: SFT + GRPO on Colab. This is a preparation sheet for a live demo: each section states the
decision, the reason, and the trade-off, followed by questions a reviewer is likely to ask.

**Status of the evidence.** Both training notebooks have run end to end on a Colab A100, but only in `SMOKE` mode
(48 evaluation tasks, 12 SFT steps, 4 GRPO steps). Those numbers prove the pipeline works; they are not evidence about
model quality. Every statement about expected improvement below is a design expectation until the full run is done.

---

## 1. One-minute summary

- **Goal.** Train a small model to act as a *planner*: read a question, call tools in one or several steps (several
  calls per step when they are independent), and stop with an answer.
- **Pipeline.** Base model -> supervised fine-tuning (SFT) -> reinforcement learning with GRPO, each stage evaluated on
  two benchmarks.
- **Environment.** Four local, deterministic tools with different costs; at most three rounds of tool calls.
- **Measurement.** BFCL for general function calling, and PlannerQA-Bench (built here) for planning with our tools.
- **Reward.** Rule-based, the same code as the benchmark score, with a penalty for unnecessary tool cost.

---

## 2. Problem definition

| Decision | Reason | Trade-off |
|---|---|---|
| Train the planner only, not the summarizer | The planner is where tool cost and multi-step errors come from | The final `Answer: ...` turn is a stand-in; composition and presentation are not scored |
| At most 3 tool rounds, 4 calls per round, 4 searches per query | Mirrors a latency and QPS budget; keeps rollouts short enough for RL | Questions needing more hops cannot be solved |
| The same loop in training, benchmark and demo | No train / test mismatch | Less freedom to tune each separately |
| Four tools with unequal cost | `knowledge_search` (1.0) models an expensive, rate-limited retriever; the others cost 0.25 | Costs are constants, not measured latency |
| `get_weather` and `get_election` share place names | They act as negatives for each other, so tool selection is testable | Synthetic, so less natural language variety |

---

## 3. Model

| Decision | Reason | Trade-off |
|---|---|---|
| Qwen3-1.7B by default, Qwen3-4B-Instruct as a switch | Fits the budget (~400 Colab units) for the whole base -> SFT -> GRPO chain with room for re-runs | Lower absolute quality than a larger model |
| LoRA (rank 32, alpha 64, all projection layers) | Small checkpoints; the SFT adapter continues directly into GRPO | Less capacity than full fine-tuning |
| 16-bit weights, no 4-bit quantisation | It fits on an A100; avoids quantisation noise under a very small RL learning rate | More memory |
| Thinking mode off by default | A planner should be low latency; all three checkpoints are compared in the same mode | Loses whatever reasoning the thinking block would add |

---

## 4. Data

### SFT (about 13k conversations)

| Source | Size | What it teaches |
|---|---|---|
| ToolACE | ~10,000 | General function calling: unseen schemas, parallel calls, declining to call |
| Hermes function calling | ~1,100 | Same, in the format of the Unsloth tool-calling notebook |
| Scripted-expert rollouts | ~1,900 | Our four tools and our task shapes |

- The expert really executes the tools, and only rollouts scoring 4 on both axes are kept. This is the simplest form
  of rejection sampling.
- The expert is deliberately plain (search with the question, then with an entity seen in the results). It scores 92%
  on the held-out set and only 69% on `kb_bridge`, which leaves room for RL to beat the demonstrations.

### GRPO (4,000 tasks, 500 per family)

- 100% in-domain. TRL's `GRPOTrainer` takes one fixed tool list, so per-example schemas (as in ToolACE) cannot be mixed in.
- Knowledge base: about 83k paragraphs from HotpotQA (distractor setting) behind BM25. HotpotQA ships gold supporting
  paragraphs, so "was the evidence retrieved" can be checked by program.
- Weather and elections are a synthetic, fictional world. The model cannot answer from memory and search cannot find
  them, so the specialised tools are required.

### Isolation

SFT uses one third of the HotpotQA training questions, GRPO the other two thirds, evaluation the validation split.
Training and evaluation regions of the synthetic world do not overlap.

---

## 5. Benchmarks

Two benchmarks, because they answer different questions.

**BFCL** (official `bfcl-eval` generation and checker)
- Measures general function calling, and whether GRPO damaged it.
- Offline categories only. The reported mean, *BFCL-core*, is not the leaderboard's overall score (the agentic group
  needs a paid search key).
- All checkpoints are evaluated in non-thinking mode through one registered model id, so the comparison is like for like.

**PlannerQA-Bench** (this repository; 800 held-out tasks, 100 per family)

| Family | Tests |
|---|---|
| `kb_bridge` | Multi-hop, query rewriting |
| `kb_compare` | Decomposition, fan-out, synthesis |
| `weather_lookup`, `election_lookup` | Tool selection, arguments, explicit date / year |
| `fanout_calc` | Parallel lookups, then `calculator` |
| `tool_bridge` | Cross-tool multi-hop |
| `composite` | Two-part questions |
| `no_tool` | Restraint: answer without calling anything |

Scoring:
- Two axes, 1 to 4 each; the final score is the **minimum**; 3 and 4 count as good. This follows the structure of the
  production autograder, keeping its two most important axes.
- **Relevance:** did some call address every part of the question, without off-topic calls?
- **Factuality:** is the answer correct *and* was the required evidence returned by a tool? A correct answer without
  evidence (a guess) gets at most 2.
- **Failure tags** (`wrong_tool`, `constraint_missed`, `not_decomposed`, `query_rewrite_failed`,
  `multi_hop_incomplete`, `synthesis_error`, ...) show which kind of error each training stage removed.

---

## 6. Reward

```
g = (min(relevance, factuality) - 1) / 3
R = g - 0.3 * min(1, excess_cost / 3) - 0.2 * invalid_fraction
```

| Decision | Reason |
|---|---|
| Same scoring code as the benchmark | The thing optimised is the thing measured |
| No LLM judge | Deterministic, free, not rate limited; about 12,800 rollouts would be costly and noisy to judge |
| Graded, not binary | Partial credit gives a learning signal on hard families where most rollouts fail |
| Minimum of the two axes, not the mean | Both must be good; guessing an answer or piling up searches cannot score well |
| Penalise only cost *above* the cheapest correct plan | Necessary calls are free; otherwise the model learns to call less rather than to call well |
| Cost penalty capped at 0.3 | A correct answer always beats a cheap wrong one |
| Separate 0.2 penalty for invalid calls | Direct signal for malformed calls and bad arguments |
| `REWARD_MODE` switch: `graded_cost`, `graded`, `binary` | Makes the ablation a one-line change |

Guards against reward hacking:
- Evidence is read only from `role == "tool"` messages, which Python writes; the model cannot fabricate it.
- Repeated identical calls still cost; going over budget gives the lowest score.
- Logged every step: share of zero-variance groups and tool calls per tool (the run's "QPS bill").

---

## 7. KL divergence: include it or not?

**Current setting: `beta = 0` (no KL term).**

Reasons not to include it:
- It matches the reference implementation (the Unsloth GRPO notebook) and TRL's default.
- Drift is already limited by a small learning rate (5e-6), LoRA-only updates, 400 steps, and the clipping in the GRPO loss.
- A KL term needs a second forward pass through a reference model, which is expensive on long multi-turn sequences.
- It pulls the policy back toward the SFT starting point, and the SFT demonstrations are what we want to beat
  (the expert reaches only 69% on `kb_bridge`).

Reasons to include it:
- GRPO is 100% in-domain with no general function-calling data as an anchor, so BFCL could drop.
- It protects against collapse into a single high-reward pattern.

**Recommendation.** Start without it and treat BFCL after GRPO as the regression check. If BFCL drops clearly, or the
share of zero-variance groups rises quickly, add a small `beta` (0.01 to 0.04) as an ablation. It is a switch decided
by data, not a matter of principle.

---

## 8. Training procedure

- **SFT first, then GRPO.** SFT is dense, per-token supervision: good for formats, schemas and the tool inventory, and
  it lifts the starting success rate to where RL has signal. GRPO trains on states the model itself reaches and
  optimises outcome and cost, which is where there are no gold labels: which rewrite works for this retriever, how the
  second hop depends on the first, when to stop.
- **Expectation (not yet verified at full scale).** BFCL moves mostly in SFT. GRPO should help `kb_bridge`,
  `tool_bridge`, `composite` and reduce tool cost.
- **TRL `GRPOTrainer` with native multi-turn tool calling**, not a custom loop. Tool results are masked out of the
  loss; a tool error is returned to the model instead of crashing the rollout.
- **8 rollouts per prompt at temperature 1.0**, because the advantage is computed within the group and needs variety.
- **`max_completion_length = 2048`**, because it counts tool results too (four searches are about 1,600 tokens).

| | SFT | GRPO |
|---|---|---|
| Learning rate | 1e-4, cosine | 5e-6 |
| Length | 2 epochs | 400 steps |
| Batch | 16 x 2 accumulation | 4 prompts x 8 rollouts per step |
| Loss on | assistant turns only | model tokens only (tool results masked) |

---

## 9. Engineering

- Logic lives in the `planner/` package with 130 CPU unit tests; the notebooks are thin.
- Checkpoints, data, and every finished evaluation stage are on Google Drive. After a disconnect, *Run all* skips
  finished stages, resumes the trainer and continues the same W&B run.
- Each checkpoint is pushed as three Hub repositories: adapter, merged 16-bit weights, GGUF.
- In the demo, the model server and the tool executor are separate: Unsloth hosts the model, and our client runs the
  four tools, because a hosted chat window can only run its own built-in tools.

---

## 10. Results so far (SMOKE run only)

48 held-out tasks (6 per family), 12 SFT steps, 4 GRPO steps. One task is about 2 percentage points.

| Metric | Base | SFT | GRPO |
|---|---|---|---|
| Good rate | 62.5% | 72.9% | 72.9% |
| Relevance (1-4) | 3.33 | 3.73 | 3.77 |
| Factuality (1-4) | 2.75 | 3.31 | 3.33 |
| Excess cost per query | 0.16 | 0.04 | 0.04 |
| Invalid call rate | 4.4% | 1.2% | 1.2% |
| Parallel rate on fan-out tasks | 50.0% | 88.9% | 94.4% |
| BFCL-core (3 smoke categories) | 86.5 | 85.4 | not run |

How to read this honestly:
- SFT moved the planning benchmark and cut invalid and excess calls, even with 12 steps.
- Four GRPO steps cannot change a policy; the GRPO column only shows that the stage runs.
- SFT regressed on `composite` (33% -> 0%) and `no_tool` (100% -> 67%) in this tiny sample. With 6 tasks per family
  this may be noise, but it is the first thing to check in the full run.
- BFCL on the GRPO checkpoint was skipped in that session because the GPU was not released; re-running the notebook
  after a restart scores it.

---

## 11. Likely questions

**Why not use an LLM judge as the reward, like your production autograder?**
Cost, rate limits and noise at about 12,800 rollouts, plus the risk of the policy learning the judge's blind spots.
On synthetic tasks with gold labels a rule-based score is exact. The plan would be to validate against the real
autograder on a sample before trusting the transfer.

**Why two axes instead of all five?**
Relevance and factuality are the ones the planner controls. Groundedness is folded into factuality (evidence must come
from tool output). Composition and presentation belong to the summarizer.

**Is the minimum too harsh as a reward?**
It is harsh on purpose: it removes the two cheap strategies (guess, or search a lot and not answer). The graded scale
keeps the signal from becoming sparse.

**Could the model game the cost penalty by not calling tools?**
No: without evidence factuality is at most 2, so `g` is at most 1/3, and the cost penalty only applies above the
cheapest correct plan.

**What does RL add over more SFT data?**
SFT can only imitate the demonstrations. RL optimises the outcome on the model's own trajectories and can learn
behaviour nobody labelled, such as query rewrites that work for this particular retriever, and cost awareness.

**Why did you not mix general function-calling tasks into GRPO?**
The trainer's tool list is fixed at construction. Forgetting is monitored with BFCL rather than prevented by design;
adding KL or an `environment_factory` with several environments are the two ways to address it.

**How do you know the benchmark is not leaking?**
Disjoint HotpotQA splits, disjoint synthetic regions, fixed seeds, and BFCL is never trained on.

**Why BM25 and not a dense retriever?**
Deterministic, fast, no GPU, and good enough to make query rewriting matter. The learned rewriting strategy is specific
to this retriever, which is a limitation.

**What would you do next?**
1. The full run, with a `binary` vs `graded` vs `graded_cost` reward ablation.
2. A KL ablation if BFCL regresses.
3. Rejection sampling with a stronger teacher, seeded from the failure tags, to improve the SFT data.
4. Validate the rule-based score against the real autograder on a sample.

---

## 12. Known limitations to state up front

- The rule-based score approximates the production autograder; transfer is unverified.
- Synthetic tools and constant costs; no real latency or rate limiting.
- Only SMOKE-scale results exist so far.
- GRPO has no anchor for general function calling other than the BFCL check.
- Answer matching is normalised containment, which can accept a few wrong answers and reject a few right ones.
