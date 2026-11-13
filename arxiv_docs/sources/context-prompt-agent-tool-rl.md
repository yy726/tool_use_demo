# 背景上下文：工具使用智能体的后训练、RL 环境、Harness 与沙箱（文献调研结论，截至 2026-10）

你将在 `rlhf-book` 仓库（Nathan Lambert 的 RLHF Book）中写代码。下面是上一个会话的文献调研结论，作为设计和实现的参考。数字都是论文作者自报；引用某篇 2026 年论文的具体细节前，请先打开 arXiv 页面核实。

## 0. 仓库约定（来自 CLAUDE.md）
- 一律用 `uv run python`，不要用裸 `python`；依赖在 `code/pyproject.toml`，需要时 `cd code/ && uv sync`。
- 现有示例在 `code/`：`policy_gradients/`（GRPO/PPO）、`reward_models/`、`direct_alignment/`、`distillation/`、`rejection_sampling/`、`instruction_tuning/`。新工作尽量复用 `policy_gradients/` 的写法和配置风格。
- 跑实验遵循 `.claude/skills/run-rlhf-code-experiment/SKILL.md`：先缩小数据跑短程，长任务放后台并监控，可用 `WANDB_MODE=disabled`。
- 书中对应章节：`book/chapters/13-tools.md`（工具使用、函数调用、MCP、工具输出 masking）。`teach/` 下的 `*.html` 被 gitignore 忽略。

## 1. 社区收敛的训练配方（实现时的默认选择）
- **奖励**：只看结果的可验证奖励（0/1），格式奖励弱化或取消（Tongyi DeepResearch 2510.24701、Tmax 2606.23321）。
- **算法**：GRPO 家族（DAPO / GSPO / DPPO），组内相对优势，不用 critic。Tmax 的设置：组大小 32，KL = 0，DPPO 会屏蔽训练与推理 logprob 偏离过大的 token。
- **冷启动 SFT 可选**：基座足够强时可以直接 RL（Tool-N1 2505.00024；Tmax 对 Qwen3.5 不做 SFT）。
- **必须做 masking**：工具或环境返回的 token 不计入损失（Search-R1 2503.09516；书第 13 章）。
- **训推一致**：按 token-in / token-out 传递，不要对 rollout 文本重新分词（GLM-5 的 TITO 网关，2602.15763）；记录推理时的 logprob 用于重要性修正或屏蔽。

## 2. 函数调用 / 多轮的奖励设计与已知坑
- ToolRL（2504.13958）：格式奖励 + 工具名、参数名、参数值的细粒度匹配。
- Tool-N1：二值奖励（格式 + 功能正确）就够用。
- CM2（2602.12268）：每轮拆成有证据支撑的二值清单（checklist）。
- PROVE（2606.03892）：加效率惩罚，抑制冗余工具调用。
- SLCA-GRPO（2609.29050）：执行奖励只分配给工具调用 token，偏好奖励只分配给总结 token。
- 已知坑：
  - “空回合”（既无代码也无答案）会导致多轮 TIR 梯度爆炸，需要过滤（SimpleTIR 2509.02479）。
  - 多轮 RL 的 “Echo Trap” 崩溃（RAGEN/StarPO 2504.20073）。
  - 全 0 / 全 1 的组没有优势信号（RC-GRPO 2602.03025）。
  - 朴素的逐回合稠密奖励最多会掉 14 分，需按区分度校准（2604.02869）。
- 信用分配的新方向：步级分组（GiGPO 2505.10978，代码在 verl-agent）、关键状态训练（2609.24985）、用参考模型对金标准答案 logprob 的时序差分做稠密奖励（TRACE 2607.13988）。

## 3. 环境设计
- 环境 = 数据集（任务 + 初始状态）+ harness（工具、上下文、控制循环）+ 验证器（测试 / rubric / 状态检查）+ 状态（沙箱里的文件、进程、数据库）。这是 NeMo Gym 的抽象。
- 接口在趋同：Gym 式 `reset / step / state`，通过 HTTP 调用、一个环境一个容器（OpenEnv）；或把环境做成可安装的 Python 模块，内含 dataset + rollout + rubric（Prime Intellect `verifiers`）；MCP 工具暴露在 `/mcp`。
- **用代码和数据库支撑状态，而不是让 LLM 模拟世界**，这样奖励可以检查（AWM 2602.10090：1,000 个 MCP 环境，按数据库状态给奖励；EnvScaler 2601.05808）。
- 环境有效性检查（DeepSeek-V3.2 2512.02556）：要求 fail-to-pass > 0 且 pass-to-fail = 0。
- 任务生成：按难度控制 × 人设 × 验证器多样化的分类体系生成（Tmax）；任务来源多样性比规模更重要（OpenThoughts-Agent 2606.24855）；可以递归加深难度（2608.05466）。
- Harness 会显著影响结果：同一模型换 harness，token 消耗最多差 40 倍（2607.22585）；同一套 RL 在不同 harness 下的增益从 +22.6 到 +0.6 不等（Polar 2605.24220）。评测时要固定并写明 harness。

## 4. Rollout 基础设施
- 长轨迹用异步、解耦架构：推理引擎（vLLM / SGLang）↔ rollout 网关（记录 token 和 logprob）↔ 沙箱池 → 样本缓冲（容忍一定陈旧度）→ 训练器 → 异步同步权重。
- 常用技巧：
  - 部分 rollout：长尾轨迹暂停，下一轮续跑（Kimi K2 / K3）。
  - 陈旧度感知 PPO（AReaL 2505.24298）。
  - 窗口 FIFO 调度，避免偏向短任务（MiniMax Forge）。
  - 共享前缀合并成树来训练。
- 沙箱：小规模用 Docker；大规模走向 Firecracker microVM 加快照和 fork（AgentENV：checkpoint 133ms，resume 49ms）。fork 一份副本跑验证器，就能无副作用地判定奖励。约 90% 的沙箱 CPU 利用率低于 5%，可以大胆超卖（DSec 2609.22978）。
- 代码执行奖励可用 SandboxFusion（`bytedance/SandboxFusion`，verl 的默认选项）。

## 5. 防奖励黑客清单（写环境和验证器时逐条对照）
- 测试与参考答案对智能体隐藏且隔离；只把测试设成只读不够，模型仍会对测试做特判（ImpossibleBench 2510.20270）。
- 沙箱默认禁网或用白名单，防止下载答案（Terminal-Bench 出现过把答案写进 AGENTS.md、上传 tests/ 的案例）。
- 构建和运行用不同账户，用 AppArmor 限制 root，禁止访问内部 RPC 和日志（DSec 实测：覆写 /bin/bash、XFS ioctl、端口扫描）。
- 限制提交次数；构建失败或伪造产物记 0 分（Kimi K3）。
- 提供“放弃 / 报告无法完成”的出口，GPT-5 作弊率从 54% 降到 9%。
- 用 hacker-fixer 对抗循环加固验证器（2606.08960：16% 的终端任务仅凭任务描述就能被破解）；用蜜罐任务自动检测作弊（HVTB 2608.22103）。
- 检查验证器能否放过错误补丁：常用 SWE 训练集中约 1/3 的任务会给看似合理的错误补丁满分（Arctic-SWE 2610.24914）。
- 学会作弊的模型会泛化出更广的失调行为（Anthropic 2511.18397），所以作弊不只是分数虚高的问题。

## 6. 评测基准
BFCL v3/v4（函数调用，gorilla 仓库）、τ²-bench（`sierra-research/tau2-bench`）、Terminal-Bench 2.0（`harbor-framework/terminal-bench-2`，用 Harbor 运行）、SWE-bench Verified、MCP-Universe、MCPMark、AppWorld、ALFWorld / WebShop（多轮 RL 的入门环境）。

## 7. 值得读代码的参考仓库（均在 github.com/）
- 训练框架：`volcengine/verl`、`langfengQ/verl-agent`、`NovaSky-AI/SkyRL`、`rllm-org/rllm`、`PrimeIntellect-ai/verifiers`、`meta-pytorch/OpenEnv`、`NVIDIA-NeMo/Gym`、`THUDM/slime`、`inclusionAI/AReaL`、`OpenPipe/ART`、`TIGER-AI-Lab/verl-tool`
- 最小可读的配方：`PeterGriffinJin/Search-R1`、`qiancheng0/ToolRL`、`NVlabs/Tool-N1`、`GAIR-NLP/ToRL`、`ltzheng/SimpleTIR`、`hamishivi/tmax`、`open-thoughts/OpenThoughts-Agent`
- 环境：`Snowflake-Labs/agent-world-model`、`RUC-NLPIR/EnvScaler`、`SalesforceAIResearch/MCP-Universe`、`SWE-Gym/SWE-Gym`、`R2E-Gym/R2E-Gym`、`kanishkg/endless-terminals`
- 沙箱：`bytedance/SandboxFusion`、`kvcache-ai/AgentENV`、`e2b-dev/E2B`

## 8. 综述（需要背景时读）
- 最贴近工具后训练的综述：Agentic Tool Use in LLMs（2604.00835），分提示 / SFT / RL 三种范式。
- 更广的综述：Agentic RL 全景（2509.02547）、Adaptation of Agentic AI（2512.16301）、Agentic Reasoning（2601.12538，有后训练工具整合一节）、Credit Assignment（2604.09459）、环境工程（2606.12191）、Harness 设计（2606.20683）。
- 没有专门针对函数调用或 TIR 的 RL 综述。

## 9. 如果要在本仓库加一个教学示例，建议做法
- 起点选单机可跑的最小例子，比如在 `code/policy_gradients/` 的 GRPO 基础上加多轮工具调用：Python 解释器或简单的有状态函数调用环境，配一个数据库或字典支撑的验证器。
- 必须包含：工具输出 masking、结果奖励（可选 ToolRL 式格式 / 参数分项奖励作对照）、空回合过滤、全 0 / 全 1 组统计、沙箱超时和资源限制。
- 汇报：奖励曲线、平均工具调用次数、无效调用率、轨迹长度，以及在留出任务上的成功率；跑法参照 run-rlhf-code-experiment skill。
