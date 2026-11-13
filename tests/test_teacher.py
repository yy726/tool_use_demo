
from planner import tasks as T
from planner.expert import build_expert_traces, expert_rollout
from planner.scoring import score_messages
from planner.teacher import as_demonstration, collect_teacher_traces, parallel_backend, teacher_targets


def _tasks(world, corpus, rows):
    return T.build_split("sft", 3, world, T.kb_items(rows, corpus) * 2, seed=3)


def test_teacher_targets_are_the_unsolved_multi_hop_tasks(world, corpus, rows):
    tasks = _tasks(world, corpus, rows)
    expert = build_expert_traces(tasks)
    unsolved = {t["task_id"] for t in tasks if t["aspects"]} - {t["task_id"] for t in expert}
    targets = teacher_targets(tasks, expert)
    assert {t["task_id"] for t in targets} <= unsolved
    assert all(t["family"] in ("kb_bridge", "kb_compare", "composite", "dual_bridge") for t in targets)
    assert {t["task_id"] for t in teacher_targets(tasks, [])} >= {t["task_id"] for t in tasks if t["family"] == "kb_bridge"}


def test_rejection_sampling_keeps_only_perfect_rollouts_and_retries(world, corpus, rows):
    tasks = [t for t in _tasks(world, corpus, rows) if t["family"] in ("tool_bridge", "chain_calc")]
    by_query = {t["query"]: t for t in tasks}
    attempts = {}

    def flaky_teacher(batch):
        """Chatty but correct from the second attempt on; the first attempt gives up without calling tools."""
        replies = []
        for messages in batch:
            task = by_query[messages[1]["content"]]
            if len(messages) == 2:
                attempts[task["task_id"]] = attempts.get(task["task_id"], 0) + 1
            if attempts[task["task_id"]] == 1:
                replies.append({"role": "assistant", "content": "I think it was sunny. Answer: sunny"})
                continue
            script = [m for m in expert_rollout(task) if m["role"] == "assistant"]
            reply = dict(script[sum(m["role"] == "assistant" for m in messages[2:])])
            reply["content"] = ("Let me look that up. " if reply.get("tool_calls") else "From the results: it adds up.\n") + reply["content"]
            replies.append(reply)
        return replies

    traces, stats = collect_teacher_traces(tasks, parallel_backend(flaky_teacher, workers=4), samples=3, verbose=False)
    assert stats == {"tasks": len(tasks), "accepted": len(tasks), "rollouts": 2 * len(tasks), "acceptance_per_family": {"tool_bridge": "3/3", "chain_calc": "3/3"}}
    for trace in traces:
        task = next(t for t in tasks if t["task_id"] == trace["task_id"])
        rollout = trace["messages"][2:]
        assert trace["source"] == "teacher" and score_messages(task, rollout).score == 4
        assert all(m["content"] == "" for m in rollout if m.get("tool_calls"))  # the teacher's chatter is dropped
        assert rollout[-1]["content"].startswith("Answer: ") and "\n" not in rollout[-1]["content"]

    none, stats = collect_teacher_traces(tasks, lambda batch: [{"role": "assistant", "content": "Answer: no idea"}] * len(batch), samples=2, verbose=False)
    assert none == [] and stats["rollouts"] == 2 * len(tasks)


def test_parallel_backend_turns_a_failed_request_into_a_rejected_rollout():
    def backend(batch):
        if "boom" in batch[0][0]["content"]:
            raise TimeoutError("server went away")
        return [{"role": "assistant", "content": "Answer: ok"}]

    replies = parallel_backend(backend, workers=2)([[{"role": "user", "content": "fine"}], [{"role": "user", "content": "boom"}]])
    assert [r["content"] for r in replies] == ["Answer: ok", ""]
    demo = as_demonstration([{"role": "assistant", "content": "<think>hm</think>Long story.\nAnswer: 42"}])
    assert demo == [{"role": "assistant", "content": "Answer: 42"}]
