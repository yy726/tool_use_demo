"""Chat-template checks against the real Qwen3 tokenizer (skipped without network / trl)."""

import pytest

from planner.chat import EMPTY_THINK, setup_chat_format
from planner.tools import TOOL_SCHEMAS

from conftest import assistant_calls

CONVERSATION = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "Q?"},
    assistant_calls(("get_weather", {"city": "X", "date": "2024-01-02"})),
    {"role": "tool", "name": "get_weather", "content": '{"temp_c": 1.0}'},
    {"role": "assistant", "content": "Answer: 1.0"},
]


def load(name):
    pytest.importorskip("trl")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer {name} not available: {exc}")


@pytest.mark.parametrize("name", ["unsloth/Qwen3-1.7B", "Qwen/Qwen3-1.7B"])
def test_thinking_model_format(name):
    from trl import chat_template_utils as ctu

    tokenizer = load(name)
    fmt = setup_chat_format(tokenizer)
    assert tokenizer.chat_template == ctu.qwen3_chat_template  # the copy TRL's GRPO tool loop recognises
    assert fmt.is_thinking_model and not fmt.enable_thinking

    full = fmt.render(CONVERSATION)
    # every assistant turn carries the (empty) think block, so SFT text matches GRPO rollouts
    assert full.count("<|im_start|>assistant\n" + EMPTY_THINK) == 2
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "X", "date": "2024-01-02"}}\n</tool_call>' in full
    assert "<tool_response>\n{\"temp_c\": 1.0}\n</tool_response>" in full
    for schema in TOOL_SCHEMAS:
        assert schema["function"]["name"] in full

    # prefix-preserving: appending tool results never changes earlier text
    first = fmt.render_prompt(CONVERSATION[:2])
    assert first.endswith("<|im_start|>assistant\n" + EMPTY_THINK)
    after_tool = fmt.render_prompt(CONVERSATION[:4])
    assert after_tool.startswith(first) and after_tool.endswith("<|im_start|>assistant\n" + EMPTY_THINK)
    assert full.startswith(after_tool)

    # the serving template (no {% generation %} markers) renders the same text
    kwargs = dict(tools=TOOL_SCHEMAS, tokenize=False)
    assert "generation %}" not in fmt.serving.replace("add_generation_prompt", "")
    assert tokenizer.apply_chat_template(CONVERSATION, chat_template=fmt.serving, **kwargs) == full
    assert tokenizer.apply_chat_template(CONVERSATION[:2], chat_template=fmt.serving, add_generation_prompt=True, **kwargs) == first

    thinking = setup_chat_format(tokenizer, enable_thinking=True)
    assert thinking.render_prompt(CONVERSATION[:2]).endswith("<|im_start|>assistant\n")


def test_instruct_model_format():
    from trl import chat_template_utils as ctu

    tokenizer = load("unsloth/Qwen3-4B-Instruct-2507")
    fmt = setup_chat_format(tokenizer, enable_thinking=True)
    assert tokenizer.chat_template == ctu.qwen3_instruct_2507_chat_template
    assert not fmt.is_thinking_model and not fmt.enable_thinking  # no thinking mode to enable
    full = fmt.render(CONVERSATION)
    assert "<think>" not in full and full.count("<|im_start|>assistant\n") == 2
    first = fmt.render_prompt(CONVERSATION[:2])
    assert fmt.render_prompt(CONVERSATION[:4]).startswith(first)
