"""Chat-template handling for Qwen3, so SFT, GRPO, the benchmark and serving all agree.

Two facts drive this module:

1. TRL's GRPO tool loop only accepts chat templates it recognises (exact string
   match), and Unsloth's tokenizers ship slightly modified Qwen3 templates. So we
   reset the tokenizer to TRL's canonical copy of the Qwen3 template.
2. For training, TRL swaps in a "training template" that always renders the
   `<think>...</think>` block of every assistant turn (empty when the model does
   not think), which keeps earlier turns byte-identical as tool results are
   appended. We render SFT text and benchmark prompts with that same template,
   so the model sees one format in all stages.

For serving (merged weights, GGUF) we save the training template with the
transformers-only `{% generation %}` markers removed and thinking off by default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .tools import TOOL_SCHEMAS

EMPTY_THINK = "<think>\n\n</think>\n\n"
_GENERATION_TAG = re.compile(r"\{%-?\s*(?:end)?generation\s*-?%\}\n?")
_THINK_SWITCH = "enable_thinking is defined and enable_thinking is false"
_THINK_SWITCH_DEFAULT_OFF = "enable_thinking is not defined or enable_thinking is false"


def serving_template(training_template: str) -> str:
    """Training template made portable: no `{% generation %}` markers, thinking off unless asked for."""
    template = _GENERATION_TAG.sub("", training_template)
    return template.replace(_THINK_SWITCH, _THINK_SWITCH_DEFAULT_OFF)


@dataclass
class ChatFormat:
    """Renders conversations the way the model is trained to see them."""

    tokenizer: object
    stock: str  # template TRL recognises (left on the tokenizer for GRPOTrainer)
    training: str  # prefix-preserving template used for SFT text and benchmark prompts
    serving: str  # what gets saved with merged / GGUF weights
    enable_thinking: bool = False
    is_thinking_model: bool = True

    def render(self, messages: Sequence[dict], tools: Sequence | None = TOOL_SCHEMAS, add_generation_prompt: bool = False) -> str:
        kwargs = {}
        if self.is_thinking_model:
            kwargs["enable_thinking"] = self.enable_thinking
        return self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.training,
            **kwargs,
        )

    def render_prompt(self, messages: Sequence[dict], tools: Sequence | None = TOOL_SCHEMAS) -> str:
        return self.render(messages, tools=tools, add_generation_prompt=True)


def setup_chat_format(tokenizer, enable_thinking: bool = False) -> ChatFormat:
    """Put TRL's canonical Qwen3 template on the tokenizer and return the matching ChatFormat.

    Works for the hybrid-thinking Qwen3 models (0.6B-32B) and for
    Qwen3-4B-Instruct-2507, which has no thinking mode.
    """
    from trl import chat_template_utils as ctu

    current = tokenizer.chat_template or ""
    is_thinking_model = "enable_thinking" in current
    tokenizer.chat_template = ctu.qwen3_chat_template if is_thinking_model else ctu.qwen3_instruct_2507_chat_template
    training = ctu.get_training_chat_template(tokenizer) or tokenizer.chat_template
    return ChatFormat(
        tokenizer=tokenizer,
        stock=tokenizer.chat_template,
        training=training,
        serving=serving_template(training),
        enable_thinking=enable_thinking and is_thinking_model,
        is_thinking_model=is_thinking_model,
    )
