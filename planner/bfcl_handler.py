"""BFCL handler for Qwen3 in non-thinking mode (imported only when bfcl-eval is installed).

Kept at module level because bfcl-eval enforces `@override`, and the `overrides`
package can only resolve the parent method for classes defined in a module.
"""

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override

EMPTY_THINK = "<think>\n\n</think>\n\n"


class QwenFCNoThinkHandler(QwenFCHandler):
    """BFCL's Qwen function-calling handler with the empty think block prefilled."""

    @override
    def _format_prompt(self, messages, function):
        return super()._format_prompt(messages, function) + EMPTY_THINK
