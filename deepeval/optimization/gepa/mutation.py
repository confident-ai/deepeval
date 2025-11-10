from __future__ import annotations
from ..types import ModuleId


class NoOpRewriter:
    """Safe default: returns the original prompt."""

    def rewrite(
        self, *, module_id: ModuleId, old_prompt: str, feedback_text: str
    ) -> str:
        return old_prompt
