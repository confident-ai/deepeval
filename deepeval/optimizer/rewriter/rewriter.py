from __future__ import annotations
import random
from typing import Optional, Tuple, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.optimizer.types import (
    ModuleId,
)
from deepeval.optimizer.configs import (
    MutationConfig,
)
from deepeval.prompt.prompt import Prompt
from deepeval.optimizer.rewriter.utils import (
    _summarize_prompt_for_rewrite,
    _compose_prompt_messages,
    _normalize_llm_output_to_text,
    _apply_rewritten_prompt,
)


class Rewriter:
    """
    Uses a provided DeepEval model to rewrite the prompt for a module,
    guided by feedback_text (μ_f).

    For LIST prompts, the target message to rewrite is chosen according to
    `list_mutation_config` and `random_state`.
    """

    def __init__(
        self,
        optimizer_model: DeepEvalBaseLLM,
        max_chars: int = 4000,
        list_mutation_config: Optional[MutationConfig] = None,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        self.optimizer_model = optimizer_model
        self.max_chars = max_chars
        self.list_mutation_config = list_mutation_config or MutationConfig()

        # Accept either an int seed or a Random instance.
        if isinstance(random_state, int):
            self.random_state: Optional[random.Random] = random.Random(
                random_state
            )
        else:
            self.random_state = random_state or random.Random()

    def _compose_messages(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Tuple[str, str]:
        current_prompt_block = _summarize_prompt_for_rewrite(
            old_prompt, self.max_chars
        )
        system_message = (
            "You are refining a prompt used in a multi-step LLM pipeline. "
            "Given the current prompt and concise feedback, produce a revised prompt "
            "that addresses the issues while preserving intent and style. "
            "Return only the new prompt text, no explanations."
        )
        user_message = f"""[Current Prompt]
{current_prompt_block}

[Feedback]
{feedback_text[:self.max_chars]}

[Instruction]
Rewrite the prompt. Keep it concise and actionable. Do not include extraneous text.
"""
        return system_message, user_message

    def rewrite(
        self,
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt:
        if not feedback_text.strip():
            return old_prompt

        system_message, user_message = self._compose_messages(
            module_id=module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )
        merged_prompt_text = _compose_prompt_messages(
            system_message, user_message
        )

        out = self.optimizer_model.generate(merged_prompt_text)
        new_text = _normalize_llm_output_to_text(out)
        return _apply_rewritten_prompt(
            old_prompt,
            new_text,
            self.random_state,
            self.list_mutation_config,
        )

    async def a_rewrite(
        self,
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt:
        if not feedback_text.strip():
            return old_prompt

        system_message, user_message = self._compose_messages(
            module_id=module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )
        merged_prompt_text = _compose_prompt_messages(
            system_message, user_message
        )

        out = await self.optimizer_model.a_generate(merged_prompt_text)
        new_text = _normalize_llm_output_to_text(out)
        return _apply_rewritten_prompt(
            old_prompt,
            new_text,
            self.random_state,
            self.list_mutation_config,
        )
