from __future__ import annotations
import asyncio
from typing import List, Optional, Tuple

from deepeval.optimization.types import MetricInfo, ModuleId, PromptRewriter
from deepeval.prompt.prompt import Prompt
from deepeval.models import DeepEvalBaseLLM


class NoOpRewriter(PromptRewriter):
    """Safe default: returns the original prompt."""

    def rewrite(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Prompt:
        return old_prompt

    async def a_rewrite(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Prompt:
        return old_prompt


class LLMRewriter(PromptRewriter):
    """
    Uses a provided DeepEval model to rewrite the prompt for a module,
    guided by feedback_text (μ_f).
    """

    def __init__(self, llm: DeepEvalBaseLLM, max_chars: int = 4000):
        self.llm = llm
        self.max_chars = max_chars

    def _compose_messages(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Tuple[str, str]:
        system_message = (
            "You are refining a prompt used in a multi-step LLM pipeline. "
            "Given the current prompt and concise feedback, produce a revised prompt "
            "that addresses the issues while preserving intent and style. "
            "Return only the new prompt text, no explanations."
        )
        user_message = f"""[Current Prompt]
{old_prompt.text_template[:self.max_chars]}

[Feedback]
{feedback_text[:self.max_chars]}

[Instruction]
Rewrite the prompt. Keep it concise and actionable. Do not include extraneous text.
"""
        return system_message, user_message

    async def _call_llm(self, system_message: str, user_message: str) -> str:
        agenerate = getattr(self.llm, "agenerate", None)
        if callable(agenerate):
            text = await agenerate(
                system_prompt=system_message, prompt=user_message
            )
        else:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(
                    system_prompt=system_message, prompt=user_message
                ),
            )
        return (text or "").strip()

    def rewrite(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Prompt:
        if not feedback_text.strip():
            return old_prompt
        system_message, user_message = self._compose_messages(
            module_id=module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )
        new_text = self.llm.generate(
            system_prompt=system_message, prompt=user_message
        )
        new_text = (new_text or "").strip()
        return old_prompt if not new_text else Prompt(text_template=new_text)

    async def a_rewrite(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Prompt:
        if not feedback_text.strip():
            return old_prompt
        system_message, user_message = self._compose_messages(
            module_id=module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )
        new_text = await self._call_llm(system_message, user_message)
        return old_prompt if not new_text else Prompt(text_template=new_text)


class MetricAwareLLMRewriter(LLMRewriter):
    """
    Uses μ_f (feedback_text) and optional metric rubrics to rewrite a module prompt.
    - metrics_info: optional list of MetricInfo(name, rubric). If provided, a
      [Metric Rubrics] block is added to the prompt for stronger guidance.
    """

    def __init__(
        self,
        llm: DeepEvalBaseLLM,
        *,
        metrics_info: Optional[List[MetricInfo]] = None,
        max_chars: int = 4000,
        max_metrics_in_prompt: int = 20,
    ):
        super().__init__(llm=llm, max_chars=max_chars)
        self.metrics_info = metrics_info or []
        self.max_metrics_in_prompt = max_metrics_in_prompt

    def _compose_messages(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Tuple[str, str]:
        # Optional rubrics block
        rubric_block = ""
        if self.metrics_info:
            lines: List[str] = []
            for metric in self.metrics_info[: self.max_metrics_in_prompt]:
                if metric.rubric and metric.rubric.strip():
                    lines.append(f"- {metric.name}: {metric.rubric.strip()}")
                else:
                    lines.append(
                        f"- {metric.name}: Optimize for this metric’s quality criteria."
                    )
            rubric_block = "\n[Metric Rubrics]\n" + "\n".join(lines)

        system_message = (
            "You are refining a prompt used in a multi-step LLM pipeline. "
            "Given the current prompt, concise feedback, and (optionally) metric rubrics, "
            "produce a revised prompt that addresses the issues while preserving intent and style. "
            "Return only the new prompt text, with no explanations."
        )

        user_message = f"""[Module]
{module_id}

[Current Prompt]
{old_prompt.text_template[:self.max_chars]}

[Feedback]
{feedback_text[:self.max_chars]}
{rubric_block}

[Instruction]
Rewrite the prompt to better satisfy the metrics and address the feedback.
Keep it concise, actionable, and faithful to the module’s role."""
        return system_message, user_message
