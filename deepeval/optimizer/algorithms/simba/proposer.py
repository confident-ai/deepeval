from __future__ import annotations
from typing import Optional

from deepeval.utils import serialize_to_json
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.prompt.prompt import Prompt
from deepeval.prompt.api import PromptMessage, PromptType
from deepeval.metrics.utils import (
    initialize_model,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)
from deepeval.optimizer.utils import _parse_prompt, _create_prompt

from .schema import SIMBARewriteSchema
from .template import SIMBATemplate


class SIMBAProposer:

    def __init__(self, optimizer_model: DeepEvalBaseLLM):
        self.model, self.using_native_model = initialize_model(optimizer_model)

    def _accrue_cost(self, cost: float) -> None:
        pass

    def _format_trajectory(
        self, inputs: str, outputs: str, score: float, feedback: str
    ) -> str:
        """Helper to cleanly format the trajectory block for the template."""
        return (
            f"Inputs: {inputs}\n"
            f"Model Output: {outputs}\n"
            f"Score: {score:.4f}\n"
            f"Evaluation Feedback: {feedback}"
        )

    def rewrite_from_introspection(
        self,
        original_prompt: Prompt,
        better_inputs: str,
        better_outputs: str,
        better_score: float,
        better_feedback: str,
        worse_inputs: str,
        worse_outputs: str,
        worse_score: float,
        worse_feedback: str,
    ) -> Prompt:
        """Strategy 1 (Sync): Introspects traces and holistically rewrites the prompt to fix the failure."""
        prompt_text = _parse_prompt(original_prompt)
        is_list = original_prompt.type == PromptType.LIST

        worse_trajectory = self._format_trajectory(
            worse_inputs, worse_outputs, worse_score, worse_feedback
        )
        better_trajectory = self._format_trajectory(
            better_inputs, better_outputs, better_score, better_feedback
        )

        template = SIMBATemplate.generate_introspection_rewrite(
            original_prompt=prompt_text,
            worse_trajectory=worse_trajectory,
            better_trajectory=better_trajectory,
            is_list_format=is_list,
        )

        try:
            rewritten_data = generate_with_schema_and_extract(
                metric=self,
                prompt=template,
                schema_cls=SIMBARewriteSchema,
                extract_schema=lambda s: s.revised_prompt,
                extract_json=lambda data: data["revised_prompt"],
            )
        except Exception:
            return original_prompt

        if not rewritten_data:
            return original_prompt

        if isinstance(rewritten_data, list):
            rewritten_data = serialize_to_json(rewritten_data)

        return _create_prompt(original_prompt, rewritten_data)

    async def a_rewrite_from_introspection(
        self,
        original_prompt: Prompt,
        better_inputs: str,
        better_outputs: str,
        better_score: float,
        better_feedback: str,
        worse_inputs: str,
        worse_outputs: str,
        worse_score: float,
        worse_feedback: str,
    ) -> Prompt:
        prompt_text = _parse_prompt(original_prompt)
        is_list = original_prompt.type == PromptType.LIST

        worse_trajectory = self._format_trajectory(
            worse_inputs, worse_outputs, worse_score, worse_feedback
        )
        better_trajectory = self._format_trajectory(
            better_inputs, better_outputs, better_score, better_feedback
        )

        template = SIMBATemplate.generate_introspection_rewrite(
            original_prompt=prompt_text,
            worse_trajectory=worse_trajectory,
            better_trajectory=better_trajectory,
            is_list_format=is_list,
        )

        try:
            rewritten_data = await a_generate_with_schema_and_extract(
                metric=self,
                prompt=template,
                schema_cls=SIMBARewriteSchema,
                extract_schema=lambda s: s.revised_prompt,
                extract_json=lambda data: data["revised_prompt"],
            )
        except Exception:
            return original_prompt

        if not rewritten_data:
            return original_prompt

        if isinstance(rewritten_data, list):
            rewritten_data = serialize_to_json(rewritten_data)

        return _create_prompt(original_prompt, rewritten_data)

    def append_a_demo(
        self,
        original_prompt: Prompt,
        inputs: str,
        outputs: str,
    ) -> Prompt:
        demo_text = f"\n\n[Example]\nInput: {inputs}\nOutput: {outputs}"
        return self._inject_text(original_prompt, demo_text)

    def _inject_text(self, prompt: Prompt, injection: str) -> Prompt:
        is_list = prompt.type == PromptType.LIST

        if is_list:
            new_messages = []
            injected = False
            for msg in prompt.messages_template:
                if not injected and msg.role == "system":
                    new_content = f"{msg.content}{injection}"
                    new_messages.append(
                        PromptMessage(role=msg.role, content=new_content)
                    )
                    injected = True
                else:
                    new_messages.append(msg)

            if not injected and new_messages:
                first = new_messages[0]
                new_messages[0] = PromptMessage(
                    role=first.role, content=f"{first.content}{injection}"
                )

            return Prompt(messages_template=new_messages)
        else:
            return Prompt(text_template=f"{prompt.text_template}{injection}")
