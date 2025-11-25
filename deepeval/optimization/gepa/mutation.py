from __future__ import annotations
import json
from typing import Callable, Dict, List, Optional, Tuple, Union

from deepeval.optimization.types import (
    MetricInfo,
    ModuleId,
)
from deepeval.optimization.utils import (
    a_invoke_model_callback,
    invoke_model_callback,
    validate_callback,
    build_model_callback_kwargs,
)
from deepeval.prompt.api import PromptType, PromptMessage
from deepeval.prompt.prompt import Prompt


##################
# Common Helpers #
##################
def _summarize_prompt_for_rewrite(old_prompt: Prompt, max_chars: int) -> str:
    """
    Produce a human-readable summary of the current prompt for the
    rewriter instruction block.

    - For TEXT prompts, this is just `text_template`.
    - For LIST prompts, this is a numbered list of (role, content) lines.
    """
    prompt_type = old_prompt.type

    # LIST prompts: show each message with its role.
    if prompt_type is PromptType.LIST and old_prompt.messages_template:
        lines: List[str] = []
        for message_index, message in enumerate(old_prompt.messages_template):
            role = message.role or ""
            content = message.content or ""
            lines.append(f"[{message_index+1}] ({role}) {content}")
        combined = "\n".join(lines)
        return combined[:max_chars]

    # Since it is not a LIST prompt, just use text_template.
    text = old_prompt.text_template or ""
    return text[:max_chars]


def _apply_rewritten_prompt(old_prompt: Prompt, new_text: str) -> Prompt:
    """
    Apply the rewritten text to a Prompt, preserving representation:

    - For TEXT prompts, update `text_template`.
    - For LIST prompts, rewrite the content of a single message while
      keeping the number of messages the same.
    - Preserve additonal Prompt meta such as `label` and `interpolation_type`
    """
    if not new_text:
        return old_prompt

    prompt_type = old_prompt.type

    if prompt_type is PromptType.LIST and old_prompt.messages_template:
        messages = old_prompt.messages_template

        # Prefer the first system message; otherwise, rewrite the first.
        target_index: int = 0
        for message_idx, message in enumerate(messages):
            role = (message.role or "").lower()
            if role == "system":
                target_index = message_idx
                break

        new_messages: List[PromptMessage] = []
        for message_index, message in enumerate(messages):
            if message_index == target_index:
                role = message.role or "system"
                new_messages.append(PromptMessage(role=role, content=new_text))
            else:
                new_messages.append(message)

        new_prompt = Prompt(
            alias=old_prompt.alias,
            text_template=None,
            messages_template=new_messages,
            model_settings=old_prompt.model_settings,
            output_type=old_prompt.output_type,
            output_schema=old_prompt.output_schema,
        )

    else:
        # Since it is not LIST, it must be TEXT type
        new_prompt = Prompt(
            alias=old_prompt.alias,
            text_template=new_text,
            model_settings=old_prompt.model_settings,
            output_type=old_prompt.output_type,
            output_schema=old_prompt.output_schema,
        )
    new_prompt.label = (old_prompt.label,)
    new_prompt.interpolation_type = (old_prompt.interpolation_type,)
    return new_prompt


def _compose_prompt_messages(system_message: str, user_message: str) -> str:
    """
    Join system and user messages into a single prompt string.
    Strips surrounding whitespace from each part; if the system message is
    empty or absent, returns just the user message.
    """
    system_text = (system_message or "").strip()
    user_text = (user_message or "").strip()
    return f"{system_text}\n\n{user_text}" if system_text else user_text


def _normalize_llm_output_to_text(
    result: Union[str, Tuple[Union[str, dict], float], dict],
) -> str:
    """
    Convert a DeepEval LLM generate() / a_generate() result to a clean string.

    Accepted inputs:
      - str                        -> returned as trimmed
      - (str|dict, float_cost)     -> first element extracted and normalized
      - dict (e.g. JSON mode)      -> JSON serialized with ensure_ascii=False

    Fallback: if serialization fails, str(value).strip() is used.
    """
    output_value: Union[str, dict]
    if isinstance(result, tuple):
        output_value = result[0]
    else:
        output_value = result

    if isinstance(output_value, str):
        return output_value.strip()

    try:
        return json.dumps(output_value, ensure_ascii=False)
    except Exception:
        return str(output_value).strip()


#################################
# Rewriters for prompt mutation #
#################################


class PromptRewriter:
    """
    Uses a provided DeepEval model to rewrite the prompt for a module,
    guided by feedback_text (μ_f).
    """

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

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
        *,
        model_callback: Callable[
            ...,
            Union[
                str,
                Dict,
                Tuple[Union[str, Dict], float],
            ],
        ],
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt:
        model_callback = validate_callback(
            component="PromptRewriter",
            model_callback=model_callback,
        )
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

        prompt_type = old_prompt.type
        prompt_type_name: Optional[str] = (
            prompt_type.name if prompt_type is not None else None
        )
        prompt_messages: Optional[List[PromptMessage]] = None
        if prompt_type is PromptType.LIST and old_prompt.messages_template:
            prompt_messages = old_prompt.messages_template

        candidate_kwargs = build_model_callback_kwargs(
            prompt=old_prompt,
            prompt_type=prompt_type_name,
            prompt_text=merged_prompt_text,
            prompt_messages=prompt_messages,
            feedback_text=feedback_text,
        )
        out = invoke_model_callback(
            hook="prompt_rewrite",
            model_callback=model_callback,
            candidate_kwargs=candidate_kwargs,
        )

        new_text = _normalize_llm_output_to_text(out)
        return _apply_rewritten_prompt(old_prompt, new_text)

    async def a_rewrite(
        self,
        *,
        model_callback: Callable[
            ...,
            Union[
                str,
                Dict,
                Tuple[Union[str, Dict], float],
            ],
        ],
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt:
        model_callback = validate_callback(
            component="PromptRewriter",
            model_callback=model_callback,
        )

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

        prompt_type = old_prompt.type
        prompt_type_name: Optional[str] = (
            prompt_type.name if prompt_type is not None else None
        )
        prompt_messages: Optional[List[PromptMessage]] = None
        if prompt_type is PromptType.LIST and old_prompt.messages_template:
            prompt_messages = old_prompt.messages_template

        candidate_kwargs = build_model_callback_kwargs(
            prompt=old_prompt,
            prompt_type=prompt_type_name,
            prompt_text=merged_prompt_text,
            prompt_messages=prompt_messages,
            feedback_text=feedback_text,
        )
        out = await a_invoke_model_callback(
            hook="prompt_rewrite",
            model_callback=model_callback,
            candidate_kwargs=candidate_kwargs,
        )

        new_text = _normalize_llm_output_to_text(out)
        return _apply_rewritten_prompt(old_prompt, new_text)


class MetricAwareLLMRewriter(PromptRewriter):
    """
    Uses μ_f (feedback_text) and optional metric rubrics to rewrite a module prompt.
    - metrics_info: optional list of MetricInfo(name, rubric). If provided, a
      [Metric Rubrics] block is added to the prompt for stronger guidance.
    """

    def __init__(
        self,
        *,
        metrics_info: Optional[List[MetricInfo]] = None,
        max_chars: int = 4000,
        max_metrics_in_prompt: int = 20,
    ):
        super().__init__(max_chars=max_chars)
        self.metrics_info = metrics_info or []
        self.max_metrics_in_prompt = max_metrics_in_prompt

    def _compose_messages(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Tuple[str, str]:

        current_prompt_block = _summarize_prompt_for_rewrite(
            old_prompt, self.max_chars
        )

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
{current_prompt_block}

[Feedback]
{feedback_text[:self.max_chars]}
{rubric_block}

[Instruction]
Rewrite the prompt to better satisfy the metrics and address the feedback.
Keep it concise, actionable, and faithful to the module’s role."""
        return system_message, user_message
