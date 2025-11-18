from __future__ import annotations
import json
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel as PydanticBaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.optimization.types import (
    MetricInfo,
    ModuleId,
)
from deepeval.optimization.utils import (
    a_invoke_model_callback,
    invoke_model_callback,
    require_model_or_callback,
    build_model_callback_kwargs,
)
from deepeval.prompt.prompt import Prompt


##################
# Common Helpers #
##################
def _compose_prompt_messages(system_message: str, user_message: str) -> str:
    """
    Join system and user messages into a single prompt string.
    Strips surrounding whitespace from each part; if the system message is
    empty/absent, returns just the user message.
    """
    system_text = (system_message or "").strip()
    user_text = (user_message or "").strip()
    return f"{system_text}\n\n{user_text}" if system_text else user_text


def _normalize_llm_output_to_text(
    result: Union[str, Tuple[Union[str, dict], float], dict]
) -> str:
    """
    Convert a DeepEval LLM generate() / a_generate() result to a clean string.

    Accepted inputs:
      - str                        -> returned as trimmed
      - (str|dict, float_cost)     -> first element extracted and normalized
      - dict (e.g., JSON mode)     -> JSON-serialized with ensure_ascii=False

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

    def rewrite(
        self,
        *,
        model: Optional[DeepEvalBaseLLM] = None,
        model_schema: Optional[PydanticBaseModel] = None,
        model_callback: Optional[
            Callable[
                ...,
                Union[
                    str,
                    Dict,
                    Tuple[Union[str, Dict], float],
                ],
            ]
        ] = None,
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt:
        model, model_callback = require_model_or_callback(
            component="PromptRewriter",
            model=model,
            model_callback=model_callback,
        )
        if not feedback_text.strip():
            return old_prompt
        system_message, user_message = self._compose_messages(
            module_id=module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )
        merged = _compose_prompt_messages(system_message, user_message)

        is_callback_async = (
            inspect.iscoroutinefunction(model_callback)
            if model_callback is not None
            else False
        )

        if model_callback is not None and not is_callback_async:
            candidate_kwargs = build_model_callback_kwargs(
                module_id=module_id,
                old_prompt=old_prompt,
                feedback_text=feedback_text,
                merged=merged,
                prompt_text=merged,
                model=model,
                model_schema=model_schema,
            )
            out = invoke_model_callback(
                hook="prompt_rewrite",
                model_callback=model_callback,
                candidate_kwargs=candidate_kwargs,
            )
        else:
            out = model.generate(merged, schema=model_schema)
        new_text = _normalize_llm_output_to_text(out)
        # if new_text == old_prompt.text_template.strip():
        #     print(
        #         f"[DEBUG][GEPA] rewrite produced NO CHANGE | module={module_id}"
        #     )
        # else:
        #     preview = (
        #         (new_text[:80] + "...") if len(new_text) > 80 else new_text
        #     )
        #     print(
        #         f"[DEBUG][GEPA] rewrite CHANGED | module={module_id} | new='{preview}'"
        #     )
        return old_prompt if not new_text else Prompt(text_template=new_text)

    async def a_rewrite(
        self,
        *,
        model: Optional[DeepEvalBaseLLM] = None,
        model_schema: Optional[PydanticBaseModel] = None,
        model_callback: Optional[
            Callable[
                ...,
                Union[
                    str,
                    Dict,
                    Tuple[Union[str, Dict], float],
                ],
            ]
        ] = None,
        module_id: ModuleId,
        old_prompt: Prompt,
        feedback_text: str,
    ) -> Prompt:
        model, model_callback = require_model_or_callback(
            component="PromptRewriter",
            model=model,
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
        if model_callback is not None:
            candidate_kwargs = build_model_callback_kwargs(
                module_id=module_id,
                old_prompt=old_prompt,
                feedback_text=feedback_text,
                prompt_text=merged_prompt_text,
                model=model,
                model_schema=model_schema,
            )
            out = await a_invoke_model_callback(
                hook="prompt_rewrite",
                model_callback=model_callback,
                candidate_kwargs=candidate_kwargs,
            )
        else:
            out = await model.a_generate(
                merged_prompt_text, schema=model_schema
            )
        new_text = _normalize_llm_output_to_text(out)
        # if new_text == old_prompt.text_template.strip():
        #     print(
        #         f"[DEBUG][GEPA] rewrite produced NO CHANGE | module={module_id}"
        #     )
        # else:
        #     preview = (
        #         (new_text[:80] + "...") if len(new_text) > 80 else new_text
        #     )
        #     print(
        #         f"[DEBUG][GEPA] rewrite CHANGED | module={module_id} | new='{preview}'"
        #     )

        return old_prompt if not new_text else Prompt(text_template=new_text)


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
