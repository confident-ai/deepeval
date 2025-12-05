from __future__ import annotations
import json
import random
from typing import List, Optional, Tuple, Union

from deepeval.errors import DeepEvalError
from deepeval.optimizer.utils import (
    validate_int_in_range,
    validate_instance,
)
from deepeval.optimizer.configs import (
    MutationConfig,
    MutationTargetType,
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

    # LIST prompts: show each message with its role.
    if old_prompt.type is PromptType.LIST and old_prompt.messages_template:
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


def _select_list_target_index(
    messages: List[PromptMessage],
    config: MutationConfig,
    random_state: random.Random,
) -> int:
    """
    Select which list message index to rewrite, based on PromptListMutationConfig.

    Rules:
    - Start with all indices in scope.
    - If target_role is set, restrict candidates to messages with that role
      (case insensitive). If no messages match, fall back to all indices.
    - target_type:
        * FIRST:       pick the first candidate index.
        * RANDOM:      pick a candidate via random_state.choice(candidates).
        * FIXED_INDEX: use target_index when valid (and consistent with role
                       filter), otherwise fall back to the first candidate.
    """
    if not messages:
        raise DeepEvalError(
            "Rewriter._select_list_target_index expected at least one "
            "message, but received an empty message list."
        )

    validate_instance(
        component="Rewriter._select_list_target_index",
        param_name="target_type",
        value=config.target_type,
        expected_types=MutationTargetType,
    )

    messages_length = len(messages)
    candidate_indices = list(range(messages_length))

    # Optional case insensitive role restriction
    if config.target_role:
        target_role_lower = config.target_role.lower()
        filtered = [
            index
            for index, message in enumerate(messages)
            if (message.role or "").lower() == target_role_lower
        ]
        if filtered:
            candidate_indices = filtered

    target_type = config.target_type

    if target_type is MutationTargetType.RANDOM:
        return random_state.choice(candidate_indices)

    if target_type is MutationTargetType.FIXED_INDEX:
        index = validate_int_in_range(
            component="Rewriter._select_list_target_index",
            param_name="target_index",
            value=int(config.target_index),
            min_inclusive=0,
            max_exclusive=len(candidate_indices),
        )
        return candidate_indices[index]

    # if you got this error it means that a new PromptListMutationTargetType was added,
    # but not handled above
    raise DeepEvalError(
        "Rewriter._select_list_target_index received unsupported "
        f"target_type={target_type!r}. Expected RANDOM or FIXED_INDEX."
    )


def _apply_rewritten_prompt(
    old_prompt: Prompt,
    new_text: str,
    random_state: random.Random,
    list_mutation_config: Optional[MutationConfig] = None,
) -> Prompt:
    """
    Apply the rewritten text to a Prompt, preserving representation:

    - For TEXT prompts, update `text_template`.
    - For LIST prompts, rewrite the content of a single message while
      keeping the number of messages the same.
    - Preserve additonal Prompt meta such as `label` and `interpolation_type`
    """
    if not new_text:
        return old_prompt

    if old_prompt.type is PromptType.LIST and old_prompt.messages_template:
        messages = old_prompt.messages_template
        config = list_mutation_config or MutationConfig()

        target_index = _select_list_target_index(
            messages=messages,
            config=config,
            random_state=random_state,
        )

        new_messages: List[PromptMessage] = []
        for message_index, message in enumerate(messages):
            if message_index == target_index:
                # Preserve the original role; do not inject a new one.
                new_messages.append(
                    PromptMessage(
                        role=message.role,
                        content=new_text,
                    )
                )
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

    new_prompt.label = old_prompt.label
    new_prompt.interpolation_type = old_prompt.interpolation_type
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
