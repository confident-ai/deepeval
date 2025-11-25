from __future__ import annotations
import inspect
import random
import re
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
    Dict,
    Set,
)

from deepeval.errors import DeepEvalError
from deepeval.metrics.base_metric import BaseMetric, BaseConversationalMetric
from deepeval.prompt.prompt import Prompt
from deepeval.optimization.types import ModuleId


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden
    from deepeval.prompt.api import PromptMessage


def split_goldens(
    goldens: Union[List[Golden], List[ConversationalGolden]],
    pareto_size: int,
    *,
    random_state: random.Random,
) -> Tuple[
    Union[List[Golden], List[ConversationalGolden]],
    Union[List[Golden], List[ConversationalGolden]],
]:
    """
    Split `goldens` into two disjoint parts:

      - d_feedback: items not selected for the Pareto validation set
      - d_pareto:   `pareto_size` items for instance-wise Pareto scoring

    The selection is deterministic given `seed`. Within each split, the
    original order from `goldens` is preserved.

    Args:
        goldens: Full list/sequence of examples.
        pareto_size: Number of items to allocate to the Pareto set bound between [0, len(goldens)].
        random_state: A shared `random.Random` instance that provides the source
            of randomness. For reproducible runs, pass the same object used by
            the GEPA loop constructed from `GEPAConfig.random_seed`

    Returns:
        (d_feedback, d_pareto)
    """
    if pareto_size < 0:
        raise ValueError("pareto_size must be >= 0")

    total = len(goldens)

    if total == 0:
        # nothing to split
        return [], []

    # With a single example, we cannot form a meaningful feedback set.
    # callers like GEPARunner should enforce a minimum of 2 goldens for
    # optimization.
    if total == 1:
        return [], list(goldens)

    # For total >= 2, ensure that we always leave at least one example
    # for d_feedback. This keeps the splits disjoint while still honoring
    # pareto_size as a target up to (total - 1).
    chosen_size = min(pareto_size, total - 1)

    indices = list(range(total))
    random_state.shuffle(indices)

    pareto_indices = set(indices[:chosen_size])

    d_pareto = [goldens[i] for i in range(total) if i in pareto_indices]
    d_feedback = [goldens[i] for i in range(total) if i not in pareto_indices]

    return d_feedback, d_pareto


################################
# Prompt normalization helpers #
################################


def _slug(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def generate_module_id(prompt: Prompt, index: int, existing: Set[str]) -> str:
    """
    Build a human readable module id stable within a single optimization run.
    Prefers alias/label; enrich with model settings provider and name; dedupe; cap to 64 chars.
    """
    parts: List[str] = []
    if prompt.alias:
        parts.append(str(prompt.alias))
    if prompt.label:
        parts.append(str(prompt.label))

    ms = prompt.model_settings
    if ms is not None:
        if ms.provider is not None:
            parts.append(ms.provider.value)
        if ms.name:
            parts.append(ms.name)

    base = "-".join(_slug(p) for p in parts if p) or f"module-{index+1}"
    base = base[:64] or f"module-{index+1}"

    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}-{suffix}"
        candidate = candidate[:64]
        suffix += 1

    existing.add(candidate)
    return candidate


def normalize_seed_prompts(
    seed_prompts: Union[Dict[ModuleId, Prompt], List[Prompt]]
) -> Dict[ModuleId, Prompt]:
    """
    Accept either {module_id: Prompt} or List[Prompt].
    If a list is given, generate human readable module ids.
    """
    if isinstance(seed_prompts, dict):
        return dict(seed_prompts)  # shallow copy

    mapping: Dict[ModuleId, Prompt] = {}
    used: Set[str] = set()
    for i, prompt in enumerate(seed_prompts):
        module_id = generate_module_id(prompt, i, used)
        mapping[module_id] = prompt
    return mapping


def build_model_callback_kwargs(
    *,
    # scoring context
    golden: Optional[Union["Golden", "ConversationalGolden"]] = None,
    # rewriter context
    feedback_text: Optional[str] = None,
    # shared
    prompt: Optional[Prompt] = None,
    prompt_type: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt_messages: Optional[List["PromptMessage"]] = None,
) -> Dict[str, Any]:
    """
    Build a superset of kwargs for GEPA model callbacks.

    All keys are present in the dict so callbacks can declare any subset of:

        hook: str           # injected by (a_)invoke_model_callback
        prompt: Prompt
        prompt_type: str
        prompt_text: str
        prompt_messages: List[PromptMessage]
        golden: Golden | ConversationalGolden
        feedback_text: str

    Non applicable fields are set to None.
    """
    return {
        # scoring context
        "golden": golden,
        # rewriter context
        "feedback_text": feedback_text,
        # shared
        "prompt": prompt,
        "prompt_type": prompt_type,
        "prompt_text": prompt_text,
        "prompt_messages": prompt_messages,
    }


def invoke_model_callback(
    *,
    hook: str,
    model_callback: Callable[
        ...,
        Union[
            str,
            Dict,
            Tuple[Union[str, Dict], float],
        ],
    ],
    candidate_kwargs: Dict[str, Any],
) -> Union[
    str,
    Dict,
    Tuple[Union[str, Dict], float],
]:
    """
    Call a user provided model_callback in a synchronous context.

    - Filters kwargs to only those the callback accepts.
    - Injects `hook` if the callback declares it.
    - Raises if the callback returns an awaitable; callers must use async
      helpers for async callbacks.
    """
    sig = inspect.signature(model_callback)
    supported = set(sig.parameters.keys())

    filtered = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in supported
    }

    if "hook" in supported:
        filtered["hook"] = hook

    result = model_callback(**filtered)
    if inspect.isawaitable(result):
        raise DeepEvalError(
            "model_callback returned an awaitable from a synchronous context. "
            "Either declare the callback as `async def` and use async GEPA, or call "
            "`model.generate(...)` instead of `model.a_generate(...)` inside a sync callback."
        )
    return result


async def a_invoke_model_callback(
    *,
    hook: str,
    model_callback: Callable[
        ...,
        Union[
            str,
            Dict,
            Tuple[Union[str, Dict], float],
        ],
    ],
    candidate_kwargs: Dict[str, Any],
) -> Union[
    str,
    Dict,
    Tuple[Union[str, Dict], float],
]:
    """
    Call a user provided model_callback in an async context.

    - Filters kwargs to only those the callback accepts.
    - Injects `hook` if the callback declares it.
    - Supports both sync and async callbacks.
    """
    sig = inspect.signature(model_callback)
    supported = set(sig.parameters.keys())

    filtered = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in supported
    }

    if "hook" in supported:
        filtered["hook"] = hook

    result = model_callback(**filtered)
    if inspect.isawaitable(result):
        return await result
    return result


##############
# Validation #
##############
def _format_type_names(types: Tuple[type, ...]) -> str:
    names = [t.__name__ for t in types]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return ", ".join(names[:-1]) + f", or {names[-1]}"


def validate_instance(
    *,
    component: str,
    param_name: str,
    value: Any,
    expected_types: Union[type, Tuple[type, ...]],
    allow_none: bool = False,
) -> Any:
    """
    Generic type validator.

    - component: Intended to help identify what is being validated.
        e.g. "PromptOptimizer.__init__", "PromptOptimizer.optimize", etc.
    - param_name: the name of the parameter being validated
    - value: the actual value passed.
    - expected_types: a type or tuple of types to accept.
    - allow_none: if True, None is allowed and returned as-is.
    """
    if value is None and allow_none:
        return value

    if not isinstance(expected_types, tuple):
        expected_types = (expected_types,)

    if not isinstance(value, expected_types):
        expected_desc = _format_type_names(expected_types)
        raise DeepEvalError(
            f"{component} expected `{param_name}` to be an instance of "
            f"{expected_desc}, but received {type(value).__name__!r} instead."
        )
    return value


def validate_sequence_of(
    *,
    component: str,
    param_name: str,
    value: Any,
    expected_item_types: Union[type, Tuple[type, ...]],
    sequence_types: Tuple[type, ...] = (list, tuple),
    allow_none: bool = False,
) -> Any:
    """
    Generic container validator.

    - Ensures `value` is one of `sequence_types` (list by default).
    - Ensures each item is an instance of `expected_item_types`.

    Returns the original `value` on success.
    """
    if value is None:
        if allow_none:
            return value
        raise DeepEvalError(
            f"{component} expected `{param_name}` to be a "
            f"{_format_type_names(sequence_types)} of "
            f"{_format_type_names(expected_item_types if isinstance(expected_item_types, tuple) else (expected_item_types,))}, "
            "but received None instead."
        )

    if not isinstance(sequence_types, tuple):
        sequence_types = (sequence_types,)

    if not isinstance(value, sequence_types):
        expected_seq = _format_type_names(sequence_types)
        raise DeepEvalError(
            f"{component} expected `{param_name}` to be a {expected_seq}, "
            f"but received {type(value).__name__!r} instead."
        )

    if not isinstance(expected_item_types, tuple):
        expected_item_types = (expected_item_types,)

    for index, item in enumerate(value):
        if not isinstance(item, expected_item_types):
            expected_items = _format_type_names(expected_item_types)
            raise DeepEvalError(
                f"{component} expected all elements of `{param_name}` to be "
                f"instances of {expected_items}, but element at index {index} "
                f"has type {type(item).__name__!r}."
            )

    return value


def validate_callback(
    *,
    component: str,
    model_callback: Optional[
        Callable[
            ...,
            Union[
                str,
                Dict,
                Tuple[Union[str, Dict], float],
            ],
        ]
    ],
) -> Callable[..., Union[str, Dict, Tuple[Union[str, Dict], float]]]:
    """
    Ensure that `model_callback` is provided.

    - `model_callback` should be a callable that performs generation and
      returns the model output.

    Returns `model_callback` unchanged on success.
    """
    if model_callback is None:
        raise DeepEvalError(
            f"{component} requires a `model_callback`.\n\n"
            "supply a custom callable via `model_callback=` that performs "
            "generation and returns the model output."
        )
    return model_callback


def validate_metrics(
    *,
    component: str,
    metrics: Union[List[BaseMetric], List[BaseConversationalMetric]],
) -> Union[List[BaseMetric], List[BaseConversationalMetric]]:

    if metrics is None or not len(metrics):
        raise DeepEvalError(
            f"{component} requires a `metrics`.\n\n"
            "supply one or more DeepEval metrics via `metrics=`"
        )

    validate_sequence_of(
        component=component,
        param_name="metrics",
        value=metrics,
        expected_item_types=(BaseMetric, BaseConversationalMetric),
        sequence_types=(list, tuple),
    )
    return list(metrics)
