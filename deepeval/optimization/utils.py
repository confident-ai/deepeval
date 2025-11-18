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
from pydantic import BaseModel as PydanticBaseModel

from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.prompt.prompt import Prompt
from deepeval.optimization.types import ModuleId


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


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
    chosen_size = min(pareto_size, total)

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
    Build a human-readable module id stable within a single optimization run.
    Prefers alias/label; enrich with provider/name; dedupe; cap to 64 chars.
    """
    parts: List[str] = []
    if prompt.alias:
        parts.append(str(prompt.alias))
    if prompt.label:
        parts.append(str(prompt.label))

    ms = prompt.model_settings
    if ms is not None:
        if ms.provider is not None:
            parts.append(ms.provider.value)  # e.g., "OPEN_AI"
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
    If a list is given, generate human-readable module ids.
    """
    if isinstance(seed_prompts, dict):
        return dict(seed_prompts)  # shallow copy

    mapping: Dict[ModuleId, Prompt] = {}
    used: Set[str] = set()
    for i, prompt in enumerate(seed_prompts):
        module_id = generate_module_id(prompt, i, used)
        mapping[module_id] = prompt
    return mapping


def require_model_or_callback(
    *,
    component: str,
    model: Optional[DeepEvalBaseLLM],
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
) -> Tuple[
    Optional[DeepEvalBaseLLM],
    Optional[Callable[..., Union[str, Dict, Tuple[Union[str, Dict], float]]]],
]:
    """
    Ensure that at least one of `model` or `model_callback` is provided.

    - `model` should be a DeepEvalBaseLLM.
    - `model_callback` should be a callable that performs generation and
      returns the model output.

    Returns the pair unchanged on success.
    """
    if model is None and model_callback is None:
        raise DeepEvalError(
            f"{component} requires either a `model` or a `model_callback`.\n\n"
            "Pass a DeepEvalBaseLLM instance via `model=` to let DeepEval call "
            "`model.generate()` / `model.a_generate()`, or supply a custom callable "
            "via `model_callback=` that performs generation and returns the model output."
        )
    return model, model_callback


def build_model_callback_kwargs(
    *,
    # scoring context
    prompt: Optional[Prompt] = None,
    prompt_text: Optional[str] = None,
    golden: Optional[Union["Golden", "ConversationalGolden"]] = None,
    prompts_by_module: Optional[Dict[ModuleId, Prompt]] = None,
    # rewriter context
    module_id: Optional[ModuleId] = None,
    old_prompt: Optional[Prompt] = None,
    feedback_text: Optional[str] = None,
    # shared
    model: Optional[DeepEvalBaseLLM] = None,
    model_schema: Optional["PydanticBaseModel"] = None,
) -> Dict[str, Any]:
    """
    Build a superset of kwargs for GEPA model callbacks.

    All keys are present in the dict so callbacks can declare any subset of:

        hook: str           # injected by invoke_model_callback / a_invoke_model_callback
        prompt: Prompt
        prompt_text: str
        golden: Golden | ConversationalGolden
        prompts_by_module: Dict[ModuleId, Prompt]
        module_id: ModuleId
        old_prompt: Prompt
        feedback_text: str
        model: DeepEvalBaseLLM
        model_schema: BaseModel

    Non applicable fields are set to None.
    """
    return {
        # scoring context
        "prompt": prompt,
        "prompt_text": prompt_text,
        "golden": golden,
        "prompts_by_module": prompts_by_module,
        # rewriter context
        "module_id": module_id,
        "old_prompt": old_prompt,
        "feedback_text": feedback_text,
        # shared
        "model": model,
        "model_schema": model_schema,
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
