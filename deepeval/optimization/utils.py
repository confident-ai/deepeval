from __future__ import annotations
import random
import re
from typing import List, Tuple, TYPE_CHECKING, Union, Dict, Set

from deepeval.prompt.prompt import Prompt
from deepeval.optimization.types import ModuleId


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


def split_goldens(
    goldens: Union[List[Golden], List[ConversationalGolden]],
    pareto_size: int,
    *,
    seed: int = 0,
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
        seed: Random seed for reproducible selection.

    Returns:
        (d_feedback, d_pareto)
    """
    if pareto_size < 0:
        raise ValueError("pareto_size must be >= 0")

    total = len(goldens)
    chosen_size = min(pareto_size, total)

    indices = list(range(total))
    random_state = random.Random(seed)
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
