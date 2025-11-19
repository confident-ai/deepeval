from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random

from deepeval.errors import DeepEvalError
from deepeval.optimization.types import PromptConfigurationId


class TieBreaker(str, Enum):
    PREFER_ROOT = "prefer_root"
    PREFER_CHILD = "prefer_child"
    RANDOM = "random"


def pick_best_with_ties(
    totals: Dict[PromptConfigurationId, float],
    parents_by_id: Dict[PromptConfigurationId, Optional[PromptConfigurationId]],
    *,
    random_state: random.Random,
    tie_tolerance: float = 1e-9,
    policy: TieBreaker = TieBreaker.PREFER_ROOT,
) -> Tuple[PromptConfigurationId, List[PromptConfigurationId], float]:
    """
    Choose the best candidate by aggregate score with deterministic tie handling.

    Returns: (chosen_id, tied_ids, max_score)
    - tied_ids includes everyone within tie_tolerance of max_score
    """
    if not totals:
        raise DeepEvalError("No candidate prompt configuration to choose from.")

    max_score = max(totals.values())
    tied = [
        prompt_configuration_id
        for prompt_configuration_id, score in totals.items()
        if abs(score - max_score) <= tie_tolerance
    ]

    if len(tied) == 1:
        return tied[0], tied, max_score

    # Resolve tie by policy
    if policy == TieBreaker.PREFER_CHILD:
        # Prefer any non root. When multiple children exist, use the most recent
        child_ids = [
            prompt_configuration_id
            for prompt_configuration_id in tied
            if parents_by_id.get(prompt_configuration_id) is not None
        ]
        if child_ids:
            # choose the newest child deterministically by order
            for prompt_configuration_id in reversed(list(totals.keys())):
                if prompt_configuration_id in child_ids:
                    return prompt_configuration_id, tied, max_score

    if policy == TieBreaker.RANDOM:
        return random_state.choice(tied), tied, max_score

    # by default prefer a root if present, otherwise the first tied
    root_ids = [
        prompt_configuration_id
        for prompt_configuration_id in tied
        if parents_by_id.get(prompt_configuration_id) is None
    ]
    chosen = root_ids[0] if root_ids else tied[0]
    return chosen, tied, max_score
