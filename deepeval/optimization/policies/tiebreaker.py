from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random

from deepeval.optimization.types import CandidateId


class TieBreaker(str, Enum):
    PREFER_ROOT = "prefer_root"
    PREFER_CHILD = "prefer_child"
    RANDOM = "random"


def pick_best_with_ties(
    totals: Dict[CandidateId, float],
    parents_by_id: Dict[CandidateId, Optional[CandidateId]],
    *,
    random_state: random.Random,
    tie_tolerance: float = 1e-9,
    policy: TieBreaker = TieBreaker.PREFER_ROOT,
) -> Tuple[CandidateId, List[CandidateId], float]:
    """
    Choose the best candidate by aggregate score with deterministic tie handling.

    Returns: (chosen_id, tied_ids, max_val)
    - tied_ids includes everyone within tie_tolerance of max_val
    """
    if not totals:
        raise ValueError("No candidates to choose from.")

    max_val = max(totals.values())
    tied = [
        cid for cid, v in totals.items() if abs(v - max_val) <= tie_tolerance
    ]

    if len(tied) == 1:
        return tied[0], tied, max_val

    # Resolve tie by policy
    if policy == TieBreaker.PREFER_CHILD:
        # Prefer any non root. When multiple children exist, use the most recent
        child_ids = [cid for cid in tied if parents_by_id.get(cid) is not None]
        if child_ids:
            # choose the newest child deterministically by order
            for cid in reversed(list(totals.keys())):
                if cid in child_ids:
                    return cid, tied, max_val

    if policy == TieBreaker.RANDOM:
        return random_state.choice(tied), tied, max_val

    # by default prefer a root if present, otherwise the first tied
    root_ids = [cid for cid in tied if parents_by_id.get(cid) is None]
    chosen = root_ids[0] if root_ids else tied[0]
    return chosen, tied, max_val
