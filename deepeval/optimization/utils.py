from __future__ import annotations
from typing import List, Sequence, Tuple, TypeVar
import random

T = TypeVar("T")


def split_dataset(
    dataset: Sequence[T],
    pareto_size: int,
    *,
    seed: int = 0,
) -> Tuple[List[T], List[T]]:
    """
    Split `dataset` into two disjoint parts:

      - d_feedback: items not selected for the Pareto validation set
      - d_pareto:   `pareto_size` items for instance-wise Pareto scoring

    The selection is deterministic given `seed`. Within each split, the
    original order from `dataset` is preserved.

    Args:
        dataset: Full list/sequence of examples.
        pareto_size: Number of items to allocate to the Pareto set (clamped to [0, len(dataset)]).
        seed: Random seed for reproducible selection.

    Returns:
        (d_feedback, d_pareto)
    """
    if pareto_size < 0:
        raise ValueError("pareto_size must be >= 0")

    total = len(dataset)
    chosen_size = min(pareto_size, total)

    indices = list(range(total))
    random_state = random.Random(seed)
    random_state.shuffle(indices)

    pareto_indices = set(indices[:chosen_size])

    d_pareto = [dataset[i] for i in range(total) if i in pareto_indices]
    d_feedback = [dataset[i] for i in range(total) if i not in pareto_indices]

    return d_feedback, d_pareto
