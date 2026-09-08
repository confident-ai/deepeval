# ruff: noqa: UP006, UP035, UP045
"""Structured output schema for ``TrajectoryDivergenceMetric``.

The metric is fully deterministic (no LLM judge), so unlike most metric
schemas this module does not model a model-generated verdict. It models the
metric's own structured result — the localization of the first sustained
divergence — so callers can consume ``score_breakdown`` as typed,
JSON-serializable data (e.g. for CI assertions or logging).
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel


class TrajectoryDivergenceResult(BaseModel):
    """Structured localization output of a trajectory comparison.

    Field semantics follow the alignment seam's ``AlignmentResult``: the
    zero-based step index where the traces first differ, the difference kind,
    the step where they realign (``None`` for an unrecovered fork), the step
    IDs that could not be matched on each side, and the reordered pairs.
    """

    matched_prefix_len: int = 0
    first_divergence: Optional[int] = None
    divergence_kind: Optional[str] = None
    resync_at: Optional[int] = None
    unmatched_baseline: List[str] = []
    unmatched_candidate: List[str] = []
    reordered: List[Tuple[str, str]] = []
    divergence_ratio: float = 0.0
