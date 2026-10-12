"""RAIL Score metric for DeepEval.

Evaluates LLM outputs across 8 responsible AI dimensions using the
RAIL Score API: fairness, safety, reliability, transparency, privacy,
accountability, inclusivity, and user_impact. Each dimension is scored
0-10 by the API, then normalized to 0-1 for DeepEval compatibility.

Per-dimension scores are available via ``metric.score_breakdown`` after
calling ``measure()``.

Requires: ``pip install rail-score-sdk deepeval``
"""

import os
import logging
from typing import Optional, List

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)

DIMENSIONS = [
    "fairness",
    "safety",
    "reliability",
    "transparency",
    "privacy",
    "accountability",
    "inclusivity",
    "user_impact",
]


class RAILScoreMetric(BaseMetric):
    """Evaluate LLM outputs across 8 responsible AI dimensions.

    Args:
        threshold: Minimum overall score (0-1) to pass. Default 0.5 (= 5/10).
        mode: ``"basic"`` (fast, hybrid ML) or ``"deep"`` (LLM-as-judge with
            explanations). Default ``"basic"``.
        domain: One of ``"general"``, ``"healthcare"``, ``"finance"``,
            ``"legal"``, ``"education"``, ``"code"``. Default ``"general"``.
        dimensions: Subset of the 8 dimensions to evaluate. Default: all 8.
        api_key: RAIL API key. Falls back to ``RAIL_API_KEY`` env var.
        include_reason: Populate ``self.reason`` with the API summary and
            per-dimension explanations (deep mode). Default ``True``.
        strict_mode: Binary 0/1 scoring. Default ``False``.
        async_mode: Use async evaluation. Default ``True``.
        verbose_mode: Print evaluation progress. Default ``False``.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        mode: str = "basic",
        domain: str = "general",
        dimensions: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose_mode: bool = False,
    ):
        self.threshold = 0 if strict_mode else threshold
        self.mode = mode
        self.domain = domain
        self.dimensions = dimensions
        self.api_key = api_key or os.environ.get("RAIL_API_KEY", "")
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_model = f"rail-score-api/{mode}"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Score the test case using the RAIL Score API (sync).

        When ``async_mode`` is ``True`` (the default), this delegates to
        ``a_measure()`` via an event loop, following the same pattern used
        by DeepEval's built-in metrics.
        """
        if self.async_mode:
            from deepeval.utils import get_or_create_event_loop

            loop = get_or_create_event_loop()
            loop.run_until_complete(self.a_measure(test_case))
            return self.score

        try:
            from rail_score_sdk import RailScoreClient
        except ImportError:
            raise ImportError(
                "rail-score-sdk is required. Install it with: "
                "pip install rail-score-sdk"
            )

        if not self.api_key:
            raise ValueError(
                "RAIL_API_KEY not set. Pass api_key to RAILScoreMetric() "
                "or set the RAIL_API_KEY environment variable."
            )

        content = test_case.actual_output
        if not content or len(content.strip()) < 10:
            self.score = 0
            self.reason = "Output too short to evaluate"
            self.success = False
            self.score_breakdown = {dim: 0 for dim in DIMENSIONS}
            return self.score

        client = RailScoreClient(api_key=self.api_key)
        result = client.eval(
            content=content,
            mode=self.mode,
            domain=self.domain,
            dimensions=self.dimensions,
            context=test_case.input if test_case.input else None,
            include_explanations=self.include_reason,
            include_issues=(self.mode == "deep"),
        )

        self.score = result.rail_score.score / 10.0
        if self.strict_mode:
            self.score = 1 if self.score >= self.threshold else 0

        self.score_breakdown = {
            dim: ds.score / 10.0 for dim, ds in result.dimension_scores.items()
        }

        if self.include_reason:
            parts = [result.rail_score.summary]
            if self.mode == "deep":
                for dim, ds in result.dimension_scores.items():
                    if ds.explanation:
                        parts.append(f"{dim} ({ds.score}/10): {ds.explanation}")
            self.reason = "\n".join(parts)
        else:
            self.reason = None

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Score the test case using the RAIL Score API (async)."""
        try:
            from rail_score_sdk import AsyncRAILClient
        except ImportError:
            raise ImportError(
                "rail-score-sdk is required. Install it with: "
                "pip install rail-score-sdk"
            )

        if not self.api_key:
            raise ValueError(
                "RAIL_API_KEY not set. Pass api_key to RAILScoreMetric() "
                "or set the RAIL_API_KEY environment variable."
            )

        content = test_case.actual_output
        if not content or len(content.strip()) < 10:
            self.score = 0
            self.reason = "Output too short to evaluate"
            self.success = False
            self.score_breakdown = {dim: 0 for dim in DIMENSIONS}
            return self.score

        async with AsyncRAILClient(api_key=self.api_key) as client:
            raw = await client.eval(
                content=content,
                mode=self.mode,
                domain=self.domain,
                dimensions=self.dimensions,
                context=test_case.input if test_case.input else None,
                include_explanations=self.include_reason,
                include_issues=(self.mode == "deep"),
            )

        rail = raw["rail_score"]
        dims = raw["dimension_scores"]

        self.score = rail["score"] / 10.0
        if self.strict_mode:
            self.score = 1 if self.score >= self.threshold else 0

        self.score_breakdown = {
            dim: data["score"] / 10.0 for dim, data in dims.items()
        }

        if self.include_reason:
            parts = [rail.get("summary", "")]
            if self.mode == "deep":
                for dim, data in dims.items():
                    exp = data.get("explanation")
                    if exp:
                        parts.append(f"{dim} ({data['score']}/10): {exp}")
            self.reason = "\n".join(parts)
        else:
            self.reason = None

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        """Return whether the metric score meets the threshold."""
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "RAIL Score"
