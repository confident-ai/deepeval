from typing import List, Dict, Optional
import asyncio

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics.api import metric_data_manager
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class TrustScoreMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        source_tiers: Dict[str, int],
        threshold: float = 0.7,
        verbose_mode: bool = False,
    ):
        self.source_tiers = source_tiers
        self.threshold = threshold
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            None,
            test_case.multimodal,
        )

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            self._calculate_score_and_reason(test_case.retrieval_context)
            self.success = self.score >= self.threshold

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Score: {self.score:.2f}",
                        f"Reason: {self.reason}",
                    ],
                )

            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # The calculation is entirely synchronous based on the context strings,
        # so we can just defer to measure().
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def _calculate_score_and_reason(self, retrieval_context: Optional[List[str]]):
        if not retrieval_context:
            self.score = 0.5
            self.reason = "Empty retrieval context, defaulting to score of 0.5."
            return

        tier_scores = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
        default_score = 0.5

        chunk_scores = []
        reason_parts = []

        for chunk in retrieval_context:
            matched_tier = None
            matched_source = None

            # Find the best (lowest number) tier that matches
            for source, tier in self.source_tiers.items():
                if source in chunk:
                    if matched_tier is None or tier < matched_tier:
                        matched_tier = tier
                        matched_source = source

            if matched_tier is not None:
                score = tier_scores.get(matched_tier, default_score)
                chunk_scores.append(score)
                reason_parts.append(f"Matched source '{matched_source}' mapped to Tier {matched_tier} (Score: {score}).")
            else:
                chunk_scores.append(default_score)
                reason_parts.append(f"Unmatched source in context chunk, defaulting to Score: {default_score}.")

        self.score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else default_score

        reason_str = " ".join(reason_parts)
        self.reason = f"Average score calculated from sources: {reason_str}"

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Trust Score"
