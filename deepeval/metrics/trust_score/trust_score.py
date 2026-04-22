from typing import Dict, List, Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

class TrustScoreMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
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
        from deepeval.metrics.utils import check_llm_test_case_params
        from deepeval.metrics.indicator import metric_progress_indicator

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
            if not test_case.retrieval_context or len(test_case.retrieval_context) == 0:
                self.score = 1.0
                self.reason = "No retrieval context provided."
                self.success = self.score >= self.threshold
                return self.score

            total_score = 0.0
            reasons = []

            for context_chunk in test_case.retrieval_context:
                matched_tier = None
                matched_source = None
                # Check for source matching (case-insensitive)
                for source, tier in self.source_tiers.items():
                    if source.lower() in context_chunk.lower():
                        matched_tier = tier
                        matched_source = source
                        break

                if matched_tier == 1:
                    chunk_score = 1.0
                elif matched_tier == 2:
                    chunk_score = 0.8
                elif matched_tier == 3:
                    chunk_score = 0.6
                elif matched_tier == 4:
                    chunk_score = 0.4
                elif matched_tier == 5:
                    chunk_score = 0.2
                else:
                    chunk_score = 0.5
                    matched_source = "Unmatched Source"
                    matched_tier = "None"

                total_score += chunk_score
                reasons.append(f"'{matched_source}' mapped to tier {matched_tier}")

            self.score = total_score / len(test_case.retrieval_context)
            self.reason = "Sources found: " + ", ".join(reasons)
            self.success = self.score >= self.threshold

            if _log_metric_to_confident:
                from deepeval.metrics.api import metric_data_manager
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
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except Exception:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Trust Score"
