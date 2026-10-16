import re
from itertools import combinations
from typing import List, Optional, Set

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import (
    LLMTestCase,
    RetrievedContextData,
    SingleTurnParams,
)


def _words(text: str) -> List[str]:
    return re.sub(r"\s+", " ", text.strip().lower()).split()


def _ngrams(words: List[str], n: int) -> Set[tuple]:
    if len(words) < n:
        return set()
    return set(zip(*[words[i:] for i in range(n)]))


def _jaccard(first: Set[tuple], second: Set[tuple]) -> float:
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


class RetrievalRedundancyMetric(BaseMetric):
    """How much of the retrieved context is near-duplicate filler?

    Computes pairwise n-gram overlap (Jaccard) between the passages in
    ``retrieval_context``. A pair counts as redundant when its overlap
    reaches ``similarity_threshold``. The score is ``1`` minus the share
    of redundant pairs, so ``1.0`` means every passage adds distinct
    content and ``0.0`` means every pair overlaps. Fully
    **deterministic**: no LLM, no API key, zero token cost.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        ngram_n: int = 2,
        threshold: Optional[float] = 0.8,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if not 0 < similarity_threshold <= 1:
            raise ValueError(
                "`similarity_threshold` must be in (0, 1], got "
                f"{similarity_threshold}."
            )
        if ngram_n < 1:
            raise ValueError(
                f"`ngram_n` must be a positive integer, got {ngram_n}."
            )
        self.similarity_threshold = similarity_threshold
        self.ngram_n = ngram_n
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.async_mode = False

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
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
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            passages = [
                (
                    passage.context
                    if isinstance(passage, RetrievedContextData)
                    else passage
                )
                for passage in (test_case.retrieval_context or [])
            ]
            ngram_sets = [
                _ngrams(_words(passage), self.ngram_n) for passage in passages
            ]
            pairs = list(combinations(range(len(passages)), 2))
            redundant = sum(
                1
                for i, j in pairs
                if _jaccard(ngram_sets[i], ngram_sets[j])
                >= self.similarity_threshold
            )
            total = len(pairs)
            diversity = 1.0 if total == 0 else 1 - redundant / total
            self.score_breakdown = {
                "diversity": diversity,
                "overlap_ratio": 0.0 if total == 0 else redundant / total,
                "redundant_pairs": float(redundant),
                "total_pairs": float(total),
            }
            self.score = (
                0
                if self.strict_mode and diversity < self.threshold
                else diversity
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(redundant, total, len(passages))
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Passages: {len(passages)}",
                    f"Redundant pairs: {redundant} of {total}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def _generate_reason(
        self, redundant: int, total: int, passages: int
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        if total == 0:
            return (
                "Fewer than two passages were retrieved, "
                "so nothing can be redundant."
            )
        if redundant == 0:
            return (
                f"All {total} passage pair(s) are distinct "
                "below the overlap threshold."
            )
        return (
            f"{redundant} of {total} passage pair(s) overlap at or "
            f"above {self.similarity_threshold} across {passages} "
            "retrieved passage(s), wasting context space."
        )

    @property
    def __name__(self):
        return "Retrieval Redundancy"
