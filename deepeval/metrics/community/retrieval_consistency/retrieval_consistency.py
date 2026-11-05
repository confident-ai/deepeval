from typing import List, Literal, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.community._consistency_utils import (
    describe_pair,
    get_runs,
    require_repeatability_evals,
    sample_pairs,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


class RetrievalConsistencyMetric(BaseMetric):
    """How consistently a retriever returns the same evidence for the same input.

    Needs no relevance labels and no LLM. Provide N >= 2 runs in
    ``metadata["retrieval_runs"]``; each run is the list of chunks one retrieval returned.

    * ``mode="id"``: each chunk is an identifier; runs are compared by Jaccard similarity of
      their ID sets. Cheap and exact, but any re-chunking or re-indexing changes the IDs.
    * ``mode="ngram"``: each chunk is its text; runs are compared by Jaccard similarity of the
      word n-grams of their pooled text, so the same evidence split differently still matches.

    Order is ignored. Consistency is not relevance: a retriever that always returns the same
    wrong chunks scores 1.0.
    """

    _required_params: List[SingleTurnParams] = [SingleTurnParams.INPUT]

    def __init__(
        self,
        mode: Literal["id", "ngram"] = "ngram",
        ngram_size: int = 3,
        threshold: Optional[float] = 0.8,
        max_comparisons: Optional[int] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if mode not in ("id", "ngram"):
            raise ValueError("mode must be 'id' or 'ngram'")
        lib = require_repeatability_evals()
        self._similarity = (
            lib.JaccardSimilarity()
            if mode == "id"
            else lib.NgramJaccard(n=ngram_size)
        )
        self.mode = mode
        self.ngram_size = ngram_size
        self.threshold = 1 if strict_mode else threshold
        self.max_comparisons = max_comparisons
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_model = None
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
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            runs = get_runs(test_case.metadata, "retrieval_runs")
            runs = [[str(c) for c in run] for run in runs]
            pairs = sample_pairs(len(runs), self.max_comparisons)
            scores = [
                self._similarity.score(runs[i], runs[j]).score for i, j in pairs
            ]
            score = sum(scores) / len(scores)
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            worst = min(range(len(pairs)), key=lambda n: scores[n])
            self.reason = (
                None
                if not self.include_reason
                else (
                    f"Compared {len(pairs)} pairs of retrieval runs using "
                    f"{'chunk-ID' if self.mode == 'id' else f'{self.ngram_size}-gram'} "
                    f"overlap; mean similarity is {score:.2f}. Most different pair: "
                    f"{describe_pair(*pairs[worst])} ({scores[worst]:.2f})."
                )
            )
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Pair scores: {[round(s, 2) for s in scores]}",
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
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    @property
    def __name__(self):
        return "Retrieval Consistency"
