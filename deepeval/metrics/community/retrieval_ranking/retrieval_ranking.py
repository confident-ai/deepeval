import math
import re
from typing import Dict, List, Optional

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

_MEASURES = ("recall", "precision", "mrr", "ndcg", "hit_rate")
_MATCH_MODES = ("exact", "contains")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _matches(relevant: str, retrieved: str, match_mode: str) -> bool:
    norm_rel = _normalize(relevant)
    norm_ret = _normalize(retrieved)
    if not norm_rel or not norm_ret:
        return False
    if match_mode == "exact":
        return norm_rel == norm_ret
    return norm_rel == norm_ret or norm_rel in norm_ret or norm_ret in norm_rel


class RetrievalRankingMetric(BaseMetric):
    """Did the retriever rank the relevant chunks near the top?

    Matches the ranked ``retrieval_context`` list against the known
    relevant ``context`` chunks, without an LLM. Fully **deterministic**,
    needs no API key, and costs zero tokens.

    Supported measures (``metric`` argument):

    - ``recall``: share of relevant chunks found in the top-k.
    - ``precision``: share of top-k retrieved chunks that are relevant.
    - ``mrr``: 1 divided by the rank of the first relevant chunk
      (0 when none is in the top-k).
    - ``ndcg``: ranking gain divided by the ideal ranking gain, with
      ``1 / log2(rank + 1)`` discounts.
    - ``hit_rate``: 1 when any relevant chunk is in the top-k, else 0.

    ``score_breakdown`` always contains every measure so one run can feed
    ``retrieval_recall@k``, ``retrieval_precision@k``, ``mrr``, ``ndcg``
    and ``hit_rate`` dashboards at once; ``score`` is the selected
    ``metric``.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.RETRIEVAL_CONTEXT,
        SingleTurnParams.CONTEXT,
    ]

    def __init__(
        self,
        k: Optional[int] = None,
        metric: str = "recall",
        match_mode: str = "contains",
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if k is not None and k <= 0:
            raise ValueError(
                "RetrievalRankingMetric requires `k` to be a positive "
                f"integer, got {k}."
            )
        if metric not in _MEASURES:
            raise ValueError(
                f"Unknown metric '{metric}'. Supported: {list(_MEASURES)}."
            )
        if match_mode not in _MATCH_MODES:
            raise ValueError(
                f"Unknown match_mode '{match_mode}'. "
                f"Supported: {list(_MATCH_MODES)}."
            )
        self.k = k
        self.metric = metric
        self.match_mode = match_mode
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
            if not test_case.context:
                raise ValueError(
                    "`context` must be a non-empty list of relevant "
                    "chunks for the 'Retrieval Ranking' metric."
                )
            retrieved_texts = [
                (
                    passage.context
                    if isinstance(passage, RetrievedContextData)
                    else passage
                )
                for passage in (test_case.retrieval_context or [])
            ]
            relevant_texts = list(test_case.context or [])
            effective_k = self.k or len(retrieved_texts)
            top_k = retrieved_texts[:effective_k] if effective_k else []

            breakdown = self._score_all(top_k, relevant_texts, effective_k)
            self.score_breakdown = breakdown
            selected = float(breakdown[self._breakdown_key(self.metric)])
            self.score = (
                0
                if self.strict_mode and selected < self.threshold
                else selected
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(
                breakdown, len(relevant_texts), effective_k
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Relevant chunks: {len(relevant_texts)}",
                    f"Retrieved passages: {len(retrieved_texts)} "
                    f"(scoring top {effective_k})",
                    f"Match mode: {self.match_mode}",
                    f"Selected metric: {self.metric} = {selected:.4f}",
                    f"Breakdown: {breakdown}",
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

    def _score_all(
        self,
        top_k: List[str],
        relevant_texts: List[str],
        effective_k: int,
    ) -> Dict[str, float]:
        # Per-retrieved relevance: 1 when the passage matches any chunk.
        retrieved_labels = [
            (
                1
                if any(
                    _matches(relevant, retrieved, self.match_mode)
                    for relevant in relevant_texts
                )
                else 0
            )
            for retrieved in top_k
        ]
        # Distinct relevant chunks recalled by at least one retrieved passage.
        recalled = sum(
            1
            for relevant in relevant_texts
            if any(
                _matches(relevant, retrieved, self.match_mode)
                for retrieved in top_k
            )
        )
        num_relevant = len(relevant_texts)

        recall = recalled / num_relevant if num_relevant else 0.0
        precision = sum(retrieved_labels) / len(top_k) if top_k else 0.0
        hit_rate = 1.0 if any(retrieved_labels) else 0.0

        first_rank = next(
            (i + 1 for i, label in enumerate(retrieved_labels) if label),
            None,
        )
        mrr = 1.0 / first_rank if first_rank else 0.0

        dcg = sum(
            label / math.log2(i + 2) for i, label in enumerate(retrieved_labels)
        )
        ideal_hits = min(num_relevant, len(top_k))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg else 0.0

        return {
            "recall@k": recall,
            "precision@k": precision,
            "mrr": mrr,
            "ndcg": ndcg,
            "hit_rate": hit_rate,
            "k": float(effective_k),
            "first_relevant_rank": float(first_rank) if first_rank else 0.0,
            "relevant_recalled": float(recalled),
            "num_relevant": float(num_relevant),
        }

    @staticmethod
    def _breakdown_key(metric: str) -> str:
        if metric in ("recall", "precision"):
            return f"{metric}@k"
        return metric

    def _generate_reason(
        self, breakdown: Dict[str, float], num_relevant: int, k: int
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        selected_key = self._breakdown_key(self.metric)
        selected = breakdown[selected_key]
        recalled = int(breakdown["relevant_recalled"])
        first_rank = int(breakdown["first_relevant_rank"])
        label = (
            f"{self.metric}@{k}"
            if selected_key.endswith("@k") or self.metric == "hit_rate"
            else selected_key
        )
        rank_note = (
            f" First relevant chunk at rank {first_rank}."
            if first_rank
            else " No relevant chunk in the top-k."
        )
        return (
            f"{label} is {selected:.2f}: {recalled} of "
            f"{num_relevant} relevant chunk(s) found in the top {k}."
            + rank_note
        )

    @property
    def __name__(self):
        return "Retrieval Ranking"
