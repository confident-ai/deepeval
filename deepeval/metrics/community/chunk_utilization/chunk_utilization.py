import re
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

# `[...]` not followed by `(`, so markdown links are not citations.
_MARKER_RE = re.compile(r"\[([^\[\]]+)\](?!\()")


class ChunkUtilizationMetric(BaseMetric):
    """What share of the retrieved chunks did the answer actually use?

    Counts the distinct ``retrieval_context`` passages cited by
    ``actual_output``. ``[N]`` refers to the N-th passage (1-based); any
    other reference must equal the ``source`` of a
    ``RetrievedContextData`` passage. The score is distinct cited
    passages divided by retrieved passages. Fully **deterministic**:
    no LLM, no API key, zero token cost.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
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
            passage_count = len(test_case.retrieval_context)
            sources = {
                passage.source
                for passage in test_case.retrieval_context
                if isinstance(passage, RetrievedContextData)
            }
            references = [
                reference.strip()
                for marker in _MARKER_RE.findall(test_case.actual_output)
                for reference in marker.split(",")
                if reference.strip()
            ]
            cited: Set[str] = set()
            broken = 0
            for reference in references:
                key = self._resolve(reference, passage_count, sources)
                if key is None:
                    broken += 1
                else:
                    cited.add(key)
            utilization = len(cited) / passage_count if passage_count else 0.0
            self.score_breakdown = {
                "utilization_rate": utilization,
                "cited_chunks": float(len(cited)),
                "total_chunks": float(passage_count),
                "broken_citations": float(broken),
            }
            self.score = (
                0
                if self.strict_mode and utilization < self.threshold
                else utilization
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(
                len(cited), passage_count, broken
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Passages in retrieval context: {passage_count}",
                    f"Cited passages: {sorted(cited)}",
                    f"Broken references: {broken}",
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

    @staticmethod
    def _resolve(
        reference: str, passage_count: int, sources: Set[str]
    ) -> Optional[str]:
        if reference.isdigit():
            index = int(reference)
            return f"#{index}" if 1 <= index <= passage_count else None
        return f"@{reference}" if reference in sources else None

    def _generate_reason(
        self, cited: int, total: int, broken: int
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        base = (
            f"{cited} of {total} retrieved chunk(s) are cited "
            "in the actual output."
        )
        if broken:
            return base + f" {broken} citation(s) match no passage."
        return base

    @property
    def __name__(self):
        return "Chunk Utilization"
