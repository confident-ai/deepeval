from typing import Any, Dict, List, Optional, Union

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


class FilterCorrectnessMetric(BaseMetric):
    """Deterministic metadata-filter correctness for retrieval.

    Checks whether every chunk in ``retrieval_context`` satisfies a metadata
    filter — e.g. that only documents for a given ``tenant_id`` or
    ``jurisdiction`` were returned.  No LLM is required.

    Parameters
    ----------
    filter_field:
        The key that must be present in each chunk's
        ``RetrievedContextData.metadata`` dict.
    allowed_values:
        A single value or a list of values that are acceptable for
        ``filter_field`` (equality / ``$in`` semantics).
    missing_metadata_fails:
        When ``True`` (the default), a chunk that carries no ``metadata``
        dict, or whose ``metadata`` dict lacks ``filter_field``, counts as a
        violation.  Set to ``False`` to ignore chunks without the field.
    threshold:
        Minimum fraction of compliant chunks required to pass.  Defaults to
        ``1.0`` (all chunks must comply).
    include_reason:
        Whether to populate ``metric.reason``.
    strict_mode:
        When ``True``, any score below ``threshold`` is zeroed out to make the
        result binary.
    verbose_mode:
        Whether to populate ``metric.verbose_logs``.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        filter_field: str,
        allowed_values: Union[Any, List[Any]],
        missing_metadata_fails: bool = True,
        threshold: Optional[float] = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if not filter_field:
            raise ValueError("`filter_field` must be a non-empty string.")
        self.filter_field = filter_field
        self.allowed_values = (
            allowed_values
            if isinstance(allowed_values, list)
            else [allowed_values]
        )
        self.missing_metadata_fails = missing_metadata_fails
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
            test_case, self._required_params, None, None, self
        )
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            chunks = test_case.retrieval_context or []
            violations = self._find_violations(chunks)
            total = len(chunks)
            score = 1.0 if total == 0 else (total - len(violations)) / total
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(chunks, violations)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Filter field: {self.filter_field!r}",
                    f"Allowed values: {self.allowed_values}",
                    f"missing_metadata_fails: {self.missing_metadata_fails}",
                    f"Total chunks: {total}",
                    f"Violations: {violations}",
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

    def _chunk_value(
        self, chunk: Union[str, RetrievedContextData]
    ) -> Optional[Any]:
        """Return the value of ``filter_field`` from a chunk, or ``None`` if absent."""
        if not isinstance(chunk, RetrievedContextData):
            return None
        if chunk.metadata is None:
            return None
        return chunk.metadata.get(self.filter_field)

    def _chunk_violates(self, chunk: Union[str, RetrievedContextData]) -> bool:
        value = self._chunk_value(chunk)
        if value is None:
            return self.missing_metadata_fails
        return value not in self.allowed_values

    def _find_violations(
        self, chunks: List[Union[str, RetrievedContextData]]
    ) -> List[Dict]:
        """Return a list of dicts describing each violating chunk."""
        violations = []
        for i, chunk in enumerate(chunks):
            if self._chunk_violates(chunk):
                value = self._chunk_value(chunk)
                violations.append(
                    {
                        "index": i,
                        "value": value,
                        "source": (
                            chunk.source
                            if isinstance(chunk, RetrievedContextData)
                            else None
                        ),
                    }
                )
        return violations

    def _generate_reason(
        self,
        chunks: List[Union[str, RetrievedContextData]],
        violations: List[Dict],
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        total = len(chunks)
        if total == 0:
            return "No retrieved chunks to evaluate."
        if not violations:
            return (
                f"All {total} chunk(s) satisfy the filter "
                f"{self.filter_field!r} in {self.allowed_values}."
            )
        offending = [
            (
                f"index {v['index']} (source={v['source']!r}, "
                f"value={v['value']!r})"
            )
            for v in violations
        ]
        return (
            f"{len(violations)} of {total} chunk(s) violated the filter "
            f"{self.filter_field!r} (allowed={self.allowed_values}): "
            f"{offending}."
        )

    @property
    def __name__(self):
        return "Filter Correctness"
