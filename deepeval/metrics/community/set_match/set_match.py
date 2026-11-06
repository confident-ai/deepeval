import json
import re
from typing import List, Optional, Set

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams


# Default separators used to split a plain-text list into items: commas,
# semicolons, and newlines. A leading bullet or numbered-list marker is
# stripped from each item after splitting.
_DEFAULT_SPLIT_RE = re.compile(r"[,\n;]+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


class SetMatchMetric(BaseMetric):
    """Order-insensitive set agreement between ``actual_output`` and
    ``expected_output``.

    ``ExactMatchMetric`` compares whole strings, so a multi-item answer is
    silently scored wrong whenever the order differs, an item is duplicated,
    or spacing/case varies (``"apple, banana"`` vs ``"Banana and apple"``),
    even though the underlying set of answers is identical. ``SetMatchMetric``
    parses both fields into a set of normalized items and scores their overlap
    with precision, recall, or F1.

    The metric is deterministic and uses no judge model, so it is reproducible
    and adds no evaluation cost. When ``expected_output`` parses to no items
    the metric is not applicable and raises ``ValueError`` rather than
    fabricating a ``0.0`` or ``1.0``.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 1.0,
        mode: str = "f1",
        case_sensitive: bool = False,
        parse_json_arrays: bool = True,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if mode not in ("f1", "recall", "precision"):
            raise ValueError(
                "`mode` must be one of 'f1', 'recall', or 'precision', "
                f"got {mode!r}."
            )
        self.threshold = 1.0 if strict_mode else threshold
        self.mode = mode
        self.case_sensitive = case_sensitive
        self.parse_json_arrays = parse_json_arrays
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky

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
            expected = self._parse_items(test_case.expected_output)
            if not expected:
                raise ValueError(
                    "`expected_output` parses to no items, so "
                    "SetMatchMetric cannot score this test case. Use "
                    "ExactMatchMetric or an LLM-judged metric for "
                    "non-list references."
                )
            actual = self._parse_items(test_case.actual_output)

            matched = expected & actual
            missing = expected - actual
            extra = actual - expected
            self.score = self._score(len(matched), len(expected), len(actual))
            self.reason = self._generate_reason(matched, missing, extra)
            self.success = self.is_successful()

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Expected items: {self._format_items(expected)}",
                        f"Actual items: {self._format_items(actual)}",
                        f"Mode: {self.mode}",
                        f"Score: {self.score:.2f}",
                        f"Reason: {self.reason}",
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

    def _parse_items(self, text: str) -> Set[str]:
        items: Set[str] = set()
        for raw in self._split(text):
            normalized = self._normalize(raw)
            if normalized:
                items.add(normalized)
        return items

    def _split(self, text: str) -> List[str]:
        stripped = text.strip()
        if self.parse_json_arrays and stripped.startswith("["):
            try:
                loaded = json.loads(stripped)
            except ValueError:
                loaded = None
            if isinstance(loaded, list):
                return [self._stringify(element) for element in loaded]
        return _DEFAULT_SPLIT_RE.split(text)

    @staticmethod
    def _stringify(element) -> str:
        if isinstance(element, str):
            return element
        return json.dumps(element, sort_keys=True)

    def _normalize(self, item: str) -> str:
        cleaned = _LIST_MARKER_RE.sub("", item).strip()
        if not self.case_sensitive:
            cleaned = cleaned.lower()
        return cleaned

    def _score(self, matched: int, n_expected: int, n_actual: int) -> float:
        recall = matched / n_expected if n_expected else 0.0
        precision = matched / n_actual if n_actual else 0.0
        if self.mode == "recall":
            return recall
        if self.mode == "precision":
            return precision
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _generate_reason(
        self, matched: Set[str], missing: Set[str], extra: Set[str]
    ) -> Optional[str]:
        if self.include_reason is False:
            return None
        parts = [f"Matched {len(matched)} item(s); mode={self.mode}."]
        if missing:
            parts.append(f"Missing from output: {self._format_items(missing)}.")
        if extra:
            parts.append(f"Extra in output: {self._format_items(extra)}.")
        return " ".join(parts)

    @staticmethod
    def _format_items(items: Set[str]) -> str:
        return "{" + ", ".join(repr(item) for item in sorted(items)) + "}"

    @property
    def __name__(self):
        return "Set Match"
