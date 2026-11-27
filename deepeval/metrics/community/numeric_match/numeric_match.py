import math
import re
from typing import List, Optional, Tuple

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams


# One numeric token: optional sign, optional currency symbol, an integer with
# optional thousands separators / a decimal / a plain integer, an optional
# scientific exponent, and a single optional trailing marker (percent or a
# magnitude suffix). Kept deliberately conservative so it reads quantities out
# of natural-language answers without swallowing arbitrary punctuation.
_NUMBER_RE = re.compile(
    r"""
    (?P<sign>[+-]?)
    [\$\u20ac\u00a3\u00a5]?\s?              # optional currency symbol
    (?P<num>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?       # 1,234,567 or 1,200.50
        | \d*\.\d+                         # 3.14 or .5
        | \d+                              # 42
    )
    (?P<exp>[eE][+-]?\d+)?                  # 1.2e3
    (?P<marker>%|[kKmMbBtT])?               # 12% or 1.2M
    """,
    re.VERBOSE,
)

_MAGNITUDE = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}


class NumericMatchMetric(BaseMetric):
    """Formatting-robust numeric agreement between ``actual_output`` and
    ``expected_output``.

    ``ExactMatchMetric`` and ``PatternMatchMetric`` compare strings, so a
    numerically correct answer is silently scored wrong when the surface form
    differs (``"1,200"`` vs ``"1200"``, ``"3.0"`` vs ``"3"``, ``"$1.2M"`` vs
    ``"1200000"``, ``"12%"`` vs ``"12"``), and a wrong answer can be scored
    right when a correct-looking number appears somewhere in a longer string.
    ``NumericMatchMetric`` extracts the numeric quantities from both fields and
    compares them by value under a configurable tolerance.

    The score is the fraction of the numbers in ``expected_output`` that have a
    tolerance-matching counterpart in ``actual_output`` (recall of the reference
    numbers). With the default threshold of ``1.0`` every reference number must
    be present. Matching is multiset-aware: each output number can satisfy at
    most one reference number, so duplicates are not double-counted.

    The metric is deterministic and uses no judge model, so it is reproducible
    and adds no evaluation cost. When ``expected_output`` contains no numeric
    value the metric is not applicable and raises ``ValueError`` rather than
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
        rel_tol: float = 1e-9,
        abs_tol: float = 0.0,
        parse_magnitude_suffixes: bool = False,
        percent_as_fraction: bool = False,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.parse_magnitude_suffixes = parse_magnitude_suffixes
        self.percent_as_fraction = percent_as_fraction
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
            expected_numbers = self._extract_numbers(test_case.expected_output)
            if not expected_numbers:
                raise ValueError(
                    "`expected_output` contains no numeric value, so "
                    "NumericMatchMetric cannot score this test case. Use "
                    "ExactMatchMetric or an LLM-judged metric for "
                    "non-numeric references."
                )
            actual_numbers = self._extract_numbers(test_case.actual_output)

            matched, missing, unmatched = self._match(
                expected_numbers, actual_numbers
            )
            self.score = matched / len(expected_numbers)
            self.reason = self._generate_reason(
                len(expected_numbers), matched, missing, unmatched
            )
            self.success = self.is_successful()

            if self.verbose_mode:
                expected_str = self._format_numbers(expected_numbers)
                actual_str = self._format_numbers(actual_numbers)
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Expected numbers: {expected_str}",
                        f"Actual numbers: {actual_str}",
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

    def _extract_numbers(self, text: str) -> List[float]:
        numbers: List[float] = []
        for match in _NUMBER_RE.finditer(text):
            value = self._parse_match(match)
            if value is not None:
                numbers.append(value)
        return numbers

    def _parse_match(self, match: "re.Match") -> Optional[float]:
        sign = -1.0 if match.group("sign") == "-" else 1.0
        raw = match.group("num").replace(",", "")
        exp = match.group("exp") or ""
        marker = match.group("marker")

        try:
            value = float(raw + exp)
        except ValueError:
            return None
        value *= sign

        if marker == "%":
            if self.percent_as_fraction:
                value /= 100.0
        elif marker is not None:
            if not self.parse_magnitude_suffixes:
                # A trailing k/m/b/t was matched but suffix parsing is off, so
                # the character is not part of the number: keep the bare value.
                return value
            value *= _MAGNITUDE[marker.lower()]

        return value

    def _match(
        self, expected: List[float], actual: List[float]
    ) -> Tuple[int, List[float], List[float]]:
        remaining = list(actual)
        matched = 0
        missing: List[float] = []
        for target in expected:
            hit_index = None
            for index, candidate in enumerate(remaining):
                if math.isclose(
                    target,
                    candidate,
                    rel_tol=self.rel_tol,
                    abs_tol=self.abs_tol,
                ):
                    hit_index = index
                    break
            if hit_index is None:
                missing.append(target)
            else:
                matched += 1
                remaining.pop(hit_index)
        return matched, missing, remaining

    def _generate_reason(
        self,
        total_expected: int,
        matched: int,
        missing: List[float],
        unmatched: List[float],
    ) -> Optional[str]:
        if self.include_reason is False:
            return None
        parts = [
            f"Matched {matched}/{total_expected} reference number(s) within "
            f"tolerance (rel_tol={self.rel_tol}, abs_tol={self.abs_tol})."
        ]
        if missing:
            missing_str = self._format_numbers(missing)
            parts.append(f"Missing from output: {missing_str}.")
        if unmatched:
            unmatched_str = self._format_numbers(unmatched)
            parts.append(f"Unmatched numbers in output: {unmatched_str}.")
        return " ".join(parts)

    @staticmethod
    def _format_numbers(numbers: List[float]) -> str:
        formatted = []
        for number in numbers:
            if number == int(number):
                formatted.append(str(int(number)))
            else:
                formatted.append(repr(number))
        return "[" + ", ".join(formatted) + "]"

    @property
    def __name__(self):
        return "Numeric Match"
