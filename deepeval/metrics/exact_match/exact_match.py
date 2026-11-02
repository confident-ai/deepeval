import re
import unicodedata
from typing import List, Optional

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams


class ExactMatchMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 1,
        verbose_mode: bool = False,
        flaky: bool = False,
        normalize: bool = False,
    ):
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.normalize = normalize

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize a piece of text before comparison.

        Raw ``==`` comparison is brittle for real LLM outputs: two strings
        that are semantically identical can differ in case, in how whitespace
        is folded (consecutive spaces / newlines / tabs), or in Unicode
        compatibility (full-width vs half-width forms, ligatures…). When
        ``normalize=True`` this collapses exactly those differences so an
        otherwise-correct output is not flagged as a false failure:

        * ``unicodedata.normalize("NFKC", ...)`` — canonical + compatibility
          decomposition, turning full-width → half-width and decomposing
          compatibility characters;
        * ``str.casefold()`` — locale-independent, more aggressive than
          ``str.lower()`` (handles e.g. ``"ß"`` → ``"ss"``);
        * collapse runs of whitespace (including newlines/tabs) to a single
          space and strip edges.

        ``re`` is imported at module level; this helper just uses it over the
        normalized string. The transformation is idempotent, so calling it
        twice is a no-op.
        """
        text = unicodedata.normalize("NFKC", text)
        text = text.casefold()
        return re.sub(r"\s+", " ", text).strip()

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
            expected = test_case.expected_output.strip()
            actual = test_case.actual_output.strip()

            if self.normalize:
                expected = self._normalize_text(expected)
                actual = self._normalize_text(actual)

            if expected == actual:
                self.score = 1.0
                self.reason = (
                    "The actual and expected outputs are exact matches."
                )
            else:
                self.score = 0.0
                self.reason = "The actual and expected outputs are different."

            self.success = self.is_successful()

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Score: {self.score:.2f}",
                        f"Reason: {self.reason}",
                        (f"Normalization: {'on' if self.normalize else 'off'}"),
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
        return "Exact Match"
