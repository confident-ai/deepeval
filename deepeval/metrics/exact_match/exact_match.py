import string
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
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        normalize_unicode: bool = False,
    ):
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        # Optional normalisations, all off by default so existing behaviour is
        # unchanged. They mirror the options on HuggingFace `evaluate`'s
        # exact_match, and address a common source of false negatives: an output
        # that is correct but differs from the reference only in case, unicode
        # form (e.g. NFC vs NFD "café"), or punctuation.
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.normalize_unicode = normalize_unicode

    def _prepare(self, text: str) -> str:
        """Apply the opted-in normalisations before comparison.

        With no options set this returns the text unchanged, so the default
        exact-match semantics are preserved byte for byte.
        """
        if self.normalize_unicode:
            # NFKC folds NFD/NFC differences and compatibility characters
            # (ligatures, full-width forms) that are visually identical.
            text = unicodedata.normalize("NFKC", text)
        if self.ignore_case:
            text = text.casefold()
        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = " ".join(text.split())
        return text

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
            expected = self._prepare(test_case.expected_output.strip())
            actual = self._prepare(test_case.actual_output.strip())

            if expected == actual:
                self.score = self.precision = self.recall = self.f1 = 1.0
                self.reason = (
                    "The actual and expected outputs are exact matches."
                )
            else:
                self.score = self.precision = self.recall = self.f1 = 0.0
                self.reason = "The actual and expected outputs are different."

            self.success = self.is_successful()

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
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

    @property
    def __name__(self):
        return "Exact Match"
        return "Exact Match"
