import re
from typing import List, Optional

from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams


class KeywordCoverageMetric(BaseMetric):
    """Deterministic required-keyword coverage with a banned-term gate.

    Scores the fraction of ``keywords`` (a required-phrase checklist) that appear
    in ``actual_output``. This gives **partial credit** — unlike
    ``ExactMatchMetric`` (whole-string equality) and ``PatternMatchMetric``
    (a single all-or-nothing regex), both of which are binary.

    If any ``forbidden`` phrase appears in the output, the metric **hard-fails**
    with a score of ``0.0`` regardless of keyword coverage. This makes it a single
    gate for the very common review requirement "must mention A, B and C, and must
    NOT mention X or Y" (rubric coverage plus banned terms, e.g. leaked internal
    codenames or competitor mentions).

    The metric does not use an LLM: it is pure string matching, deterministic,
    needs no API key, and costs nothing to run.

    Scoring:

        score = 0.0                                   if any forbidden term is present
        score = (# keywords present) / (# keywords)   otherwise

    With the default ``threshold`` of ``1.0``, every required keyword must be
    present (and no forbidden term) for the test case to pass.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        keywords: List[str],
        forbidden: Optional[List[str]] = None,
        ignore_case: bool = True,
        whole_word: bool = False,
        threshold: Optional[float] = 1.0,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.keywords = [k.strip() for k in (keywords or []) if k and k.strip()]
        if not self.keywords:
            raise ValueError(
                "KeywordCoverageMetric requires at least one non-empty keyword."
            )
        self.forbidden = [
            f.strip() for f in (forbidden or []) if f and f.strip()
        ]
        self.ignore_case = ignore_case
        self.whole_word = whole_word
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.flaky = flaky

    def _contains(self, text: str, term: str) -> bool:
        if self.whole_word:
            flags = re.IGNORECASE if self.ignore_case else 0
            return re.search(rf"\b{re.escape(term)}\b", text, flags) is not None
        if self.ignore_case:
            return term.lower() in text.lower()
        return term in text

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
            actual = test_case.actual_output.strip()

            self.present = [
                k for k in self.keywords if self._contains(actual, k)
            ]
            self.missing = [
                k for k in self.keywords if not self._contains(actual, k)
            ]
            self.forbidden_present = [
                f for f in self.forbidden if self._contains(actual, f)
            ]

            if self.forbidden_present:
                self.score = 0.0
                self.reason = "Forbidden term(s) present: " + ", ".join(
                    repr(f) for f in self.forbidden_present
                )
            else:
                self.score = len(self.present) / len(self.keywords)
                if self.missing:
                    self.reason = (
                        f"Covered {len(self.present)}/{len(self.keywords)} "
                        "required keywords; missing: "
                        + ", ".join(repr(k) for k in self.missing)
                    )
                else:
                    self.reason = (
                        f"All {len(self.keywords)} required keywords present"
                        + ("; no forbidden terms." if self.forbidden else ".")
                    )

            self.success = self.is_successful()

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Required keywords: {self.keywords}",
                        f"Present: {self.present}",
                        f"Missing: {self.missing}",
                        f"Forbidden present: {self.forbidden_present}",
                        f"Score: {self.score:.2f}\nReason: {self.reason}",
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
        return "Keyword Coverage"
