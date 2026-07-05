from typing import List, Optional, Dict, Any

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


class JuryEvalMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        metric_fn=None,
        metric_name: str = "JuryEval",
        threshold: float = 0.5,
        verbose_mode: bool = False,
        **kwargs,
    ):
        try:
            import juryeval  # noqa: F401
        except ImportError:
            raise ImportError(
                "juryeval is required for JuryEvalMetric. "
                "Install it with: pip install juryeval"
            )
        self.metric_fn = metric_fn
        self._name = metric_name
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.kwargs = kwargs

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        from juryeval import PairwiseJudge, PointwiseJudge

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
            if isinstance(self.metric_fn, PairwiseJudge):
                self.score = self._run_pairwise(test_case)
            elif isinstance(self.metric_fn, PointwiseJudge):
                self.score = self._run_pointwise(test_case)
            elif callable(self.metric_fn):
                result = self.metric_fn(
                    predictions=[test_case.actual_output],
                    references=[test_case.expected_output] if test_case.expected_output else None,
                    **self.kwargs,
                )
                if isinstance(result, dict):
                    self.score = float(list(result.values())[0])
                else:
                    self.score = float(result)
            else:
                raise ValueError(
                    "metric_fn must be a callable, PairwiseJudge, or PointwiseJudge instance"
                )

            self.reason = f"{self._name} score: {self.score:.4f}"
            self.success = self.score >= self.threshold

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[f"Score: {self.score:.4f}", f"Reason: {self.reason}"],
                )

            return self.score

    def _run_pairwise(self, test_case: LLMTestCase) -> float:
        result = self.metric_fn.compare(
            answer_a=test_case.actual_output,
            answer_b=test_case.expected_output or "",
            question=test_case.input,
        )
        return result.get("score", 0.0)

    def _run_pointwise(self, test_case: LLMTestCase) -> float:
        result = self.metric_fn.score(
            output=test_case.actual_output,
            question=test_case.input,
            reference=test_case.expected_output,
        )
        return result.get("score", 0.0)

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

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except Exception:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return self._name


class PairwiseJudgeMetric(JuryEvalMetric):
    def __init__(
        self,
        model: str = "gpt-4",
        threshold: float = 0.5,
        verbose_mode: bool = False,
        **judge_kwargs,
    ):
        from juryeval import PairwiseJudge

        judge = PairwiseJudge(model=model, **judge_kwargs)
        super().__init__(
            metric_fn=judge,
            metric_name="Pairwise Judge",
            threshold=threshold,
            verbose_mode=verbose_mode,
        )


class PointwiseJudgeMetric(JuryEvalMetric):
    def __init__(
        self,
        model: str = "gpt-4",
        threshold: float = 0.5,
        verbose_mode: bool = False,
        **judge_kwargs,
    ):
        from juryeval import PointwiseJudge

        judge = PointwiseJudge(model=model, **judge_kwargs)
        super().__init__(
            metric_fn=judge,
            metric_name="Pointwise Judge",
            threshold=threshold,
            verbose_mode=verbose_mode,
        )
