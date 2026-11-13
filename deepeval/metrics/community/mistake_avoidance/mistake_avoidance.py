from typing import List, Optional, Union

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.metrics.base_metric import Verdict, YES_NO
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    prepare_measure,
    construct_verbose_logs,
    initialize_model,
    generate_qag_verdicts,
    a_generate_qag_verdicts,
    score_qag_verdicts,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.community.mistake_avoidance.template import (
    MistakeAvoidanceTemplate,
)
from deepeval.metrics.community.mistake_avoidance.schema import (
    MistakeAvoidanceVerdict,
    Verdicts,
)


class MistakeAvoidanceMetric(BaseMetric):
    """Mistake avoidance: does the answer repeat a mistake that is already known?

    Systems that learn from failure (a RAG pipeline with a memory of wrong
    answers, an agent with a lessons-learned store, a regression suite built
    from past incidents) keep a record of what went wrong before. This metric
    checks whether ``actual_output`` makes any of those mistakes again.

    Pass the known mistakes as ``context``, one mistake per item. The judge
    returns one verdict per mistake, and the score is the fraction of known
    mistakes the answer avoided:

        score = mistakes avoided / known mistakes

    A score of ``1.0`` means no known mistake was repeated. The metric does
    not judge correctness in any other way: an answer can avoid every known
    mistake and still be wrong in a new way, so pair it with a correctness
    or faithfulness metric.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.CONTEXT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case, multimodal=False)
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                self.verdicts: List[MistakeAvoidanceVerdict] = (
                    self._generate_verdicts(test_case)
                )
                self._finish()

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case, multimodal=False)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.verdicts: List[MistakeAvoidanceVerdict] = (
                await self._a_generate_verdicts(test_case)
            )
            self._finish()
            return self.score

    def _finish(self) -> None:
        self.score = self._calculate_score()
        self.reason = self._generate_reason()
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Verdicts:\n{prettify_list(self.verdicts)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _build_prompt(self, test_case: LLMTestCase) -> str:
        return MistakeAvoidanceTemplate.generate_verdicts(
            input=test_case.input,
            actual_output=test_case.actual_output,
            mistakes=test_case.context,
        )

    async def _a_generate_verdicts(
        self, test_case: LLMTestCase
    ) -> List[MistakeAvoidanceVerdict]:
        return await a_generate_qag_verdicts(
            metric=self,
            prompt=self._build_prompt(test_case),
            verdict_cls=MistakeAvoidanceVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
        )

    def _generate_verdicts(
        self, test_case: LLMTestCase
    ) -> List[MistakeAvoidanceVerdict]:
        return generate_qag_verdicts(
            metric=self,
            prompt=self._build_prompt(test_case),
            verdict_cls=MistakeAvoidanceVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
        )

    def _generate_reason(self) -> Optional[str]:
        """Built from the judge's own per-mistake reasons, so the reason
        costs no extra LLM call."""
        if self.include_reason is False:
            return None

        total = len(self.verdicts)
        if total == 0:
            return "There were no known mistakes to check the answer against."

        repeated = [
            verdict.reason
            for verdict in self.verdicts
            if verdict.verdict != Verdict.YES
        ]
        avoided = total - len(repeated)
        summary = (
            f"The answer avoided {avoided} of {total} known "
            f"mistake{'s' if total != 1 else ''}."
        )
        if not repeated:
            return summary
        return f"{summary} Repeated: " + " ".join(repeated)

    def _calculate_score(self) -> float:
        return score_qag_verdicts(
            self,
            self.verdicts,
            passing=(Verdict.YES,),
        )

    @property
    def __name__(self):
        return "Mistake Avoidance"
