from typing import List, Optional, Union

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    prepare_measure,
    construct_verbose_logs,
    initialize_model,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.community.fluency_coherence.schema import (
    FluencyJudgment,
)


class FluencyCoherenceMetric(BaseMetric):
    """How readable and logically ordered is the response?

    An LLM judge rates ``actual_output`` (in the context of ``input``)
    from 1 (broken or incoherent) to 5 (polished and well ordered);
    the metric score is ``(rating - 1) / 4``. Anything expressible as a
    custom ``GEval`` criterion can also cover this — use this metric
    for a zero-setup readability gate.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
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
        prepare_measure(self, test_case)
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
                judgment = self._judge(test_case)
                self._fill(judgment)
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            prompt = self._build_prompt(test_case)
            judgment = await a_generate_with_schema_and_extract(
                metric=self,
                prompt=prompt,
                schema_cls=FluencyJudgment,
                extract_schema=lambda result: result,
                extract_json=lambda data: FluencyJudgment(**data),
            )
            self._fill(judgment)
            return self.score

    def _judge(self, test_case: LLMTestCase) -> FluencyJudgment:
        prompt = self._build_prompt(test_case)
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=FluencyJudgment,
            extract_schema=lambda result: result,
            extract_json=lambda data: FluencyJudgment(**data),
        )

    def _fill(self, judgment: FluencyJudgment):
        rating = min(5.0, max(1.0, float(judgment.score)))
        score = (rating - 1.0) / 4.0
        self.score = 0 if self.strict_mode and score < self.threshold else score
        self.success = self.is_successful()
        self.reason = (
            f"Rated {rating:.1f}/5. {judgment.reason}"
            if self.include_reason
            else None
        )
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Rating: {rating:.1f}/5",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    @staticmethod
    def _build_prompt(test_case: LLMTestCase) -> str:
        return (
            "You rate the readability and logical flow of an answer. "
            "Reply with JSON only: "
            '{"score": <1-5>, "reason": "<one sentence>"}. '
            "5 is polished and well ordered, 3 is understandable but "
            "awkward, 1 is broken or incoherent.\n"
            f"Question: {test_case.input}\n"
            f"Answer: {test_case.actual_output}"
        )

    @property
    def __name__(self):
        return "Fluency Coherence"
