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
from deepeval.test_case import (
    LLMTestCase,
    RetrievedContextData,
    SingleTurnParams,
)
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.community.context_sufficiency.schema import (
    SufficiencyJudgment,
)


class ContextSufficiencyMetric(BaseMetric):
    """Does the retrieved context contain enough to answer the question?

    An LLM judge rates whether ``retrieval_context`` holds the facts
    needed to answer ``input``. Unlike ``ContextualRecallMetric``, no
    ``expected_output`` is needed, so it works for unlabelled
    production traces. Scores ``1.0`` when fully answerable down to
    ``0.0`` when nothing relevant was retrieved.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
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
                self._fill(test_case, judgment)
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
                schema_cls=SufficiencyJudgment,
                extract_schema=lambda result: result,
                extract_json=lambda data: SufficiencyJudgment(**data),
            )
            self._fill(test_case, judgment)
            return self.score

    def _judge(self, test_case: LLMTestCase) -> SufficiencyJudgment:
        prompt = self._build_prompt(test_case)
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=SufficiencyJudgment,
            extract_schema=lambda result: result,
            extract_json=lambda data: SufficiencyJudgment(**data),
        )

    def _fill(self, test_case: LLMTestCase, judgment: SufficiencyJudgment):
        score = min(1.0, max(0.0, float(judgment.score)))
        self.score = 0 if self.strict_mode and score < self.threshold else score
        self.success = self.is_successful()
        self.reason = judgment.reason if self.include_reason else None
        passages = self._passages(test_case)
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Input: {test_case.input}",
                f"Passages: {len(passages)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    @staticmethod
    def _passages(test_case: LLMTestCase) -> List[str]:
        return [
            (
                passage.context
                if isinstance(passage, RetrievedContextData)
                else passage
            )
            for passage in (test_case.retrieval_context or [])
        ]

    @staticmethod
    def _build_prompt(test_case: LLMTestCase) -> str:
        passages = ContextSufficiencyMetric._passages(test_case)
        numbered = "\n".join(
            f"[{i + 1}] {passage}" for i, passage in enumerate(passages)
        )
        return (
            "You rate whether retrieved passages contain enough "
            "information to answer the question. Reply with JSON only: "
            '{"score": <0.0-1.0>, "reason": "<one sentence>"}. '
            "Score 1.0 when fully answerable, 0.5 when partially, "
            "0.0 when nothing relevant was retrieved.\n"
            f"Question: {test_case.input}\n"
            f"Passages:\n{numbered}"
        )

    @property
    def __name__(self):
        return "Context Sufficiency"
