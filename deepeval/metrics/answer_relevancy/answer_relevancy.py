from typing import Optional, List, Union, Type

from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.base_metric import Verdict, YES_NO_BORDERLINE
from deepeval.metrics.utils import (
    generate_qag_verdicts,
    a_generate_qag_verdicts,
    SystemOneEvalSpec,
    SystemOneVerdictSpec,
    initialize_system_one_model,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    score_qag_verdicts,
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.test_case import LLMTestCase, SingleTurnParams, MLLMImage
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.answer_relevancy.schema import (
    Statements,
    AnswerRelevancyVerdict,
    Verdicts,
    AnswerRelevancyScoreReason,
)
from deepeval.templates import make_template_class


AnswerRelevancyTemplate = make_template_class("AnswerRelevancyMetric")


class AnswerRelevancyMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[
            AnswerRelevancyTemplate
        ] = AnswerRelevancyTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template

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
            self.model,
            test_case.multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
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
                if run_system_one_eval(self, test_case):
                    return self.score

                input = test_case.input
                actual_output = test_case.actual_output

                self.statements: List[str] = self._generate_statements(
                    actual_output, test_case.multimodal
                )
                self.verdicts: List[AnswerRelevancyVerdict] = (
                    self._generate_verdicts(input, test_case.multimodal)
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(input, test_case.multimodal)
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{prettify_list(self.statements)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
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

        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score

            input = test_case.input
            actual_output = test_case.actual_output

            self.statements: List[str] = await self._a_generate_statements(
                actual_output, test_case.multimodal
            )
            self.verdicts: List[AnswerRelevancyVerdict] = (
                await self._a_generate_verdicts(input, test_case.multimodal)
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(
                input, test_case.multimodal
            )
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Statements:\n{prettify_list(self.statements)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, input: str, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.NO:
                irrelevant_statements.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            multimodal=multimodal,
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AnswerRelevancyScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, input: str, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.NO:
                irrelevant_statements.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            multimodal=multimodal,
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AnswerRelevancyScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(
        self, input: str, multimodal: bool
    ) -> List[AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = self._get_prompt(
            "generate_verdicts",
            multimodal=multimodal,
            input=input,
            statements=self.statements,
        )

        return await a_generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=AnswerRelevancyVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO_BORDERLINE,
            system_one=self._experimental_system_one_spec(input),
        )

    def _generate_verdicts(
        self, input: str, multimodal: bool
    ) -> List[AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = self._get_prompt(
            "generate_verdicts",
            multimodal=multimodal,
            input=input,
            statements=self.statements,
        )

        return generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=AnswerRelevancyVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO_BORDERLINE,
            system_one=self._experimental_system_one_spec(input),
        )

    def _experimental_system_one_spec(self, input: str) -> SystemOneVerdictSpec:
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=self.statements,
            item_key="statement",
            state={"input": input},
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `input` and `actual_output`; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=self._required_params,
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    def _generate_statements(
        self,
        actual_output: str,
        multimodal: bool,
    ) -> List[str]:
        prompt = self._get_prompt(
            "generate_statements",
            multimodal=multimodal,
            actual_output=actual_output,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Statements,
            extract_schema=lambda s: s.statements
            + [ele for ele in actual_output if isinstance(ele, MLLMImage)],
            extract_json=lambda d: d["statements"]
            + [ele for ele in actual_output if isinstance(ele, MLLMImage)],
        )

    async def _a_generate_statements(
        self,
        actual_output: str,
        multimodal: bool,
    ) -> List[str]:
        prompt = self._get_prompt(
            "generate_statements",
            multimodal=multimodal,
            actual_output=actual_output,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Statements,
            extract_schema=lambda s: s.statements
            + [ele for ele in actual_output if isinstance(ele, MLLMImage)],
            extract_json=lambda d: d["statements"]
            + [ele for ele in actual_output if isinstance(ele, MLLMImage)],
        )

    def _calculate_score(self):
        return score_qag_verdicts(
            self,
            self.verdicts,
            passing=(Verdict.YES, Verdict.BORDERLINE),
        )

    @property
    def __name__(self):
        return "Answer Relevancy"
