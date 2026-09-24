from typing import Optional, List, Union, Type

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.base_metric import Verdict, YES_NO
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
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ToolCall,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.argument_correctness.schema import (
    ArgumentCorrectnessVerdict,
    Verdicts,
    ArgumentCorrectnessScoreReason,
)
from deepeval.templates import make_template_class


ArgumentCorrectnessTemplate = make_template_class("ArgumentCorrectnessMetric")


class ArgumentCorrectnessMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.TOOLS_CALLED,
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
            ArgumentCorrectnessTemplate
        ] = ArgumentCorrectnessTemplate,
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
                if len(test_case.tools_called) == 0:
                    self.verdicts = []
                    self.score = 1.0
                    self.reason = "No tool calls provided"
                elif run_system_one_eval(self, test_case):
                    return self.score
                else:
                    self.verdicts: List[ArgumentCorrectnessVerdict] = (
                        self._generate_verdicts(
                            test_case.input,
                            test_case.tools_called,
                            test_case.multimodal,
                        )
                    )
                    self.score = self._calculate_score()
                    self.reason = self._generate_reason(
                        test_case.input, test_case.multimodal
                    )
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
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
            if len(test_case.tools_called) == 0:
                self.verdicts = []
                self.score = 1.0
                self.reason = "No tool calls provided"
            elif await a_run_system_one_eval(self, test_case):
                return self.score
            else:
                self.verdicts: List[ArgumentCorrectnessVerdict] = (
                    await self._a_generate_verdicts(
                        test_case.input,
                        test_case.tools_called,
                        test_case.multimodal,
                    )
                )
                self.score = self._calculate_score()
                self.reason = await self._a_generate_reason(
                    test_case.input, test_case.multimodal
                )
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, input: str, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        incorrect_tool_calls_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.NO:
                incorrect_tool_calls_reasons.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            incorrect_tool_calls_reasons=incorrect_tool_calls_reasons,
            input=input,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ArgumentCorrectnessScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, input: str, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        incorrect_tool_calls_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.NO:
                incorrect_tool_calls_reasons.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            incorrect_tool_calls_reasons=incorrect_tool_calls_reasons,
            input=input,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ArgumentCorrectnessScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(
        self, input: str, tools_called: List[ToolCall], multimodal: bool
    ) -> List[ArgumentCorrectnessVerdict]:
        prompt = self._get_prompt(
            "generate_verdicts",
            input=input,
            stringified_tools_called=tools_called,
            multimodal=multimodal,
        )

        return await a_generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=ArgumentCorrectnessVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(input, tools_called),
        )

    def _generate_verdicts(
        self, input: str, tools_called: List[ToolCall], multimodal: bool
    ) -> List[ArgumentCorrectnessVerdict]:
        prompt = self._get_prompt(
            "generate_verdicts",
            input=input,
            stringified_tools_called=tools_called,
            multimodal=multimodal,
        )

        return generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=ArgumentCorrectnessVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(input, tools_called),
        )

    def _experimental_system_one_spec(
        self, input: str, tools_called: List[ToolCall]
    ) -> SystemOneVerdictSpec:
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=[repr(tool_call) for tool_call in tools_called],
            item_key="tool_call",
            state={"input": input},
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `input` and the structured `tools_called`; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=self._required_params,
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    def _calculate_score(self):
        return score_qag_verdicts(
            self,
            self.verdicts,
            passing=(Verdict.YES,),
        )

    @property
    def __name__(self):
        return "Argument Correctness"
