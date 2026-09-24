from typing import List, Optional, Union, Type

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
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
from deepeval.metrics.pii_leakage.schema import (
    PIILeakageVerdict,
    Verdicts,
    ExtractedPII,
    PIILeakageScoreReason,
)
from deepeval.templates import make_template_class


PIILeakageTemplate = make_template_class("PIILeakageMetric")


class PIILeakageMetric(BaseMetric):
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
        evaluation_template: Type[PIILeakageTemplate] = PIILeakageTemplate,
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

                self.extracted_pii: List[str] = self._extract_pii(
                    test_case.actual_output, multimodal=test_case.multimodal
                )
                self.verdicts: List[PIILeakageVerdict] = (
                    self._generate_verdicts()
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Extracted PII:\n{prettify_list(self.extracted_pii)}",
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

            self.extracted_pii: List[str] = await self._a_extract_pii(
                test_case.actual_output, multimodal=test_case.multimodal
            )
            self.verdicts: List[PIILeakageVerdict] = (
                await self._a_generate_verdicts()
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Extracted PII:\n{prettify_list(self.extracted_pii)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        privacy_violations = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.YES:
                privacy_violations.append(verdict.reason)

        prompt: dict = self._get_prompt(
            "generate_reason",
            privacy_violations=privacy_violations,
            score=format(self.score, ".2f"),
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=PIILeakageScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        privacy_violations = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.YES:
                privacy_violations.append(verdict.reason)

        prompt: dict = self._get_prompt(
            "generate_reason",
            privacy_violations=privacy_violations,
            score=format(self.score, ".2f"),
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=PIILeakageScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(self) -> List[PIILeakageVerdict]:
        if len(self.extracted_pii) == 0:
            return []

        prompt = self._get_prompt(
            "generate_verdicts",
            extracted_pii=self.extracted_pii,
        )
        return await a_generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=PIILeakageVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(),
        )

    def _generate_verdicts(self) -> List[PIILeakageVerdict]:
        if len(self.extracted_pii) == 0:
            return []

        prompt = self._get_prompt(
            "generate_verdicts",
            extracted_pii=self.extracted_pii,
        )
        return generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=PIILeakageVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(),
        )

    def _experimental_system_one_spec(self) -> SystemOneVerdictSpec:
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=self.extracted_pii,
            item_key="statement",
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `actual_output`; higher probabilities mean less leakage, matching
        the metric's 1-is-a-pass direction. See EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    async def _a_extract_pii(
        self, actual_output: str, *, multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "extract_pii",
            actual_output=actual_output,
            multimodal=multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ExtractedPII,
            extract_schema=lambda s: s.extracted_pii,
            extract_json=lambda data: data["extracted_pii"],
        )

    def _extract_pii(
        self, actual_output: str, *, multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "extract_pii",
            actual_output=actual_output,
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ExtractedPII,
            extract_schema=lambda s: s.extracted_pii,
            extract_json=lambda data: data["extracted_pii"],
        )

    def _calculate_score(self) -> float:
        return score_qag_verdicts(
            self,
            self.verdicts,
            passing=(Verdict.NO,),
        )

    @property
    def __name__(self):
        return "PII Leakage"
