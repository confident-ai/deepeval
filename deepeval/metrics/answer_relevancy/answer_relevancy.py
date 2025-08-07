from typing import Optional, List, Type, Union
import statistics

from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
    prettify_nested_list,
    get_repeat,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.answer_relevancy.template import AnswerRelevancyTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.answer_relevancy.schema import *


class AnswerRelevancyMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            AnswerRelevancyTemplate
        ] = AnswerRelevancyTemplate,
        repeat: Optional[int] = None,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
        self.repeat = repeat or get_repeat()

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
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
            elif self.repeat == 1:
                self.statements: List[str] = self._generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[AnswerRelevancyVerdict] = (
                    self._generate_verdicts(test_case.input)
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{prettify_nested_list(self.statements)}",
                        f"Verdicts:\n{prettify_nested_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score
            else:
                statements, verdicts, scores = [], [], []
                for _ in range(self.repeat):
                    self.statements: List[str] = self._generate_statements(
                        test_case.actual_output
                    )
                    self.verdicts: List[AnswerRelevancyVerdict] = (
                        self._generate_verdicts(test_case.input)
                    )
                    self.score = self._calculate_score()
                    self.success = self.score >= self.threshold

                    # append to lists
                    scores.append(self.score)
                    statements.append(self.statements)
                    verdicts.append(self.verdicts)

                recent_statements = statements[-1]

                self.statements = statements
                self.verdicts = verdicts
                self.score = sum(scores) / len(scores)
                self.reason = self._generate_aggregate_reason(
                    test_case.input, recent_statements
                )
                self.success = self.score >= self.threshold
                self.standard_deviation = statistics.stdev(scores)
                self.verbose_logs = self._construct_repeat_verbose_logs(
                    scores=scores
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if self.repeat == 1:
                self.statements: List[str] = await self._a_generate_statements(
                    test_case.actual_output
                )
                self.verdicts: List[AnswerRelevancyVerdict] = (
                    await self._a_generate_verdicts(test_case.input)
                )
                self.score = self._calculate_score()
                self.reason = await self._a_generate_reason(test_case.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Statements:\n{prettify_list(self.statements)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            else:
                statements, verdicts, scores = [], [], []
                for _ in range(self.repeat):
                    self.statements: List[str] = (
                        await self._a_generate_statements(
                            test_case.actual_output
                        )
                    )
                    self.verdicts: List[AnswerRelevancyVerdict] = (
                        await self._a_generate_verdicts(test_case.input)
                    )
                    self.score = self._calculate_score()
                    self.success = self.score >= self.threshold

                    # append to lists
                    scores.append(self.score)
                    statements.append(self.statements)
                    verdicts.append(self.verdicts)

                recent_statements = statements[-1]

                self.statements = statements
                self.verdicts = verdicts
                self.score = sum(scores) / len(scores)
                self.reason = await self._a_generate_aggregate_reason(
                    test_case.input, recent_statements
                )
                self.success = self.score >= self.threshold
                self.standard_deviation = statistics.stdev(scores)
                self.verbose_logs = self._construct_repeat_verbose_logs(
                    scores=scores
                )

            return self.score

    async def _a_generate_aggregate_reason(
        self, input: str, statements: List[str]
    ) -> str:
        if self.include_reason is False:
            return None

        prompt = self.evaluation_template.generate_aggregate_reason(
            statements=statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=AnswerRelevancyScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: AnswerRelevancyScoreReason = await self.model.a_generate(
                    prompt=prompt, schema=AnswerRelevancyScoreReason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_aggregate_reason(
        self, input: str, statements: List[str]
    ) -> str:
        if self.include_reason is False:
            return None

        prompt = self.evaluation_template.generate_aggregate_reason(
            statements=statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=AnswerRelevancyScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: AnswerRelevancyScoreReason = self.model.generate(
                    prompt=prompt, schema=AnswerRelevancyScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=AnswerRelevancyScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: AnswerRelevancyScoreReason = await self.model.a_generate(
                    prompt=prompt, schema=AnswerRelevancyScoreReason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, input: str) -> str:
        if self.include_reason is False:
            return None

        irrelevant_statements = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                irrelevant_statements.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            irrelevant_statements=irrelevant_statements,
            input=input,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=AnswerRelevancyScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: AnswerRelevancyScoreReason = self.model.generate(
                    prompt=prompt, schema=AnswerRelevancyScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(
        self, input: str
    ) -> List[AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            statements=self.statements,
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            return [item for item in res.verdicts]
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                return [item for item in res.verdicts]
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return [
                    AnswerRelevancyVerdict(**item) for item in data["verdicts"]
                ]

    def _generate_verdicts(self, input: str) -> List[AnswerRelevancyVerdict]:
        if len(self.statements) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            statements=self.statements,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            return [item for item in res.verdicts]
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                return [item for item in res.verdicts]
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return [
                    AnswerRelevancyVerdict(**item) for item in data["verdicts"]
                ]

    async def _a_generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = self.evaluation_template.generate_statements(
            actual_output=actual_output,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Statements)
            self.evaluation_cost += cost
            return res.statements
        else:
            try:
                res: Statements = await self.model.a_generate(
                    prompt, schema=Statements
                )
                return res.statements
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["statements"]

    def _generate_statements(
        self,
        actual_output: str,
    ) -> List[str]:
        prompt = self.evaluation_template.generate_statements(
            actual_output=actual_output,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Statements)
            self.evaluation_cost += cost
            return res.statements
        else:
            try:
                res: Statements = self.model.generate(prompt, schema=Statements)
                return res.statements
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["statements"]

    def _construct_repeat_verbose_logs(
        self,
        scores: List[float],
    ):
        repetition_details = []
        for i in range(len(scores)):
            ordinal = (
                "1st"
                if i == 0
                else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1}th"
            )
            repetition_details.append(f"{'-'*20} {ordinal} Repetition {'-'*20}")
            repetition_details.append(
                f"Statements:\n{prettify_list(self.statements[i])}"
            )
            repetition_details.append(
                f"Verdicts:\n{prettify_list(self.verdicts[i])}"
            )
            repetition_details.append(f"Score: {scores[i]:.2f}")

        return construct_verbose_logs(
            self,
            steps=[
                f"{'-'*20} Summary {'-'*20}",
                f"Repetitions: {self.repeat}\nAverage Score: {self.score:.2f}\nStandard Deviation: {self.standard_deviation:.2f}\nReason: {self.reason}\n",
                *repetition_details,
            ],
            repeat=True,
        )

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"