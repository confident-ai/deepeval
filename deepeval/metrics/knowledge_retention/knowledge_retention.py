from typing import Optional, Union, List

from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.knowledge_retention.template import (
    KnowledgeRetentionTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.knowledge_retention.schema import (
    Knowledge,
    KnowledgeRetentionVerdict,
    Reason,
)
from deepeval.utils import get_or_create_event_loop, prettify_list


class KnowledgeRetentionMetric(BaseConversationalMetric):
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
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self, test_case: ConversationalTestCase, _show_indicator: bool = True
    ):
        check_conversational_test_case_params(
            test_case, self._required_params, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.knowledges: List[Knowledge] = self._generate_knowledges(
                    test_case.turns
                )
                self.verdicts: List[KnowledgeRetentionVerdict] = (
                    self._generate_verdicts(test_case.turns)
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Knowledges:\n{prettify_list(self.knowledges)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
    ) -> float:
        check_conversational_test_case_params(
            test_case, self._required_params, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            self.knowledges: List[Knowledge] = (
                await self._a_generate_knowledges(test_case.turns)
            )
            self.verdicts: List[KnowledgeRetentionVerdict] = (
                await self._a_generate_verdicts(test_case.turns)
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Knowledges:\n{prettify_list(self.knowledges)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        attritions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                attritions.append(verdict.reason)

        prompt: dict = KnowledgeRetentionTemplate.generate_reason(
            attritions=attritions,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        attritions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                attritions.append(verdict.reason)

        prompt: dict = KnowledgeRetentionTemplate.generate_reason(
            attritions=attritions,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdicts(
        self, llm_test_cases: List[LLMTestCase]
    ) -> List[KnowledgeRetentionVerdict]:
        verdicts: List[KnowledgeRetentionVerdict] = []
        for index, llm_test_case in enumerate(llm_test_cases):
            previous_knowledge = self.knowledges[index].data

            prompt = KnowledgeRetentionTemplate.generate_verdict(
                llm_message=llm_test_case.actual_output,
                previous_knowledge=previous_knowledge,
            )
            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                verdict = KnowledgeRetentionVerdict(index=index, **data)
            else:
                try:
                    verdict: KnowledgeRetentionVerdict = (
                        await self.model.a_generate(
                            prompt, schema=KnowledgeRetentionVerdict
                        )
                    )
                    verdict.index = index
                except TypeError:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    verdict = KnowledgeRetentionVerdict(index=index, **data)
            verdicts.append(verdict)

        return verdicts

    def _generate_verdicts(
        self, llm_test_cases: List[LLMTestCase]
    ) -> List[KnowledgeRetentionVerdict]:
        verdicts: List[KnowledgeRetentionVerdict] = []
        for index, llm_test_case in enumerate(llm_test_cases):
            previous_knowledge = self.knowledges[index].data

            prompt = KnowledgeRetentionTemplate.generate_verdict(
                llm_message=llm_test_case.actual_output,
                previous_knowledge=previous_knowledge,
            )
            if self.using_native_model:
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                verdict = KnowledgeRetentionVerdict(index=index, **data)
            else:
                try:
                    verdict: KnowledgeRetentionVerdict = self.model.generate(
                        prompt, schema=KnowledgeRetentionVerdict
                    )
                    verdict.index = index
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    verdict = KnowledgeRetentionVerdict(index=index, **data)
            verdicts.append(verdict)

        return verdicts

    async def _a_generate_knowledges(
        self, llm_test_cases: List[LLMTestCase]
    ) -> List[Knowledge]:
        knowledges: List[Knowledge] = []
        for index, llm_test_case in enumerate(llm_test_cases):
            previous_knowledge = knowledges[-1].data if knowledges else {}
            llm_turn = (
                llm_test_cases[index - 1].actual_output if index > 0 else ""
            )

            prompt = KnowledgeRetentionTemplate.extract_data(
                llm_message=llm_turn,
                user_message=llm_test_case.input,
                previous_knowledge=previous_knowledge,
            )

            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                knowledge: Knowledge = Knowledge(data=data)
            else:
                try:
                    knowledge: Knowledge = await self.model.a_generate(
                        prompt, schema=Knowledge
                    )
                except TypeError:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    knowledge = Knowledge(data=data)
            knowledges.append(knowledge)

        return knowledges

    def _generate_knowledges(
        self, llm_test_cases: List[LLMTestCase]
    ) -> List[Knowledge]:
        knowledges: List[Knowledge] = []
        for index, llm_test_case in enumerate(llm_test_cases):
            previous_knowledge = knowledges[-1].data if knowledges else {}
            llm_turn = (
                llm_test_cases[index - 1].actual_output if index > 0 else ""
            )

            prompt = KnowledgeRetentionTemplate.extract_data(
                llm_message=llm_turn,
                user_message=llm_test_case.input,
                previous_knowledge=previous_knowledge,
            )

            if self.using_native_model:
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                knowledge: Knowledge = Knowledge(data=data)
            else:
                try:
                    knowledge: Knowledge = self.model.generate(
                        prompt, schema=Knowledge
                    )
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    knowledge = Knowledge(data=data)
            knowledges.append(knowledge)

        return knowledges

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        retention_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                retention_count += 1

        score = retention_count / number_of_verdicts

        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Knowledge Retention"
