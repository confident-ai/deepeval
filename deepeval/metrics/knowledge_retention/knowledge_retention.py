from typing import Optional, Union, Dict, List
from pydantic import BaseModel, Field

from deepeval.test_case import ConversationalTestCase
from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.utils import (
    validate_conversational_test_case,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.knowledge_retention.template import (
    KnowledgeRetentionTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.knowledge_retention.schema import *


class KnowledgeRetentionMetric(BaseConversationalMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.strict_mode = strict_mode

    def measure(self, test_case: ConversationalTestCase):
        validate_conversational_test_case(test_case, self)
        with metric_progress_indicator(self):
            self.knowledges: List[Knowledge] = self._generate_knowledges(
                test_case
            )
            self.verdicts: List[KnowledgeRetentionVerdict] = (
                self._generate_verdicts(test_case)
            )
            knowledge_retention_score = self._calculate_score()
            self.reason = self._generate_reason(knowledge_retention_score)

            self.success = knowledge_retention_score >= self.threshold
            self.score = knowledge_retention_score
            return self.score

    def _generate_reason(self, score: float) -> str:
        if self.include_reason is False:
            return None

        attritions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                attritions.append(verdict.reason)

        prompt: dict = KnowledgeRetentionTemplate.generate_reason(
            attritions=attritions,
            score=format(score, ".2f"),
        )
        if self.using_native_model:
            res, _ = self.model.generate(prompt)
        else:
            res = self.model.generate(prompt)
        return res

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

    def _generate_verdicts(
        self, test_case: ConversationalTestCase
    ) -> List[KnowledgeRetentionVerdict]:
        verdicts: List[KnowledgeRetentionVerdict] = []
        for index, message in enumerate(test_case.messages):
            previous_knowledge = self.knowledges[index].data

            prompt = KnowledgeRetentionTemplate.generate_verdict(
                llm_message=message.actual_output,
                previous_knowledge=previous_knowledge,
            )
            if self.using_native_model:
                res, _ = self.model.generate(prompt)
            else:
                res = self.model.generate(prompt)
            data = trimAndLoadJson(res, self)
            verdict = KnowledgeRetentionVerdict(index=index, **data)
            verdicts.append(verdict)

        return verdicts

    def _generate_knowledges(
        self, test_case: ConversationalTestCase
    ) -> List[Knowledge]:
        knowledges: List[Knowledge] = []
        for index, message in enumerate(test_case.messages):
            previous_knowledge = knowledges[-1].data if knowledges else {}
            llm_message = (
                test_case.messages[index - 1].actual_output if index > 0 else ""
            )

            prompt = KnowledgeRetentionTemplate.extract_data(
                llm_message=llm_message,
                user_message=message.input,
                previous_knowledge=previous_knowledge,
            )

            if self.using_native_model:
                res, _ = self.model.generate(prompt)
            else:
                res = self.model.generate(prompt)
            data = trimAndLoadJson(res, self)
            knowledge = Knowledge(data=data)
            knowledges.append(knowledge)

        return knowledges

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
