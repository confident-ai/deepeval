from typing import Optional, List
from pydantic import BaseModel, Field
import json

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel
from deepeval.templates import AnswerRelevancyTemplate


class AnswerRelvancyVerdict(BaseModel):
    verdict: str
    mock_question: str = Field(default=None)


class AnswerRelevancyMetric(BaseMetric):
    def __init__(
        self,
        minimum_score: float = 0.5,
        model: Optional[str] = None,
    ):
        self.minimum_score = minimum_score
        self.chat_model = GPTModel(model_name=model)
        self.n = 5

    def measure(self, test_case: LLMTestCase) -> float:
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.retrieval_context is None
        ):
            raise ValueError(
                "Input, actual output, or retrieval context cannot be None"
            )

        self.mock_questions: List[str] = self._generate_mock_questions(
            test_case.actual_output, "\n".join(test_case.retrieval_context)
        )
        self.meta_questions: List[str] = self._generate_meta_questions(
            test_case.input
        )
        self.verdicts: List[AnswerRelvancyVerdict] = self._generate_verdicts()

        answer_relevancy_score = self._generate_score()

        # generate mock questions based on the answer and context, these don't have to be close ended
        # generate meta questions that are close ended
        # generate close ended answers to meta questions and a reason to it. The reason should focus on the potential answer to the question being relevant or not to the actual question
        # single out 'no' and 'idk' answers and generate a final reason eg while there are ambigity from idk, it could work. however, .... (talk about no)
        # final score is num(yes or idk)/total
        self.reason = self._generate_reason(
            test_case.input, answer_relevancy_score
        )
        self.success = answer_relevancy_score >= self.minimum_score
        self.score = answer_relevancy_score

        return self.score

    def _generate_score(self):
        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() != "no":
                relevant_count += 1

        return relevant_count / len(self.verdicts)

    def _generate_reason(self, original_question: str, score: float) -> str:
        # for each ireelevancy and ambiguity, suggest a point being made that might not be relevant to the original question
        irrelevant_questions = []
        ambiguous_questions = []

        for verdict in self.verdicts:
            if verdict.verdict.lower() == "no":
                irrelevant_questions.append(verdict.mock_question)
            elif verdict.verdict.lower == "idk":
                ambiguous_questions.append(verdict.mock_question)

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_questions=irrelevant_questions,
            ambiguous_questions=ambiguous_questions,
            original_question=original_question,
            score=format(score, ".2f"),
        )

        res = self.chat_model(prompt)
        return res.content

    def _generate_verdicts(self) -> List[AnswerRelvancyVerdict]:
        prompt = AnswerRelevancyTemplate.generate_verdicts(
            meta_questions=self.meta_questions
        )
        res = self.chat_model(prompt)
        json_output = trimToJson(res.content)
        data = json.loads(json_output)
        verdicts = [AnswerRelvancyVerdict(**item) for item in data["verdicts"]]
        if len(verdicts) != len(self.meta_questions):
            raise ValueError("Number of verdicts generated does not equal.")

        for i in range(len(verdicts)):
            verdicts[i].mock_question = self.mock_questions[i]

        return verdicts

    def _generate_meta_questions(self, original_question: str) -> List[str]:
        # TODO: create
        meta_questions = []
        for mock_question in self.mock_questions:
            meta_question = AnswerRelevancyTemplate.generate_meta_question(
                original_question=original_question,
                mock_questions=mock_question,
            )
            meta_questions.append(meta_question)

        return meta_questions

    def _generate_mock_questions(
        self, answer: str, retrieval_context: str
    ) -> List[str]:
        # TODO: create
        prompt = AnswerRelevancyTemplate.generate_mock_questions(
            answer=answer, retrieval_context=retrieval_context
        )
        res = self.chat_model(prompt)
        return res.content

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Answer Relevancy"
