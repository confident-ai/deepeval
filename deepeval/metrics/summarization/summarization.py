import asyncio
from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.utils import trimAndLoadJson, get_or_create_event_loop
from deepeval.metrics.summarization.template import SummarizationTemplate
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from deepeval.progress_context import metrics_progress_context
from deepeval.telemetry import capture_metric_type


class SummarizationAlignmentVerdict(BaseModel):
    # yes, no, or idk
    verdict: str
    reason: str = Field(default=None)


class SummarizationCoverageVerdict(BaseModel):
    summary_verdict: str
    original_verdict: str
    question: str = Field(default=None)


class ScoreType(Enum):
    ALIGNMENT = "Alignment"
    COVERAGE = "Coverage"


class SummarizationMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        n: int = 5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        assessment_questions: Optional[List[str]] = None,
        include_reason: bool = True,
        run_async=True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

        if assessment_questions is not None and len(assessment_questions) == 0:
            self.assessment_questions = None
        else:
            self.assessment_questions = assessment_questions

        self.run_async = run_async
        self.include_reason = include_reason
        self.n = n
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase):
        if test_case.input is None or test_case.actual_output is None:
            raise ValueError("Input or actual output cannot be None")

        with metrics_progress_context(
            self.__name__, self.evaluation_model, self.strict_mode
        ):
            if self.run_async:
                loop = get_or_create_event_loop()
                self.truths, self.claims = loop.run_until_complete(
                    asyncio.gather(
                        self._a_generate_claims(test_case.input),
                        self._a_generate_claims(test_case.actual_output),
                    )
                )
                self.coverage_verdicts, self.alignment_verdicts = (
                    loop.run_until_complete(
                        asyncio.gather(
                            self._a_generate_coverage_verdicts(test_case),
                            self._a_generate_alignment_verdicts(),
                        )
                    )
                )
                alignment_score = self._generate_score(ScoreType.ALIGNMENT)
                coverage_score = self._generate_score(ScoreType.COVERAGE)

                self.score_breakdown = {
                    ScoreType.ALIGNMENT.value: alignment_score,
                    ScoreType.COVERAGE.value: coverage_score,
                }
                self.score = min(alignment_score, coverage_score)
                self.reason = loop.run_until_complete(self._a_generate_reason())

            else:
                self.truths: List[str] = self._generate_claims(test_case.input)
                self.claims: List[str] = self._generate_claims(
                    test_case.actual_output
                )
                self.coverage_verdicts: List[SummarizationCoverageVerdict] = (
                    self._generate_coverage_verdicts(test_case)
                )
                self.alignment_verdicts: List[SummarizationAlignmentVerdict] = (
                    self._generate_alignment_verdicts()
                )
                alignment_score = self._generate_score(ScoreType.ALIGNMENT)
                coverage_score = self._generate_score(ScoreType.COVERAGE)

                self.score_breakdown = {
                    ScoreType.ALIGNMENT.value: alignment_score,
                    ScoreType.COVERAGE.value: coverage_score,
                }
                self.score = min(alignment_score, coverage_score)
                self.reason = self._generate_reason()

            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        redundancies = []
        for verdict in self.alignment_verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            elif verdict.verdict.strip().lower() == "idk":
                redundancies.append(verdict.reason)

        questions = []
        if self.coverage_verdicts:
            for verdict in self.coverage_verdicts:
                if (
                    verdict.original_verdict.strip().lower() == "yes"
                    and verdict.summary_verdict.strip().lower() == "no"
                ):
                    questions.append(verdict.question)

        prompt: dict = SummarizationTemplate.generate_reason(
            contradictions=contradictions,
            redundancies=redundancies,
            questions=questions,
            score=format(self.score, ".2f"),
        )

        if len(questions) > 0:
            prompt += f"""Questions the original text can answer but not the summary:
{questions}

"""
        prompt += """Reason:"""

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        return res

    def _generate_reason(self) -> str:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_reason)

    def _generate_score(self, score_type: ScoreType) -> float:
        if score_type == ScoreType.ALIGNMENT:
            total = len(self.alignment_verdicts)
            if total == 0:
                return 0
            faithfulness_count = 0
            for verdict in self.alignment_verdicts:
                # Different from the faithfulness score, this
                # penalizes 'idk' (full of fluff) summaries
                if verdict.verdict.strip().lower() == "yes":
                    faithfulness_count += 1

            score = faithfulness_count / total

        else:
            if self.assessment_questions is None:
                return 1
            total = 0
            coverage_count = 0
            for verdict in self.coverage_verdicts:
                if verdict.original_verdict.strip().lower() == "yes":
                    total += 1
                    if verdict.summary_verdict.strip().lower() == "yes":
                        coverage_count += 1

            if total == 0:
                return 0

            score = coverage_count / total

        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_answers(self, text: str) -> List[str]:
        prompt = SummarizationTemplate.generate_answers(
            questions=self.assessment_questions, text=text
        )

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res)
        return data["answers"]

    def _generate_answers(self, text: str) -> List[str]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_answers(text))

    async def _a_generate_assessment_questions(self, text: str):
        prompt = SummarizationTemplate.generate_questions(text=text, n=self.n)
        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res)
        return data["questions"]

    def _generate_assessment_questions(self, text: str):
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self._a_generate_assessment_questions(text)
        )

    async def _a_generate_coverage_verdicts(
        self, test_case: LLMTestCase
    ) -> List[SummarizationCoverageVerdict]:
        if self.assessment_questions is None:
            if self.run_async:
                self.assessment_questions = (
                    await self._a_generate_assessment_questions(test_case.input)
                )
            else:
                self.assessment_questions = self._generate_assessment_questions(
                    test_case.input
                )

        if self.run_async:
            tasks = [
                self._a_generate_answers(test_case.input),
                self._a_generate_answers(test_case.actual_output),
            ]
            results = await asyncio.gather(*tasks)
            original_answers = results[0]
            summary_answers = results[1]
        else:
            original_answers = self._generate_answers(test_case.input)
            summary_answers = self._generate_answers(test_case.actual_output)

        if len(original_answers) != len(summary_answers):
            raise ValueError("Number of verdicts generated does not equal.")

        coverage_veridcts: List[SummarizationCoverageVerdict] = []
        for i in range(len(original_answers)):
            coverage_veridcts.append(
                SummarizationCoverageVerdict(
                    summary_verdict=summary_answers[i],
                    original_verdict=original_answers[i],
                    question=self.assessment_questions[i],
                )
            )

        return coverage_veridcts

    def _generate_coverage_verdicts(
        self, test_case: LLMTestCase
    ) -> List[SummarizationCoverageVerdict]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self._a_generate_coverage_verdicts(test_case)
        )

    async def _a_generate_alignment_verdicts(
        self,
    ) -> List[SummarizationAlignmentVerdict]:
        verdicts: List[SummarizationAlignmentVerdict] = []
        prompt = SummarizationTemplate.generate_alignment_verdicts(
            summary_claims=self.claims, orignal_text="\n\n".join(self.truths)
        )

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res)
        verdicts = [
            SummarizationAlignmentVerdict(**item) for item in data["verdicts"]
        ]
        return verdicts

    def _generate_alignment_verdicts(
        self,
    ) -> List[SummarizationAlignmentVerdict]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_alignment_verdicts())

    async def _a_generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template
        prompt = FaithfulnessTemplate.generate_claims(text=text)

        if self.run_async:
            res = await self.model.a_generate(prompt)
        else:
            res = self.model.generate(prompt)

        data = trimAndLoadJson(res)
        return data["claims"]

    def _generate_claims(self, text: str) -> List[str]:
        loop = get_or_create_event_loop()
        return loop.run_until_complete(self._a_generate_claims(text))

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Summarization"
