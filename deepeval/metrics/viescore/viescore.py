import asyncio
from typing import Optional, List, Tuple, Union
import math
import textwrap

from deepeval.metrics import BaseMultimodalMetric
from deepeval.test_case import MLLMTestCaseParams, MLLMTestCase, MLLMImage
from deepeval.metrics.viescore.template import VIEScoreTemplate
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_mllm_test_case_params,
    initialize_multimodal_model,
)
from deepeval.models import DeepEvalBaseMLLM
from deepeval.metrics.viescore.schema import ReasonScore
from deepeval.metrics.viescore.task import VIEScoreTask
from deepeval.metrics.indicator import metric_progress_indicator

required_params: List[MLLMTestCaseParams] = [
    MLLMTestCaseParams.INPUT,
    MLLMTestCaseParams.ACTUAL_OUTPUT,
]


class VIEScore(BaseMultimodalMetric):
    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseMLLM]] = None,
        task: VIEScoreTask = VIEScoreTask.TEXT_TO_IMAGE_GENERATION,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_VIEScore_task_name: bool = True,
    ):
        self.model, self.using_native_model = initialize_multimodal_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.task = task
        self._include_VIEScore_task_name = _include_VIEScore_task_name

    def measure(
        self, test_case: MLLMTestCase, _show_indicator: bool = True
    ) -> float:
        if self.task == VIEScoreTask.TEXT_TO_IMAGE_GENERATION:
            check_mllm_test_case_params(test_case, required_params, 0, 1, self)
        elif self.task == VIEScoreTask.TEXT_TO_IMAGE_EDITING:
            check_mllm_test_case_params(test_case, required_params, 1, 1, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                input_texts, input_images = self.separate_images_from_text(
                    test_case.input
                )
                _, output_images = self.separate_images_from_text(
                    test_case.actual_output
                )

                self.SC_scores, self.SC_reasoning = (
                    self._evaluate_semantic_consistency(
                        "\n".join(input_texts),
                        None if len(input_images) == 0 else input_images[0],
                        output_images[0],
                    )
                )
                self.PQ_scores, self.PQ_reasoning = (
                    self._evaluate_perceptual_quality(output_images[0])
                )
                self.score = self._calculate_score()
                self.score = (
                    0
                    if self.strict_mode and self.score < self.threshold
                    else self.score
                )
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Semantic Consistency Scores:\n{self.SC_scores}",
                        f"Semantic Consistency Reasoning:\n{self.SC_reasoning}",
                        f"Perceptual Quality Scores:\n{self.SC_scores}",
                        f"Perceptual Quality Reasoning:\n{self.PQ_reasoning}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score

    async def a_measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
    ) -> float:
        if self.task == VIEScoreTask.TEXT_TO_IMAGE_GENERATION:
            check_mllm_test_case_params(test_case, required_params, 0, 1, self)
        elif self.task == VIEScoreTask.TEXT_TO_IMAGE_EDITING:
            check_mllm_test_case_params(test_case, required_params, 1, 1, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            input_texts, input_images = self.separate_images_from_text(
                test_case.input
            )
            _, output_images = self.separate_images_from_text(
                test_case.actual_output
            )
            (self.SC_scores, self.SC_reasoning), (
                self.PQ_scores,
                self.PQ_reasoning,
            ) = await asyncio.gather(
                self._a_evaluate_semantic_consistency(
                    "\n".join(input_texts),
                    None if len(input_images) == 0 else input_images[0],
                    output_images[0],
                ),
                self._a_evaluate_perceptual_quality(output_images[0]),
            )
            self.score = self._calculate_score()
            self.score = (
                0
                if self.strict_mode and self.score < self.threshold
                else self.score
            )
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Semantic Consistency Scores:\n{self.SC_scores}",
                    f"Semantic Consistency Reasoning:\n{self.SC_reasoning}",
                    f"Perceptual Quality Scores:\n{self.SC_scores}",
                    f"Perceptual Quality Reasoning:\n{self.PQ_reasoning}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def separate_images_from_text(
        self, multimodal_list: List[Union[MLLMImage, str]]
    ) -> Tuple[List[str], List[MLLMImage]]:
        images: List[MLLMImage] = []
        texts: List[str] = []
        for item in multimodal_list:
            if isinstance(item, MLLMImage):
                images.append(item)
            elif isinstance(item, str):
                texts.append(item)
        return texts, images

    async def _a_evaluate_semantic_consistency(
        self,
        text_prompt: str,
        image_input: MLLMImage,
        actual_image_output: MLLMImage,
    ) -> Tuple[List[int], str]:
        images: List[MLLMImage] = []
        if self.task == VIEScoreTask.TEXT_TO_IMAGE_GENERATION:
            images.append(image_input)
        elif self.task == VIEScoreTask.TEXT_TO_IMAGE_EDITING:
            images.extend([image_input, actual_image_output])
        prompt = [
            VIEScoreTemplate.generate_semantic_consistency_evaluation_results(
                text_prompt=text_prompt, task=self.task
            )
        ]
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt + images)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["score"], data["reasoning"]
        else:
            try:
                res: ReasonScore = await self.model.a_generate(
                    prompt + images, schema=ReasonScore
                )
                return res.score, res.reasoning
            except TypeError:
                res = await self.model.a_generate(
                    prompt + images, input_text=prompt
                )
                data = trimAndLoadJson(res, self)
                return data["score"], data["reasoning"]

    def _evaluate_semantic_consistency(
        self,
        text_prompt: str,
        image_input: MLLMImage,
        actual_image_output: MLLMImage,
    ) -> Tuple[List[int], str]:
        images: List[MLLMImage] = []
        if self.task == VIEScoreTask.TEXT_TO_IMAGE_GENERATION:
            images.append(image_input)
        elif self.task == VIEScoreTask.TEXT_TO_IMAGE_EDITING:
            images.extend([image_input, actual_image_output])
        prompt = [
            VIEScoreTemplate.generate_semantic_consistency_evaluation_results(
                text_prompt=text_prompt, task=self.task
            )
        ]
        if self.using_native_model:
            res, cost = self.model.generate(prompt + images)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["score"], data["reasoning"]
        else:
            try:
                res: ReasonScore = self.model.generate(
                    prompt + images, schema=ReasonScore
                )
                return res.score, res.reasoning
            except TypeError:
                res = self.model.generate(prompt + images)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reasoning"]

    async def _a_evaluate_perceptual_quality(
        self, actual_image_output: MLLMImage
    ) -> Tuple[List[int], str]:
        images: List[MLLMImage] = [actual_image_output]
        prompt = [
            VIEScoreTemplate.generate_perceptual_quality_evaluation_results()
        ]
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt + images)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["score"], data["reasoning"]
        else:
            try:
                res: ReasonScore = await self.model.a_generate(
                    prompt + images, schema=ReasonScore
                )
                return res.score, res.reasoning
            except TypeError:
                res = await self.model.a_generate(prompt + images)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reasoning"]

    def _evaluate_perceptual_quality(
        self, actual_image_output: MLLMImage
    ) -> Tuple[List[int], str]:
        images: List[MLLMImage] = [actual_image_output]
        prompt = [
            VIEScoreTemplate.generate_perceptual_quality_evaluation_results()
        ]
        if self.using_native_model:
            res, cost = self.model.generate(prompt + images)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["score"], data["reasoning"]
        else:
            try:
                res: ReasonScore = self.model.generate(
                    prompt + images, schema=ReasonScore
                )
                return res.score, res.reasoning
            except TypeError:
                res = self.model.generate(prompt + images)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reasoning"]

    def _calculate_score(self) -> List[str]:
        min_SC_score = min(self.SC_scores)
        min_PQ_score = min(self.PQ_scores)
        return math.sqrt(min_SC_score * min_PQ_score) / 10

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score >= self.threshold
            except:
                self.success = False
        return self.success

    def _generate_reason(
        self,
    ) -> Tuple[List[float], str]:
        return textwrap.dedent(
            f"""
            The overall score is {self.score:.2f} because the lowest score from semantic consistency was {min(self.SC_scores)} 
            and the lowest score from perceptual quality was {min(self.PQ_scores)}. These scores were combined to reflect the 
            overall effectiveness and quality of the AI-generated image(s).
            Reason for Semantic Consistency score: {self.SC_reasoning}
            Reason for Perceptual Quality score: {self.PQ_reasoning}
        """
        )

    @property
    def __name__(self):
        if self._include_VIEScore_task_name:
            return f"{self.task.value} (VIEScore)"
        else:
            return "VIEScore"
