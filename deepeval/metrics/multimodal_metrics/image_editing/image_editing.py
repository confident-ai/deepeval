import asyncio
from typing import Optional, List, Tuple, Union, Type
import math
import textwrap

from deepeval.metrics import BaseMetric
from deepeval.test_case import SingleTurnParams, LLMTestCase, MLLMImage
from deepeval.metrics.jev_eval.utils import extract_images
from deepeval.utils import (
    get_or_create_event_loop,
    convert_to_multi_modal_array,
)
from deepeval.metrics.utils import (
    SystemOneEvalSpec,
    initialize_image_system_one_model,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    system_one_reads,
    construct_verbose_logs,
    prepare_measure,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.config.eval_mode import EvalModeName
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.multimodal_metrics.image_editing.schema import ReasonScore
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.templates import make_template_class


ImageEditingTemplate = make_template_class("ImageEditingMetric")


class ImageEditingMetric(BaseMetric):

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: Optional[float] = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[ImageEditingTemplate] = ImageEditingTemplate,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
    ):
        # Under `system_one` the images go to a System One model that reads
        # them (Drex Max) and the reason is built from its answers.
        self.eval_mode, self.system_one_model = (
            initialize_image_system_one_model(system_one_model, eval_mode)
        )
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = True
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(
            self,
            test_case,
            input_image_count=1,
            actual_output_image_count=1,
        )
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
                input = convert_to_multi_modal_array(test_case.input)
                actual_output = convert_to_multi_modal_array(
                    test_case.actual_output
                )
                input_texts, input_images = self.separate_images_from_text(
                    input
                )
                _, output_images = self.separate_images_from_text(actual_output)

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
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Semantic Consistency Scores:\n{self.SC_scores}",
                        f"Semantic Consistency Reasoning:\n{self.SC_reasoning}",
                        f"Perceptual Quality Scores:\n{self.PQ_scores}",
                        f"Perceptual Quality Reasoning:\n{self.PQ_reasoning}",
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
        prepare_measure(
            self,
            test_case,
            input_image_count=1,
            actual_output_image_count=1,
        )
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score
            input = convert_to_multi_modal_array(test_case.input)
            actual_output = convert_to_multi_modal_array(
                test_case.actual_output
            )
            input_texts, input_images = self.separate_images_from_text(input)
            _, output_images = self.separate_images_from_text(actual_output)
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
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Semantic Consistency Scores:\n{self.SC_scores}",
                    f"Semantic Consistency Reasoning:\n{self.SC_reasoning}",
                    f"Perceptual Quality Scores:\n{self.PQ_scores}",
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
        images.extend([image_input, actual_image_output])
        prompt = [
            self._get_prompt(
                "generate_semantic_consistency_evaluation_results",
                text_prompt=text_prompt,
            )
        ]
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=f"{prompt} {images}",
            schema_cls=ReasonScore,
            extract_schema=lambda s: (s.score, s.reasoning),
            extract_json=lambda data: (data["score"], data["reasoning"]),
        )

    def _evaluate_semantic_consistency(
        self,
        text_prompt: str,
        image_input: MLLMImage,
        actual_image_output: MLLMImage,
    ) -> Tuple[List[int], str]:
        images: List[MLLMImage] = []
        images.extend([image_input, actual_image_output])
        prompt = [
            self._get_prompt(
                "generate_semantic_consistency_evaluation_results",
                text_prompt=text_prompt,
            )
        ]
        return generate_with_schema_and_extract(
            metric=self,
            prompt=f"{prompt} {images}",
            schema_cls=ReasonScore,
            extract_schema=lambda s: (s.score, s.reasoning),
            extract_json=lambda data: (data["score"], data["reasoning"]),
        )

    async def _a_evaluate_perceptual_quality(
        self, actual_image_output: MLLMImage
    ) -> Tuple[List[int], str]:
        images: List[MLLMImage] = [actual_image_output]
        prompt = [
            self._get_prompt(
                "generate_perceptual_quality_evaluation_results",
            )
        ]
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=f"{prompt} {images}",
            schema_cls=ReasonScore,
            extract_schema=lambda s: (s.score, s.reasoning),
            extract_json=lambda data: (data["score"], data["reasoning"]),
        )

    def _evaluate_perceptual_quality(
        self, actual_image_output: MLLMImage
    ) -> Tuple[List[int], str]:
        images: List[MLLMImage] = [actual_image_output]
        prompt = [
            self._get_prompt(
                "generate_perceptual_quality_evaluation_results",
            )
        ]
        return generate_with_schema_and_extract(
            metric=self,
            prompt=f"{prompt} {images}",
            schema_cls=ReasonScore,
            extract_schema=lambda s: (s.score, s.reasoning),
            extract_json=lambda data: (data["score"], data["reasoning"]),
        )

    def _calculate_score(self) -> float:
        min_SC_score = min(self.SC_scores)
        min_PQ_score = min(self.PQ_scores)
        return math.sqrt(min_SC_score * min_PQ_score) / 10

    def _generate_reason(
        self,
    ) -> str:
        return textwrap.dedent(
            f"""
            The overall score is {self.score:.2f} because the lowest score from semantic consistency was {min(self.SC_scores)} 
            and the lowest score from perceptual quality was {min(self.PQ_scores)}. These scores were combined to reflect the 
            overall effectiveness and quality of the AI-generated image(s).
            Reason for Semantic Consistency score: {self.SC_reasoning}
            Reason for Perceptual Quality score: {self.PQ_reasoning}
        """
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the edited image in `actual_output` judged against the original and the instruction in `input`, for overediting, naturalness and artifacts, in one request to a System One model
        that reads images (Drex Max)."""
        if not system_one_reads(self, test_case):
            return None
        _, input_images = extract_images(test_case.input)
        _, output_images = extract_images(test_case.actual_output)
        if (len(input_images), len(output_images)) != (1, 1):
            raise ValueError(
                f"{self.__name__} judges test cases with 1 input "
                f"image and 1 actual_output image; found "
                f"{len(input_images)} and {len(output_images)}."
            )
        return SystemOneEvalSpec(
            evaluation_params=self._required_params,
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    @property
    def __name__(self):
        return "Image Editing"
