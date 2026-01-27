import asyncio
from typing import Optional, List, Tuple, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase, MLLMImage
from deepeval.metrics.multimodal_metrics.image_helpfulness.template import (
    ImageHelpfulnessTemplate,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.multimodal_metrics.image_helpfulness.schema import (
    ReasonScore,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import (
    get_or_create_event_loop,
    convert_to_multi_modal_array,
)


class ImageHelpfulnessMetric(BaseMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_context_size: Optional[int] = None,
    ):
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.max_context_size = max_context_size

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
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
                        _log_metric_to_confident=_log_metric_to_confident,
                    )
                )
            else:
                actual_output = convert_to_multi_modal_array(
                    test_case.actual_output
                )
                self.contexts_above = []
                self.contexts_below = []
                self.scores = []
                self.reasons = []
                image_indices = self.get_image_indices(actual_output)
                if not image_indices:
                    raise ValueError(
                        f"The test case must have atleast one image in the `actual_output` to calculate {self.__name__} score"
                    )
                for image_index in image_indices:
                    context_above, context_below = self.get_image_context(
                        image_index, actual_output
                    )
                    image = actual_output[image_index]
                    score, reason = self.evaluate_image_helpfulness(
                        image, context_above, context_below
                    )
                    score = score / 10
                    self.contexts_above.append(context_above)
                    self.contexts_below.append(context_below)
                    self.scores.append(score)
                    self.reasons.append(reason)

                self.score = self.calculate_score(self.scores)
                self.score = (
                    0
                    if self.strict_mode and self.score < self.threshold
                    else self.score
                )
                self.reason = "\n".join(
                    f"Reason for image {i}: {reason}"
                    for i, reason in enumerate(self.reasons)
                )
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        (
                            (
                                (
                                    f"Context Above Image: {self.contexts_above[0][:20]}...\n"
                                    if self.contexts_above
                                    and self.contexts_above[0]
                                    else ""
                                )
                                + (
                                    f"Context Below Image: {self.contexts_below[0][:20]}...\n"
                                    if self.contexts_below
                                    and self.contexts_below[0]
                                    else ""
                                )
                                + f"Score: {self.scores[0]}\nReason: {self.reasons[0]}\n"
                            )
                            if len(self.scores) == 1
                            else (
                                (
                                    f"Context Above Image {i + 1}: {self.contexts_above[i][:20]}...\n"
                                    if self.contexts_above
                                    and self.contexts_above[i]
                                    else ""
                                )
                                + (
                                    f"Context Below Image {i + 1}: {self.contexts_below[i][:20]}...\n"
                                    if self.contexts_below
                                    and self.contexts_below[i]
                                    else ""
                                )
                                + f"Image {i + 1} Score: {self.scores[i]}\nImage {i + 1} Reason: {self.reasons[i]}\n"
                            )
                        )
                        for i in range(len(self.scores))
                    ]
                    + (
                        [f"Score (Average): {self.score}"]
                        if len(self.scores) > 1
                        else []
                    ),
                )
                return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
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
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            actual_output = convert_to_multi_modal_array(
                test_case.actual_output
            )
            self.contexts_above = []
            self.contexts_below = []
            self.scores = []
            self.reasons = []

            tasks = []
            image_indices = self.get_image_indices(actual_output)
            if not image_indices:
                raise ValueError(
                    f"The test case must have atleast one image in the `actual_output` to calculate {self.__name__} score"
                )
            for image_index in image_indices:
                context_above, context_below = self.get_image_context(
                    image_index, actual_output
                )
                image = actual_output[image_index]
                tasks.append(
                    self.a_evaluate_image_helpfulness(
                        image, context_above, context_below
                    )
                )
                # Append contexts immediately
                self.contexts_above.append(context_above)
                self.contexts_below.append(context_below)
            results = await asyncio.gather(*tasks)

            for score, reason in results:
                score = score / 10
                self.scores.append(score)
                self.reasons.append(reason)

            self.score = self.calculate_score(self.scores)
            self.score = (
                0
                if self.strict_mode and self.score < self.threshold
                else self.score
            )
            self.reason = "\n".join(
                f"Reason for image {i}: {reason}"
                for i, reason in enumerate(self.reasons)
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    (
                        (
                            (
                                f"Context Above Image: {self.contexts_above[0][:20]}...\n"
                                if self.contexts_above
                                and self.contexts_above[0]
                                else ""
                            )
                            + (
                                f"Context Below Image: {self.contexts_below[0][:20]}...\n"
                                if self.contexts_below
                                and self.contexts_below[0]
                                else ""
                            )
                            + f"Score: {self.scores[0]}\nReason: {self.reasons[0]}\n"
                        )
                        if len(self.scores) == 1
                        else (
                            (
                                f"Context Above Image {i + 1}: {self.contexts_above[i][:20]}...\n"
                                if self.contexts_above
                                and self.contexts_above[i]
                                else ""
                            )
                            + (
                                f"Context Below Image {i + 1}: {self.contexts_below[i][:20]}...\n"
                                if self.contexts_below
                                and self.contexts_below[i]
                                else ""
                            )
                            + f"Image {i + 1} Score: {self.scores[i]}\nImage {i + 1} Reason: {self.reasons[i]}\n"
                        )
                    )
                    for i in range(len(self.scores))
                ]
                + (
                    [f"Score (Average): {self.score}"]
                    if len(self.scores) > 1
                    else []
                ),
            )
            return self.score

    def evaluate_image_helpfulness(
        self,
        image: MLLMImage,
        context_above: Optional[str] = None,
        context_below: Optional[str] = None,
    ) -> Tuple[float, str]:
        instructions = ImageHelpfulnessTemplate.evaluate_image_helpfulness(
            context_above, context_below
        )
        prompt = f"{instructions} \nImages: {image}"
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ReasonScore,
            extract_schema=lambda s: (s.score, s.reasoning),
            extract_json=lambda data: (data["score"], data["reasoning"]),
        )

    async def a_evaluate_image_helpfulness(
        self,
        image: MLLMImage,
        context_above: Optional[str] = None,
        context_below: Optional[str] = None,
    ) -> Tuple[float, str]:
        instructions = ImageHelpfulnessTemplate.evaluate_image_helpfulness(
            context_above, context_below
        )
        prompt = f"{instructions} \nImages: {image}"
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ReasonScore,
            extract_schema=lambda s: (s.score, s.reasoning),
            extract_json=lambda data: (data["score"], data["reasoning"]),
        )

    def get_image_context(
        self, image_index: int, actual_output: List[Union[str, MLLMImage]]
    ) -> Tuple[str, str]:
        context_above = None
        context_below = None

        # Find context_above (last characters until max_context_size)
        for i in range(image_index - 1, -1, -1):  # Iterate backward
            if isinstance(actual_output[i], str):
                context_above = actual_output[i]
                if self.max_context_size:
                    context_above = context_above[-self.max_context_size :]
                break

        # Find context_below (first characters until max_context_size)
        for i in range(image_index + 1, len(actual_output)):  # Iterate forward
            if isinstance(actual_output[i], str):
                context_below = actual_output[i]
                if self.max_context_size:
                    context_below = context_below[: self.max_context_size]
                break

        return context_above, context_below

    def get_image_indices(
        self, actual_output: List[Union[str, MLLMImage]]
    ) -> List[int]:
        return [
            index
            for index, element in enumerate(actual_output)
            if isinstance(element, MLLMImage)
        ]

    def calculate_score(self, scores: List[float]) -> float:
        return sum(scores) / len(scores)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Image Helpfulness"
