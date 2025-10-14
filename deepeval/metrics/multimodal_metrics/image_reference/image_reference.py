import asyncio
from typing import Optional, List, Tuple, Union

from deepeval.metrics import BaseMultimodalMetric
from deepeval.test_case import MLLMTestCaseParams, MLLMTestCase, MLLMImage
from deepeval.metrics.multimodal_metrics.image_reference.template import (
    ImageReferenceTemplate,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_mllm_test_case_params,
    initialize_multimodal_model,
)
from deepeval.models import DeepEvalBaseMLLM
from deepeval.metrics.multimodal_metrics.image_reference.schema import (
    ReasonScore,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.utils import get_or_create_event_loop


class ImageReferenceMetric(BaseMultimodalMetric):

    _required_params: List[MLLMTestCaseParams] = [
        MLLMTestCaseParams.INPUT,
        MLLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseMLLM]] = None,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_context_size: Optional[int] = None,
    ):
        self.model, self.using_native_model = initialize_multimodal_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.max_context_size = max_context_size

    def measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_mllm_test_case_params(
            test_case, self._required_params, None, None, self
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
                actual_output = test_case.actual_output
                self.contexts_above = []
                self.contexts_below = []
                self.scores = []
                self.reasons = []
                for image_index in self.get_image_indices(actual_output):
                    context_above, context_below = self.get_image_context(
                        image_index, actual_output
                    )
                    image = actual_output[image_index]
                    score, reason = self.evaluate_image_reference(
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
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_mllm_test_case_params(
            test_case, self._required_params, None, None, self
        )
        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            actual_output = test_case.actual_output
            self.contexts_above = []
            self.contexts_below = []
            self.scores = []
            self.reasons = []

            tasks = []
            image_indices = self.get_image_indices(actual_output)
            for image_index in image_indices:
                context_above, context_below = self.get_image_context(
                    image_index, actual_output
                )
                image = actual_output[image_index]
                tasks.append(
                    self.a_evaluate_image_reference(
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

    def evaluate_image_reference(
        self,
        image: MLLMImage,
        context_above: Optional[str] = None,
        context_below: Optional[str] = None,
    ) -> Tuple[float, str]:
        instructions = ImageReferenceTemplate.evaluate_image_reference(
            context_above, context_below
        )
        prompt = [instructions] + [image]
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=ReasonScore)
            self.evaluation_cost += cost
            return res.score, res.reasoning
        else:
            try:
                res: ReasonScore = self.model.generate(
                    prompt, schema=ReasonScore
                )
                return res.score, res.reasoning
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reasoning"]

    async def a_evaluate_image_reference(
        self,
        image: MLLMImage,
        context_above: Optional[str] = None,
        context_below: Optional[str] = None,
    ) -> Tuple[float, str]:
        instructions = ImageReferenceTemplate.evaluate_image_reference(
            context_above, context_below
        )
        prompt = [instructions] + [image]
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=ReasonScore)
            self.evaluation_cost += cost
            return res.score, res.reasoning
        else:
            try:
                res: ReasonScore = await self.model.a_generate(
                    prompt, schema=ReasonScore
                )
                return res.score, res.reasoning
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reasoning"]

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

    def calculate_score(self, scores: List[float]):
        return sum(scores) / len(scores)

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
        return "Image Reference"
