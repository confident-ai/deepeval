import asyncio
from typing import Optional, List, Tuple, Union, Type

from deepeval.metrics import BaseMetric
from deepeval.test_case import SingleTurnParams, LLMTestCase, MLLMImage
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
from deepeval.metrics.multimodal_metrics.image_helpfulness.schema import (
    ReasonScore,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.jev_eval.utils import extract_images
from deepeval.utils import (
    get_or_create_event_loop,
    convert_to_multi_modal_array,
)
from deepeval.templates import make_template_class


ImageHelpfulnessTemplate = make_template_class("ImageHelpfulnessMetric")


class ImageHelpfulnessMetric(BaseMetric):

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
        max_context_size: Optional[int] = None,
        flaky: bool = False,
        evaluation_template: Type[
            ImageHelpfulnessTemplate
        ] = ImageHelpfulnessTemplate,
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
        self.max_context_size = max_context_size
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
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
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score
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
        instructions = self._get_prompt(
            "evaluate_image_helpfulness",
            context_above=context_above,
            context_below=context_below,
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
        instructions = self._get_prompt(
            "evaluate_image_helpfulness",
            context_above=context_above,
            context_below=context_below,
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

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: how much every image in `actual_output` helps a reader with the text around it, in one request to a System One
        model that reads images (Drex Max)."""
        if not system_one_reads(self, test_case):
            return None
        _, images = extract_images(test_case.actual_output)
        if not images:
            raise ValueError(
                f"The test case must have atleast one image in the `actual_output` to calculate {self.__name__} score"
            )
        return SystemOneEvalSpec(
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            questions=parse_questions(
                self._get_prompt(
                    "_experimental_system_one_questions",
                    image_count=len(images),
                )
            ),
        )

    @property
    def __name__(self):
        return "Image Helpfulness"
