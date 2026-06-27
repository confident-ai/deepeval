import asyncio
import inspect
from typing import Awaitable, Callable, Dict, List, Optional, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.metrics.adversarial_robustness.schema import (
    Perturbation,
    Perturbations,
    RobustnessVerdict,
    Verdicts,
    AdversarialRobustnessScoreReason,
)

# A callable that maps a (perturbed) input prompt to the system-under-test's
# output. May be synchronous or asynchronous.
ModelCallbackType = Callable[[str], Union[str, Awaitable[str]]]

VALID_PERTURBATION_TYPES = ("semantic", "orthographic")


class AdversarialRobustnessMetric(BaseMetric):
    """Measures how robust an LLM is to adversarial, meaning-preserving
    perturbations of its input.

    Inspired by the RoMA (Robustness Measurement and Assessment) framework
    (https://arxiv.org/abs/2504.17723), this is a black-box metric: it only
    needs to call the system under test, never inspect its weights.

    For each test case the metric:
      1. Uses the evaluation model to generate meaning-preserving adversarial
         perturbations of ``test_case.input`` (``semantic`` synonym/rephrasing
         swaps and ``orthographic`` typo-style character noise).
      2. Runs ``model_callback`` on every perturbation to obtain the system's
         response to each.
      3. Uses the evaluation model to judge whether each perturbed response is
         semantically consistent with the reference ``test_case.actual_output``.

    The score is the fraction of perturbations the system stayed consistent on
    (``1.0`` = perfectly robust). Higher is better, so a test case passes when
    ``score >= threshold``.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        model_callback: ModelCallbackType,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        num_perturbations: int = 5,
        perturbation_types: Optional[List[str]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        if not callable(model_callback):
            raise ValueError(
                "`model_callback` must be a callable that maps a perturbed "
                "input string to the system under test's output string. It is "
                "what `AdversarialRobustnessMetric` probes with adversarial "
                "inputs."
            )
        if num_perturbations < 1:
            raise ValueError("`num_perturbations` must be at least 1.")

        perturbation_types = perturbation_types or list(
            VALID_PERTURBATION_TYPES
        )
        invalid = [
            t for t in perturbation_types if t not in VALID_PERTURBATION_TYPES
        ]
        if invalid or not perturbation_types:
            raise ValueError(
                "`perturbation_types` must be a non-empty subset of "
                f"{list(VALID_PERTURBATION_TYPES)}, got {perturbation_types}."
            )

        self.model_callback = model_callback
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.num_perturbations = num_perturbations
        self.perturbation_types = list(perturbation_types)
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

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
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
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
                self.perturbations: List[Perturbation] = (
                    self._generate_perturbations(test_case.input)
                )
                self.responses: List[str] = [
                    self._invoke_model(p.perturbed_input)
                    for p in self.perturbations
                ]
                self.verdicts: List[RobustnessVerdict] = (
                    self._generate_verdicts(
                        test_case.input, test_case.actual_output
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Perturbations:\n{prettify_list(self.perturbations)}",
                        f"Responses:\n{prettify_list(self.responses)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
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
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.perturbations: List[Perturbation] = (
                await self._a_generate_perturbations(test_case.input)
            )
            self.responses: List[str] = await asyncio.gather(
                *[
                    self._a_invoke_model(p.perturbed_input)
                    for p in self.perturbations
                ]
            )
            self.verdicts: List[RobustnessVerdict] = (
                await self._a_generate_verdicts(
                    test_case.input, test_case.actual_output
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Perturbations:\n{prettify_list(self.perturbations)}",
                    f"Responses:\n{prettify_list(self.responses)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def _invoke_model(self, perturbed_input: str) -> str:
        result = self.model_callback(perturbed_input)
        if inspect.isawaitable(result):
            loop = get_or_create_event_loop()
            result = loop.run_until_complete(result)
        return result

    async def _a_invoke_model(self, perturbed_input: str) -> str:
        if inspect.iscoroutinefunction(self.model_callback):
            return await self.model_callback(perturbed_input)
        result = self.model_callback(perturbed_input)
        if inspect.isawaitable(result):
            return await result
        return result

    def _generate_perturbations(self, input: str) -> List[Perturbation]:
        prompt = self._get_prompt(
            "generate_perturbations",
            input=input,
            num_perturbations=self.num_perturbations,
            perturbation_types=self.perturbation_types,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Perturbations,
            extract_schema=lambda r: list(r.perturbations),
            extract_json=lambda data: [
                Perturbation(**item) for item in data["perturbations"]
            ],
        )

    async def _a_generate_perturbations(self, input: str) -> List[Perturbation]:
        prompt = self._get_prompt(
            "generate_perturbations",
            input=input,
            num_perturbations=self.num_perturbations,
            perturbation_types=self.perturbation_types,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Perturbations,
            extract_schema=lambda r: list(r.perturbations),
            extract_json=lambda data: [
                Perturbation(**item) for item in data["perturbations"]
            ],
        )

    def _perturbation_pairs(self) -> List[Dict[str, str]]:
        return [
            {
                "perturbed_input": perturbation.perturbed_input,
                "perturbation_type": perturbation.perturbation_type,
                "perturbed_output": response,
            }
            for perturbation, response in zip(
                self.perturbations, self.responses
            )
        ]

    def _generate_verdicts(
        self, input: str, actual_output: str
    ) -> List[RobustnessVerdict]:
        if len(self.perturbations) == 0:
            return []
        prompt = self._get_prompt(
            "generate_verdicts",
            input=input,
            original_output=actual_output,
            perturbations=self._perturbation_pairs(),
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda r: list(r.verdicts),
            extract_json=lambda data: [
                RobustnessVerdict(**item) for item in data["verdicts"]
            ],
        )

    async def _a_generate_verdicts(
        self, input: str, actual_output: str
    ) -> List[RobustnessVerdict]:
        if len(self.perturbations) == 0:
            return []
        prompt = self._get_prompt(
            "generate_verdicts",
            input=input,
            original_output=actual_output,
            perturbations=self._perturbation_pairs(),
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda r: list(r.verdicts),
            extract_json=lambda data: [
                RobustnessVerdict(**item) for item in data["verdicts"]
            ],
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None
        fragilities = [
            verdict.reason
            for verdict in self.verdicts
            if verdict.verdict.strip().lower() == "no"
        ]
        prompt = self._get_prompt(
            "generate_reason",
            fragilities=fragilities,
            score=format(self.score, ".2f"),
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AdversarialRobustnessScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None
        fragilities = [
            verdict.reason
            for verdict in self.verdicts
            if verdict.verdict.strip().lower() == "no"
        ]
        prompt = self._get_prompt(
            "generate_reason",
            fragilities=fragilities,
            score=format(self.score, ".2f"),
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AdversarialRobustnessScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1.0

        robust_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                robust_count += 1

        score = robust_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

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
        return "Adversarial Robustness"
