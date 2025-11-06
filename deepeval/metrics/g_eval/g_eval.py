"""LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

import asyncio

from typing import Optional, List, Tuple, Union, Type
from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics.g_eval.template import GEvalTemplate
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
    check_llm_test_case_params,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.g_eval import schema as gschema
from deepeval.metrics.g_eval.utils import (
    Rubric,
    construct_g_eval_params_string,
    construct_test_case_string,
    format_rubrics,
    no_log_prob_support,
    calculate_weighted_summed_score,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
    number_evaluation_steps,
    get_score_range,
)
from deepeval.metrics.api import metric_data_manager
from deepeval.config.settings import get_settings


class GEval(BaseMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: List[LLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        rubric: Optional[List[Rubric]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        top_logprobs: int = 20,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[GEvalTemplate] = GEvalTemplate,
        _include_g_eval_suffix: bool = True,
    ):
        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)
        self.name = name
        self.evaluation_params = evaluation_params
        self.criteria = criteria
        self.rubric = validate_and_sort_rubrics(rubric)
        self.score_range = get_score_range(self.rubric)
        self.score_range_span = self.score_range[1] - self.score_range[0]
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = (
            evaluation_steps
            if evaluation_steps and len(evaluation_steps) > 0
            else None
        )
        self.threshold = 1 if strict_mode else threshold
        self.top_logprobs = top_logprobs
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self._include_g_eval_suffix = _include_g_eval_suffix
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
        _additional_context: Optional[str] = None,
    ) -> float:
        check_llm_test_case_params(test_case, self.evaluation_params, self)
        self.evaluation_cost = 0 if self.using_native_model else None

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                coro = self.a_measure(
                    test_case,
                    _show_indicator=False,
                    _in_component=_in_component,
                    _additional_context=_additional_context,
                )
                loop.run_until_complete(
                    asyncio.wait_for(
                        coro,
                        timeout=get_settings().DEEPEVAL_PER_TASK_TIMEOUT_SECONDS,
                    )
                )
            else:
                self.evaluation_steps: List[str] = (
                    self._generate_evaluation_steps()
                )
                g_score, reason = self._evaluate(
                    test_case, _additional_context=_additional_context
                )
                self.score = (
                    (float(g_score) - self.score_range[0])
                    / self.score_range_span
                    if not self.strict_mode
                    else int(g_score)
                )
                self.success = self.score >= self.threshold

                self.reason = reason
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Criteria:\n{self.criteria}",
                        f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                        f"Rubric:\n{format_rubrics(self.rubric)}",
                        f"Score: {self.score}",
                        f"Reason: {self.reason}",
                    ],
                )
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
        _additional_context: Optional[str] = None,
    ) -> float:
        check_llm_test_case_params(test_case, self.evaluation_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.evaluation_steps: List[str] = (
                await self._a_generate_evaluation_steps()
            )
            g_score, reason = await self._a_evaluate(
                test_case, _additional_context=_additional_context
            )
            self.score = (
                (float(g_score) - self.score_range[0]) / self.score_range_span
                if not self.strict_mode
                else int(g_score)
            )
            self.success = self.score >= self.threshold

            self.reason = reason
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Criteria:\n{self.criteria}",
                    f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                    f"Rubric:\n{format_rubrics(self.rubric)}",
                    (f"Score: {self.score}"),
                    f"Reason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def _a_generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = self.evaluation_template.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
        else:
            try:
                res: gschema.Steps = await self.model.a_generate(
                    prompt, schema=gschema.Steps
                )
                return res.steps
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    def _generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = self.evaluation_template.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
        else:
            try:
                res: gschema.Steps = self.model.generate(
                    prompt, schema=gschema.Steps
                )
                return res.steps
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    async def _a_evaluate(
        self, test_case: LLMTestCase, _additional_context: Optional[str] = None
    ) -> Tuple[Union[int, float], str]:
        test_case_content = construct_test_case_string(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=number_evaluation_steps(self.evaluation_steps),
                test_case_content=test_case_content,
                parameters=g_eval_params_str,
                rubric=rubric_str,
                score_range=self.score_range,
                _additional_context=_additional_context,
            )
        else:
            prompt = (
                self.evaluation_template.generate_strict_evaluation_results(
                    evaluation_steps=number_evaluation_steps(
                        self.evaluation_steps
                    ),
                    test_case_content=test_case_content,
                    parameters=g_eval_params_str,
                    _additional_context=_additional_context,
                )
            )
        try:
            # don't use log probabilities for unsupported gpt models
            if no_log_prob_support(self.model):
                raise AttributeError("log_probs unsupported.")

            # Don't have to check for using native model
            # since generate raw response only exist for deepeval's native model
            res, cost = await self.model.a_generate_raw_response(
                prompt, top_logprobs=self.top_logprobs
            )

            if self.evaluation_cost is not None:
                self.evaluation_cost += cost

            data = trimAndLoadJson(res.choices[0].message.content, self)

            reason = data["reason"]
            score = data["score"]
            if self.strict_mode:
                return score, reason

            try:
                weighted_summed_score = calculate_weighted_summed_score(
                    score, res
                )
                return weighted_summed_score, reason
            except (KeyError, AttributeError, TypeError, ValueError):
                return score, reason
        except (
            AttributeError
        ):  # This catches the case where a_generate_raw_response doesn't exist.
            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]
            else:
                try:
                    res: gschema.ReasonScore = await self.model.a_generate(
                        prompt, schema=gschema.ReasonScore
                    )
                    return res.score, res.reason
                except TypeError:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    return data["score"], data["reason"]

    def _evaluate(
        self, test_case: LLMTestCase, _additional_context: Optional[str] = None
    ) -> Tuple[Union[int, float], str]:
        test_case_content = construct_test_case_string(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )

        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=number_evaluation_steps(self.evaluation_steps),
                test_case_content=test_case_content,
                parameters=g_eval_params_str,
                rubric=rubric_str,
                score_range=self.score_range,
                _additional_context=_additional_context,
            )
        else:
            prompt = (
                self.evaluation_template.generate_strict_evaluation_results(
                    evaluation_steps=number_evaluation_steps(
                        self.evaluation_steps
                    ),
                    test_case_content=test_case_content,
                    parameters=g_eval_params_str,
                    _additional_context=_additional_context,
                )
            )

        try:
            # don't use log probabilities for unsupported gpt models
            if no_log_prob_support(self.model):
                raise AttributeError("log_probs unsupported.")

            res, cost = self.model.generate_raw_response(
                prompt, top_logprobs=self.top_logprobs
            )
            self.evaluation_cost += cost
            data = trimAndLoadJson(res.choices[0].message.content, self)

            reason = data["reason"]
            score = data["score"]
            if self.strict_mode:
                return score, reason

            try:
                weighted_summed_score = calculate_weighted_summed_score(
                    score, res
                )
                return weighted_summed_score, reason
            except (KeyError, AttributeError, TypeError, ValueError):
                return score, reason
        except AttributeError:
            # This catches the case where a_generate_raw_response doesn't exist.
            if self.using_native_model:
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]
            else:
                try:
                    res: gschema.ReasonScore = self.model.generate(
                        prompt, schema=gschema.ReasonScore
                    )
                    return res.score, res.reason
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    return data["score"], data["reason"]

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
        if self._include_g_eval_suffix:
            return f"{self.name} [GEval]"
        else:
            return self.name
