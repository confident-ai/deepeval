"""LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

from typing import Optional, List, Tuple, Type, Union
from deepeval.models import DeepEvalBaseMLLM
from deepeval.metrics import BaseMultimodalMetric
from deepeval.test_case import (
    MLLMTestCaseParams,
    MLLMTestCase,
)
from deepeval.metrics.multimodal_metrics.multimodal_g_eval.template import (
    MultimodalGEvalTemplate,
)
import deepeval.metrics.multimodal_metrics.multimodal_g_eval.schema as mgschema
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    initialize_multimodal_model,
    check_mllm_test_case_params,
    construct_verbose_logs,
    a_gen_and_extract,
    gen_and_extract,
    trimAndLoadJson,
)
from deepeval.metrics.multimodal_metrics.multimodal_g_eval.utils import (
    construct_test_case_list,
    no_multimodal_log_prob_support,
    construct_g_eval_params_string,
)
from deepeval.metrics.g_eval.utils import (
    Rubric,
    format_rubrics,
    calculate_weighted_summed_score,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
    number_evaluation_steps,
    get_score_range,
)


class MultimodalGEval(BaseMultimodalMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: List[MLLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        rubric: Optional[List[Rubric]] = None,
        model: Optional[Union[str, DeepEvalBaseMLLM]] = None,
        threshold: float = 0.5,
        top_logprobs: int = 20,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            MultimodalGEvalTemplate
        ] = MultimodalGEvalTemplate,
        _include_g_eval_suffix: bool = True,
    ):
        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)
        self.name = name
        self.evaluation_params = evaluation_params
        self.criteria = criteria
        self.rubric = validate_and_sort_rubrics(rubric)
        self.model, self.using_native_model = initialize_multimodal_model(model)
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
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
        _additional_context: Optional[str] = None,
    ) -> float:

        check_mllm_test_case_params(
            test_case, self.evaluation_params, None, None, self
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
                        _additional_context=_additional_context,
                    )
                )
            else:
                self.evaluation_steps: List[str] = (
                    self._generate_evaluation_steps()
                )
                g_score, reason = self._evaluate(
                    test_case, _additional_context=_additional_context
                )
                self.reason = reason
                self.score = float(g_score) / 10
                self.score = (
                    0
                    if self.strict_mode and self.score < self.threshold
                    else self.score
                )
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Criteria:\n{self.criteria}",
                        f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                        f"Rubric:\n{format_rubrics(self.rubric)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _additional_context: Optional[str] = None,
        _log_metric_to_confident: bool = True,
    ) -> float:

        check_mllm_test_case_params(
            test_case, self.evaluation_params, None, None, self
        )

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
            self.reason = reason
            self.score = (
                float(g_score) / 10 if not self.strict_mode else int(g_score)
            )
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Criteria:\n{self.criteria}",
                    f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                    f"Rubric:\n{format_rubrics(self.rubric)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
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

        return await a_gen_and_extract(
            self,
            prompt,
            mgschema.Steps,
            extract_schema=lambda r: r.steps,
            extract_json=lambda d: d["steps"],
        )

    def _generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = self.evaluation_template.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )

        return gen_and_extract(
            self,
            prompt,
            mgschema.Steps,
            extract_schema=lambda r: r.steps,
            extract_json=lambda d: d["steps"],
        )

    async def _a_evaluate(
        self, test_case: MLLMTestCase, _additional_context: Optional[str] = None
    ) -> Tuple[Union[int, float], str]:
        test_case_list = construct_test_case_list(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )

        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=number_evaluation_steps(self.evaluation_steps),
                test_case_list=test_case_list,
                parameters=g_eval_params_str,
                rubric=rubric_str,
                score_range=get_score_range(self.rubric),
                _additional_context=_additional_context,
            )
        else:
            prompt = (
                self.evaluation_template.generate_strict_evaluation_results(
                    evaluation_steps=number_evaluation_steps(
                        self.evaluation_steps
                    ),
                    test_case_list=test_case_list,
                    parameters=g_eval_params_str,
                    _additional_context=_additional_context,
                )
            )
        try:
            # don't use log probabilities for unsupported gpt models
            if no_multimodal_log_prob_support(self.model):
                raise AttributeError("log_probs unsupported.")

            # Don't have to check for using native model
            # since generate raw response only exist for deepeval's native model
            result, cost = await self.model.a_generate_raw_response(
                prompt, top_logprobs=self.top_logprobs
            )
            if self.evaluation_cost is not None:
                self.evaluation_cost += cost
            data = trimAndLoadJson(result.choices[0].message.content, self)

            reason = data["reason"]
            score = data["score"]
            if self.strict_mode:
                return score, reason

            try:
                weighted_summed_score = calculate_weighted_summed_score(
                    score, result
                )
                return weighted_summed_score, reason
            except Exception:
                return score, reason
        except (
            AttributeError
        ):  # This catches the case where a_generate_raw_response doesn't exist.

            return await a_gen_and_extract(
                self,
                prompt,
                mgschema.ReasonScore,
                extract_schema=lambda r: (r.score, r.reason),
                extract_json=lambda d: (d["score"], d["reason"]),
            )

    def _evaluate(
        self, test_case: MLLMTestCase, _additional_context: Optional[str] = None
    ) -> Tuple[Union[int, float], str]:
        test_case_list = construct_test_case_list(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )

        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=number_evaluation_steps(self.evaluation_steps),
                test_case_list=test_case_list,
                parameters=g_eval_params_str,
                rubric=rubric_str,
                score_range=get_score_range(self.rubric),
                _additional_context=_additional_context,
            )
        else:
            prompt = (
                self.evaluation_template.generate_strict_evaluation_results(
                    evaluation_steps=number_evaluation_steps(
                        self.evaluation_steps
                    ),
                    test_case_list=test_case_list,
                    parameters=g_eval_params_str,
                    _additional_context=_additional_context,
                )
            )

        try:
            # don't use log probabilities for unsupported gpt models
            if no_multimodal_log_prob_support(self.model):
                raise AttributeError("log_probs unsupported.")

            result, cost = self.model.generate_raw_response(
                prompt, top_logprobs=self.top_logprobs
            )
            if self.evaluation_cost is not None:
                self.evaluation_cost += cost
            data = trimAndLoadJson(result.choices[0].message.content, self)

            reason = data["reason"]
            score = data["score"]
            if self.strict_mode:
                return score, reason

            try:
                weighted_summed_score = calculate_weighted_summed_score(
                    score, result
                )
                return weighted_summed_score, reason
            except Exception:
                return score, reason
        except AttributeError:
            # This catches the case where a_generate_raw_response doesn't exist.

            return gen_and_extract(
                self,
                prompt,
                mgschema.ReasonScore,
                extract_schema=lambda r: (r.score, r.reason),
                extract_json=lambda d: (d["score"], d["reason"]),
            )

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except Exception:
                self.success = False
        return self.success

    @property
    def __name__(self):
        if self._include_g_eval_suffix:
            return f"{self.name} [GEval]"
        else:
            return self.name
