"""A slightly modified tailored version of the LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional, List, Tuple, Union, Dict, Type
import math
from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.g_eval.utils import (
    Rubric,
    construct_conversational_g_eval_turn_params_string,
    construct_non_turns_test_case_string,
    format_rubrics,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
)
from deepeval.test_case import (
    TurnParams,
    ConversationalTestCase,
)
from deepeval.metrics.conversational_g_eval.template import (
    ConversationalGEvalTemplate,
)
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
    convert_turn_to_dict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
import deepeval.metrics.conversational_g_eval.schema as cgschema
from deepeval.metrics.api import metric_data_manager


class ConversationalGEval(BaseConversationalMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: Optional[List[TurnParams]] = None,
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        rubric: Optional[List[Rubric]] = None,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            ConversationalGEvalTemplate
        ] = ConversationalGEvalTemplate,
        _include_g_eval_suffix: bool = True,
    ):
        if evaluation_params is not None and len(evaluation_params) == 0:
            raise ValueError("evaluation_params cannot be an empty list.")

        self.name = name
        if evaluation_params is None:
            evaluation_params = [TurnParams.CONTENT, TurnParams.ROLE]

        if TurnParams.CONTENT not in evaluation_params:
            evaluation_params.append(TurnParams.CONTENT)
        if TurnParams.ROLE not in evaluation_params:
            evaluation_params.append(TurnParams.ROLE)

        self.evaluation_params = evaluation_params

        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)
        self.criteria = criteria
        self.rubric = validate_and_sort_rubrics(rubric)
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = (
            evaluation_steps
            if evaluation_steps and len(evaluation_steps) > 0
            else None
        )
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
        self._include_g_eval_suffix = _include_g_eval_suffix

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_conversational_test_case_params(
            test_case, self.evaluation_params, self
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
                self.evaluation_steps: List[str] = (
                    self._generate_evaluation_steps()
                )
                g_score, reason = self.evaluate(test_case)
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
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_conversational_test_case_params(
            test_case, self.evaluation_params, self
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
            g_score, reason = await self._a_evaluate(test_case)
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
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    async def _a_generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_conversational_g_eval_turn_params_string(
            self.evaluation_params
        )
        prompt = self.evaluation_template.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=cgschema.Steps
            )
            self.evaluation_cost += cost
            return res.steps
        else:
            try:
                res: cgschema.Steps = await self.model.a_generate(
                    prompt, schema=cgschema.Steps
                )
                return res.steps
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    def _generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_conversational_g_eval_turn_params_string(
            self.evaluation_params
        )
        prompt = self.evaluation_template.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=cgschema.Steps)
            self.evaluation_cost += cost
            return res.steps
        else:
            try:
                res: cgschema.Steps = self.model.generate(
                    prompt, schema=cgschema.Steps
                )
                return res.steps
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    async def _a_evaluate(
        self, test_case: ConversationalTestCase
    ) -> Tuple[Union[int, float], str]:
        test_case_content = construct_non_turns_test_case_string(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_conversational_g_eval_turn_params_string(
            self.evaluation_params
        )
        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                test_case_content=test_case_content,
                turns=[
                    convert_turn_to_dict(turn, self.evaluation_params)
                    for turn in test_case.turns
                ],
                parameters=g_eval_params_str,
                rubric=rubric_str,
            )
        else:
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                test_case_content=test_case_content,
                turns=[
                    convert_turn_to_dict(turn, self.evaluation_params)
                    for turn in test_case.turns
                ],
                parameters=g_eval_params_str,
            )
        try:
            res, cost = await self.model.a_generate_raw_response(
                prompt, top_logprobs=20
            )
            self.evaluation_cost += cost
            data = trimAndLoadJson(res.choices[0].message.content, self)

            reason = data["reason"]
            score = data["score"]
            if self.strict_mode:
                return score, reason

            try:
                weighted_summed_score = self.generate_weighted_summed_score(
                    score, res
                )
                return weighted_summed_score, reason
            except (KeyError, AttributeError, TypeError, ValueError):
                return score, reason
        except (
            AttributeError
        ):  # This catches the case where a_generate_raw_response doesn't exist.
            if self.using_native_model:
                res, cost = await self.model.a_generate(
                    prompt, schema=cgschema.ReasonScore
                )
                self.evaluation_cost += cost
                return res.score, res.reason
            else:
                try:
                    res: cgschema.ReasonScore = await self.model.a_generate(
                        prompt, schema=cgschema.ReasonScore
                    )
                    return res.score, res.reason
                except TypeError:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    return data["score"], data["reason"]

    def evaluate(
        self, test_case: ConversationalTestCase
    ) -> Tuple[Union[int, float], str]:
        test_case_content = construct_non_turns_test_case_string(
            self.evaluation_params, test_case
        )
        g_eval_params_str = construct_conversational_g_eval_turn_params_string(
            self.evaluation_params
        )
        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                test_case_content=test_case_content,
                turns=[
                    convert_turn_to_dict(turn, self.evaluation_params)
                    for turn in test_case.turns
                ],
                parameters=g_eval_params_str,
                rubric=rubric_str,
            )
        else:
            prompt = self.evaluation_template.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                test_case_content=test_case_content,
                turns=[
                    convert_turn_to_dict(turn, self.evaluation_params)
                    for turn in test_case.turns
                ],
                parameters=g_eval_params_str,
            )
        try:
            res, cost = self.model.generate_raw_response(
                prompt, top_logprobs=20
            )
            self.evaluation_cost += cost
            data = trimAndLoadJson(res.choices[0].message.content, self)

            reason = data["reason"]
            score = data["score"]
            if self.strict_mode:
                return score, reason

            try:
                weighted_summed_score = self.generate_weighted_summed_score(
                    score, res
                )
                return weighted_summed_score, reason
            except (KeyError, AttributeError, TypeError, ValueError):
                return score, reason
        except AttributeError:
            # This catches the case where a_generate_raw_response doesn't exist.
            if self.using_native_model:
                res, cost = self.model.generate(
                    prompt, schema=cgschema.ReasonScore
                )
                self.evaluation_cost += cost
                return res.score, res.reason
            else:
                try:
                    res: cgschema.ReasonScore = self.model.generate(
                        prompt, schema=cgschema.ReasonScore
                    )
                    return res.score, res.reason
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    return data["score"], data["reason"]

    def generate_weighted_summed_score(
        self, raw_score: int, raw_response: ChatCompletion
    ) -> Union[int, float]:
        generated_logprobs = raw_response.choices[0].logprobs.content
        # First, locate the token that we care for logprobs, i.e., the token matching the score
        score_logprobs = None
        for token_logprobs in generated_logprobs:
            if token_logprobs.token == str(raw_score):
                score_logprobs = token_logprobs
                break
        # Then, calculate the score based on the logprobs
        token_linear_probability: Dict[int, float] = {}
        sum_linear_probability = 0
        # Filter out tokens with <1% linear probability, i.e., logprobs < math.log(0.01)
        min_logprob = math.log(0.01)
        for token_logprob in score_logprobs.top_logprobs:
            logprob = token_logprob.logprob

            # Filter out low probability tokens
            if logprob < min_logprob:
                continue
            # Filter out non-decimal token to prevent errors in later int(token) conversion
            if not token_logprob.token.isdecimal():
                continue

            # Calculate the linear probability
            linear_prob = math.exp(logprob)
            token_score = int(token_logprob.token)
            if token_linear_probability.get(token_score):
                token_linear_probability[token_score] += linear_prob
            else:
                token_linear_probability[token_score] = linear_prob
            sum_linear_probability += linear_prob

        sum_of_weighted_scores = 0.0
        for score, prob in token_linear_probability.items():
            sum_of_weighted_scores += score * prob

        # Scale the sum of linear probability to 1
        weighted_summed_score = sum_of_weighted_scores / sum_linear_probability
        return weighted_summed_score

    def number_evaluation_steps(self):
        evaluation_steps = """"""
        for index, string in enumerate(self.evaluation_steps, start=1):
            evaluation_steps += f"{index}. {string}\n"
        return evaluation_steps

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
            return f"{self.name} [Conversational GEval]"
        else:
            return self.name
