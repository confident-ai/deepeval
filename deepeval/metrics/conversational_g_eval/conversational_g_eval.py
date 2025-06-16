"""A slightly modified tailored version of the LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional, List, Tuple, Union, Dict
import math
from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.g_eval.utils import (
    Rubric,
    construct_conversational_g_eval_turn_params_string,
    format_rubrics,
)
from deepeval.test_case import (
    Turn,
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
from deepeval.metrics.conversational_g_eval.schema import *


class ConversationalGEval(BaseConversationalMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: Optional[List[TurnParams]] = [TurnParams.CONTENT],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        rubric: Optional[List[Rubric]] = None,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_g_eval_suffix: bool = True,
    ):
        if evaluation_params is not None and len(evaluation_params) == 0:
            raise ValueError("evaluation_params cannot be an empty list.")

        self.name = name
        self.evaluation_params = evaluation_params

        # Check if both criteria and evaluation_steps are not None at the same time
        if criteria is None and evaluation_steps is None:
            raise ValueError(
                "Either 'criteria' or 'evaluation_steps' must be provided."
            )

        # Check if criteria is provided, it cannot be an empty string
        if criteria is not None and not criteria.strip():
            raise ValueError("Criteria provided cannot be an empty string.")

        # Check if evaluation_steps is provided, it cannot be an empty list
        if evaluation_steps is not None and len(evaluation_steps) == 0:
            raise ValueError(
                "'evaluation_steps' must not be an empty list. Either omit evaluation steps or include a non-empty list of steps."
            )

        self.criteria = criteria
        self.rubric = rubric
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = evaluation_steps
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self._include_g_eval_suffix = _include_g_eval_suffix

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_conversational_test_case_params(test_case, self)

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
                    )
                )
            else:
                self.evaluation_steps: List[str] = (
                    self._generate_evaluation_steps()
                )
                g_score, reason = self.evaluate(test_case.turns)
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
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_conversational_test_case_params(test_case, self)

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
            g_score, reason = await self._a_evaluate(test_case.turns)
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

    async def _a_generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_conversational_g_eval_turn_params_string(
            self.evaluation_params
        )
        prompt = ConversationalGEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Steps)
            self.evaluation_cost += cost
            return res.steps
        else:
            try:
                res: Steps = await self.model.a_generate(prompt, schema=Steps)
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
        prompt = ConversationalGEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Steps)
            self.evaluation_cost += cost
            return res.steps
        else:
            try:
                res: Steps = self.model.generate(prompt, schema=Steps)
                return res.steps
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    async def _a_evaluate(
        self, turns: List[Turn]
    ) -> Tuple[Union[int, float], str]:
        g_eval_params_str = construct_conversational_g_eval_turn_params_string(
            self.evaluation_params
        )
        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = ConversationalGEvalTemplate.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                turns=[convert_turn_to_dict(turn) for turn in turns],
                parameters=g_eval_params_str,
                rubric=rubric_str,
            )
        else:
            prompt = ConversationalGEvalTemplate.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                turns=[convert_turn_to_dict(turn) for turn in turns],
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
            except:
                return score, reason
        except (
            AttributeError
        ):  # This catches the case where a_generate_raw_response doesn't exist.
            if self.using_native_model:
                res, cost = await self.model.a_generate(
                    prompt, schema=ReasonScore
                )
                self.evaluation_cost += cost
                return res.score, res.reason
            else:
                try:
                    res: ReasonScore = await self.model.a_generate(
                        prompt, schema=ReasonScore
                    )
                    return res.score, res.reason
                except TypeError:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    return data["score"], data["reason"]

    def evaluate(self, turns: List[Turn]) -> Tuple[Union[int, float], str]:
        g_eval_params_str = construct_conversational_g_eval_turn_params_string(
            self.evaluation_params
        )
        if not self.strict_mode:
            rubric_str = format_rubrics(self.rubric) if self.rubric else None
            prompt = ConversationalGEvalTemplate.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                turns=[convert_turn_to_dict(turn) for turn in turns],
                parameters=g_eval_params_str,
                rubric=rubric_str,
            )
        else:
            prompt = ConversationalGEvalTemplate.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                turns=[convert_turn_to_dict(turn) for turn in turns],
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
            except:
                return score, reason
        except AttributeError:
            # This catches the case where a_generate_raw_response doesn't exist.
            if self.using_native_model:
                res, cost = self.model.generate(prompt, schema=ReasonScore)
                self.evaluation_cost += cost
                return res.score, res.reason
            else:
                try:
                    res: ReasonScore = self.model.generate(
                        prompt, schema=ReasonScore
                    )
                    return res.score, res.reason
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    return data["score"], data["reason"]

    def generate_weighted_summed_score(
        self, raw_score: int, raw_response: ChatCompletion
    ) -> Union[int, float]:
        try:
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
            weighted_summed_score = (
                sum_of_weighted_scores / sum_linear_probability
            )
            return weighted_summed_score
        except:
            raise

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
                self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        if self._include_g_eval_suffix:
            return f"{self.name} (Conversational GEval)"
        else:
            return self.name
