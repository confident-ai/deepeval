"""LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

from typing import Optional, List, Tuple, Union, Dict
from langchain.schema import AIMessage
import math
from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    ToolCall,
)

from deepeval.metrics.g_eval.template import GEvalTemplate
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.models.providers.gpt_model import unsupported_log_probs_gpt_models
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.g_eval.schema import *

G_EVAL_PARAMS = {
    LLMTestCaseParams.INPUT: "Input",
    LLMTestCaseParams.ACTUAL_OUTPUT: "Actual Output",
    LLMTestCaseParams.EXPECTED_OUTPUT: "Expected Output",
    LLMTestCaseParams.CONTEXT: "Context",
    LLMTestCaseParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    LLMTestCaseParams.EXPECTED_TOOLS: "Expected Tools",
    LLMTestCaseParams.TOOLS_CALLED: "Tools Called",
}


def no_log_prob_support(model: Union[str, DeepEvalBaseLLM]):

    if isinstance(model, str) and model in unsupported_log_probs_gpt_models:
        return True
    elif (
        isinstance(model, GPTModel)
        and model.model_name in unsupported_log_probs_gpt_models
    ):
        return True

    return False


def construct_g_eval_params_string(
    llm_test_case_params: List[LLMTestCaseParams],
):
    g_eval_params = [G_EVAL_PARAMS[param] for param in llm_test_case_params]

    if len(g_eval_params) == 1:
        g_eval_params_str = g_eval_params[0]
    elif len(g_eval_params) == 2:
        g_eval_params_str = " and ".join(g_eval_params)
    else:
        g_eval_params_str = (
            ", ".join(g_eval_params[:-1]) + ", and " + g_eval_params[-1]
        )

    return g_eval_params_str


class GEval(BaseMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: List[LLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        top_logprobs: int = 20,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_g_eval_suffix: bool = True,
    ):
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
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = evaluation_steps
        self.threshold = 1 if strict_mode else threshold
        self.top_logprobs = top_logprobs
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self._include_g_eval_suffix = _include_g_eval_suffix

    def measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[-1]
        check_llm_test_case_params(test_case, self.evaluation_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
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
                        f"Evaluation Steps:\n{prettify_list(self.evaluation_steps)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[-1]
        check_llm_test_case_params(test_case, self.evaluation_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.evaluation_steps: List[str] = (
                await self._a_generate_evaluation_steps()
            )
            g_score, reason = await self._a_evaluate(test_case)
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
        prompt = GEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
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

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = GEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["steps"]
        else:
            try:
                res: Steps = self.model.generate(prompt, schema=Steps)
                return res.steps
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["steps"]

    async def _a_evaluate(
        self, test_case: LLMTestCase
    ) -> Tuple[Union[int, float], str]:
        text = """"""
        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            if isinstance(value, ToolCall):
                value = repr(value)
            text += f"{G_EVAL_PARAMS[param]}:\n{value} \n\n"

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )

        if not self.strict_mode:
            prompt = GEvalTemplate.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                text=text,
                parameters=g_eval_params_str,
            )
        else:
            prompt = GEvalTemplate.generate_strict_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                text=text,
                parameters=g_eval_params_str,
            )

        try:
            # don't use log probabilities for unsupported gpt models
            if no_log_prob_support(self.model):
                raise AttributeError("log_probs unsupported.")

            # Don't have to check for using native model
            # since generate raw response only exist for deepeval's native model
            res, cost = await self.model.a_generate_raw_response(
                prompt, logprobs=True, top_logprobs=self.top_logprobs
            )
            self.evaluation_cost += cost
            data = trimAndLoadJson(res.content, self)

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
                res, cost = await self.model.a_generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]
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

    def evaluate(self, test_case: LLMTestCase) -> Tuple[Union[int, float], str]:
        text = """"""
        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            if isinstance(value, ToolCall):
                value = repr(value)
            text += f"{param.value}: {value} \n\n"

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )

        if not self.strict_mode:
            prompt = GEvalTemplate.generate_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                text=text,
                parameters=g_eval_params_str,
            )
        else:
            prompt = GEvalTemplate.generate_strict_evaluation_results(
                evaluation_steps=self.number_evaluation_steps(),
                text=text,
                parameters=g_eval_params_str,
            )

        try:
            # don't use log probabilities for unsupported gpt models
            if no_log_prob_support(self.model):
                raise AttributeError("log_probs unsupported.")

            res, cost = self.model.generate_raw_response(
                prompt, logprobs=True, top_logprobs=self.top_logprobs
            )
            self.evaluation_cost += cost
            data = trimAndLoadJson(res.content, self)

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
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]
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
        self, raw_score: int, raw_response: AIMessage
    ) -> Union[int, float]:
        """
        Example raw_response.response_metadata["logprobs"]["content"]
        [
            {
                'token': '9',
                'bytes': [57],
                'logprob': -0.18066935,
                'top_logprobs': [
                    {'token': '9', 'bytes': [57], 'logprob': -0.18066935},
                    {'token': '8', 'bytes': [56], 'logprob': -1.8056693},
                    {'token': '10', 'bytes': [49, 48], 'logprob': -7.1337943},
                    {'token': '7', 'bytes': [55], 'logprob': -8.977545},
                    {'token': ' ', 'bytes': [32], 'logprob': -15.477545},
                    {'token': '6', 'bytes': [54], 'logprob': -17.133795},
                    {'token': '5', 'bytes': [53], 'logprob': -20.352545},
                    {'token': '09', 'bytes': [48, 57], 'logprob': -21.83692},
                    {'token': '0', 'bytes': [48], 'logprob': -22.383795},
                    {'token': ' nine', 'bytes': [32, 110, 105, 110, 101], 'logprob': -22.74317},
                    {'token': '4', 'bytes': [52], 'logprob': -22.875982},
                    {'token': '08', 'bytes': [48, 56], 'logprob': -22.99317},
                    {'token': '<|end|>', 'bytes': None, 'logprob': -23.469732},
                    {'token': '９', 'bytes': [239, 188, 153], 'logprob': -23.625982},
                    {'token': '\xa0', 'bytes': [194, 160], 'logprob': -24.079107},
                    {'token': '3', 'bytes': [51], 'logprob': -24.125982},
                    {'token': ' eight',
                    'bytes': [32, 101, 105, 103, 104, 116],
                    'logprob': -24.39942},
                    {'token': '90', 'bytes': [57, 48], 'logprob': -24.454107},
                    {'token': '８', 'bytes': [239, 188, 152], 'logprob': -24.89942},
                    {'token': '1', 'bytes': [49], 'logprob': -25.329107}
                ]
            },
            { next token in generation with top_logprobs ...}
        ]
        """
        try:
            generated_logprobs = raw_response.response_metadata["logprobs"][
                "content"
            ]
            # First, locate the token that we care for logprobs, i.e., the token matching the score
            score_logprobs = None
            for token_logprobs in generated_logprobs:
                if token_logprobs["token"] == str(raw_score):
                    score_logprobs = token_logprobs
                    break
            # Then, calculate the score based on the logprobs
            token_linear_probability: Dict[int, float] = {}
            sum_linear_probability = 0
            # Filter out tokens with <1% linear probability, i.e., logprobs < math.log(0.01)
            min_logprob = math.log(0.01)
            for token_logprob in score_logprobs["top_logprobs"]:
                logprob = token_logprob["logprob"]

                # Filter out low probability tokens
                if logprob < min_logprob:
                    continue
                # Filter out non-decimal token to prevent errors in later int(token) conversion
                if not token_logprob["token"].isdecimal():
                    continue

                # Calculate the linear probability
                linear_prob = math.exp(logprob)
                token_score = int(token_logprob["token"])
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
            return f"{self.name} (GEval)"
        else:
            return self.name
