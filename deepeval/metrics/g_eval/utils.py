from typing import List, Optional, Union, Tuple, Dict
from openai.types.chat.chat_completion import ChatCompletion
import math

from deepeval.models import DeepEvalBaseLLM, GPTModel, AzureOpenAIModel
from deepeval.test_case import (
    LLMTestCaseParams,
    TurnParams,
    LLMTestCase,
    ToolCall,
)
from deepeval.models.llms.openai_model import unsupported_log_probs_gpt_models
from pydantic import BaseModel, field_validator

from deepeval.test_case.conversational_test_case import ConversationalTestCase


class Rubric(BaseModel):
    score_range: Tuple[int, int]
    expected_outcome: str

    @field_validator("score_range")
    def validate_score_range(cls, value):
        start, end = value
        if not (0 <= start <= 10 and 0 <= end <= 10):
            raise ValueError(
                "Both Rubric's 'score_range' values must be between 0 and 10 inclusive."
            )
        if start > end:
            raise ValueError(
                "Rubric's 'score_range' start must be less than or equal to end."
            )
        return value


G_EVAL_PARAMS = {
    LLMTestCaseParams.INPUT: "Input",
    LLMTestCaseParams.ACTUAL_OUTPUT: "Actual Output",
    LLMTestCaseParams.EXPECTED_OUTPUT: "Expected Output",
    LLMTestCaseParams.CONTEXT: "Context",
    LLMTestCaseParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    LLMTestCaseParams.EXPECTED_TOOLS: "Expected Tools",
    LLMTestCaseParams.TOOLS_CALLED: "Tools Called",
}

CONVERSATIONAL_G_EVAL_PARAMS = {
    TurnParams.CONTENT: "Content",
    TurnParams.ROLE: "Role",
    TurnParams.TOOLS_CALLED: "Tools Called",
    TurnParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    TurnParams.EXPECTED_OUTCOME: "Expected Outcome",
    TurnParams.SCENARIO: "Scenario",
}


def validate_criteria_and_evaluation_steps(
    criteria: Optional[str] = None,
    evaluation_steps: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[List[str]]]:
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


def validate_and_sort_rubrics(
    rubrics: Optional[List[Rubric]] = None,
) -> Optional[List[Rubric]]:
    if rubrics is None or len(rubrics) == 0:
        return None

    # Sort rubrics by start of range
    sorted_rubrics = sorted(rubrics, key=lambda r: r.score_range[0])

    # Full overlap check
    for i in range(len(sorted_rubrics)):
        a_start, a_end = sorted_rubrics[i].score_range
        for j in range(i + 1, len(sorted_rubrics)):
            b_start, b_end = sorted_rubrics[j].score_range
            # Check if ranges overlap
            if a_end >= b_start:
                raise ValueError(
                    f"Overlapping score ranges: {sorted_rubrics[i].score_range} and {sorted_rubrics[j].score_range}"
                )

    return sorted_rubrics


def format_rubrics(rubrics: Optional[List[Rubric]]) -> Optional[str]:
    if rubrics is None:
        return None

    return "\n".join(
        (
            f"{start}: {rubric.expected_outcome}"
            if start == end
            else f"{start}-{end}: {rubric.expected_outcome}"
        )
        for rubric in rubrics
        for start, end in [rubric.score_range]
    )


def no_log_prob_support(model: Union[str, DeepEvalBaseLLM]):

    if isinstance(model, str) and model in unsupported_log_probs_gpt_models:
        return True
    elif (
        isinstance(model, GPTModel)
        and model.model_name in unsupported_log_probs_gpt_models
    ):
        return True
    elif (
        isinstance(model, AzureOpenAIModel)
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


def construct_conversational_g_eval_turn_params_string(
    turn_params: List[TurnParams],
):
    g_eval_params = [
        CONVERSATIONAL_G_EVAL_PARAMS[param] for param in turn_params
    ]

    if len(g_eval_params) == 1:
        g_eval_params_str = g_eval_params[0]
    elif len(g_eval_params) == 2:
        g_eval_params_str = " and ".join(g_eval_params)
    else:
        g_eval_params_str = (
            ", ".join(g_eval_params[:-1]) + ", and " + g_eval_params[-1]
        )

    return g_eval_params_str


def construct_non_turns_test_case_string(
    turn_params: List[TurnParams], test_case: ConversationalTestCase
) -> str:
    text = """"""
    for param in turn_params:
        if (
            param == TurnParams.RETRIEVAL_CONTEXT
            or param == TurnParams.TOOLS_CALLED
            or param == TurnParams.CONTENT
            or param == TurnParams.ROLE
        ):
            continue

        value = getattr(test_case, param.value)
        text += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value} \n\n"
    return text


def construct_test_case_string(
    evaluation_params: List[LLMTestCaseParams], test_case: LLMTestCase
) -> str:
    text = """"""
    for param in evaluation_params:
        value = getattr(test_case, param.value)
        if isinstance(value, ToolCall):
            value = repr(value)
        text += f"{G_EVAL_PARAMS[param]}:\n{value} \n\n"
    return text


def calculate_weighted_summed_score(
    raw_score: int, raw_response: ChatCompletion
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
        weighted_summed_score = sum_of_weighted_scores / sum_linear_probability
        return weighted_summed_score
    except:
        raise


def number_evaluation_steps(evaluation_steps: List[str]) -> str:
    formatted_evaluation_steps = """"""
    for index, string in enumerate(evaluation_steps, start=1):
        formatted_evaluation_steps += f"{index}. {string}\n"
    return formatted_evaluation_steps


def number_test_case_contents(test_case_contents: List[str]) -> str:
    formatted_test_case_contents = """"""
    for index, string in enumerate(test_case_contents):
        formatted_test_case_contents += f"{index}. {string}\n"
    return formatted_test_case_contents


def get_score_range(rubric: Optional[List[Rubric]]) -> Tuple[int, int]:
    if rubric is None:
        return (0, 10)

    return rubric[0].score_range[0], rubric[-1].score_range[1]
