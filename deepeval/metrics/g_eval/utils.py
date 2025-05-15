from typing import List, Optional, Union
from deepeval.models import DeepEvalBaseLLM, GPTModel, AzureOpenAIModel
from deepeval.test_case import LLMTestCaseParams
from deepeval.models.llms.openai_model import unsupported_log_probs_gpt_models
from pydantic import BaseModel, field_validator
from typing import Tuple


class Rubric(BaseModel):
    score_range: Tuple[int, int]
    expected_outcome: str

    @field_validator("score_range")
    def validate_score_range(cls, value):
        start, end = value
        if not (0 <= start <= 10 and 0 <= end <= 10):
            raise ValueError(
                "Both score_range values must be between 0 and 10 inclusive."
            )
        if start >= end:
            raise ValueError("score_range start must be less than end.")
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


def validate_and_sort_rubrics(
    rubrics: Optional[List[Rubric]] = None,
) -> Optional[List[Rubric]]:
    if rubrics is None:
        return None

    for rubric in rubrics:
        start, end = rubric.score_range
        if not (0 <= start <= 10 and 0 <= end <= 10):
            raise ValueError(
                f"Score range {rubric.score_range} must be between 0 and 10 inclusive."
            )
        if start >= end:
            raise ValueError(
                f"Invalid score range {rubric.score_range}: start must be less than end."
            )

    # Sort rubrics by start of range
    sorted_rubrics = sorted(rubrics, key=lambda r: r.score_range[0])

    # Full overlap check
    for i in range(len(sorted_rubrics)):
        a_start, a_end = sorted_rubrics[i].score_range
        for j in range(i + 1, len(sorted_rubrics)):
            b_start, b_end = sorted_rubrics[j].score_range
            # Check if ranges overlap
            if a_end > b_start:
                raise ValueError(
                    f"Overlapping score ranges: {sorted_rubrics[i].score_range} and {sorted_rubrics[j].score_range}"
                )

    return sorted_rubrics


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
