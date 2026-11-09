from typing import List, Optional, Union, Tuple, Dict
from openai.types.chat.chat_completion import ChatCompletion
import math

from deepeval.models import DeepEvalBaseLLM, OpenAIModel, AzureOpenAIModel
from deepeval.test_case import (
    SingleTurnParams,
    MultiTurnParams,
    LLMTestCase,
    ToolCall,
)
from pydantic import BaseModel, field_validator
from deepeval.models.llms.constants import OPENAI_MODELS_DATA

from deepeval.test_case.conversational_test_case import ConversationalTestCase


from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


class APIRubric(BaseModel):
    scoreRange: Tuple[float, float]
    expectedOutcome: str


class MetricPullResponse(BaseModel):
    id: Optional[str] = None
    criteria: Optional[str] = None
    evaluationSteps: Optional[List[str]] = None
    requiredParameters: List[str] = Field(default_factory=list)
    rubric: Optional[List[APIRubric]] = None


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
    SingleTurnParams.INPUT: "Input",
    SingleTurnParams.ACTUAL_OUTPUT: "Actual Output",
    SingleTurnParams.EXPECTED_OUTPUT: "Expected Output",
    SingleTurnParams.CONTEXT: "Context",
    SingleTurnParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    SingleTurnParams.METADATA: "Metadata",
    SingleTurnParams.TAGS: "Tags",
    SingleTurnParams.EXPECTED_TOOLS: "Expected Tools",
    SingleTurnParams.TOOLS_CALLED: "Tools Called",
}

CONVERSATIONAL_G_EVAL_PARAMS = {
    MultiTurnParams.CONTENT: "Content",
    MultiTurnParams.ROLE: "Role",
    MultiTurnParams.METADATA: "Metadata",
    MultiTurnParams.TAGS: "Tags",
    MultiTurnParams.TOOLS_CALLED: "Tools Called",
    MultiTurnParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    MultiTurnParams.EXPECTED_OUTCOME: "Expected Outcome",
    MultiTurnParams.SCENARIO: "Scenario",
    MultiTurnParams.AUDIO: "Audio",
}

G_EVAL_API_PARAMS = {
    SingleTurnParams.INPUT: "input",
    SingleTurnParams.ACTUAL_OUTPUT: "actualOutput",
    SingleTurnParams.EXPECTED_OUTPUT: "expectedOutput",
    SingleTurnParams.CONTEXT: "context",
    SingleTurnParams.RETRIEVAL_CONTEXT: "retrievalContext",
    SingleTurnParams.METADATA: "metadata",
    SingleTurnParams.TAGS: "tags",
    SingleTurnParams.EXPECTED_TOOLS: "expectedTools",
    SingleTurnParams.TOOLS_CALLED: "toolsCalled",
}

CONVERSATIONAL_G_EVAL_API_PARAMS = {
    MultiTurnParams.ROLE: "role",
    MultiTurnParams.CONTENT: "content",
    MultiTurnParams.METADATA: "metadata",
    MultiTurnParams.TAGS: "tags",
    MultiTurnParams.SCENARIO: "scenario",
    MultiTurnParams.EXPECTED_OUTCOME: "expectedOutcome",
    MultiTurnParams.RETRIEVAL_CONTEXT: "retrievalContext",
    MultiTurnParams.TOOLS_CALLED: "toolsCalled",
    MultiTurnParams.AUDIO: "audio",
}


def construct_geval_pull_evaluation_params(
    required_parameters: List[str], multi_turn: bool
) -> List[Union[SingleTurnParams, MultiTurnParams]]:
    if not required_parameters:
        raise ValueError(
            "This metric has no evaluation parameters and cannot be pulled."
        )

    if multi_turn:
        reverse_params = {
            value: key
            for key, value in CONVERSATIONAL_G_EVAL_API_PARAMS.items()
        }
    else:
        reverse_params = {
            value: key for key, value in G_EVAL_API_PARAMS.items()
        }

    unsupported_params = [
        param for param in required_parameters if param not in reverse_params
    ]
    if unsupported_params:
        raise ValueError(
            f"Unsupported evaluation params encountered while pulling metric: {', '.join(unsupported_params)}."
        )

    return [reverse_params[param] for param in required_parameters]


def construct_geval_upload_payload(
    name: str,
    evaluation_params: List[SingleTurnParams],
    g_eval_api_params: Dict,
    criteria: Optional[str] = None,
    evaluation_steps: Optional[List[str]] = None,
    multi_turn: bool = False,
    rubric: Optional[List[Rubric]] = None,
) -> Dict:
    if not evaluation_params:
        raise ValueError("GEval requires at least one evaluation parameter.")

    unsupported_params = [
        param for param in evaluation_params if param not in g_eval_api_params
    ]
    if unsupported_params:
        raise ValueError(
            "Unsupported evaluation params for GEval upload: "
            + ", ".join(param.name for param in unsupported_params)
        )

    payload = {
        "name": name,
        "evaluationParams": [
            g_eval_api_params[param] for param in evaluation_params
        ],
        "multiTurn": multi_turn,
    }

    if criteria is not None:
        payload["criteria"] = criteria
    else:
        payload["evaluationSteps"] = evaluation_steps

    if rubric is not None:
        payload["rubric"] = [
            {
                "scoreRange": list(r.score_range),
                "expectedOutcome": r.expected_outcome,
            }
            for r in rubric
        ]

    return payload


def ensure_required_params(
    evaluation_params: Optional[List],
    criteria: Optional[str],
    evaluation_steps: Optional[List[str]],
    *,
    operation: str = "evaluate",
) -> None:
    if not evaluation_params:
        raise ValueError(
            f"GEval requires evaluation_params. Provide them at initialization or call pull() before {operation}."
        )
    validate_criteria_and_evaluation_steps(criteria, evaluation_steps)


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

    if isinstance(model, str):
        model_data = OPENAI_MODELS_DATA.get(model)
        if not model_data.supports_log_probs:
            return True
    elif (
        isinstance(model, OpenAIModel)
        and not model.model_data.supports_log_probs
    ):
        return True
    elif (
        isinstance(model, AzureOpenAIModel)
        and not model.model_data.supports_log_probs
    ):
        return True

    return False


def construct_g_eval_params_string(
    llm_test_case_params: List[SingleTurnParams],
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
    turn_params: List[MultiTurnParams],
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
    turn_params: List[MultiTurnParams], test_case: ConversationalTestCase
) -> str:
    body = """"""
    for param in turn_params:
        if (
            param == MultiTurnParams.RETRIEVAL_CONTEXT
            or param == MultiTurnParams.TOOLS_CALLED
            or param == MultiTurnParams.CONTENT
            or param == MultiTurnParams.ROLE
            or param == MultiTurnParams.AUDIO
        ):
            continue

        value = getattr(test_case, param.value)
        body += f"{CONVERSATIONAL_G_EVAL_PARAMS[param]}:\n{value} \n\n"

    if not body:
        return ""

    return f"Conversation-level fields:\n{body}"


def construct_test_case_string(
    evaluation_params: List[SingleTurnParams], test_case: LLMTestCase
) -> str:
    text = """"""
    for param in evaluation_params:
        value = getattr(test_case, param.value)
        if isinstance(value, ToolCall):
            value = repr(value)
        text += f"{G_EVAL_PARAMS[param]}:\n{value} \n\n"
    return text


# Candidates under 1% probability are ignored.
_MIN_LOGPROB = math.log(0.01)


def _find_score_span(tokens, raw_score: int) -> Optional[Tuple[int, int]]:
    """Start and length of the last run of tokens spelling ``raw_score``. The
    reason may repeat its digits, and some tokenizers split "10" into "1", "0".
    """
    score_str = str(raw_score)
    for start in reversed(range(len(tokens))):
        for length in range(len(score_str), 0, -1):
            span = tokens[start : start + length]
            spelled = "".join(t.token for t in span)
            if len(span) == length and spelled == score_str:
                return start, length
    return None


def _is_prefix_of_larger_score(token: str, score_max: int) -> bool:
    """Whether ``token`` could be the first digit of a larger score, like "1"
    of 10."""
    return any(
        str(score).startswith(token)
        for score in range(int(token) + 1, score_max + 1)
    )


def calculate_weighted_summed_score(
    raw_score: int,
    raw_response: ChatCompletion,
    score_max: int = 10,
    score_min: int = 0,
) -> Union[int, float]:
    """Average the scores the judge considered, weighted by their probability.
    Alternative scores outside ``score_min``..``score_max`` are ignored; the
    emitted ``raw_score`` itself is kept as is. Falls back to ``raw_score``
    when there is nothing to weight."""
    tokens = raw_response.choices[0].logprobs.content
    span = _find_score_span(tokens, raw_score)
    if span is None:
        return raw_score
    start, length = span
    score_token = tokens[start]

    weighted_sum = 0.0
    total_prob = 0.0
    for candidate in score_token.top_logprobs:
        if candidate.logprob < _MIN_LOGPROB or not candidate.token.isdecimal():
            continue
        if int(candidate.token) > score_max:
            continue

        if candidate.token == score_token.token:
            # The emitted score, including the rest of its span.
            score = raw_score
            logprob = candidate.logprob + sum(
                t.logprob for t in tokens[start + 1 : start + length]
            )
        elif _is_prefix_of_larger_score(candidate.token, score_max):
            continue
        else:
            score = int(candidate.token)
            logprob = candidate.logprob
            if score < score_min:
                # e.g. "0" under a 1-5 rubric would pull the score below 1
                continue

        prob = math.exp(logprob)
        weighted_sum += score * prob
        total_prob += prob

    if total_prob == 0:
        return raw_score
    return weighted_sum / total_prob


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
