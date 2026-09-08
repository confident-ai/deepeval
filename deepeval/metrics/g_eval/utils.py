from typing import Dict, Iterable, List, Optional, Tuple, Union
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
    """A score band and the outcome it represents.

    Bounds are arbitrary finite numbers, so a rubric can be expressed on any
    scale — 0-10, 0-1 with decimals, 1-5, 0-100. The metric normalizes whatever
    the judge returns to 0-1 using the rubric's overall span, so the scale is a
    presentation choice for the judge rather than something callers must undo.
    """

    score_range: Tuple[float, float]
    expected_outcome: str

    @field_validator("score_range")
    def validate_score_range(cls, value):
        start, end = value
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(
                "Both Rubric's 'score_range' values must be finite numbers."
            )
        if start > end:
            raise ValueError(
                "Rubric's 'score_range' start must be less than or equal to end."
            )
        return value


def is_integral_rubric_scale(rubric: Optional[List[Rubric]]) -> bool:
    """Whether *every* rubric bound is a whole number.

    Note this looks at all bands, not just the outer bounds: a 0-1 rubric split
    at `0.3`/`0.4` spans a whole-numbered 0.0 to 1.0 but is still a decimal
    scale, and the judge must be asked for a decimal accordingly.

    Also gates the log-prob weighting: `calculate_weighted_summed_score` matches
    the score token by `str(raw_score)` and calls `int(token)`, so it only ever
    works on integer scores.
    """
    if rubric is None:
        return True

    return all(
        float(bound).is_integer() for r in rubric for bound in r.score_range
    )


def _as_number(value: float) -> Union[int, float]:
    """`0.0` -> `0`, `0.5` -> `0.5`."""
    return int(value) if float(value).is_integer() else float(value)


def _render_bounds(
    values: Iterable[float], integral: bool
) -> Tuple[Union[int, float], ...]:
    """Render rubric bounds consistently across a whole scale.

    Bounds are floats internally, but an integral scale must keep rendering and
    serializing as ints — `0-5` in a prompt, `[0, 5]` in an upload payload —
    since emitting `0.0-5.0` would silently change every existing prompt. A
    decimal scale stays decimal throughout so `0.0-0.3` and `1.0` don't sit
    next to a bare `1`.
    """
    return tuple(
        _as_number(value) if integral else float(value) for value in values
    )


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
        integral = is_integral_rubric_scale(rubric)
        payload["rubric"] = [
            {
                "scoreRange": list(_render_bounds(r.score_range, integral)),
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

    integral = is_integral_rubric_scale(rubrics)

    return "\n".join(
        (
            f"{start}: {rubric.expected_outcome}"
            if start == end
            else f"{start}-{end}: {rubric.expected_outcome}"
        )
        for rubric in rubrics
        for start, end in [_render_bounds(rubric.score_range, integral)]
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


def calculate_weighted_summed_score(
    raw_score: int, raw_response: ChatCompletion
) -> Union[int, float]:
    try:
        generated_logprobs = raw_response.choices[0].logprobs.content
        # First, locate the final token matching the score. The reasoning may
        # contain the same token before the model emits its final score.
        score_logprobs = None
        for token_logprobs in reversed(generated_logprobs):
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

        # If all tokens were filtered out, fall back to the raw score
        if sum_linear_probability == 0:
            return raw_score

        # Scale the sum of linear probability to 1
        weighted_summed_score = sum_of_weighted_scores / sum_linear_probability
        return weighted_summed_score
    except Exception:
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


def get_score_range(
    rubric: Optional[List[Rubric]],
) -> Tuple[Union[int, float], Union[int, float]]:
    """The scale the judge is asked to score on: the rubric's outer bounds.

    Values go straight into the prompt, so they are rendered to match the
    rubric's own scale.
    """
    if rubric is None:
        return (0, 10)

    start, end = rubric[0].score_range[0], rubric[-1].score_range[1]
    return _render_bounds((start, end), is_integral_rubric_scale(rubric))


def normalize_score(
    raw_score: Union[int, float], score_range: Tuple[float, float]
) -> float:
    """Map a judge score on `score_range` onto 0-1, clamped."""
    start, end = float(score_range[0]), float(score_range[1])
    span = end - start
    if span <= 0:
        # Degenerate rubric collapsed to a single point — nothing to interpolate.
        return 1.0 if float(raw_score) >= start else 0.0

    return min(1.0, max(0.0, (float(raw_score) - start) / span))
