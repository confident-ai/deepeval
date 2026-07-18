from typing import List, Optional, Union, Tuple, Dict
from openai.types.chat.chat_completion import ChatCompletion
import math
import re

from deepeval.models import DeepEvalBaseLLM, GPTModel, AzureOpenAIModel
from deepeval.test_case import (
    SingleTurnParams,
    MultiTurnParams,
    LLMTestCase,
    RetrievedContextData,
    ToolCall,
)
from pydantic import BaseModel, Field, field_validator
from deepeval.models.llms.constants import OPENAI_MODELS_DATA

from deepeval.test_case.conversational_test_case import ConversationalTestCase


class APIRubric(BaseModel):
    scoreRange: Tuple[float, float]
    expectedOutcome: str


class MetricPullResponse(BaseModel):
    id: Optional[str] = None
    criteria: Optional[str] = None
    evaluationSteps: Optional[List[str]] = None
    requiredParameters: List[str] = Field(default_factory=list)
    rubric: Optional[List[APIRubric]] = None


class RetrievalContextChunkBudget(BaseModel):
    index: int
    source: Optional[str]
    original_tokens: int
    rendered_tokens: int
    relevance_score: float = 0.0
    omitted: bool = False


class RetrievalContextEvidenceCoverage(BaseModel):
    query_terms_count: int
    covered_terms: List[str]
    missing_terms: List[str]
    coverage_ratio: float
    warning: Optional[str] = None


class RetrievalContextBudgetReport(BaseModel):
    original_tokens: int
    rendered_tokens: int
    budget_tokens: int
    total_chunks: int
    visible_chunks: int
    omitted_chunks: int
    compression_ratio: float
    exceeded_budget: bool
    rendered_context: str
    chunks: List[RetrievalContextChunkBudget]
    evidence_coverage: RetrievalContextEvidenceCoverage


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
        isinstance(model, GPTModel) and not model.model_data.supports_log_probs
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


TOKEN_CHAR_RATIO = 4
MIN_CONTEXT_WINDOW_TOKENS = 32
RELEVANCE_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "does",
    "from",
    "have",
    "into",
    "only",
    "should",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def estimate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / TOKEN_CHAR_RATIO))


def _normalize_retrieval_context_item(
    item: Union[str, RetrievedContextData],
) -> Tuple[Optional[str], str]:
    if isinstance(item, RetrievedContextData):
        return item.source, item.context
    return None, str(item)


def _relevance_terms(text: Optional[str]) -> set[str]:
    if not text:
        return set()

    return {
        _normalize_relevance_term(term)
        for term in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        if _normalize_relevance_term(term) not in RELEVANCE_STOPWORDS
    }


def _normalize_relevance_term(term: str) -> str:
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    return term


def _retrieval_context_relevance_score(
    context: str,
    relevance_terms: set[str],
) -> float:
    if not relevance_terms:
        return 0.0

    context_terms = _relevance_terms(context)
    if not context_terms:
        return 0.0

    overlap = context_terms & relevance_terms
    coverage = len(overlap) / len(relevance_terms)
    density = len(overlap) / max(1, len(context_terms))
    return round(coverage + density, 4)


def _build_evidence_coverage(
    original_context: str,
    rendered_context: str,
    relevance_terms: set[str],
) -> RetrievalContextEvidenceCoverage:
    if not relevance_terms:
        return RetrievalContextEvidenceCoverage(
            query_terms_count=0,
            covered_terms=[],
            missing_terms=[],
            coverage_ratio=1.0,
        )

    original_terms = _relevance_terms(original_context)
    rendered_terms = _relevance_terms(rendered_context)
    evidence_terms = relevance_terms & original_terms
    if not evidence_terms:
        return RetrievalContextEvidenceCoverage(
            query_terms_count=len(relevance_terms),
            covered_terms=[],
            missing_terms=[],
            coverage_ratio=1.0,
        )

    covered_terms = sorted(evidence_terms & rendered_terms)
    missing_terms = sorted(evidence_terms - rendered_terms)
    coverage_ratio = round(len(covered_terms) / len(evidence_terms), 4)
    warning = None
    if missing_terms:
        warning = (
            "Some relevance-query terms appeared in the original retrieval "
            "context but were not present after GEval compaction."
        )
    return RetrievalContextEvidenceCoverage(
        query_terms_count=len(relevance_terms),
        covered_terms=covered_terms,
        missing_terms=missing_terms,
        coverage_ratio=coverage_ratio,
        warning=warning,
    )


def build_retrieval_relevance_query(
    evaluation_params: List[SingleTurnParams],
    test_case: LLMTestCase,
) -> str:
    query_parts: List[str] = []
    for param in (
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ):
        if param not in evaluation_params:
            continue
        value = getattr(test_case, param.value, None)
        if value:
            query_parts.append(str(value))

    return "\n".join(query_parts)


def _truncate_middle(text: str, max_tokens: int, label: str) -> str:
    max_chars = max_tokens * TOKEN_CHAR_RATIO
    if len(text) <= max_chars:
        return text

    marker = (
        f"\n\n[... omitted ~{estimate_token_count(text) - max_tokens} "
        f"tokens from {label} to fit GEval retrieval context budget ...]\n\n"
    )
    if max_chars <= len(marker) + 16:
        return text[:max_chars]

    remaining_chars = max_chars - len(marker)
    head_chars = math.ceil(remaining_chars * 0.6)
    tail_chars = remaining_chars - head_chars

    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def format_retrieval_context_with_budget(
    retrieval_context: List[Union[str, RetrievedContextData]],
    max_retrieval_context_tokens: int,
    relevance_query: Optional[str] = None,
) -> str:
    return build_retrieval_context_budget_report(
        retrieval_context,
        max_retrieval_context_tokens,
        relevance_query=relevance_query,
    ).rendered_context


def build_retrieval_context_budget_report(
    retrieval_context: List[Union[str, RetrievedContextData]],
    max_retrieval_context_tokens: int,
    relevance_query: Optional[str] = None,
) -> RetrievalContextBudgetReport:
    if max_retrieval_context_tokens <= 0:
        raise ValueError("max_retrieval_context_tokens must be greater than 0.")

    relevance_terms = _relevance_terms(relevance_query)
    normalized_contexts = [
        (
            index,
            *_normalize_retrieval_context_item(item),
        )
        for index, item in enumerate(retrieval_context, start=1)
    ]
    total_tokens = sum(
        estimate_token_count(context) for _, _, context in normalized_contexts
    )
    original_context = "\n\n".join(
        context for _, _, context in normalized_contexts
    )

    if total_tokens <= max_retrieval_context_tokens:
        rendered_context = str(retrieval_context)
        rendered_tokens = estimate_token_count(rendered_context)
        return RetrievalContextBudgetReport(
            original_tokens=total_tokens,
            rendered_tokens=rendered_tokens,
            budget_tokens=max_retrieval_context_tokens,
            total_chunks=len(normalized_contexts),
            visible_chunks=len(normalized_contexts),
            omitted_chunks=0,
            compression_ratio=1.0,
            exceeded_budget=False,
            rendered_context=rendered_context,
            chunks=[
                RetrievalContextChunkBudget(
                    index=index,
                    source=source,
                    original_tokens=estimate_token_count(context),
                    rendered_tokens=estimate_token_count(context),
                    relevance_score=_retrieval_context_relevance_score(
                        context, relevance_terms
                    ),
                    omitted=False,
                )
                for index, source, context in normalized_contexts
            ],
            evidence_coverage=_build_evidence_coverage(
                original_context,
                rendered_context,
                relevance_terms,
            ),
        )

    if not normalized_contexts:
        rendered_context = str(retrieval_context)
        return RetrievalContextBudgetReport(
            original_tokens=0,
            rendered_tokens=estimate_token_count(rendered_context),
            budget_tokens=max_retrieval_context_tokens,
            total_chunks=0,
            visible_chunks=0,
            omitted_chunks=0,
            compression_ratio=1.0,
            exceeded_budget=False,
            rendered_context=rendered_context,
            chunks=[],
            evidence_coverage=_build_evidence_coverage(
                original_context,
                rendered_context,
                relevance_terms,
            ),
        )

    context_count = len(normalized_contexts)
    visible_context_count = min(
        context_count,
        max(1, max_retrieval_context_tokens // MIN_CONTEXT_WINDOW_TOKENS),
    )
    scored_contexts = [
        (
            index,
            source,
            context,
            _retrieval_context_relevance_score(context, relevance_terms),
        )
        for index, source, context in normalized_contexts
    ]
    ranked_contexts = sorted(
        scored_contexts,
        key=lambda item: (-item[3], item[0]),
    )
    selected_indices = {
        index for index, _, _, _ in ranked_contexts[:visible_context_count]
    }
    visible_contexts = [
        item for item in scored_contexts if item[0] in selected_indices
    ]
    omitted_context_count = context_count - visible_context_count
    context_token_budget = max(
        1,
        max_retrieval_context_tokens // visible_context_count,
    )

    rendered_contexts = [
        (
            "[retrieval_context compacted for GEval: "
            f"estimated {total_tokens} tokens across {context_count} chunks; "
            f"budget {max_retrieval_context_tokens} tokens; "
            f"kept {visible_context_count} highest-relevance chunks]"
        )
    ]
    chunk_reports: List[RetrievalContextChunkBudget] = []
    for index, source, context, relevance_score in visible_contexts:
        label = f"retrieval chunk {index}"
        source_label = f" source={source}" if source else ""
        rendered_chunk = _truncate_middle(context, context_token_budget, label)
        rendered_contexts.append(
            f"[{index}{source_label}]\n" f"{rendered_chunk}"
        )
        chunk_reports.append(
            RetrievalContextChunkBudget(
                index=index,
                source=source,
                original_tokens=estimate_token_count(context),
                rendered_tokens=estimate_token_count(rendered_chunk),
                relevance_score=relevance_score,
                omitted=estimate_token_count(context)
                > estimate_token_count(rendered_chunk),
            )
        )

    if omitted_context_count > 0:
        rendered_contexts.append(
            f"[... omitted {omitted_context_count} retrieval chunks because "
            "the GEval retrieval context budget was reached ...]"
        )
        for index, source, context, relevance_score in scored_contexts:
            if index in selected_indices:
                continue
            chunk_reports.append(
                RetrievalContextChunkBudget(
                    index=index,
                    source=source,
                    original_tokens=estimate_token_count(context),
                    rendered_tokens=0,
                    relevance_score=relevance_score,
                    omitted=True,
                )
            )

    rendered_context = "\n\n".join(rendered_contexts)
    rendered_tokens = estimate_token_count(rendered_context)
    return RetrievalContextBudgetReport(
        original_tokens=total_tokens,
        rendered_tokens=rendered_tokens,
        budget_tokens=max_retrieval_context_tokens,
        total_chunks=context_count,
        visible_chunks=visible_context_count,
        omitted_chunks=omitted_context_count,
        compression_ratio=(
            round(rendered_tokens / total_tokens, 4)
            if total_tokens > 0
            else 1.0
        ),
        exceeded_budget=True,
        rendered_context=rendered_context,
        chunks=chunk_reports,
        evidence_coverage=_build_evidence_coverage(
            original_context,
            rendered_context,
            relevance_terms,
        ),
    )


def construct_test_case_string(
    evaluation_params: List[SingleTurnParams],
    test_case: LLMTestCase,
    max_retrieval_context_tokens: Optional[int] = None,
) -> str:
    text = """"""
    for param in evaluation_params:
        value = getattr(test_case, param.value)
        if isinstance(value, ToolCall):
            value = repr(value)
        elif (
            param == SingleTurnParams.RETRIEVAL_CONTEXT
            and max_retrieval_context_tokens is not None
            and isinstance(value, list)
        ):
            value = format_retrieval_context_with_budget(
                value,
                max_retrieval_context_tokens,
                relevance_query=build_retrieval_relevance_query(
                    evaluation_params, test_case
                ),
            )
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


def get_score_range(rubric: Optional[List[Rubric]]) -> Tuple[int, int]:
    if rubric is None:
        return (0, 10)

    return rubric[0].score_range[0], rubric[-1].score_range[1]
