"""Context-budget pre-check for System One requests.

Jev has a bounded context window and the SDK ships neither a tokenizer nor a
dedicated exception for overflow (it surfaces as a 400/422 the retry policy
rightly refuses to retry). Estimating the size before the call gives the user a
clear "too large for Jev" error instead of an opaque 400/422.
`check_drex_budget` does the same for a Drex model, whose limits differ per
model and also cap the number of questions.
"""

import json
from typing import Any, Dict, Optional

from deepeval.errors import ContextWindowExceededError
from deepeval.models.system_one.constants import (
    DREX_TOKENS_PER_IMAGE,
    JEV_MAX_REQUEST_TOKENS,
    JEV_MAX_STATE_TOKENS,
    DrexLimits,
)

# Characters per token. English prose averages roughly 4; JSON with its
# punctuation and short keys tokenizes denser, so 3 keeps the estimate on the
# high side and the pre-check conservative.
_CHARS_PER_TOKEN = 3


class SystemOneContextLimitError(ContextWindowExceededError):

    def __init__(
        self,
        message: str,
        *,
        estimated_tokens: int,
        limit_tokens: Optional[int],
    ):
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.limit_tokens = limit_tokens


def estimate_tokens(value: Any) -> int:
    """Upper-bound token estimate for a JSON-serialisable value."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return -(-len(text) // _CHARS_PER_TOKEN)  # ceil division


def check_context_budget(
    state: Any,
    questions: Dict[str, Any],
    *,
    max_state_tokens: int = JEV_MAX_STATE_TOKENS,
    max_request_tokens: int = JEV_MAX_REQUEST_TOKENS,
) -> None:
    """Raise ``SystemOneContextLimitError`` if the request is likely to exceed
    the model's context budget. ``questions`` are the SDK-shaped dicts or the
    pydantic question models; either serialises."""
    state_tokens = estimate_tokens(state)
    question_tokens = [
        estimate_tokens(_dumpable(q)) for q in questions.values()
    ]
    longest = max(question_tokens) if question_tokens else 0
    if state_tokens + longest > max_state_tokens:
        raise SystemOneContextLimitError(
            f"System One state ({state_tokens} est. tokens) plus the longest "
            f"question ({longest}) exceeds the {max_state_tokens}-token state "
            f"budget. Send less state or judge this test case with the LLM.",
            estimated_tokens=state_tokens + longest,
            limit_tokens=max_state_tokens,
        )
    total = state_tokens + sum(question_tokens)
    if total > max_request_tokens:
        raise SystemOneContextLimitError(
            f"System One request ({total} est. tokens across "
            f"{len(question_tokens)} questions) exceeds the "
            f"{max_request_tokens}-token request budget. Ask fewer questions "
            f"per request or judge this test case with the LLM.",
            estimated_tokens=total,
            limit_tokens=max_request_tokens,
        )


class SystemOneQuestionLimitError(SystemOneContextLimitError):
    """More questions than the model answers in one request."""

    def __init__(
        self,
        message: str,
        *,
        question_count: int,
        max_questions: int,
        estimated_tokens: int,
    ):
        super().__init__(
            message, estimated_tokens=estimated_tokens, limit_tokens=None
        )
        self.question_count = question_count
        self.max_questions = max_questions


def check_drex_budget(
    state: Any,
    questions: Dict[str, Any],
    *,
    model: str,
    limits: DrexLimits,
    image_count: int = 0,
) -> None:
    """``check_context_budget`` for a Drex model: its question cap, its state
    and row limits, and the image tokens Drex Max adds to every question
    (https://console.nace.ai/docs/drex/reference/limits)."""
    state_tokens = estimate_tokens(state)
    question_tokens = [
        estimate_tokens(_dumpable(q)) for q in questions.values()
    ]
    image_tokens = image_count * DREX_TOKENS_PER_IMAGE
    if len(question_tokens) > limits.max_questions:
        raise SystemOneQuestionLimitError(
            f"{model} answers at most {limits.max_questions} questions per "
            f"request; this one has {len(question_tokens)}. Ask fewer "
            f"questions per request or judge this test case with the LLM.",
            question_count=len(question_tokens),
            max_questions=limits.max_questions,
            estimated_tokens=state_tokens + sum(question_tokens),
        )
    if state_tokens > limits.max_state_tokens:
        raise SystemOneContextLimitError(
            f"System One state ({state_tokens} est. tokens) exceeds the "
            f"{limits.max_state_tokens}-token state limit of {model}. Send "
            f"less state or judge this test case with the LLM.",
            estimated_tokens=state_tokens,
            limit_tokens=limits.max_state_tokens,
        )
    longest = max(question_tokens) if question_tokens else 0
    row = state_tokens + longest + image_tokens
    if row > limits.max_row_tokens:
        raise SystemOneContextLimitError(
            f"System One state plus the longest question"
            f"{' and the images' if image_count else ''} ({row} est. tokens) "
            f"exceeds the {limits.max_row_tokens}-token limit of {model}. "
            f"Send less state or judge this test case with the LLM.",
            estimated_tokens=row,
            limit_tokens=limits.max_row_tokens,
        )
    total = state_tokens + sum(question_tokens) + image_tokens
    if (
        limits.max_request_tokens is not None
        and total > limits.max_request_tokens
    ):
        raise SystemOneContextLimitError(
            f"System One request ({total} est. tokens across "
            f"{len(question_tokens)} questions) exceeds the "
            f"{limits.max_request_tokens}-token request limit of {model}. "
            f"Ask fewer questions per request or judge this test case with "
            f"the LLM.",
            estimated_tokens=total,
            limit_tokens=limits.max_request_tokens,
        )


def _dumpable(question: Any) -> Any:
    dump: Optional[Any] = getattr(question, "model_dump", None)
    if callable(dump):
        return dump()
    return question
