"""A metric whose prompt overflows the evaluation model's context window raises
`ContextWindowExceededError` instead of the provider's raw error."""

import asyncio

import anthropic
import httpx
import openai
import pytest
import tenacity
from google.genai import errors as genai_errors

from deepeval.errors import ContextWindowExceededError
from deepeval.metrics import AnswerRelevancyMetric, GEval, TaskCompletionMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


TRACE = {
    "name": "support_agent",
    "type": "agent",
    "input": {"input": "Where is my order #1234?"},
    "output": "Your order is in transit.",
    "children": [],
}


def _response(url: str) -> httpx.Response:
    return httpx.Response(400, request=httpx.Request("POST", url))


def _openai_overflow() -> Exception:
    body = {
        "message": "This model's maximum context length is 128000 tokens. "
        "However, your messages resulted in 300000 tokens.",
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
    }
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {body}}}",
        response=_response("https://api.openai.com/v1/chat/completions"),
        body=body,
    )


def _anthropic_overflow() -> Exception:
    body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 215000 tokens > 200000 maximum",
        },
    }
    return anthropic.BadRequestError(
        f"Error code: 400 - {body}",
        response=_response("https://api.anthropic.com/v1/messages"),
        body=body,
    )


def _gemini_overflow() -> Exception:
    return genai_errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": "The input token count (1200000) exceeds the "
                "maximum number of tokens allowed (1048576).",
                "status": "INVALID_ARGUMENT",
            }
        },
    )


def _exhausted_retry(error: Exception) -> Exception:
    attempt = tenacity.Future(attempt_number=3)
    attempt.set_exception(error)
    retry_error = tenacity.RetryError(attempt)
    retry_error.__cause__ = error
    return retry_error


OVERFLOWS = [
    pytest.param(_openai_overflow, id="openai"),
    pytest.param(_anthropic_overflow, id="anthropic"),
    pytest.param(_gemini_overflow, id="gemini"),
    pytest.param(
        lambda: _exhausted_retry(_openai_overflow()), id="exhausted-retry"
    ),
]


class RaisingLLM(DeepEvalBaseLLM):
    def __init__(self, error: Exception):
        super().__init__("raising-llm")
        self.error = error

    def load_model(self, *args, **kwargs):
        return None

    def get_model_name(self):
        return "raising-llm"

    def generate(self, *args, **kwargs):
        raise self.error

    async def a_generate(self, *args, **kwargs):
        raise self.error


def _test_case(with_trace: bool) -> LLMTestCase:
    test_case = LLMTestCase(
        input="Where is my order #1234?",
        actual_output="Your order is in transit.",
    )
    if with_trace:
        test_case._trace_dict = TRACE
    return test_case


def _measure(metric, test_case, async_mode, show_indicator=True):
    if async_mode:
        return asyncio.run(
            metric.a_measure(test_case, _show_indicator=show_indicator)
        )
    return metric.measure(test_case, _show_indicator=show_indicator)


@pytest.mark.parametrize("show_indicator", [True, False])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("make_error", OVERFLOWS)
def test_provider_overflow_raises_context_window_exceeded(
    make_error, async_mode, show_indicator
):
    error = make_error()
    metric = TaskCompletionMetric(
        model=RaisingLLM(error), async_mode=async_mode
    )

    with pytest.raises(
        ContextWindowExceededError, match="Too large to evaluate"
    ) as info:
        _measure(
            metric, _test_case(with_trace=True), async_mode, show_indicator
        )

    assert info.value.__cause__ is error


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "make_metric, with_trace",
    [
        pytest.param(TaskCompletionMetric, True, id="trace-task-completion"),
        pytest.param(
            lambda **kwargs: GEval(
                name="Helpfulness",
                criteria="Did the agent resolve the request?",
                **kwargs,
            ),
            True,
            id="trace-g-eval",
        ),
        pytest.param(AnswerRelevancyMetric, False, id="answer-relevancy"),
    ],
)
def test_overflow_is_translated_for_every_metric(
    make_metric, with_trace, async_mode
):
    metric = make_metric(
        model=RaisingLLM(_openai_overflow()), async_mode=async_mode
    )

    with pytest.raises(ContextWindowExceededError):
        _measure(metric, _test_case(with_trace), async_mode)


@pytest.mark.parametrize("async_mode", [False, True])
def test_other_errors_propagate_unchanged(async_mode):
    error = openai.BadRequestError(
        "Error code: 400 - invalid 'temperature'",
        response=_response("https://api.openai.com/v1/chat/completions"),
        body=None,
    )
    metric = TaskCompletionMetric(
        model=RaisingLLM(error), async_mode=async_mode
    )

    with pytest.raises(openai.BadRequestError) as info:
        _measure(metric, _test_case(with_trace=True), async_mode)

    assert info.value is error
