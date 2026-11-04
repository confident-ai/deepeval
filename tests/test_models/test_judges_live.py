"""Live checks for every judge whose registry entry or request handling changed.

These call the real provider APIs and cost money, so they only run with
``DEEPEVAL_LIVE_JUDGE_TESTS=1`` plus the provider's API key:

    DEEPEVAL_LIVE_JUDGE_TESTS=1 OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \
    GOOGLE_API_KEY=... pytest tests/test_models/test_judges_live.py -v
"""

import os
from typing import List

import pytest
from pydantic import BaseModel

from deepeval.metrics import FaithfulnessMetric, NonAdviceMetric
from deepeval.models.llms.anthropic_model import AnthropicModel
from deepeval.models.llms.gemini_model import GeminiModel
from deepeval.models.llms.openai_model import OpenAIModel
from deepeval.test_case import LLMTestCase

pytestmark = pytest.mark.skipif(
    os.getenv("DEEPEVAL_LIVE_JUDGE_TESTS") != "1",
    reason="set DEEPEVAL_LIVE_JUDGE_TESTS=1 to call real provider APIs",
)


def _needs(env_var):
    return pytest.mark.skipif(
        not (os.getenv(env_var) or "").strip(), reason=f"{env_var} is not set"
    )


JUDGES = (
    [
        pytest.param(OpenAIModel, name, marks=_needs("OPENAI_API_KEY"), id=name)
        for name in (
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
        )
    ]
    + [
        pytest.param(GeminiModel, name, marks=_needs("GOOGLE_API_KEY"), id=name)
        for name in (
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-2.5-flash",
        )
    ]
    + [
        pytest.param(
            AnthropicModel, name, marks=_needs("ANTHROPIC_API_KEY"), id=name
        )
        for name in (
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        )
    ]
)


class _Verdict(BaseModel):
    index: int
    verdict: str
    reason: str


class _Verdicts(BaseModel):
    verdicts: List[_Verdict]


VERDICT_COUNT = 60
LONG_VERDICT_PROMPT = (
    f"Return a JSON object with a 'verdicts' list of exactly {VERDICT_COUNT} "
    "items. Item i has 'index' = i (0-based), 'verdict' = 'yes' when i is "
    "even and 'no' when i is odd, and a 'reason' of one full sentence "
    "explaining the verdict."
)


def _check_cost(model, cost):
    """Cost must be priced from the registry, not left unknown."""
    assert cost is not None and cost > 0
    assert cost.input_tokens and cost.output_tokens
    expected = (
        cost.input_tokens * model.model_data.input_price
        + cost.output_tokens * model.model_data.output_price
    )
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize("model_cls, model_name", JUDGES)
def test_judge_returns_long_structured_verdict_list(model_cls, model_name):
    """A long verdict list parses without truncation, costs are reported, and
    the provider accepts the request as DeepEval builds it (no rejected
    temperature, and for Claude no fallback off native structured outputs)."""
    model = model_cls(model=model_name)

    output, cost = model.generate(LONG_VERDICT_PROMPT, _Verdicts)

    assert isinstance(output, _Verdicts)
    assert len(output.verdicts) == VERDICT_COUNT
    assert [v.index for v in output.verdicts] == list(range(VERDICT_COUNT))
    _check_cost(model, cost)
    if isinstance(model, AnthropicModel):
        assert model._structured_outputs_rejected is False


@pytest.mark.parametrize("model_cls, model_name", JUDGES)
async def test_judge_async_generate(model_cls, model_name):
    model = model_cls(model=model_name)

    output, cost = await model.a_generate(
        "Return verdicts for 3 items as described: index 0-2, all 'yes'.",
        _Verdicts,
    )

    assert isinstance(output, _Verdicts)
    assert len(output.verdicts) == 3
    _check_cost(model, cost)


@pytest.mark.parametrize("model_cls, model_name", JUDGES)
def test_judge_runs_non_advice_metric(model_cls, model_name):
    """The configured advice type decides the verdict: crypto advice is
    flagged when only 'crypto' is configured, and not when only 'medical' is."""
    case = LLMTestCase(
        input="What should I do with my savings?",
        actual_output=(
            "Put all of your savings into Dogecoin this week, it will 10x."
        ),
    )
    model = model_cls(model=model_name)

    flagged = NonAdviceMetric(
        advice_types=["crypto"], model=model, eval_mode="llm", async_mode=False
    )
    flagged.measure(case)
    ignored = NonAdviceMetric(
        advice_types=["medical"],
        model=model,
        eval_mode="llm",
        async_mode=False,
    )
    ignored.measure(case)

    assert flagged.score < 1
    assert ignored.score == 1
    assert flagged.evaluation_cost and flagged.evaluation_cost > 0


@pytest.mark.parametrize("model_cls, model_name", JUDGES)
def test_judge_runs_faithfulness_metric(model_cls, model_name):
    model = model_cls(model=model_name)
    metric = FaithfulnessMetric(model=model, eval_mode="llm", async_mode=False)

    metric.measure(
        LLMTestCase(
            input="Tell me about Einstein.",
            actual_output=(
                "Einstein was born in Ulm in 1879. He won the Nobel Prize "
                "in Physics in 1935."
            ),
            retrieval_context=[
                "Albert Einstein was born in Ulm, Germany, on 14 March 1879. "
                "He received the 1921 Nobel Prize in Physics."
            ],
        )
    )

    assert 0 < metric.score < 1
    assert metric.evaluation_cost and metric.evaluation_cost > 0
