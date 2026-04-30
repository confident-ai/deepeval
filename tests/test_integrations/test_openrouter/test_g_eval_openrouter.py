import os
import pytest
from deepeval.metrics import GEval
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval import evaluate

# ---------------------------------------------------------------------------
# Guard: skip the entire module if OPENROUTER_API_KEY is not configured
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    os.getenv("OPENROUTER_API_KEY") is None
    or not os.getenv("OPENROUTER_API_KEY").strip(),
    reason="OPENROUTER_API_KEY is not set",
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
# Override the model name via the env var OPENROUTER_TEST_MODEL, or fall back
# to a free/cheap default that is available on OpenRouter.
_MODEL_NAME = os.getenv("OPENROUTER_TEST_MODEL", "openai/gpt-4o-mini")


def _make_model() -> OpenRouterModel:
    """Return a fresh OpenRouterModel instance for each test."""
    return OpenRouterModel(model=_MODEL_NAME)


def _make_test_case() -> LLMTestCase:
    return LLMTestCase(
        input="What if these shoes don't fit?",
        expected_output="We offer a 30-day full refund at no extra cost.",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra cost."
        ],
        context=[
            "All customers are eligible for a 30 day full refund at no extra cost."
        ],
        tools_called=[
            ToolCall(name="PolicyLookup"),
            ToolCall(name="OrderQuery"),
        ],
        expected_tools=[ToolCall(name="PolicyLookup")],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGEvalOpenRouter:
    """GEval metric integration tests using OpenRouterModel."""

    # ------------------------------------------------------------------
    # 1. Synchronous path — model passed explicitly
    # ------------------------------------------------------------------
    def test_sync_metric_measure_explicit_model(self):
        """measure() works synchronously when OpenRouterModel is passed directly."""
        test_case = _make_test_case()
        metric = GEval(
            name="Relevancy (sync, explicit model)",
            model=_make_model(),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            criteria="Check if the actual output is relevant to the input",
            async_mode=False,
        )
        metric.measure(test_case)

        assert metric.score is not None, "score should not be None"
        assert 0.0 <= metric.score <= 1.0, f"score out of range: {metric.score}"
        assert metric.reason is not None, "reason should not be None"

    # ------------------------------------------------------------------
    # 2. Asynchronous path — model passed explicitly
    # ------------------------------------------------------------------
    def test_async_metric_measure_explicit_model(self):
        """measure() works in async mode when OpenRouterModel is passed directly."""
        test_case = _make_test_case()
        metric = GEval(
            name="Relevancy (async, explicit model)",
            model=_make_model(),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            criteria="Check if the actual output is relevant to the input",
            async_mode=True,
        )
        metric.measure(test_case)

        assert metric.score is not None
        assert 0.0 <= metric.score <= 1.0
        assert metric.reason is not None

    # ------------------------------------------------------------------
    # 3. Verify is_native_model() recognises OpenRouterModel
    # ------------------------------------------------------------------
    def test_openrouter_is_native_model(self):
        """OpenRouterModel instances must be flagged as native by initialize_model()."""
        from deepeval.metrics.utils import initialize_model, is_native_model

        model = _make_model()
        assert is_native_model(model), (
            "OpenRouterModel should be recognised as a native model"
        )

        returned_model, using_native = initialize_model(model)
        assert using_native is True, "initialize_model must return using_native=True"
        assert returned_model is model, (
            "initialize_model should return the same model instance"
        )

    # ------------------------------------------------------------------
    # 4. Cost tracking — OpenRouterModel.calculate_cost() returns a value
    # ------------------------------------------------------------------
    def test_cost_is_tracked(self):
        """After measure(), total_cost should be a finite float or None (no crash)."""
        test_case = _make_test_case()
        metric = GEval(
            name="Relevancy (cost check)",
            model=_make_model(),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            criteria="Check if the actual output is relevant to the input",
            async_mode=False,
        )
        metric.measure(test_case)

        # total_cost may be None if OpenRouter does not return pricing info
        # but it must never raise and must be a number if present
        if metric.total_cost is not None:
            assert isinstance(metric.total_cost, float)
            assert metric.total_cost >= 0.0

    # ------------------------------------------------------------------
    # 5. evaluate() helper — batch evaluation works end-to-end
    # ------------------------------------------------------------------
    def test_evaluate_helper(self):
        """deepeval.evaluate() runs cleanly with an OpenRouter-backed GEval metric."""
        test_case = _make_test_case()
        metric = GEval(
            name="Relevancy (evaluate helper)",
            model=_make_model(),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            criteria="Check if the actual output is relevant to the input",
        )

        results = evaluate([test_case], [metric])
        assert results is not None
