"""
Agent Evaluation Example with DeepEval v3.0

This example demonstrates how to evaluate an AI agent using:
1. TaskCompletionMetric - Did the agent complete the user's task?
2. AnswerRelevancyMetric - Is the response relevant?
3. Custom GEval - Domain-specific evaluation criteria
4. Component-level evaluation with @observe

Run: python test_agent_eval.py
  or: deepeval test run test_agent_eval.py
"""

import pytest
from deepeval import assert_test, evaluate
from deepeval.metrics import (
    TaskCompletionMetric,
    AnswerRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_span


# ── Simulated Agent Components ─────────────────────────
# Replace these with your actual agent's retriever, LLM, tools, etc.


def mock_retriever(query: str) -> str:
    """Simulates a retrieval step that fetches context."""
    knowledge = {
        "refund": "Refunds are available within 30 days of purchase.",
        "shipping": "Standard shipping: 5-7 days. Express: 1-2 days ($9.99).",
        "hours": "Support hours: Mon-Fri, 9am-5pm EST.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No relevant information found."


def mock_agent(query: str) -> str:
    """Simulates an agent that retrieves context and generates a response."""
    context = mock_retriever(query)
    # In a real agent, this would be an LLM call using the context
    return f"Based on our records: {context}"


# ── Define Metrics ─────────────────────────────────────

task_completion = TaskCompletionMetric(threshold=0.5)

answer_relevancy = AnswerRelevancyMetric(threshold=0.5)

# Custom metric: evaluate whether the response is grounded in facts (not hallucinating beyond what the retriever returned)
groundedness = GEval(
    name="Response Groundedness",
    criteria=(
        "Evaluate whether the response only contains information that "
        "could be derived from a customer support knowledge base. "
        "The response should not make up policies, prices, or deadlines "
        "that aren't typically found in standard business documentation."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.5,
)


# ── Test Cases ─────────────────────────────────────────

test_cases = [
    LLMTestCase(
        input="What is your refund policy?",
        actual_output=mock_agent("What is your refund policy?"),
        expected_output="Refunds are available within 30 days of purchase.",
    ),
    LLMTestCase(
        input="How long does shipping take?",
        actual_output=mock_agent("How long does shipping take?"),
        expected_output="Standard shipping takes 5-7 days. Express is 1-2 days for $9.99.",
    ),
    LLMTestCase(
        input="When is customer support available?",
        actual_output=mock_agent("When is customer support available?"),
        expected_output="Support is available Monday through Friday, 9am to 5pm EST.",
    ),
]


# ── Option 1: Run with evaluate() ─────────────────────


def run_with_evaluate():
    """Run evaluation using the evaluate() function."""
    print("Running agent evaluation with evaluate()...")
    evaluate(
        test_cases=test_cases,
        metrics=[task_completion, answer_relevancy, groundedness],
    )


# ── Option 2: Run with pytest (deepeval test run) ─────

dataset = EvaluationDataset(test_cases=test_cases)


@pytest.mark.parametrize("test_case", dataset)
def test_agent(test_case: LLMTestCase):
    """Pytest-compatible test for CI/CD integration."""
    assert_test(
        test_case,
        [task_completion, answer_relevancy, groundedness],
    )


# ── Option 3: Component-level evaluation with @observe ─


@observe(metrics=[answer_relevancy])
def traced_retriever(query: str):
    """Evaluate the retriever component independently."""
    result = mock_retriever(query)
    update_current_span(
        test_case=LLMTestCase(input=query, actual_output=result)
    )
    return result


@observe(metrics=[task_completion])
def traced_agent(query: str):
    """Evaluate the full agent pipeline."""
    context = traced_retriever(query)
    response = f"Based on our records: {context}"
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
            expected_output="A helpful, grounded response.",
        )
    )
    return response


def run_component_level():
    """Run component-level evaluation with tracing."""
    print("Running component-level agent evaluation...")
    goldens = [
        Golden(input="What is your refund policy?"),
        Golden(input="How long does shipping take?"),
    ]
    evaluate(
        observed_callback=traced_agent,
        goldens=goldens,
    )


if __name__ == "__main__":
    run_with_evaluate()
    print("\n" + "=" * 50 + "\n")
    run_component_level()
