"""
Agent Evaluation Example with DeepEval

This example demonstrates how to evaluate an AI agent using:
1. @observe decorators for tracing agent, tool, and LLM components
2. update_current_span for creating test cases at runtime
3. evals_iterator for running evaluations over a dataset of Goldens
4. Trace-level metrics (TaskCompletionMetric) - evaluates the entire agent
5. Span-level metrics (AnswerRelevancyMetric) - evaluates individual components
6. Custom GEval for domain-specific evaluation criteria

Run: python test_agent_eval.py
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    TaskCompletionMetric,
    AnswerRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_span


# ── Knowledge Base (simulates a vector DB) ─────────────

KNOWLEDGE_BASE = {
    "refund": "Refunds are available within 30 days of purchase. Items must be unused.",
    "shipping": "Standard shipping: 5-7 days. Express: 1-2 days ($9.99).",
    "hours": "Support hours: Mon-Fri, 9am-5pm EST.",
    "return": "Email returns@example.com with your order number for a prepaid label.",
}


# ── Observed Agent Components ──────────────────────────
#
# @observe creates spans that form a trace. The nesting of
# function calls determines the trace structure:


@observe(type="tool")
def retrieve_context(query: str) -> str:
    """Search the knowledge base. Traced as a tool span."""
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return value
    return "No relevant information found."


@observe(metrics=[AnswerRelevancyMetric(threshold=0.5)])
def generate_response(query: str, context: str) -> str:
    """Generate a response from retrieved context.

    AnswerRelevancyMetric is attached at the span level — it evaluates
    THIS component's output, not the entire agent trace.

    In a real app, replace this with an actual LLM call:
        response = openai.chat.completions.create(...)
    """
    response = f"Based on our records: {context}"

    # Create the test case for this span's metric evaluation.
    # update_current_span tells DeepEval what input/output to evaluate.
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=[context],
        )
    )
    return response


@observe(type="agent")
def support_agent(query: str) -> str:
    """Customer support agent — trace root.

    TaskCompletionMetric evaluates the entire trace (all spans together).
    It's passed to evals_iterator(), not to @observe().
    """
    context = retrieve_context(query)
    response = generate_response(query, context)
    return response


# ── Metrics ────────────────────────────────────────────

task_completion = TaskCompletionMetric(threshold=0.5)

# Custom GEval: is the response grounded in facts?
groundedness = GEval(
    name="Response Groundedness",
    criteria=(
        "Evaluate whether the response only contains information "
        "that could be derived from a customer support knowledge base. "
        "The response should not fabricate policies, prices, or deadlines."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.5,
)


# ── Evaluation Dataset ─────────────────────────────────

goldens = [
    Golden(input="What is your refund policy?"),
    Golden(input="How long does shipping take?"),
    Golden(input="When is customer support available?"),
    Golden(input="How do I return an item?"),
    Golden(input="How much does express shipping cost?"),
]

dataset = EvaluationDataset(goldens=goldens)


# ── Option 1: Component-level eval with evals_iterator ─
#
# This is the recommended approach for agent evaluation.
# - Trace-level metrics go in evals_iterator(metrics=[...])
# - Span-level metrics go in @observe(metrics=[...])


def run_component_level():
    """Run component-level evaluation using evals_iterator."""
    print("Running component-level agent evaluation...")
    print(f"  Dataset: {len(goldens)} goldens")
    print(f"  Trace-level: TaskCompletionMetric")
    print(f"  Span-level:  AnswerRelevancyMetric (on generate_response)")
    print("-" * 50)

    for golden in dataset.evals_iterator(metrics=[task_completion]):
        support_agent(golden.input)


# ── Option 2: End-to-end eval with pytest ──────────────
#
# For CI/CD integration using `deepeval test run`.
# This treats the agent as a black box.

e2e_test_cases = [
    LLMTestCase(
        input="What is your refund policy?",
        actual_output=support_agent("What is your refund policy?"),
        expected_output="Refunds are available within 30 days of purchase.",
    ),
    LLMTestCase(
        input="How long does shipping take?",
        actual_output=support_agent("How long does shipping take?"),
        expected_output="Standard shipping takes 5-7 days. Express is 1-2 days for $9.99.",
    ),
]

e2e_dataset = EvaluationDataset(test_cases=e2e_test_cases)


@pytest.mark.parametrize("test_case", e2e_dataset)
def test_agent_e2e(test_case: LLMTestCase):
    """End-to-end pytest test for CI/CD pipelines."""
    assert_test(test_case, [task_completion, groundedness])


# ── Main ───────────────────────────────────────────────

if __name__ == "__main__":
    run_component_level()
