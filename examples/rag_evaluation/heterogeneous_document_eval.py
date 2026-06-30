"""Heterogeneous Document RAG Evaluation with DeepEval.

This example demonstrates how to evaluate a financial RAG pipeline that
retrieve chunks from mixed document types (10-K filings, earnings call
transcripts, balance sheets) using document-type-specific thresholds.

Key features shown:
1. Using LLMTestCase metadata to tag chunks by document_type.
2. Using threshold_overrides (per PR #2785) to set different pass/fail
   thresholds per document type.
3. Combining FaithfulnessMetric, ContextualPrecisionMetric, and
   ContextualRecallMetric in a single heterogeneous test run.
4. Interpreting results: why structured documents (balance sheets) need
   higher thresholds than narrative documents (earnings calls).

Requires: OPENAI_API_KEY environment variable.
Install: pip install deepeval
"""

import os
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Document-type threshold configuration
# ---------------------------------------------------------------------------
# Different document types warrant different pass/fail thresholds:
#
# - balance_sheet: High precision required (0.95). Structured numeric data
#   must be retrieved exactly; hallucinated figures are dangerous.
#
# - annual_report (10-K narrative): Moderate threshold (0.80). Some
#   paraphrase of narrative text is acceptable.
#
# - earnings_call: Lower threshold (0.70). Transcripts contain hedged
#   language and forward-looking statements that LLMs may paraphrase.
#
# - default: Applied when document_type metadata is absent (0.75).

THRESHOLD_OVERRIDES = {
    "balance_sheet": 0.95,
    "annual_report": 0.80,
    "earnings_call": 0.70,
    "default": 0.75,
}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def build_test_cases():
    """Build a mixed set of test cases representing a financial RAG pipeline.

    Each test case includes a document_type metadata key that will be used
    by the threshold_overrides parameter (available in DeepEval >= 2.0,
    once PR #2785 is merged).
    """
    test_cases = []

    # --- Balance sheet query ---
    # High precision required: exact numeric retrieval from structured table.
    balance_sheet_case = LLMTestCase(
        input="What were total assets and total liabilities for FY2023?",
        actual_output=(
            "Total assets for FY2023 were $18.7 billion. "
            "Total liabilities were $11.2 billion."
        ),
        expected_output=(
            "Total assets: $18.7B. Total liabilities: $11.2B (FY2023)."
        ),
        retrieval_context=[
            "| Metric | FY2023 | FY2022 |\n"
            "|--------|--------|--------|\n"
            "| Total Assets | $18.7B | $16.4B |\n"
            "| Total Liabilities | $11.2B | $10.1B |\n"
            "| Stockholders' Equity | $7.5B | $6.3B |",
        ],
        # Tag this test case with document type for threshold routing
        additional_metadata={"document_type": "balance_sheet"},
    )
    test_cases.append(balance_sheet_case)

    # --- Annual report (10-K narrative) query ---
    # Moderate precision: narrative paraphrase acceptable.
    annual_report_case = LLMTestCase(
        input="What drove revenue growth in FY2023 according to the 10-K?",
        actual_output=(
            "Revenue growth in FY2023 was primarily driven by the cloud segment, "
            "which grew 34% year-over-year and contributed $1.4B to the total "
            "revenue increase."
        ),
        expected_output=(
            "The cloud segment drove revenue growth, growing 34% YoY "
            "and contributing $1.4B to overall revenue gains."
        ),
        retrieval_context=[
            "Revenue for FY2023 was $4.2 billion, up 12% year-over-year. "
            "The primary growth driver was the cloud segment, which grew 34% "
            "year-over-year and contributed $1.4 billion to the revenue increase. "
            "Enterprise software and services also grew 8%, while legacy hardware "
            "revenue declined 5%.",
        ],
        additional_metadata={"document_type": "annual_report"},
    )
    test_cases.append(annual_report_case)

    # --- Earnings call query ---
    # Lower threshold: forward-looking statements and hedged language.
    earnings_call_case = LLMTestCase(
        input="What guidance did management provide for FY2024 revenue?",
        actual_output=(
            "Management guided for FY2024 revenue in the range of $4.6B to $4.8B, "
            "representing 10-14% growth, subject to macroeconomic conditions."
        ),
        expected_output=(
            "FY2024 revenue guidance: $4.6B–4.8B (10–14% growth), "
            "conditional on macroeconomic environment."
        ),
        retrieval_context=[
            "Looking ahead, we are guiding to fiscal 2024 revenue in the range of "
            "$4.6 to $4.8 billion, which represents growth of approximately 10 to 14 "
            "percent year-over-year. This outlook assumes stable macroeconomic "
            "conditions and does not account for potential FX headwinds beyond "
            "current rates. We feel confident in our pipeline but remain cautious "
            "given the broader environment.",
        ],
        additional_metadata={"document_type": "earnings_call"},
    )
    test_cases.append(earnings_call_case)

    return test_cases


# ---------------------------------------------------------------------------
# Metrics with threshold_overrides
# ---------------------------------------------------------------------------
# Note: threshold_overrides is available once PR #2785 is merged.
# Until then, the metrics will use the default threshold.
# The metadata["document_type"] key is used to select the override.

def build_metrics():
    """Build metrics with document-type threshold overrides.

    threshold_overrides: dict mapping document_type values to thresholds.
    The metric checks test_case.additional_metadata["document_type"] and
    applies the matching threshold, falling back to the default threshold
    if the key is absent or unrecognised.
    """
    faithfulness = FaithfulnessMetric(
        threshold=THRESHOLD_OVERRIDES["default"],
        # threshold_overrides will route to the right threshold per test case
        # threshold_overrides=THRESHOLD_OVERRIDES,  # Uncomment post-PR #2785
        verbose_mode=True,
    )
    precision = ContextualPrecisionMetric(
        threshold=THRESHOLD_OVERRIDES["default"],
        # threshold_overrides=THRESHOLD_OVERRIDES,  # Uncomment post-PR #2785
        verbose_mode=True,
    )
    recall = ContextualRecallMetric(
        threshold=THRESHOLD_OVERRIDES["default"],
        # threshold_overrides=THRESHOLD_OVERRIDES,  # Uncomment post-PR #2785
        verbose_mode=True,
    )
    return [faithfulness, precision, recall]


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_heterogeneous_eval():
    """Run the full heterogeneous document evaluation.

    Results will show per-test-case scores. The key insight:
    - Balance sheet cases should be evaluated at 0.95 threshold.
    - Earnings call cases pass at 0.70 threshold even with hedged language.
    - Without threshold_overrides, all cases use the same threshold,
      leading to false failures on earnings calls or false passes on
      balance sheets.
    """
    test_cases = build_test_cases()
    metrics = build_metrics()

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        # Optional: group results by document type for easier analysis
        # run_async=False,  # Useful for debugging
    )

    print("\n--- Evaluation Results ---")
    for result in results.test_results:
        doc_type = result.additional_metadata.get("document_type", "unknown")
        threshold_used = THRESHOLD_OVERRIDES.get(
            doc_type, THRESHOLD_OVERRIDES["default"]
        )
        print(f"\nDocument type: {doc_type}")
        print(f"Expected threshold: {threshold_used}")
        for metric_result in result.metrics_data:
            status = "✅" if metric_result.success else "❌"
            print(
                f"  {status} {metric_result.name}: "
                f"{metric_result.score:.3f} "
                f"(threshold: {metric_result.threshold})"
            )

    return results


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export your API key before running:\n"
            "  export OPENAI_API_KEY=sk-..."
        )
    run_heterogeneous_eval()
