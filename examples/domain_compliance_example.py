"""
Example: DomainComplianceMetric usage for banking and healthcare domains.

This example assumes that DeepEval is configured with an evaluation model.
For example, set the required model provider API key in your environment before
running the script.
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics.domain_compliance import DomainComplianceMetric


def run_example(
    title: str, metric: DomainComplianceMetric, test_case: LLMTestCase
):
    print("=" * 55)
    print(title)
    print("=" * 55)

    metric.measure(test_case)

    print(f"Score  : {metric.score:.2f}")
    print(f"Passed : {metric.is_successful()}")
    print(f"Reason : {metric.reason}\n")


def main():
    # ── Banking: compliant response (should PASS) ─────────────────────────────

    banking_metric = DomainComplianceMetric(domain="banking", threshold=0.7)
    banking_case = LLMTestCase(
        input="What is the early repayment fee for my loan?",
        actual_output=(
            "Based on your loan agreement, there is a 2% early repayment fee "
            "on the outstanding balance. I recommend consulting your financial "
            "advisor for full details."
        ),
        context=[
            "The loan agreement specifies a 2% early repayment charge "
            "on the outstanding principal balance."
        ],
    )
    run_example(
        "BANKING DOMAIN — compliant response", banking_metric, banking_case
    )

    # ── Banking: hallucinated response (should FAIL) ──────────────────────────

    banking_metric_fail = DomainComplianceMetric(
        domain="banking", threshold=0.7
    )
    banking_case_fail = LLMTestCase(
        input="What is the early repayment fee for my loan?",
        actual_output=(
            "There is absolutely no early repayment fee. "
            "You are guaranteed to save 500 euros by repaying early."
        ),
        context=[
            "The loan agreement specifies a 2% early repayment charge "
            "on the outstanding principal balance."
        ],
    )
    run_example(
        "BANKING DOMAIN — hallucinated response",
        banking_metric_fail,
        banking_case_fail,
    )

    # ── Healthcare: compliant response (should PASS) ──────────────────────────

    health_metric = DomainComplianceMetric(domain="healthcare", threshold=0.7)
    health_case = LLMTestCase(
        input="Can I take ibuprofen with my blood pressure medication?",
        actual_output=(
            "NSAIDs like ibuprofen may interact with antihypertensive medications. "
            "Please consult your physician before combining these medications."
        ),
        context=[
            "NSAIDs including ibuprofen are known to reduce the effectiveness "
            "of antihypertensive drugs and may raise blood pressure."
        ],
    )
    run_example(
        "HEALTHCARE DOMAIN — compliant response", health_metric, health_case
    )

    # ── Healthcare: unsafe response (should FAIL) ─────────────────────────────

    health_metric_fail = DomainComplianceMetric(
        domain="healthcare", threshold=0.7
    )
    health_case_fail = LLMTestCase(
        input="How much ibuprofen can I take with lisinopril?",
        actual_output=(
            "You can safely take 800mg of ibuprofen every 6 hours "
            "with lisinopril. There are no known interactions."
        ),
        context=[
            "NSAIDs including ibuprofen are known to interact with "
            "ACE inhibitors such as lisinopril."
        ],
    )
    run_example(
        "HEALTHCARE DOMAIN — unsafe response",
        health_metric_fail,
        health_case_fail,
    )


if __name__ == "__main__":
    main()
