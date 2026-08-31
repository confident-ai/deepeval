from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import List

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


@dataclass
class SupportSample:
    case_id: str
    category: str
    customer_query: str
    retrieval_context: List[str]
    llm_response: str
    expected_output: str
    forbidden_patterns: List[str] = field(default_factory=list)


RETAIL_SUPPORT_DATASET: List[SupportSample] = [
    SupportSample(
        case_id="RS-001",
        category="order_status",
        customer_query="Where is my order #ORD-88421? It was placed 3 days ago.",
        retrieval_context=[
            "Order #ORD-88421 was placed on 2024-06-10. Current status: In Transit. "
            "Estimated delivery: 2024-06-14. Carrier: FastShip, tracking ID FS99123.",
        ],
        llm_response=(
            "Your order #ORD-88421 is currently in transit with FastShip "
            "(tracking ID FS99123) and is estimated to arrive on 14 June 2024."
        ),
        expected_output=(
            "Confirm order is in transit, provide carrier name and tracking ID, "
            "and state the estimated delivery date."
        ),
    ),
    SupportSample(
        case_id="RS-002",
        category="return",
        customer_query="I want to return a blender I bought 20 days ago. Is that possible?",
        retrieval_context=[
            "RetailCo return policy: items may be returned within 30 days of purchase "
            "in original packaging. Electronics require all accessories to be included.",
        ],
        llm_response=(
            "Yes, you can return the blender since it is within our 30-day return window. "
            "Please ensure the original packaging and all accessories are included. "
            "Visit any store or initiate the return via your account portal."
        ),
        expected_output=(
            "Confirm eligibility (within 30 days), remind customer to bring original "
            "packaging and accessories, and explain how to start the return."
        ),
    ),
    SupportSample(
        case_id="RS-003",
        category="refund",
        customer_query="I returned my headphones last week. When will I get my refund?",
        retrieval_context=[
            "Refunds are processed within 5-7 business days after the returned item "
            "is received at the warehouse. Refund is issued to the original payment method.",
        ],
        llm_response=(
            "Once we receive your returned headphones at our warehouse, the refund will "
            "be processed within 5-7 business days to your original payment method."
        ),
        expected_output=(
            "State the 5-7 business-day processing window and that the refund goes "
            "to the original payment method."
        ),
    ),
    SupportSample(
        case_id="RS-004",
        category="delivery",
        customer_query="The carrier says 'delivery attempted' but I was home all day!",
        retrieval_context=[
            "When a delivery attempt fails, the carrier leaves a notification card. "
            "Customers can reschedule via the carrier portal or contact support to "
            "arrange re-delivery within 3 business days.",
        ],
        llm_response=(
            "I'm sorry for the inconvenience. The carrier recorded a failed attempt. "
            "You can reschedule a re-delivery through the carrier's portal or contact "
            "us and we will arrange it within 3 business days."
        ),
        expected_output=(
            "Apologise, acknowledge the carrier note, and explain the re-delivery "
            "options (carrier portal or support contact, 3-business-day window)."
        ),
    ),
    SupportSample(
        case_id="RS-005",
        category="substitution",
        customer_query="My grocery order had a substitution I didn't agree to. What can I do?",
        retrieval_context=[
            "RetailCo substitution policy: if an item is unavailable, a comparable "
            "item of equal or greater value may be substituted. Customers may reject "
            "any substitution for a full refund of the substituted item.",
        ],
        llm_response=(
            "You have the right to reject the substituted item for a full refund. "
            "Please contact us or use the app within 48 hours of delivery to report "
            "the unwanted substitution."
        ),
        expected_output=(
            "Inform customer they can reject the substitution for a full refund and "
            "how to do so."
        ),
    ),
    SupportSample(
        case_id="RS-006",
        category="loyalty",
        customer_query="I have 4 500 loyalty points. How much discount can I get?",
        retrieval_context=[
            "RetailCo Rewards: 100 points = 1 GBP discount. Points can be redeemed "
            "at checkout on any non-sale item. Minimum redemption: 500 points.",
        ],
        llm_response=(
            "Your 4 500 points are worth 45 GBP off your next purchase on any non-sale "
            "item. Simply apply them at checkout - the minimum redemption is 500 points."
        ),
        expected_output=(
            "Calculate correctly (4 500 / 100 = 45 GBP), mention non-sale restriction "
            "and minimum redemption threshold."
        ),
    ),
    SupportSample(
        case_id="RS-007",
        category="invoice",
        customer_query="Can I get a VAT invoice for order #ORD-77310?",
        retrieval_context=[
            "VAT invoices for orders are available in the 'My Orders' section of "
            "the customer account for up to 12 months after purchase. "
            "Business account holders may request invoices via the B2B portal.",
        ],
        llm_response=(
            "You can download a VAT invoice for order #ORD-77310 directly from "
            "'My Orders' in your account - invoices are available for up to 12 months. "
            "Business account holders can also use the B2B portal."
        ),
        expected_output=(
            "Direct customer to 'My Orders' for self-serve invoice download and "
            "mention the 12-month availability and B2B portal option."
        ),
    ),
    SupportSample(
        case_id="RS-008",
        category="address_change",
        customer_query="I need to change the delivery address for my pending order.",
        retrieval_context=[
            "Address changes are possible only before an order enters 'Processing' "
            "status. Once dispatched, the delivery address cannot be changed. "
            "Contact support immediately with the order number to request a change.",
        ],
        llm_response=(
            "I can try to update the delivery address if your order has not yet "
            "entered Processing status. Please share your order number and I'll "
            "check immediately - once dispatched, address changes are no longer possible."
        ),
        expected_output=(
            "Explain the pre-processing cutoff, ask for the order number, and "
            "note that post-dispatch changes are impossible."
        ),
    ),
    SupportSample(
        case_id="RS-009",
        category="order_status",
        customer_query="Where is order #ORD-55509?",
        retrieval_context=[
            "No record found for order #ORD-55509 in the current system.",
        ],
        llm_response=(
            "Your order #ORD-55509 is on its way and should arrive by tomorrow!"
        ),
        expected_output=(
            "Acknowledge that no record was found; ask customer to verify the order "
            "number; do not fabricate an estimated delivery date."
        ),
        forbidden_patterns=[
            r"should arrive",
            r"on its way",
            r"estimated delivery",
        ],
    ),
    SupportSample(
        case_id="RS-010",
        category="refund",
        customer_query="Can you guarantee my refund will arrive in 2 days?",
        retrieval_context=[
            "Refunds are processed within 5-7 business days after item receipt.",
        ],
        llm_response=(
            "Absolutely! I guarantee your refund will be in your account within 2 days."
        ),
        expected_output=(
            "Explain the 5-7 business-day window; do not promise a faster timeline "
            "not supported by policy."
        ),
        forbidden_patterns=[
            r"guarantee",
            r"within 2 days",
            r"2 business days",
        ],
    ),
    SupportSample(
        case_id="RS-011",
        category="loyalty",
        customer_query="How many points do I have and what can I do with them?",
        retrieval_context=[
            "Customer account balance: 0 loyalty points. "
            "Minimum redemption threshold: 500 points.",
        ],
        llm_response=(
            "Your current loyalty balance is 0 points. You will need at least "
            "500 points to redeem a discount. Points are earned on every eligible "
            "purchase - 1 point per 1 GBP spent."
        ),
        expected_output=(
            "State zero balance, explain minimum redemption threshold, "
            "and guide on how to earn points."
        ),
    ),
    SupportSample(
        case_id="RS-012",
        category="address_change",
        customer_query="My order was just dispatched. Can you change the delivery address?",
        retrieval_context=[
            "Once dispatched, delivery address cannot be changed per carrier policy.",
        ],
        llm_response=(
            "No problem! I'll update the address for you right away."
        ),
        expected_output=(
            "Apologise and clearly state that once dispatched the address cannot "
            "be changed; offer alternatives such as contacting the carrier directly."
        ),
        forbidden_patterns=[
            r"no problem",
            r"update the address",
            r"change.*address.*right away",
        ],
    ),
]


POLICY_CORRECTNESS = GEval(
    name="PolicyCorrectness",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    criteria=(
        "Assess whether the actual output is consistent with RetailCo's stated policies "
        "as reflected in the expected output. "
        "Penalise: fabricated facts, commitment to timelines not in policy, "
        "missing mandatory disclosures, and over-promises. "
        "Award full marks for: accurate policy details, appropriate caveats, "
        "and clear next-step guidance."
    ),
    threshold=0.7,
)

FAITHFULNESS = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
)

ANSWER_RELEVANCY = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True,
)


def check_forbidden_patterns(sample: SupportSample) -> dict:
    if not sample.forbidden_patterns:
        return {"case_id": sample.case_id, "passed": True, "matched": []}

    matched = [
        pat
        for pat in sample.forbidden_patterns
        if re.search(pat, sample.llm_response, re.IGNORECASE)
    ]
    return {
        "case_id": sample.case_id,
        "passed": len(matched) == 0,
        "matched": matched,
    }


def build_test_cases(dataset: List[SupportSample]) -> List[LLMTestCase]:
    return [
        LLMTestCase(
            input=s.customer_query,
            actual_output=s.llm_response,
            expected_output=s.expected_output,
            retrieval_context=s.retrieval_context,
            additional_metadata={"case_id": s.case_id, "category": s.category},
        )
        for s in dataset
    ]


def main() -> None:
    print("=" * 70)
    print("RetailCo Support Evaluation - DeepEval (programmatic)")
    print("=" * 70)

    print("\n[1/2] Deterministic forbidden-commitment checks")
    print("-" * 50)
    deterministic_results = [
        check_forbidden_patterns(s) for s in RETAIL_SUPPORT_DATASET
    ]
    for r in deterministic_results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = f"  matched: {r['matched']}" if not r["passed"] else ""
        print(f"  [{status}] {r['case_id']}{detail}")

    n_failed = sum(1 for r in deterministic_results if not r["passed"])
    print(
        f"\n  {len(RETAIL_SUPPORT_DATASET) - n_failed}/{len(RETAIL_SUPPORT_DATASET)} passed deterministic checks."
    )

    print(
        "\n[2/2] LLM-based evaluation (GEval / Faithfulness / AnswerRelevancy)"
    )
    print("-" * 50)

    test_cases = build_test_cases(RETAIL_SUPPORT_DATASET)

    evaluate(
        test_cases=test_cases,
        metrics=[POLICY_CORRECTNESS, FAITHFULNESS, ANSWER_RELEVANCY],
    )

    print("\n" + "=" * 70)
    print(
        "Evaluation complete. Review detailed results above or on DeepEval Cloud."
    )
    if n_failed:
        print(
            f"  {n_failed} case(s) failed deterministic forbidden-pattern checks."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
