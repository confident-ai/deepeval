from __future__ import annotations

from dataclasses import dataclass

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams


@dataclass(frozen=True)
class RetailSupportCase:
    case_id: str
    category: str
    customer_input: str
    actual_output: str
    retrieval_context: tuple[str, ...]
    must_include: tuple[str, ...]
    must_not_include: tuple[str, ...]
    policy_outcome: str


RETAIL_SUPPORT_CASES = (
    RetailSupportCase(
        "retail-01",
        "order_status",
        "Can you check the status of order ORD-1001?",
        "Order ORD-1001 is in transit. You can follow it with tracking ID TRK-1001.",
        ("ORD-1001 is in transit. Tracking ID: TRK-1001.",),
        ("in transit", "tracking ID TRK-1001"),
        ("guaranteed delivery",),
        "provide_status_only",
    ),
    RetailSupportCase(
        "retail-02",
        "returns",
        "I bought a kettle 20 days ago. Can I return it?",
        "The purchase is within the 30-day return window. Please return the kettle with its original packaging.",
        ("Returns are accepted within 30 days with the original packaging.",),
        ("30-day return window", "original packaging"),
        ("refund approved",),
        "explain_return_eligibility",
    ),
    RetailSupportCase(
        "retail-03",
        "refund_status",
        "When will the refund for my returned lamp be processed?",
        "After the return reaches the warehouse, processing takes 5-7 business days and the refund goes to the original payment method.",
        (
            "Refunds are processed 5-7 business days after warehouse receipt and returned to the original payment method.",
        ),
        ("5-7 business days", "original payment method"),
        ("refund approved today",),
        "explain_refund_process",
    ),
    RetailSupportCase(
        "retail-04",
        "delivery_dispute",
        "The carrier marked my parcel as attempted, but I was home.",
        "I'm sorry for the inconvenience. The carrier recorded a delivery attempt. You can check its notice or use the carrier portal to arrange another attempt.",
        (
            "For a failed delivery attempt, the carrier leaves a notice. Customers can arrange another attempt in the carrier portal.",
        ),
        ("delivery attempt", "carrier portal"),
        ("guaranteed delivery today",),
        "offer_carrier_reschedule",
    ),
    RetailSupportCase(
        "retail-05",
        "substitution",
        "I did not want the substitute item in my grocery order.",
        "You may reject the substituted item for a full refund of that item. Contact support with your order number to start the request.",
        (
            "Customers may reject a substituted item for a full refund of that item. Support needs the order number to review the request.",
        ),
        ("reject the substituted item", "full refund"),
        ("replacement approved",),
        "explain_substitution_options",
    ),
    RetailSupportCase(
        "retail-06",
        "loyalty_points",
        "How much is a balance of 4,500 rewards points worth?",
        "At 100 points per GBP 1, 4,500 points equal a GBP 45 discount. You can redeem them at checkout on a non-sale item.",
        (
            "100 points equal GBP 1. Points can be used at checkout on non-sale items. At least 500 points are required.",
        ),
        ("4,500 points", "GBP 45", "non-sale item"),
        ("cash refund",),
        "explain_points_redemption",
    ),
    RetailSupportCase(
        "retail-07",
        "invoice_copy",
        "Where can I download the VAT invoice for order ORD-1007?",
        "Open My Orders in your account to download the VAT invoice. It is available for 12 months after purchase.",
        (
            "VAT invoices are available in My Orders for up to 12 months after purchase.",
        ),
        ("My Orders", "12 months"),
        ("invoice attached",),
        "self_serve_invoice",
    ),
    RetailSupportCase(
        "retail-08",
        "address_change",
        "Can I change the address on my order?",
        "Address changes are possible only before the order enters processing. Please share the order number so support can check its status.",
        (
            "Address changes are possible before processing starts. After dispatch, the delivery address cannot be changed.",
        ),
        ("before the order enters processing", "order number"),
        ("changed the address",),
        "verify_order_status",
    ),
    RetailSupportCase(
        "retail-09",
        "damaged_item",
        "A mug arrived chipped. What should I send you?",
        "Please send the order number and a photo of the chipped mug so support can review the available options.",
        (
            "For a damaged item, support needs the order number and a photo before reviewing available options.",
        ),
        ("order number", "photo"),
        ("replacement approved", "refund approved"),
        "collect_damage_evidence",
    ),
    RetailSupportCase(
        "retail-10",
        "refund_timing",
        "Can you guarantee that my refund will arrive in two days?",
        "I can't guarantee that timing. Refund processing takes 5-7 business days after the returned item reaches the warehouse.",
        (
            "Refund processing takes 5-7 business days after warehouse receipt. No faster arrival time is guaranteed.",
        ),
        ("5-7 business days", "can't guarantee"),
        ("guarantee your refund",),
        "state_refund_window_without_guarantee",
    ),
    RetailSupportCase(
        "retail-11",
        "quality_triage",
        "A packaged snack tastes unusual. Can you look into it?",
        "I'm sorry to hear that. Please share the batch code and a photo of the packaging so support can review the report.",
        (
            "Quality reports are reviewed with the product batch code and a photo of its packaging.",
        ),
        ("batch code", "photo"),
        ("refund approved", "replacement approved"),
        "collect_quality_details",
    ),
    RetailSupportCase(
        "retail-12",
        "cancellation_after_dispatch",
        "My order has been dispatched. Can you cancel it?",
        "The order cannot be cancelled after dispatch. Once it arrives, you can follow the carrier's return process if you no longer need it.",
        (
            "Orders cannot be cancelled after dispatch. Customers may use the carrier's return process after delivery.",
        ),
        ("cannot be cancelled", "carrier's return process"),
        ("cancellation confirmed",),
        "explain_post_dispatch_options",
    ),
)


FORBIDDEN_COMMITMENT_EXAMPLES = (
    (
        "Your refund is approved and will arrive tomorrow.",
        ("refund is approved", "arrive tomorrow"),
    ),
    ("I guarantee delivery tomorrow.", ("guarantee delivery",)),
    ("I have replaced the item for you.", ("replaced the item",)),
    (
        "I cancelled your dispatched order.",
        ("cancelled your dispatched order",),
    ),
    ("I updated your delivery address.", ("updated your delivery address",)),
)


GEVAL_CRITERIA = """Evaluate whether the actual output is a policy-aligned retail support response for the customer's request.

A passing response should:
- Directly address the customer's support task.
- Use only the provided retail context, policy notes, and evidence.
- Follow the expected policy path in the expected output.
- Include required facts or next steps.
- Avoid forbidden commitments such as unsupported refunds, cancellations, replacements, delivery guarantees, or account changes.

Penalise responses that:
- Fabricate information not present in the context.
- Misidentify the customer's task.
- Omit required next steps or evidence.
- Promise an action that the provided context does not support."""
GEVAL_EVALUATION_PARAMS = (
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
    SingleTurnParams.RETRIEVAL_CONTEXT,
)


def serialize_expected_output(case: RetailSupportCase) -> str:
    return (
        f"Response must include: {', '.join(case.must_include)}.\n"
        f"Response must not include: {', '.join(case.must_not_include)}.\n"
        f"Expected policy path: {case.policy_outcome}."
    )


def build_test_cases() -> list[LLMTestCase]:
    return [
        LLMTestCase(
            input=case.customer_input,
            actual_output=case.actual_output,
            expected_output=serialize_expected_output(case),
            retrieval_context=list(case.retrieval_context),
            metadata={
                "case_id": case.case_id,
                "category": case.category,
            },
        )
        for case in RETAIL_SUPPORT_CASES
    ]


def find_forbidden_commitments(
    response: str, forbidden_phrases: tuple[str, ...]
) -> list[str]:
    lowered_response = response.casefold()
    return [
        phrase
        for phrase in forbidden_phrases
        if phrase.casefold() in lowered_response
    ]


def build_metrics():
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )

    return (
        GEval(
            name="PolicyAlignedRetailSupport",
            criteria=GEVAL_CRITERIA,
            evaluation_params=GEVAL_EVALUATION_PARAMS,
            threshold=0.7,
        ),
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
    )


def test_retail_support_with_deepeval():
    metrics = build_metrics()
    for test_case in build_test_cases():
        assert_test(test_case, metrics)
