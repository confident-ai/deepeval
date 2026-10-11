from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    ROOT
    / "examples"
    / "community"
    / "retail_support_evaluation"
    / "retail_support.py"
)


def load_example():
    assert EXAMPLE_PATH.is_file(), "the retail support example is missing"
    spec = importlib.util.spec_from_file_location(
        "retail_support", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retail_support_example_contract():
    example = load_example()

    assert len(example.RETAIL_SUPPORT_CASES) == 12
    assert len({case.case_id for case in example.RETAIL_SUPPORT_CASES}) == 12
    assert {case.category for case in example.RETAIL_SUPPORT_CASES} == {
        "order_status",
        "returns",
        "refund_status",
        "delivery_dispute",
        "substitution",
        "loyalty_points",
        "invoice_copy",
        "address_change",
        "damaged_item",
        "refund_timing",
        "quality_triage",
        "cancellation_after_dispatch",
    }
    assert example.GEVAL_EVALUATION_PARAMS == (
        example.SingleTurnParams.INPUT,
        example.SingleTurnParams.ACTUAL_OUTPUT,
        example.SingleTurnParams.EXPECTED_OUTPUT,
        example.SingleTurnParams.RETRIEVAL_CONTEXT,
    )
    assert "Use only the provided retail context" in example.GEVAL_CRITERIA

    test_cases = example.build_test_cases()
    assert len(test_cases) == len(example.RETAIL_SUPPORT_CASES)

    for case, test_case in zip(example.RETAIL_SUPPORT_CASES, test_cases):
        assert test_case.input == case.customer_input
        assert test_case.actual_output == case.actual_output
        assert test_case.expected_output == example.serialize_expected_output(
            case
        )
        assert test_case.retrieval_context == list(case.retrieval_context)
        assert test_case.metadata == {
            "case_id": case.case_id,
            "category": case.category,
        }
        assert all(
            phrase.casefold() in case.actual_output.casefold()
            for phrase in case.must_include
        )
        assert not example.find_forbidden_commitments(
            case.actual_output, case.must_not_include
        )

    assert example.serialize_expected_output(
        example.RETAIL_SUPPORT_CASES[0]
    ) == (
        "Response must include: in transit, tracking ID TRK-1001.\n"
        "Response must not include: guaranteed delivery.\n"
        "Expected policy path: provide_status_only."
    )


def test_forbidden_commitment_check_detects_negative_examples():
    example = load_example()

    for text, forbidden in example.FORBIDDEN_COMMITMENT_EXAMPLES:
        assert example.find_forbidden_commitments(text, forbidden)
