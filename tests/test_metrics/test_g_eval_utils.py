import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.g_eval.utils import (
    CONVERSATIONAL_G_EVAL_API_PARAMS,
    G_EVAL_API_PARAMS,
    construct_geval_upload_payload,
    construct_non_turns_test_case_string,
    construct_test_case_string,
    format_retrieval_context_with_budget,
    build_retrieval_context_budget_report,
)
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    check_llm_test_case_params,
    convert_turn_to_dict,
)
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    RetrievedContextData,
    SingleTurnParams,
    Turn,
    MultiTurnParams,
)


class DummyMetric:
    __name__ = "DummyMetric"
    error = None


class DummyConversationalMetric:
    __name__ = "DummyConversationalMetric"
    error = None


def test_geval_accepts_metadata_and_tags():
    test_case = LLMTestCase(
        input="input",
        metadata={"source": "unit"},
        tags=["tag"],
    )

    text = construct_test_case_string(
        [SingleTurnParams.METADATA, SingleTurnParams.TAGS],
        test_case,
    )
    payload = construct_geval_upload_payload(
        name="metadata-test",
        evaluation_params=[SingleTurnParams.METADATA, SingleTurnParams.TAGS],
        g_eval_api_params=G_EVAL_API_PARAMS,
        criteria="criteria",
    )

    assert "Metadata" in text
    assert "Tags" in text
    assert payload["evaluationParams"] == ["metadata", "tags"]


def test_geval_retrieval_context_budget_compacts_large_chunks():
    large_context = (
        "refund policy starts here. "
        + ("middle evidence that should be compacted. " * 200)
        + "refund policy ends here."
    )
    test_case = LLMTestCase(
        input="What is the refund window?",
        actual_output="30 days",
        retrieval_context=[large_context],
    )

    text = construct_test_case_string(
        [SingleTurnParams.RETRIEVAL_CONTEXT],
        test_case,
        max_retrieval_context_tokens=80,
    )

    assert "retrieval_context compacted for GEval" in text
    assert "refund policy starts here" in text
    assert "refund policy ends here" in text
    assert "omitted" in text
    assert len(text) < len(large_context)


def test_geval_retrieval_context_budget_keeps_source_labels():
    retrieval_context = [
        RetrievedContextData(
            source="docs/refunds.md",
            context="Refunds are available within 30 days. " * 120,
        ),
        "Warranty claims require a receipt. " * 120,
    ]

    text = format_retrieval_context_with_budget(
        retrieval_context,
        max_retrieval_context_tokens=96,
    )

    assert "[1 source=docs/refunds.md]" in text
    assert "[2]" in text
    assert "Refunds are available" in text
    assert "Warranty claims" in text


def test_geval_retrieval_context_budget_marks_omitted_chunks():
    retrieval_context = [
        f"retrieval chunk {index} " + ("evidence " * 120) for index in range(8)
    ]

    text = format_retrieval_context_with_budget(
        retrieval_context,
        max_retrieval_context_tokens=96,
    )

    assert "retrieval chunk 0" in text
    assert "omitted" in text
    assert "retrieval chunks because" in text


def test_geval_retrieval_context_budget_report_quantifies_compaction():
    retrieval_context = [
        RetrievedContextData(
            source="kb/refunds.md",
            context=(
                "Refund policy anchor. "
                + ("irrelevant body " * 500)
                + "Refund policy tail evidence."
            ),
        ),
        RetrievedContextData(
            source="kb/warranty.md",
            context="Warranty claims need receipts. " * 400,
        ),
        "Shipping delays are excluded from refunds. " * 400,
    ]

    report = build_retrieval_context_budget_report(
        retrieval_context,
        max_retrieval_context_tokens=120,
    )

    assert report.exceeded_budget is True
    assert report.original_tokens > report.rendered_tokens
    assert report.compression_ratio < 0.25
    assert report.total_chunks == 3
    assert report.visible_chunks >= 1
    assert any(chunk.source == "kb/refunds.md" for chunk in report.chunks)
    assert "Refund policy anchor" in report.rendered_context
    assert "tail evidence" in report.rendered_context
    assert "source=kb/refunds.md" in report.rendered_context
    assert report.evidence_coverage.coverage_ratio == 1.0


def test_geval_retrieval_context_budget_prioritizes_relevant_chunks():
    retrieval_context = [
        RetrievedContextData(
            source="kb/security.md",
            context="SAML setup and audit logging configuration. " * 160,
        ),
        RetrievedContextData(
            source="kb/sales.md",
            context="Pipeline stages and account ownership notes. " * 160,
        ),
        RetrievedContextData(
            source="kb/refunds.md",
            context=(
                "Enterprise refunds are available within 30 days after purchase. "
                "Onboarding fees are non-refundable."
            ),
        ),
        RetrievedContextData(
            source="kb/procurement.md",
            context="Vendor review and procurement workflow details. " * 160,
        ),
    ]

    report = build_retrieval_context_budget_report(
        retrieval_context,
        max_retrieval_context_tokens=64,
        relevance_query=(
            "Can enterprise customers get a refund after onboarding? "
            "Enterprise customers can request a refund within 30 days, "
            "but onboarding fees are non-refundable."
        ),
    )

    assert "kb/refunds.md" in report.rendered_context
    assert "Enterprise refunds are available" in report.rendered_context
    assert "kb/procurement.md" not in report.rendered_context
    refund_chunk = next(
        chunk for chunk in report.chunks if chunk.source == "kb/refunds.md"
    )
    assert refund_chunk.omitted is False
    assert refund_chunk.relevance_score > 0
    assert report.omitted_chunks == 2
    assert "enterprise" in report.evidence_coverage.covered_terms
    assert "refund" in report.evidence_coverage.covered_terms
    assert report.evidence_coverage.missing_terms == []


def test_geval_retrieval_context_budget_reports_missing_evidence_terms():
    retrieval_context = [
        RetrievedContextData(
            source="kb/refunds.md",
            context=(
                "enterprise refund policy starts. "
                + ("background notes " * 120)
                + "nonrefundable onboarding fee."
            ),
        )
    ]

    report = build_retrieval_context_budget_report(
        retrieval_context,
        max_retrieval_context_tokens=18,
        relevance_query=(
            "Enterprise customers can request a refund, "
            "but onboarding is nonrefundable."
        ),
    )

    assert report.evidence_coverage.coverage_ratio < 1.0
    assert "enterprise" in report.evidence_coverage.covered_terms
    assert "nonrefundable" in report.evidence_coverage.missing_terms
    assert report.evidence_coverage.warning is not None


def test_geval_retrieval_context_budget_rejects_invalid_budget():
    with pytest.raises(ValueError, match="greater than 0"):
        format_retrieval_context_with_budget(
            ["context"],
            max_retrieval_context_tokens=0,
        )


def test_geval_requires_metadata_when_selected():
    test_case = LLMTestCase(input="input", tags=["tag"])

    with pytest.raises(MissingTestCaseParamsError):
        check_llm_test_case_params(
            test_case,
            [SingleTurnParams.METADATA],
            None,
            None,
            DummyMetric(),
        )


def test_conversational_geval_accepts_metadata_and_tags():
    case_metadata = {"case": "metadata"}
    case_tags = ["tag"]
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
        metadata=case_metadata,
        tags=case_tags,
    )

    non_turn_text = construct_non_turns_test_case_string(
        [MultiTurnParams.METADATA, MultiTurnParams.TAGS],
        test_case,
    )
    turn_dict = convert_turn_to_dict(
        test_case.turns[0],
        [
            MultiTurnParams.CONTENT,
            MultiTurnParams.ROLE,
            MultiTurnParams.METADATA,
            MultiTurnParams.TAGS,
        ],
    )
    payload = construct_geval_upload_payload(
        name="conversational-metadata-test",
        evaluation_params=[MultiTurnParams.METADATA, MultiTurnParams.TAGS],
        g_eval_api_params=CONVERSATIONAL_G_EVAL_API_PARAMS,
        criteria="criteria",
        multi_turn=True,
    )

    assert "Metadata" in non_turn_text
    assert "case" in non_turn_text
    assert "Tags" in non_turn_text
    assert "tag" in non_turn_text
    assert "metadata" not in turn_dict
    assert "tags" not in turn_dict
    assert payload["evaluationParams"] == ["metadata", "tags"]


def test_conversational_geval_requires_metadata_when_selected():
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
        tags=["tag"],
    )

    with pytest.raises(MissingTestCaseParamsError):
        check_conversational_test_case_params(
            test_case,
            [MultiTurnParams.METADATA],
            DummyConversationalMetric(),
        )


def test_conversational_geval_requires_tags_when_selected():
    test_case = ConversationalTestCase(
        turns=[Turn(role="user", content="hello")],
        metadata={"case": "metadata"},
    )

    with pytest.raises(MissingTestCaseParamsError):
        check_conversational_test_case_params(
            test_case,
            [MultiTurnParams.TAGS],
            DummyConversationalMetric(),
        )
