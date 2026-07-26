import pytest

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    LLMTestCase,
    RetrievedContextData,
    SingleTurnParams,
)


class LocalEchoModel(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def generate(self, *args, **kwargs) -> str:
        return '{"score": 10, "reason": "ok"}'

    async def a_generate(self, *args, **kwargs) -> str:
        return '{"score": 10, "reason": "ok"}'

    def get_model_name(self, *args, **kwargs) -> str:
        return "local-echo"


def build_large_rag_case() -> LLMTestCase:
    return LLMTestCase(
        input="Can enterprise customers get a refund after onboarding?",
        actual_output=(
            "Enterprise customers can request a refund within 30 days, "
            "but onboarding fees are non-refundable."
        ),
        retrieval_context=[
            RetrievedContextData(
                source="handbook/refunds.md",
                context=(
                    "Refunds are allowed for enterprise customers within "
                    "30 days. "
                    + ("Unrelated account-management details. " * 700)
                    + "Onboarding fees are non-refundable."
                ),
            ),
            RetrievedContextData(
                source="handbook/security.md",
                context="Security review requirements. " * 650,
            ),
            RetrievedContextData(
                source="handbook/support.md",
                context="Support escalation playbook. " * 650,
            ),
            RetrievedContextData(
                source="handbook/billing.md",
                context="Billing ownership details. " * 650,
            ),
            RetrievedContextData(
                source="handbook/sales.md",
                context="Sales handoff details. " * 650,
            ),
            RetrievedContextData(
                source="handbook/legal.md",
                context="Legal escalation details. " * 650,
            ),
            RetrievedContextData(
                source="handbook/procurement.md",
                context="Procurement workflow details. " * 650,
            ),
            RetrievedContextData(
                source="handbook/renewals.md",
                context="Renewal workflow details. " * 650,
            ),
        ],
    )


def build_metric() -> GEval:
    return GEval(
        name="RAG Faithfulness",
        evaluation_steps=[
            "Check whether the actual output is supported by retrieval context.",
            "Penalize unsupported refund or onboarding claims.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        max_retrieval_context_tokens=160,
        model=LocalEchoModel(),
    )


def test_geval_preview_prompt_bounds_large_rag_context_without_model_call():
    test_case = build_large_rag_case()
    metric = build_metric()

    prompt = metric.preview_evaluation_prompt(test_case)
    report = metric.get_retrieval_context_budget_report(test_case)
    coverage = metric.get_retrieval_context_evidence_coverage(test_case)

    assert report is not None
    assert coverage is not None
    assert report.original_tokens > 3000
    assert report.rendered_tokens < 260
    assert report.compression_ratio < 0.1
    assert coverage.coverage_ratio > 0
    assert "refund" in coverage.covered_terms
    assert "handbook/refunds.md" in prompt
    assert "Refunds are allowed" in prompt
    assert "n-refundable" in prompt
    assert "retrieval_context compacted for GEval" in prompt
    assert report.omitted_chunks > 0
    assert "Legal escalation details" not in prompt


def test_geval_preview_prompt_requires_steps_to_avoid_hidden_llm_call():
    metric = GEval(
        name="RAG Faithfulness",
        criteria="Check whether the answer is grounded in retrieval context.",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        max_retrieval_context_tokens=160,
        model=LocalEchoModel(),
    )

    with pytest.raises(ValueError, match="requires evaluation_steps"):
        metric.preview_evaluation_prompt(build_large_rag_case())
