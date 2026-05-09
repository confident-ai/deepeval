"""
Unit tests for DomainComplianceMetric.

These tests are deterministic and CI-safe. They use a mock DeepEval model instead
of requiring a local Ollama server or an external API key.

Tests cover:
  - Banking domain: compliant and non-compliant outputs
  - Healthcare domain: compliant and non-compliant outputs
  - Missing context error handling
  - Invalid domain error handling
  - Domain instantiation
  - Async measurement

Run with:
    deepeval test run tests/test_domain_compliance.py
    # or standard pytest:
    pytest tests/test_domain_compliance.py -v
"""

import json

import pytest
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics.domain_compliance import DomainComplianceMetric


class MockDomainComplianceModel(DeepEvalBaseLLM):
    """Deterministic mock model for testing DomainComplianceMetric."""

    def __init__(
        self,
        score: float = 1.0,
        reason: str = "Mock domain-compliance evaluation reason.",
    ):
        self.mock_score = score
        self.mock_reason = reason

    def load_model(self):
        return None

    def get_model_name(self):
        return "mock-domain-compliance-model"

    def generate(self, prompt: str, schema=None):
        response = {
            "score": self.mock_score,
            "reason": self.mock_reason,
        }

        if schema is not None:
            return schema(**response)

        return json.dumps(response)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)


# ── Banking test cases ────────────────────────────────────────────────────────


class TestBankingDomainCompliance:

    def test_compliant_banking_response(self):
        """Output that correctly hedges and stays faithful to context."""
        metric = DomainComplianceMetric(
            domain="banking",
            threshold=0.7,
            model=MockDomainComplianceModel(score=9),
        )
        test_case = LLMTestCase(
            input="What is the early repayment fee for my loan?",
            actual_output=(
                "Based on your loan agreement, there is a 2% early repayment "
                "fee applied to the outstanding balance. I recommend consulting "
                "your financial advisor to understand the full implications."
            ),
            context=[
                "The loan agreement specifies a 2% early repayment charge "
                "on the outstanding principal balance at the time of repayment."
            ],
        )

        score = metric.measure(test_case)

        # assert score == 0.9
        # assert metric.score == 0.9
        assert score == pytest.approx(0.9)
        assert metric.score == pytest.approx(0.9)
        assert metric.reason == "Mock domain-compliance evaluation reason."
        assert metric.is_successful() is True

    def test_hallucinated_banking_response(self):
        """Output that fabricates a fee not present in context should fail."""
        metric = DomainComplianceMetric(
            domain="banking",
            threshold=0.7,
            model=MockDomainComplianceModel(score=2),
        )
        test_case = LLMTestCase(
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

        score = metric.measure(test_case)

        assert score == pytest.approx(0.2)
        assert metric.score == pytest.approx(0.2)
        assert metric.is_successful() is False

    def test_banking_missing_context_raises(self):
        """DomainComplianceMetric must raise ValueError if context is missing."""
        metric = DomainComplianceMetric(
            domain="banking",
            threshold=0.7,
            model=MockDomainComplianceModel(score=9),
        )
        test_case = LLMTestCase(
            input="What is the interest rate?",
            actual_output="The interest rate is 3.5%.",
            # context intentionally omitted
        )

        with pytest.raises(ValueError, match="context"):
            metric.measure(test_case)


# ── Healthcare test cases ─────────────────────────────────────────────────────


class TestHealthcareDomainCompliance:

    def test_compliant_healthcare_response(self):
        """Clinically cautious output grounded in context should pass."""
        metric = DomainComplianceMetric(
            domain="healthcare",
            threshold=0.7,
            model=MockDomainComplianceModel(score=9.5),
        )
        test_case = LLMTestCase(
            input="Can I take ibuprofen with my blood pressure medication?",
            actual_output=(
                "Based on clinical guidelines, NSAIDs such as ibuprofen may "
                "interact with certain antihypertensive medications and could "
                "affect blood pressure control. Please consult your physician "
                "or pharmacist before combining these medications."
            ),
            context=[
                "NSAIDs including ibuprofen are known to reduce the "
                "effectiveness of antihypertensive drugs and may raise "
                "blood pressure in some patients."
            ],
        )

        score = metric.measure(test_case)

        assert score == pytest.approx(0.95)
        assert metric.score == pytest.approx(0.95)
        assert metric.is_successful() is True

    def test_hallucinated_healthcare_response(self):
        """Output that prescribes a specific dosage not in context should fail."""
        metric = DomainComplianceMetric(
            domain="healthcare",
            threshold=0.7,
            model=MockDomainComplianceModel(score=1),
        )
        test_case = LLMTestCase(
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

        score = metric.measure(test_case)

        assert score == pytest.approx(0.1)
        assert metric.score == pytest.approx(0.1)
        assert metric.is_successful() is False


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestDomainComplianceEdgeCases:

    def test_invalid_domain_raises(self):
        """Unsupported domain should raise ValueError on instantiation."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            DomainComplianceMetric(domain="legal")

    def test_telco_domain_instantiates(self):
        """Telco domain should instantiate without errors."""
        metric = DomainComplianceMetric(
            domain="telco",
            threshold=0.6,
            model=MockDomainComplianceModel(score=8),
        )

        assert metric.domain == "telco"
        assert metric.threshold == 0.6

    def test_manufacturing_domain_instantiates(self):
        """Manufacturing domain should instantiate without errors."""
        metric = DomainComplianceMetric(
            domain="manufacturing",
            threshold=0.6,
            model=MockDomainComplianceModel(score=9),
        )

        assert metric.domain == "manufacturing"
        assert metric.threshold == 0.6

    @pytest.mark.asyncio
    async def test_async_measure_banking(self):
        """Async measurement should return a valid score."""
        metric = DomainComplianceMetric(
            domain="banking",
            threshold=0.7,
            model=MockDomainComplianceModel(score=8.5),
        )
        test_case = LLMTestCase(
            input="What is the penalty for overdraft?",
            actual_output=(
                "According to your account terms, an overdraft fee of €15 "
                "applies per transaction that exceeds your limit. "
                "Please contact your advisor for personalised guidance."
            ),
            context=["Overdraft transactions incur a €15 fee per occurrence."],
        )

        score = await metric.a_measure(test_case)

        assert score == pytest.approx(0.85)
        assert metric.score == pytest.approx(0.85)
        assert metric.is_successful() is True
