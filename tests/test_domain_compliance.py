"""
Unit tests for DomainComplianceMetric.

Tests cover:
  - Banking domain: compliant and non-compliant outputs
  - Healthcare domain: compliant and non-compliant outputs
  - Missing context error handling
  - Invalid domain error handling
  - Async measurement

Run with:
    deepeval test run tests/test_domain_compliance.py
    # or standard pytest:
    pytest tests/test_domain_compliance.py -v
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics.domain_compliance import DomainComplianceMetric

#-----------------------------------------------
#no API for GPT, so any off-the-shelf model
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval

# Use Ollama (free, runs locally)
# First: ollama pull llama3
import ollama

class LocalLlamaModel(DeepEvalBaseLLM):
    def load_model(self): return self
    def generate(self, prompt):
        return ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )["message"]["content"]
    async def a_generate(self, prompt): return self.generate(prompt)
    def get_model_name(self): return "llama3-local"

local_model = LocalLlamaModel()
#-----------------------------------------------------------
# ── Banking test cases ────────────────────────────────────────────────────────

class TestBankingDomainCompliance:

    def test_compliant_banking_response(self):
        """Output that correctly hedges and stays faithful to context."""
        metric = DomainComplianceMetric(domain="banking", threshold=0.7, model=local_model)
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
        assert_test(test_case, [metric])

    def test_hallucinated_banking_response(self):
        """Output that fabricates a fee not present in context — should FAIL."""
        metric = DomainComplianceMetric(domain="banking", threshold=0.7, model=local_model)
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
        metric.measure(test_case)
        assert not metric.is_successful(), (
            f"Expected failure for hallucinated output, got score: {metric.score}"
        )

    def test_banking_missing_context_raises(self):
        """DomainComplianceMetric must raise ValueError if context is missing."""
        metric = DomainComplianceMetric(domain="banking", threshold=0.7, model=local_model)
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
        """Clinically cautious output grounded in context — should PASS."""
        metric = DomainComplianceMetric(domain="healthcare", threshold=0.7, model=local_model)
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
        assert_test(test_case, [metric])

    def test_hallucinated_healthcare_response(self):
        """Output that prescribes a specific dosage not in context — should FAIL."""
        metric = DomainComplianceMetric(domain="healthcare", threshold=0.7, model=local_model)
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
        metric.measure(test_case)
        assert not metric.is_successful(), (
            f"Expected failure for unsafe clinical advice, got score: {metric.score}"
        )


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestDomainComplianceEdgeCases:

    def test_invalid_domain_raises(self):
        """Unsupported domain should raise ValueError on instantiation."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            DomainComplianceMetric(domain="legal")  # not yet supported

    def test_telco_domain_instantiates(self):
        """Telco domain should instantiate without errors."""
        metric = DomainComplianceMetric(domain="telco", threshold=0.6, model=local_model)
        assert metric.domain == "telco"

    def test_manufacturing_domain_instantiates(self):
        """Manufacturing domain should instantiate without errors."""
        metric = DomainComplianceMetric(domain="manufacturing", threshold=0.6, model=local_model)
        assert metric.domain == "manufacturing"

    @pytest.mark.asyncio
    async def test_async_measure_banking(self):
        """Async measurement should return a valid score."""
        metric = DomainComplianceMetric(domain="banking", threshold=0.7, model=local_model)
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
        assert 0.0 <= score <= 1.0
