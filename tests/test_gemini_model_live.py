"""Integration tests for Gemini models using live API calls.

These tests require valid Google Cloud credentials and will make actual API calls.
Skip these tests if you don't want to incur API costs.
"""

import os
import pytest
from deepeval.models import GeminiModel, MultimodalGeminiModel
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    MultimodalContextualRecallMetric,
    MultimodalContextualRelevancyMetric,
    MultimodalContextualPrecisionMetric,
    MultimodalAnswerRelevancyMetric,
    MultimodalFaithfulnessMetric,
)

# Skip all tests if credentials are not set
SKIP_LIVE_TESTS = not (os.getenv("GOOGLE_CLOUD_PROJECT") and os.getenv("GOOGLE_CLOUD_LOCATION"))

# Create a test case for multimodal evaluation
multimodal_test_case = LLMTestCase(
    input="Tell me about some landmarks in France",
    actual_output="The Eiffel Tower is located in Paris, France. It is a wrought-iron lattice tower on the Champ de Mars. The Statue of Liberty was a gift from France to the United States.",
    expected_output="The Eiffel Tower is located in Paris, France. It is a wrought-iron lattice tower on the Champ de Mars. The Statue of Liberty was a gift from France to the United States.",
    context=[
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "It is named after the engineer Gustave Eiffel.",
        "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.",
    ]
)

@pytest.mark.skipif(SKIP_LIVE_TESTS, reason="Google Cloud credentials not set")
class TestGeminiModelLive:
    """Live API tests for GeminiModel."""
    
    def test_structured_output_generation(self):
        """Test generation with structured output schema."""
        from pydantic import BaseModel
        
        class CityInfo(BaseModel):
            city: str
            country: str
            population: int
        
        # Initialize the model
        model = GeminiModel(
            model_name="gemini-1.5-pro",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION")
        )
        
        response = model.generate(
            "Give me information about Paris, France",
            schema=CityInfo
        )
        
        # Verify we get structured output
        assert isinstance(response, CityInfo)
        assert response.city == "Paris"
        assert response.country == "France"
        assert response.population > 1000000  # Paris has over 1M people


    def test_simple_evaluation(self):
        """Test simple evaluation with AnswerRelevancyMetric."""
        # Initialize the model
        model = GeminiModel(
            model_name="gemini-1.5-pro",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION")
        )
        
        # Create and run test case
        test_input = "What is the capital of France?"
        actual_output = "The capital of France is Paris."
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output
        )
        
        # Evaluate using metrics
        metric = AnswerRelevancyMetric(model=model, threshold=0.7)
        # Assert test case passes
        assert_test(test_case, [metric])
        

@pytest.mark.skipif(SKIP_LIVE_TESTS, reason="Google Cloud credentials not set")
class TestMultimodalGeminiModelLive:
    """Live API tests for MultimodalGeminiModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = MultimodalGeminiModel(
            model_name="gemini-1.5-pro",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION")
        )
    
    def test_multimodal_evaluation(self):
        """Test using MultimodalGeminiModel as judge for various metrics."""
        
        # Create metrics using Gemini as judge
        metrics = [
            MultimodalContextualRecallMetric(llm=self.model),
            MultimodalContextualRelevancyMetric(llm=self.model),
            MultimodalContextualPrecisionMetric(llm=self.model),
            MultimodalAnswerRelevancyMetric(llm=self.model),
            MultimodalFaithfulnessMetric(llm=self.model),
        ]
        
        # Assert all metric evaluations using the imported rag_test_case
        assert_test(
            multimodal_test_case,
            metrics,
            run_async=True,
        )
