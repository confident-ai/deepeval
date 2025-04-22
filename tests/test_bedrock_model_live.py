import os
import pytest
from deepeval.models import BedrockModel, MultimodalBedrockModel
from deepeval.test_case import LLMTestCase, MLLMTestCase, MLLMImage
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    MultimodalContextualPrecisionMetric,
    MultimodalAnswerRelevancyMetric
)

SKIP_LIVE_TESTS = not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("AWS_REGION"))

simple_test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris."
)

multimodal_test_case = MLLMTestCase(
    input=["Tell me about some landmarks in France"],
    actual_output=[
        "The Eiffel Tower is located in Paris, France.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
        ),
        "The Statue of Liberty was a gift from France to the United States.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg/375px-Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg"
        ),
    ],
    expected_output=[
        "The Eiffel Tower is located in Paris, France.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
        ),
        "The Statue of Liberty was a gift from France to the United States.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg/375px-Front_view_of_Statue_of_Liberty_with_pedestal_and_base_2024.jpg"
        ),
    ],
    context=[
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "It is named after the engineer Gustave Eiffel.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
        ),
        "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Liberty-from-behind-2024.jpg/330px-Liberty-from-behind-2024.jpg"
        ),
    ],
    retrieval_context=[
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "It is named after the engineer Gustave Eiffel.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/375px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
        ),
        "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.",
        MLLMImage(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Liberty-from-behind-2024.jpg/330px-Liberty-from-behind-2024.jpg"
        ),
    ],
)

@pytest.mark.skipif(SKIP_LIVE_TESTS, reason="AWS credentials not set")
class TestBedrockModelLive:
    """Live API tests for BedrockModel."""

    def test_structured_output_generation(self):
        """Test generation with structured output schema."""
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            city: str
            country: str
            population: int

        model = BedrockModel(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            region=os.getenv("AWS_REGION")
        )

        response = model.generate(
            "Give me information about Paris, France",
            schema=CityInfo
        )

        print(f"response: {response}")

        assert isinstance(response, CityInfo)
        assert response.city == "Paris"
        assert response.country == "France"
        assert response.population > 1000000  

    def test_simple_evaluation(self):
        """Test simple evaluation with AnswerRelevancyMetric."""
        model = BedrockModel(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            region=os.getenv("AWS_REGION")
        )

        metric = AnswerRelevancyMetric(model=model, threshold=0.8)
        assert_test(simple_test_case, [metric])


@pytest.mark.skipif(SKIP_LIVE_TESTS, reason="AWS credentials not set")
class TestMultimodalBedrockModelLive:
    """Live API tests for MultimodalBedrockModel."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = MultimodalBedrockModel(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            region=os.getenv("AWS_REGION")
        )

    def test_multimodal_evaluation(self):
        """Test using MultimodalBedrockModel as judge for various metrics."""

        metrics = [
            MultimodalContextualPrecisionMetric(model=self.model),
            MultimodalAnswerRelevancyMetric(model=self.model)
        ]

        assert_test(
            multimodal_test_case,
            metrics,
            run_async=True,
        )
