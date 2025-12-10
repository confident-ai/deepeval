import os
import pytest
from deepeval.metrics import TurnFaithfulnessMetric
from deepeval.test_case import ConversationalTestCase, MLLMImage, Turn
from deepeval import evaluate

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


class TestTurnFaithfulnessMetric:
    """Tests for answer relevancy metric"""

    def test_normal_sync_metric_measure(self):
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content="What if these shoes don't fit?"),
                Turn(
                    role="assistant", 
                    content="We offer a 30-day full refund at no extra cost.", 
                    retrieval_context=[
                        "All customers are eligible for a 30 day full refund at no extra cost."
                    ]
                )
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant"
        )
        metric = TurnFaithfulnessMetric(async_mode=False)
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is False

    def test_normal_async_metric_measure(self):
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content="What if these shoes don't fit?"),
                Turn(
                    role="assistant", 
                    content="We offer a 30-day full refund at no extra cost.", 
                    retrieval_context=[
                        "All customers are eligible for a 30 day full refund at no extra cost."
                    ]
                )
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant"
        )
        metric = TurnFaithfulnessMetric()
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is False

    def test_multimodal_async_metric_measure(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content=f"What's shown in this image? {image}'",),
                Turn(
                    role="assistant", 
                    content=f"That's an image of a car",
                    retrieval_context=[
                        f"Cars are great to look at {image}"
                    ]
                )
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant"
        )
        metric = TurnFaithfulnessMetric()
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is True

    def test_multimodal_sync_metric_measure(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content=f"What's shown in this image? {image}'",),
                Turn(
                    role="assistant", 
                    content=f"That's an image of a car",
                    retrieval_context=[
                        f"Cars are great to look at {image}"
                    ]
                )
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant"
        )
        metric = TurnFaithfulnessMetric(async_mode=False)
        metric.measure(convo_test_case)

        assert metric.score is not None
        assert metric.reason is not None
        assert convo_test_case.multimodal is True

    def test_invalid_model_throws_error_for_multimodal(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content=f"What's shown in this image? {image}'",),
                Turn(
                    role="assistant", 
                    content=f"That's an image of a car",
                    retrieval_context=[
                        f"Cars are great to look at {image}"
                    ]
                )
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant"
        )
        with pytest.raises(ValueError):
            metric = TurnFaithfulnessMetric(
                model="gpt-3.5-turbo"
            )
            metric.measure(convo_test_case)

    def test_normal_evaluate_method(self):
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content="What if these shoes don't fit?"),
                Turn(
                    role="assistant", 
                    content="We offer a 30-day full refund at no extra cost.", 
                    retrieval_context=[
                        "All customers are eligible for a 30 day full refund at no extra cost."
                    ]
                )
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant"
        )
        metric = TurnFaithfulnessMetric()
        
        results = evaluate([convo_test_case], [metric])

        assert results is not None

    def test_multimodal_evaluate_method(self):
        image = MLLMImage(url=CAR)
        convo_test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content=f"What's shown in this image? {image}'",),
                Turn(
                    role="assistant", 
                    content=f"That's an image of a car",
                    retrieval_context=[
                        f"Cars are great to look at {image}"
                    ]
                )
            ],
            expected_outcome="The chatbot must explain the store policies like refunds, discounts, ..etc.",
            chatbot_role="A helpful assistant"
        )
        metric = TurnFaithfulnessMetric()

        
        results = evaluate([convo_test_case], [metric])

        assert results is not None
