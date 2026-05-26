from deepeval.metrics import TurnContextualRecallMetric
from deepeval.metrics.turn_contextual_recall.schema import (
    ContextualRecallScoreReason,
    ContextualRecallVerdict,
    Verdicts,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


EXPECTED_OUTCOME = "The assistant must mention the refund policy."


class FakeTurnRecallModel(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def get_model_name(self):
        return "fake-turn-recall-model"

    def generate(self, prompt, schema=None):
        return self._generate_response(prompt, schema)

    async def a_generate(self, prompt, schema=None):
        return self._generate_response(prompt, schema)

    def _generate_response(self, prompt, schema):
        if schema is Verdicts:
            verdict = "yes" if EXPECTED_OUTCOME in prompt else "no"
            return Verdicts(
                verdicts=[
                    ContextualRecallVerdict(
                        verdict=verdict,
                        reason="Expected outcome was passed into the prompt.",
                    )
                ]
            )
        if schema is ContextualRecallScoreReason:
            return ContextualRecallScoreReason(
                reason="The retrieved context supports the expected outcome."
            )
        raise AssertionError(f"Unexpected schema: {schema}")


def build_test_case():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What if these shoes do not fit?"),
            Turn(
                role="assistant",
                content="You can return them within 30 days.",
                retrieval_context=[
                    "Customers can return shoes for a refund within 30 days."
                ],
            ),
        ],
        expected_outcome=EXPECTED_OUTCOME,
        chatbot_role="A helpful assistant",
    )


def test_turn_contextual_recall_async_uses_expected_outcome():
    metric = TurnContextualRecallMetric(model=FakeTurnRecallModel())
    score = metric.measure(build_test_case())

    assert score == 1
    assert metric.score == 1
    assert metric.success is True
