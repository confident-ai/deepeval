import pytest

from deepeval import assert_test
from deepeval.dataset import ConversationalGolden
from deepeval.metrics import (
    AgentResponsivenessMetric,
    ConversationCompletenessMetric,
    TurnRelevancyMetric,
    TurnTakingNaturalnessMetric,
)
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import ConversationalTestCase
from deepeval.voice import LiveKitConnector, VoiceConfig

goldens = [
    ConversationalGolden(
        scenario=(
            "A caller wants to move their dental cleaning from Tuesday "
            "morning to Thursday afternoon."
        ),
        expected_outcome=(
            "The cleaning is moved to Thursday afternoon and the new time "
            "is confirmed."
        ),
    ),
    ConversationalGolden(
        scenario=(
            "A caller asks whether the clinic accepts Delta Dental and how "
            "much a cleaning costs without insurance."
        ),
        expected_outcome=(
            "The caller learns whether Delta Dental is accepted and the "
            "price of a cleaning."
        ),
    ),
]

simulator = ConversationSimulator(
    voice_config=VoiceConfig(
        connector=LiveKitConnector(agent_name="clinic-receptionist"),
    ),
    max_concurrent=2,
)
test_cases = simulator.simulate(
    conversational_goldens=goldens, max_user_simulations=8
)


@pytest.mark.parametrize("test_case", test_cases)
def test_voice_agent(test_case: ConversationalTestCase):
    assert_test(
        test_case=test_case,
        metrics=[
            ConversationCompletenessMetric(),
            TurnRelevancyMetric(),
            AgentResponsivenessMetric(),
            TurnTakingNaturalnessMetric(),
        ],
    )
