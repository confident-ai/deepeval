import pytest

from deepeval import assert_test
from deepeval.dataset import ConversationalGolden
from deepeval.metrics import (
    AgentResponsivenessMetric,
    TurnTakingNaturalnessMetric,
    VoiceReliabilityMetric,
)
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio, ConversationalTestCase
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import ConnectorTurn
from tests.test_core.test_simulator.helpers import StaticSimulatorModel
from tests.test_core.test_voice.helpers import RATE, EchoAgent, StubSTT, StubTTS


class SilentAgent:
    async def __call__(self, audio: Audio) -> ConnectorTurn:
        pcm = b"\x00\x00" * int(RATE * 0.5)
        return ConnectorTurn(
            audio=Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(pcm, sample_rate=RATE),
                "audio/wav",
                sampleRate=RATE,
                encoding="wav",
                duration=0.5,
            )
        )


def _simulate(agent, goldens: int = 2):
    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(agent),
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    )
    return simulator.simulate(
        [
            ConversationalGolden(scenario=f"Caller asks for refund {index}")
            for index in range(goldens)
        ],
        max_user_simulations=2,
    )


def _voice_metrics():
    return [
        AgentResponsivenessMetric(),
        TurnTakingNaturalnessMetric(),
        VoiceReliabilityMetric(),
    ]


test_cases = _simulate(EchoAgent())


@pytest.mark.parametrize("test_case", test_cases)
def test_a_healthy_voice_agent_passes(test_case: ConversationalTestCase):
    assert_test(test_case=test_case, metrics=_voice_metrics())


def test_a_silent_voice_agent_fails():
    [test_case] = _simulate(SilentAgent(), goldens=1)

    with pytest.raises(AssertionError, match="Agent Responsiveness"):
        assert_test(test_case=test_case, metrics=_voice_metrics())


async def test_voice_simulations_run_inside_async_tests():
    [test_case] = _simulate(EchoAgent(), goldens=1)

    assert_test(test_case=test_case, metrics=_voice_metrics())
