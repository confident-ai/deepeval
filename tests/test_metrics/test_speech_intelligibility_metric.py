from deepeval.metrics import SpeechIntelligibilityMetric
from tests.test_metrics.helpers import (
    agent_speech,
    measure,
    with_noise,
    with_reply_audio,
)


def test_loud_noise_lowers_speech_intelligibility():
    quiet = measure(
        SpeechIntelligibilityMetric(),
        with_reply_audio(1, with_noise(agent_speech(), -50)),
    )
    loud = measure(
        SpeechIntelligibilityMetric(),
        with_reply_audio(1, with_noise(agent_speech(), -30)),
    )
    assert loud.score < quiet.score
