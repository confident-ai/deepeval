"""Voice behavior driven by the golden's `Persona`."""

from typing import List, Optional

import pytest

from deepeval.dataset import (
    BackgroundNoiseSettings,
    ConversationalGolden,
    InterruptionBehavior,
    Persona,
)
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio
from deepeval.voice import VoiceConfig
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.callback import (
    CallbackVoiceConnector,
)
from deepeval.voice.connectors.types import ConnectorTurn
from tests.test_core.test_simulator.helpers import StaticSimulatorModel


def _wav_audio(sample: int = 1000, samples: int = 240) -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            sample.to_bytes(2, "little", signed=True) * samples,
            sample_rate=24000,
        ),
        "audio/wav",
    )


class RecordingTTS:
    def __init__(self):
        self.calls: List[dict] = []

    async def a_synthesize(self, text: str, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return _wav_audio(), None


class RecordingSTT:
    def __init__(self, transcripts: Optional[List[str]] = None):
        self.calls: List[dict] = []
        self.transcripts = list(transcripts or [])

    async def a_transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        if self.transcripts:
            return self.transcripts.pop(0), None
        return "Agent reply", None


class RecordingAgent:
    """A voice agent that echoes back fixed audio and records the uplink."""

    def __init__(self, transcript: Optional[str] = "Agent reply"):
        self.uplinks: List[Audio] = []
        self.transcript = transcript

    async def __call__(self, audio: Audio) -> ConnectorTurn:
        self.uplinks.append(audio)
        return ConnectorTurn(audio=_wav_audio(), transcript=self.transcript)


def _simulator(agent, tts=None, stt=None, **voice_kwargs):
    return ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(agent),
            tts_model=tts or RecordingTTS(),
            stt_model=stt or RecordingSTT(),
            output_dir=None,
            combine_audio_files=False,
            **voice_kwargs,
        ),
    )


def test_persona_voice_reaches_the_tts_model():
    tts = RecordingTTS()
    simulator = _simulator(RecordingAgent(), tts=tts)
    golden = ConversationalGolden(
        scenario="Refund",
        persona=Persona(characteristics="An impatient caller.", voice="coral"),
    )

    simulator.simulate([golden], max_user_simulations=1)

    assert tts.calls and tts.calls[0]["voice"] == "coral"


def test_multilingual_stt_requests_language_detection():
    stt = RecordingSTT()
    agent = RecordingAgent(transcript=None)
    simulator = _simulator(agent, stt=stt)
    golden = ConversationalGolden(
        scenario="Refund",
        persona=Persona(
            characteristics="A bilingual caller.", multilingual_stt=True
        ),
    )

    simulator.simulate([golden], max_user_simulations=1)

    assert stt.calls and stt.calls[0] == {"language": "auto"}


def test_single_language_personas_leave_stt_alone():
    stt = RecordingSTT()
    simulator = _simulator(RecordingAgent(transcript=None), stt=stt)
    golden = ConversationalGolden(
        scenario="Refund", persona=Persona(characteristics="A caller.")
    )

    simulator.simulate([golden], max_user_simulations=1)

    assert stt.calls and stt.calls[0] == {}


def test_a_persona_that_waits_lets_the_agent_open():
    tts = RecordingTTS()
    simulator = _simulator(
        RecordingAgent(transcript="Hello, how can I help?"), tts=tts
    )
    golden = ConversationalGolden(
        scenario="Refund",
        persona=Persona(characteristics="A quiet caller.", speaks_first=False),
    )

    turns = simulator.simulate([golden], max_user_simulations=1)[0].turns

    assert turns[0].role == "assistant"
    assert turns[0].content == "Hello, how can I help?"
    assert turns[1].role == "user"
    # The greeting is heard on silence, so nothing is synthesized for it.
    assert len(tts.calls) == 1


def test_a_muted_persona_never_speaks():
    tts = RecordingTTS()
    agent = RecordingAgent(transcript="Are you still there?")
    simulator = _simulator(agent, tts=tts)
    golden = ConversationalGolden(
        scenario="Dead air",
        persona=Persona(
            characteristics="A caller who says nothing.", muted=True
        ),
    )

    turns = simulator.simulate([golden], max_user_simulations=2)[0].turns

    assert tts.calls == []
    assert [turn.content for turn in turns if turn.role == "user"] == ["", ""]
    assert all(turn.audio is not None for turn in turns if turn.role == "user")
    assert len(agent.uplinks) == 2


def test_hold_timeout_hangs_up_on_dead_air():
    simulator = _simulator(
        RecordingAgent(transcript=None), stt=RecordingSTT([""] * 10)
    )
    golden = ConversationalGolden(
        scenario="Hold music",
        persona=Persona(
            characteristics="A caller who will not wait.", hold_timeout=1e-6
        ),
    )

    turns = simulator.simulate([golden], max_user_simulations=5)[0].turns

    # One user turn, one silent agent turn, then the caller hangs up.
    assert len(turns) == 2


def test_a_talkative_agent_resets_the_hold_timer():
    simulator = _simulator(RecordingAgent(transcript="Still here!"))
    golden = ConversationalGolden(
        scenario="Hold music",
        persona=Persona(
            characteristics="A caller who will not wait.", hold_timeout=1e-6
        ),
    )

    turns = simulator.simulate([golden], max_user_simulations=3)[0].turns

    assert len(turns) == 6


def test_background_audio_is_mixed_into_the_uplink(tmp_path):
    bed = tmp_path / "cafe.wav"
    bed.write_bytes(
        audio_utils.pcm16_to_wav_bytes(
            (500).to_bytes(2, "little", signed=True) * 240, sample_rate=24000
        )
    )
    agent = RecordingAgent()
    simulator = _simulator(agent)
    golden = ConversationalGolden(
        scenario="Refund",
        persona=Persona(
            characteristics="A caller in a cafe.",
            background_noise=BackgroundNoiseSettings(
                audio=str(bed), volume=1.0
            ),
        ),
    )

    simulator.simulate([golden], max_user_simulations=1)

    pcm, _, _ = audio_utils.wav_bytes_to_pcm16(agent.uplinks[0].get_bytes())
    assert int.from_bytes(pcm[:2], "little", signed=True) == 1500


def test_persona_interruption_behavior_selects_the_duplex_path():
    simulator = _simulator(RecordingAgent())
    golden = ConversationalGolden(
        scenario="Refund",
        persona=Persona(
            characteristics="A caller who talks over you.",
            interruption_behavior=InterruptionBehavior(frequency="frequent"),
        ),
    )

    policy, floor = simulator._build_interruption(golden)

    assert policy is not None and floor is not None
    assert policy.level == "frequent"


def test_no_interruption_behavior_keeps_the_half_duplex_path():
    simulator = _simulator(RecordingAgent())
    golden = ConversationalGolden(
        scenario="Refund", persona=Persona(characteristics="A calm caller.")
    )

    policy, floor = simulator._build_interruption(golden)

    assert policy is None and floor is None


def test_deprecated_voice_config_interruption_settings_still_applies():
    with pytest.warns(DeprecationWarning, match="interruption_settings"):
        simulator = _simulator(
            RecordingAgent(),
            interruption_settings=InterruptionBehavior(frequency="rare"),
        )
    golden = ConversationalGolden(
        scenario="Refund", persona=Persona(characteristics="A calm caller.")
    )

    policy, _ = simulator._build_interruption(golden)

    assert policy is not None and policy.level == "rare"
