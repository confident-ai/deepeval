"""Agent turns say whether their text is the agent's own transcript or ours."""

from deepeval.dataset.golden import ConversationalGolden, Persona
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.transcript_source import (
    TRANSCRIPT_SOURCE_KEY,
    transcript_source_metadata,
)
from tests.test_core.test_voice.test_voice_simulation import (
    _StaticVoiceModel,
    _StubSTT,
    _StubTTS,
    _tone,
)


def _agent_turns(transcript):
    async def agent(_audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=_tone(0.4), transcript=transcript)

    simulator = ConversationSimulator(
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(agent, turn_detection="eager"),
            tts_model=_StubTTS(),
            stt_model=_StubSTT(),
            output_dir=None,
        ),
        simulator_model=_StaticVoiceModel(),
    )
    golden = ConversationalGolden(
        scenario="Ask a question", persona=Persona(characteristics="A caller.")
    )
    case = simulator.simulate([golden], max_user_simulations=2)[0]
    return [turn for turn in case.turns if turn.role == "assistant"]


def test_the_agents_own_transcript_is_marked_as_the_agents():
    turns = _agent_turns("The agent's words.")

    assert {turn.metadata[TRANSCRIPT_SOURCE_KEY] for turn in turns} == {"agent"}


def test_a_reply_we_transcribed_is_marked_as_a_transcription():
    turns = _agent_turns(None)

    assert {turn.metadata[TRANSCRIPT_SOURCE_KEY] for turn in turns} == {
        "transcription"
    }


def test_the_source_joins_existing_metadata():
    assert transcript_source_metadata("agent", {"intended_content": "x"}) == {
        "intended_content": "x",
        TRANSCRIPT_SOURCE_KEY: "agent",
    }
    assert transcript_source_metadata(None, {"a": 1}) == {"a": 1}
