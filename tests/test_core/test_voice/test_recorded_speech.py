"""Recorded caller lines and voice events on simulation graph nodes."""

import asyncio

import pytest

from deepeval.dataset.golden import ConversationalGolden, Persona
from deepeval.simulator import ConversationSimulator, SimulationNode
from deepeval.simulator.events import (
    Hold,
    MidSentencePause,
    Prosody,
    SideSpeech,
)
from deepeval.test_case import Audio, Turn
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.recorded_speech import assemble_utterance
from deepeval.voice.timeline import audio_duration
from tests.test_core.test_voice.test_voice_simulation import (
    _RATE,
    _StaticVoiceModel,
    _StubSTT,
    _StubTTS,
    _tone,
)


async def _spoken(text: str) -> Audio:
    return _tone(0.2)


def _assemble(text, recording, events):
    return asyncio.run(
        assemble_utterance(text, recording, events, _RATE, _spoken)
    )


def test_a_recording_is_used_as_the_line():
    audio, records = _assemble(
        "My number is four four one seven.", _tone(1.0), []
    )

    assert audio_duration(audio) == pytest.approx(1.0)
    assert records == []


def test_a_pause_is_laid_into_the_line_where_its_words_fall():
    audio, records = _assemble(
        "one two three four",
        _tone(2.0),
        [MidSentencePause(seconds=1.5, after_words=1)],
    )

    assert audio_duration(audio) == pytest.approx(3.5)
    assert records[0]["type"] == "MidSentencePause"
    assert records[0]["offset_seconds"] == pytest.approx(0.5)
    assert records[0]["seconds"] == pytest.approx(1.5)


def test_an_aside_and_a_hold_come_before_the_line_in_the_order_given():
    audio, records = _assemble(
        "Okay, found it.",
        _tone(1.0),
        [
            SideSpeech(text="Where's the letter?"),
            Hold(seconds=5.0),
            Prosody(cue="tired"),
        ],
    )

    aside, hold, prosody = records
    assert aside["offset_seconds"] == 0.0
    assert hold["offset_seconds"] == pytest.approx(0.2 + 0.4)
    assert hold["seconds"] == 5.0
    assert audio_duration(audio) == pytest.approx(0.2 + 0.4 + 5.0 + 1.0)
    # A label-only event is recorded without a place in the audio.
    assert prosody == {"type": "Prosody", "cue": "tired"}


def test_a_node_that_returns_a_recording_is_played_not_synthesized():
    heard = []

    async def agent(audio: Audio) -> ConnectorTurn:
        heard.append(audio_duration(audio))
        return ConnectorTurn(audio=_tone(0.4), transcript="the agent speaks")

    def give_number():
        return Turn(
            role="user", content="Four four one seven.", audio=_tone(1.0)
        )

    node = SimulationNode(
        action=give_number,
        ending="done",
        events=[Hold(seconds=2.0)],
    )
    tts = _StubTTS()
    simulator = ConversationSimulator(
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(agent, turn_detection="eager"),
            tts_model=tts,
            stt_model=_StubSTT(),
            output_dir=None,
        ),
        simulator_model=_StaticVoiceModel(),
        simulation_graph=node,
    )
    golden = ConversationalGolden(
        scenario="Give a card number",
        persona=Persona(characteristics="A caller."),
    )

    case = simulator.simulate([golden], max_user_simulations=2)[0]

    assert tts.spoken == []
    assert heard[0] == pytest.approx(3.0, abs=0.1)
    user_turn = case.turns[0]
    assert user_turn.metadata["events"] == [
        {"type": "Hold", "seconds": 2.0, "offset_seconds": 0.0}
    ]
    assert case.metadata["Simulation ending"] == "done"
