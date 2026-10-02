"""What the agent heard the caller say, read off each connector.

`Turn.user_transcript` is the agent's own STT of the caller's audio: the
hypothesis a transcription-accuracy metric scores against what was actually
spoken. It sits on the assistant turn that answers that speech, and is kept
apart from `content`, the agent's own words, because comparing the two is the
whole reason for capturing either.
"""

import asyncio
import json
import time

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Audio, Turn
from deepeval.voice import ElevenLabsConnector, WebSocketConnector
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.callback import (
    CallbackVoiceConnector,
)
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.duplex import DuplexExchange
from deepeval.voice.floor_control import FloorController
from deepeval.voice.interruption import interruption_policy


def _audio(samples: int = 2400) -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(
            b"\xe8\x03" * samples, sample_rate=24000
        ),
        "audio/wav",
    )


class _FailIfCalledSTT:
    def supports_streaming(self) -> bool:
        return False

    async def a_transcribe(self, audio):
        raise AssertionError(
            "STT should be skipped when transcript is supplied"
        )


# --------------------------------------------------------------- ElevenLabs


def test_elevenlabs_reports_what_it_heard_the_caller_say():
    connector = ElevenLabsConnector(agent_id="agent-1", api_key="key-1")

    event = connector._decode_inbound(
        json.dumps(
            {
                "type": "user_transcript",
                "user_transcription_event": {
                    "user_transcript": "I can start in two weeks"
                },
            }
        )
    )

    assert event.user_transcript == "I can start in two weeks"
    assert event.transcript is None


def test_elevenlabs_keeps_the_agents_own_reply_out_of_what_it_heard():
    connector = ElevenLabsConnector(agent_id="agent-1", api_key="key-1")

    event = connector._decode_inbound(
        json.dumps(
            {
                "type": "agent_response",
                "agent_response_event": {"agent_response": "Noted."},
            }
        )
    )

    assert event.transcript == "Noted."
    assert event.user_transcript is None


def test_elevenlabs_ignores_an_empty_caller_transcript():
    connector = ElevenLabsConnector(agent_id="agent-1", api_key="key-1")

    assert (
        connector._decode_inbound(
            json.dumps(
                {
                    "type": "user_transcript",
                    "user_transcription_event": {"user_transcript": "  "},
                }
            )
        )
        is None
    )


# ----------------------------------------------------- generic WebSocket agent


def test_a_custom_agent_can_name_where_it_reports_what_it_heard():
    connector = WebSocketConnector(
        "wss://agent.example/socket",
        receive_transcript_key="reply.text",
        receive_user_transcript_key="heard.text",
    )

    event = connector._decode_inbound(
        json.dumps({"heard": {"text": "fifteen years"}})
    )

    assert event.user_transcript == "fifteen years"
    assert event.transcript is None


def test_a_custom_agent_that_names_no_such_key_reports_nothing_heard():
    connector = WebSocketConnector(
        "wss://agent.example/socket", receive_transcript_key="reply.text"
    )

    event = connector._decode_inbound(
        json.dumps({"heard": {"text": "fifteen years"}})
    )

    assert event.user_transcript is None


# ------------------------------------------------------------------ callback


@pytest.mark.asyncio
async def test_a_text_agent_reports_the_text_it_was_handed():
    """`from_text_agent` transcribes for the agent, so it knows what it heard."""

    class _STT:
        async def a_transcribe(self, audio, **kwargs):
            return "fifty years", None

    class _TTS:
        sample_rate = 24000

        async def a_synthesize(self, text, voice=None):
            return _audio(), None

    connector = CallbackVoiceConnector.from_text_agent(
        lambda heard: f"You said {heard}", tts=_TTS(), stt=_STT()
    )

    turn = await connector.agent(_audio())

    assert turn.user_transcript == "fifty years"
    assert turn.transcript == "You said fifty years"


@pytest.mark.asyncio
async def test_duplex_records_what_the_agent_heard_on_the_reply_it_prompted():
    reply = _audio()

    async def agent(_audio_in):
        return ConnectorTurn(
            audio=reply,
            transcript="Two weeks works.",
            user_transcript="I can start in two months",
        )

    connector = CallbackVoiceConnector(agent)
    policy = interruption_policy("normal")
    assert policy is not None
    exchange = DuplexExchange(
        connector=connector,
        tts_model=object(),
        stt_model=_FailIfCalledSTT(),
        policy=policy,
        floor=FloorController(policy=policy),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=None,
        call_started_at=time.perf_counter(),
    )
    spoken = _audio()
    turns = [
        Turn(role="user", content="I can start in two weeks", audio=spoken)
    ]

    async with connector:
        await connector.stream_uplink(spoken)
        result = await asyncio.wait_for(
            exchange.run(
                turns=turns,
                sent_at=time.perf_counter(),
                barges_this_conversation=0,
            ),
            timeout=2,
        )

    assert len(result.turns) == 1
    assistant = result.turns[0]
    # What was said and what was heard disagree, which is the case the metric
    # exists to find: the reply is coherent, the hearing is not.
    assert assistant.content == "Two weeks works."
    assert assistant.user_transcript == "I can start in two months"
    assert turns[0].content == "I can start in two weeks"


# ---------------------------------------------------------------- Turn model


def test_what_the_agent_heard_stays_out_of_metric_prompts():
    """Every conversational metric reads this dump; none asked for a new key."""
    turn = Turn(
        role="assistant",
        content="Two weeks works.",
        user_transcript="I can start in two months",
    )

    dumped = turn.model_dump_for_prompt()

    assert "user_transcript" not in dumped
    assert dumped["content"] == "Two weeks works."


def test_what_the_agent_heard_survives_a_round_trip():
    turn = Turn(
        role="assistant",
        content="Two weeks works.",
        user_transcript="I can start in two months",
    )

    assert (
        Turn.model_validate(turn.model_dump()).user_transcript
        == "I can start in two months"
    )
    assert (
        Turn.model_validate(
            {
                "role": "assistant",
                "content": "Two weeks works.",
                "userTranscript": "I can start in two months",
            }
        ).user_transcript
        == "I can start in two months"
    )


def test_a_turn_without_one_defaults_to_none():
    assert Turn(role="user", content="Hello").user_transcript is None
