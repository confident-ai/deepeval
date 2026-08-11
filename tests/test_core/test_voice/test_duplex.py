import asyncio
import time

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Audio, Turn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.callback import CallbackVoiceConnector
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.duplex import DuplexExchange
from deepeval.voice.floor_control import FloorController
from deepeval.voice.interruption import interruption_policy


class FailIfCalledSTT:
    async def a_transcribe(self, audio):
        raise AssertionError(
            "STT should be skipped when transcript is supplied"
        )


@pytest.mark.asyncio
async def test_duplex_waits_for_delayed_callback_event_without_closing_stream():
    reply_wav = audio_utils.pcm16_to_wav_bytes(
        b"\xe8\x03" * 2400, sample_rate=24000
    )
    reply_audio = Audio.from_bytes(reply_wav, "audio/wav")

    async def delayed_agent(_audio):
        await asyncio.sleep(0.1)
        return ConnectorTurn(audio=reply_audio, transcript="Agent reply")

    connector = CallbackVoiceConnector(delayed_agent)
    policy = interruption_policy("normal")
    assert policy is not None
    exchange = DuplexExchange(
        connector=connector,
        tts_model=object(),
        stt_model=FailIfCalledSTT(),
        voice_config=object(),
        policy=policy,
        floor=FloorController(policy=policy),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=None,
        call_started_at=time.perf_counter(),
    )
    input_audio = Audio.from_bytes(reply_wav, "audio/wav")
    turns = [Turn(role="user", content="Hello", audio=input_audio)]

    async with connector:
        await connector.stream_uplink(input_audio)
        result = await asyncio.wait_for(
            exchange.run(
                turns=turns,
                sent_at=time.perf_counter(),
                barges_this_conversation=0,
            ),
            timeout=2,
        )

    assert len(result.turns) == 1
    assert result.turns[0].role == "assistant"
    assert result.turns[0].content == "Agent reply"
    assert result.turns[0].audio is not None


@pytest.mark.asyncio
async def test_uplink_drops_downlink_left_over_from_the_previous_reply():
    """A new turn must not inherit the previous reply's queued frames.

    The reply task paces frames in real time, so it keeps enqueueing long
    after an exchange has finished. Leaking them makes the next assistant turn
    start at the same instant as the user turn and carry the wrong audio.
    """
    long_reply = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(b"\xe8\x03" * 48000, sample_rate=24000),
        "audio/wav",
    )

    async def agent(_audio):
        return ConnectorTurn(audio=long_reply, transcript="Agent reply")

    connector = CallbackVoiceConnector(agent)
    probe = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(b"\xe8\x03" * 240, sample_rate=24000),
        "audio/wav",
    )

    async with connector:
        await connector.stream_uplink(probe)
        # Let the reply task queue a few frames without consuming them.
        await asyncio.sleep(0.1)
        assert not connector._events.empty()

        await connector.stream_uplink(probe)
        assert connector._events.empty()
