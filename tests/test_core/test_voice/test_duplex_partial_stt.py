import asyncio
import math
import struct
import time

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.voice.connectors.types import AgentEvent
from deepeval.voice.duplex import DuplexExchange
from deepeval.voice.floor_control import FloorController

RATE = 24000
SPEECH_S = 5.0


class _SlowSTT:
    truncated_audio_pad_seconds = 0.0

    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.calls = 0

    async def a_transcribe(self, audio, **_kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return f"heard {audio.duration:.2f}s.", 1.0


class _PhoneLikeConnector:
    recv_sample_rate = RATE
    silence_threshold_rms = 300
    end_of_turn_silence_ms = 1000
    max_turn_timeout_s = 30.0
    signals_turn_complete = False

    def __init__(self):
        self.queue = asyncio.Queue()

    async def iter_agent_events(self):
        while True:
            yield await self.queue.get()

    async def stop_uplink(self):
        return None


def _frame(voiced: bool) -> bytes:
    amplitude = 8000 if voiced else 0
    return b"".join(
        struct.pack(
            "<h", int(amplitude * math.sin(2 * math.pi * 440 * n / RATE))
        )
        for n in range(RATE // 100)
    )


async def _speak(connector, seconds: float):
    voiced, silent = _frame(True), _frame(False)
    for _ in range(int(seconds * 100)):
        await connector.queue.put(
            AgentEvent(audio=voiced, received_at=time.perf_counter())
        )
        await asyncio.sleep(0.01)
    for _ in range(300):
        await connector.queue.put(
            AgentEvent(audio=silent, received_at=time.perf_counter())
        )
        await asyncio.sleep(0.01)


async def _run(stt):
    connector = _PhoneLikeConnector()
    speaker = asyncio.create_task(_speak(connector, SPEECH_S))
    turns = []
    started = time.perf_counter()
    result = await DuplexExchange(
        connector=connector,
        tts_model=None,
        stt_model=stt,
        policy=None,
        floor=FloorController(),
        golden=ConversationalGolden(scenario="Listen to the agent"),
        language="English",
        a_generate_schema=None,
        call_started_at=started,
    ).run(turns=turns, sent_at=started, barges_this_conversation=0)
    speaker.cancel()
    return turns, result


@pytest.mark.asyncio
async def test_a_partial_transcript_that_finishes_mid_reply_is_kept():
    stt = _SlowSTT(delay_s=1.5)

    _, result = await _run(stt)

    assert stt.calls >= 3
    assert result.stt_cost >= stt.calls - 1
