"""Tests for cancelable agent turn collection."""

import asyncio

import pytest

from deepeval.voice.connectors.turn_engine import collect_agent_turn
from deepeval.voice.connectors.types import AgentEvent


@pytest.mark.asyncio
async def test_collect_agent_turn_cancel_event():
    q: asyncio.Queue = asyncio.Queue()
    cancel = asyncio.Event()

    async def producer():
        # Non-silent frame (high amplitude)
        await q.put(AgentEvent(audio=b"\xff\x7f" * 480))
        await asyncio.sleep(0.05)
        cancel.set()
        await q.put(AgentEvent(audio=b"\xff\x7f" * 480))

    asyncio.create_task(producer())
    pcm, first_at = await collect_agent_turn(
        q,
        sample_rate=24000,
        end_of_turn_silence_ms=5000,
        frame_gap_timeout_s=1.0,
        max_turn_timeout_s=5.0,
        cancel_event=cancel,
    )
    assert first_at is not None
    assert len(pcm) > 0


@pytest.mark.asyncio
async def test_collect_agent_turn_complete_event():
    q: asyncio.Queue = asyncio.Queue()
    await q.put(AgentEvent(audio=b"\xff\x7f" * 480))
    await q.put(AgentEvent(turn_complete=True))
    pcm, first_at = await collect_agent_turn(
        q,
        sample_rate=24000,
        end_of_turn_silence_ms=5000,
        frame_gap_timeout_s=1.0,
        max_turn_timeout_s=5.0,
    )
    assert first_at is not None
    assert len(pcm) > 0
