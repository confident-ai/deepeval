import time
import asyncio
from typing import Optional, Tuple, Union

from deepeval.voice.connectors.audio_utils import is_silent, DEFAULT_SILENCE_RMS
from deepeval.voice.connectors.types import AgentEvent


async def collect_agent_turn(
    frames: asyncio.Queue,
    *,
    sample_rate: int,
    end_of_turn_silence_ms: float,
    frame_gap_timeout_s: float,
    max_turn_timeout_s: float,
    silence_threshold_rms: float = DEFAULT_SILENCE_RMS,
    cancel_event: Optional[asyncio.Event] = None,
) -> Tuple[bytes, Optional[float]]:
    """Buffer agent audio until end-of-turn, optional cancel, or timeout.

    Queue items may be raw PCM `bytes`, `AgentEvent`, or `None` / an event
    with `turn_complete=True` as an end sentinel.
    """
    collected = bytearray()
    started = False
    trailing_silence_ms = 0.0
    first_audio_at: Optional[float] = None
    deadline = time.perf_counter() + max_turn_timeout_s

    while True:
        if cancel_event is not None and cancel_event.is_set():
            break
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(
                frames.get(), timeout=min(frame_gap_timeout_s, remaining)
            )
        except asyncio.TimeoutError:
            if started:
                break  # gap after speech -> end of turn
            continue  # still waiting for the agent to start speaking

        pcm, turn_complete = _coerce_frame(item)
        if turn_complete:
            break
        if pcm is None:
            continue

        frame_ms = (len(pcm) / 2 / sample_rate) * 1000.0
        silent = is_silent(pcm, silence_threshold_rms)

        if not started:
            if silent:
                continue  # skip leading silence
            started = True
            first_audio_at = time.perf_counter()

        collected.extend(pcm)
        if silent:
            trailing_silence_ms += frame_ms
            if trailing_silence_ms >= end_of_turn_silence_ms:
                break
        else:
            trailing_silence_ms = 0.0

    return bytes(collected), first_audio_at


def _coerce_frame(
    item: Union[None, bytes, bytearray, AgentEvent],
) -> Tuple[Optional[bytes], bool]:
    if item is None:
        return None, True
    if isinstance(item, AgentEvent):
        if item.turn_complete:
            return item.audio, True
        return item.audio, False
    if isinstance(item, (bytes, bytearray)):
        return bytes(item), False
    return None, False
