import time
import asyncio
from typing import Optional, Tuple

from deepeval.voice.connectors.audio_utils import is_silent, DEFAULT_SILENCE_RMS


async def collect_agent_turn(
    frames: asyncio.Queue,
    *,
    sample_rate: int,
    end_of_turn_silence_ms: float,
    frame_gap_timeout_s: float,
    max_turn_timeout_s: float,
    silence_threshold_rms: float = DEFAULT_SILENCE_RMS,
) -> Tuple[bytes, Optional[float]]:
    collected = bytearray()
    started = False
    trailing_silence_ms = 0.0
    first_audio_at: Optional[float] = None
    deadline = time.perf_counter() + max_turn_timeout_s

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            pcm = await asyncio.wait_for(
                frames.get(), timeout=min(frame_gap_timeout_s, remaining)
            )
        except asyncio.TimeoutError:
            if started:
                break  # gap after speech -> end of turn
            continue  # still waiting for the agent to start speaking

        if pcm is None:
            break  # stream closed sentinel

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
