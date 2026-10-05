import logging
import time
import asyncio
from typing import Optional, Tuple, Union

from deepeval.voice.connectors.audio_utils import (
    DEFAULT_SILENCE_RMS,
    rms_pcm16,
)
from deepeval.voice.connectors.types import AgentEvent

logger = logging.getLogger(__name__)


async def collect_agent_turn(
    frames: asyncio.Queue,
    *,
    sample_rate: int,
    end_of_turn_silence_ms: float,
    frame_gap_timeout_s: float,
    max_turn_timeout_s: float,
    silence_threshold_rms: float = DEFAULT_SILENCE_RMS,
    cancel_event: Optional[asyncio.Event] = None,
    hold_event: Optional[asyncio.Event] = None,
) -> Tuple[bytes, Optional[float]]:
    """Buffer agent audio until end-of-turn, optional cancel, or timeout.

    Queue items may be raw PCM `bytes`, `AgentEvent`, or `None` / an event
    with `turn_complete=True` as an end sentinel.

    `hold_event`, while set, means the agent has gone quiet waiting on
    something deepeval is doing for it — running one of its client tools, say —
    rather than because it finished speaking. Silence proves nothing then, so
    neither the gap nor the ceiling may end the turn, and the time spent
    waiting is credited back to the deadline once the tool returns.
    """
    collected = bytearray()
    started = False
    peak_rms = 0.0
    trailing_silence_ms = 0.0
    first_audio_at: Optional[float] = None
    deadline = time.perf_counter() + max_turn_timeout_s
    held_since: Optional[float] = None

    while True:
        if cancel_event is not None and cancel_event.is_set():
            break

        now = time.perf_counter()
        holding = hold_event is not None and hold_event.is_set()
        if holding and held_since is None:
            held_since = now
        elif not holding and held_since is not None:
            deadline += now - held_since
            held_since = None
            trailing_silence_ms = 0.0

        remaining = deadline - now
        if not holding and remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(
                frames.get(),
                timeout=(
                    frame_gap_timeout_s
                    if holding
                    else min(frame_gap_timeout_s, remaining)
                ),
            )
        except asyncio.TimeoutError:
            # Re-read the hold: the wait is where the gap elapses, so a tool
            # call that began during it would otherwise be missed and the
            # silence it caused mistaken for the agent finishing.
            holding = hold_event is not None and hold_event.is_set()
            if started and not holding:
                break  # gap after speech -> end of turn
            continue  # still waiting for the agent to start speaking

        pcm, turn_complete = _coerce_frame(item)
        if turn_complete:
            break
        if pcm is None:
            continue

        frame_ms = (len(pcm) / 2 / sample_rate) * 1000.0
        rms = rms_pcm16(pcm)
        peak_rms = max(peak_rms, rms)
        silent = rms < silence_threshold_rms

        if not started:
            if silent:
                continue  # skip leading silence
            started = True
            first_audio_at = time.perf_counter()

        collected.extend(pcm)
        if silent:
            trailing_silence_ms += frame_ms
            if trailing_silence_ms >= end_of_turn_silence_ms and not holding:
                break
        else:
            trailing_silence_ms = 0.0

    logger.debug(
        "Agent turn collected: %.1fs of audio, peak_rms=%.0f, "
        "silence_threshold_rms=%.0f",
        len(collected) / 2 / sample_rate,
        peak_rms,
        silence_threshold_rms,
    )
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
