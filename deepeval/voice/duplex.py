"""Duplex voice exchange: LLM barge judge + floor control."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Audio, Turn
from deepeval.voice.background import mix_background
from deepeval.voice.connectors import audio_utils
from deepeval.voice.floor_control import FloorController, FloorState
from deepeval.voice.interruption import (
    InterruptionPolicy,
    should_poll_judge,
)
from deepeval.simulator.interrupt_template import SimulatorInterruptTemplate
from deepeval.simulator.schema import InterruptDecision

if TYPE_CHECKING:
    from deepeval.models.base_model import DeepEvalBaseSTT, DeepEvalBaseTTS
    from deepeval.voice.connectors.transports.base import BaseVoiceConnector
    from deepeval.voice.config import VoiceConfig

logger = logging.getLogger(__name__)


@dataclass
class DuplexExchangeResult:
    """Turns produced by one duplex user→agent exchange (may include barges)."""

    turns: List[Turn] = field(default_factory=list)
    tts_cost: float = 0.0
    stt_cost: float = 0.0
    frustrated: bool = False
    barges: int = 0


def _pcm_to_audio(pcm: bytes, sample_rate: int) -> Audio:
    wav = audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1)
    duration = (len(pcm) / 2) / sample_rate if sample_rate else None
    return Audio.from_bytes(
        wav,
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=duration,
    )


def _recv_rate(connector: "BaseVoiceConnector") -> int:
    rate = getattr(connector, "recv_sample_rate", None)
    if rate:
        return int(rate)
    fmt = connector.audio_format
    return int(fmt[0])


class DuplexExchange:
    """Runs one duplex listen/barge cycle after the initial user audio is sent."""

    def __init__(
        self,
        *,
        connector: "BaseVoiceConnector",
        tts_model: "DeepEvalBaseTTS",
        stt_model: "DeepEvalBaseSTT",
        voice_config: "VoiceConfig",
        policy: InterruptionPolicy,
        floor: FloorController,
        golden: ConversationalGolden,
        language: str,
        a_generate_schema,
        call_started_at: float,
    ):
        self.connector = connector
        self.tts_model = tts_model
        self.stt_model = stt_model
        self.voice_config = voice_config
        self.policy = policy
        self.floor = floor
        self.golden = golden
        self.language = language
        self.a_generate_schema = a_generate_schema
        self.sample_rate = _recv_rate(connector)
        self.call_started_at = call_started_at
        self.tts_kwargs = (
            golden.persona.tts_kwargs() if golden.persona is not None else {}
        )
        self.background_noise = (
            golden.persona.background_noise
            if golden.persona is not None
            else None
        )

    async def run(
        self,
        *,
        turns: List[Turn],
        sent_at: float,
        barges_this_conversation: int,
    ) -> DuplexExchangeResult:
        result = DuplexExchangeResult(frustrated=self.floor.frustrated)
        self.floor.reset_turn()

        agent_pcm = bytearray()
        partial_transcript = ""
        first_audio_at: Optional[float] = None
        trailing_silence_ms = 0.0
        agent_started = False
        last_judged_len = 0
        last_judge_at: Optional[float] = None
        barges_this_agent_turn = 0
        judge_task: Optional[asyncio.Task] = None
        uplink_task: Optional[asyncio.Task] = None
        pending_decision: Optional[InterruptDecision] = None
        stt_task: Optional[asyncio.Task] = None
        event_task: Optional[asyncio.Task] = None
        last_stt_pcm_len = 0
        connector_transcript_seen = False
        eot_silence = getattr(self.connector, "end_of_turn_silence_ms", 800)
        max_timeout = getattr(self.connector, "max_turn_timeout_s", 30.0)
        deadline = time.perf_counter() + max_timeout
        exchange_done = False
        interrupted_assistant = False

        async def _maybe_stop_uplink() -> None:
            nonlocal uplink_task
            await self.connector.stop_uplink()
            self.floor.on_user_uplink_stop()
            if uplink_task is not None and not uplink_task.done():
                uplink_task.cancel()
                try:
                    await uplink_task
                except (asyncio.CancelledError, Exception):
                    pass
            uplink_task = None

        async def _start_barge(utterance: str, *, frustrated: bool) -> None:
            nonlocal uplink_task, barges_this_agent_turn, barges_this_conversation
            barge_audio, tts_cost = await self.tts_model.a_synthesize(
                utterance, **self.tts_kwargs
            )
            if tts_cost is not None:
                result.tts_cost += tts_cost
            barge_audio = mix_background(barge_audio, self.background_noise)
            now = time.perf_counter()
            barge_audio.start_time = max(0.0, now - self.call_started_at)
            meta = {"barge_in": True}
            if frustrated:
                meta["frustrated"] = True
            turns.append(
                Turn(
                    role="user",
                    content=utterance,
                    audio=barge_audio,
                    metadata=meta,
                )
            )
            self.floor.on_user_barge_start(now)
            barges_this_agent_turn += 1
            barges_this_conversation += 1
            result.barges += 1
            # No trailing silence on barge — we want overlap, not VAD pad.
            uplink_task = asyncio.create_task(
                self.connector.stream_uplink(
                    barge_audio, trailing_silence=False
                )
            )

        async def _finalize_assistant(
            *, interrupted: bool, content: Optional[str] = None
        ) -> None:
            nonlocal agent_pcm, first_audio_at, partial_transcript
            text = content if content is not None else partial_transcript
            has_audio = len(agent_pcm) > 0
            audio = (
                _pcm_to_audio(bytes(agent_pcm), self.sample_rate)
                if has_audio
                else None
            )
            if audio is not None and first_audio_at is not None:
                audio.start_time = max(
                    0.0, first_audio_at - self.call_started_at
                )
            if not text and has_audio:
                text, stt_cost = await self.stt_model.a_transcribe(audio)
                if stt_cost is not None:
                    result.stt_cost += stt_cost
            latency_ms = (
                (first_audio_at - sent_at) * 1000.0
                if first_audio_at is not None
                else None
            )
            turns.append(
                Turn(
                    role="assistant",
                    content=text or "",
                    audio=audio,
                    latency_ms=latency_ms,
                    interrupted=interrupted if interrupted else None,
                )
            )
            result.turns.append(turns[-1])
            # Reset buffers for a possible post-barge agent reply.
            agent_pcm = bytearray()
            partial_transcript = ""
            first_audio_at = None
            # Keep sent_at as original for first reply latency only.

        events = self.connector.iter_agent_events()
        try:
            while not exchange_done and time.perf_counter() < deadline:
                now = time.perf_counter()

                # Consume pending judge result without blocking the audio loop.
                if judge_task is not None and judge_task.done():
                    try:
                        pending_decision = judge_task.result()
                    except Exception:
                        logger.exception("Interrupt judge failed")
                        pending_decision = None
                    judge_task = None

                if (
                    pending_decision is not None
                    and pending_decision.should_interrupt
                    and self.floor.can_run_judge
                    and pending_decision.utterance
                ):
                    decision = pending_decision
                    pending_decision = None
                    await _start_barge(
                        decision.utterance, frustrated=self.floor.frustrated
                    )

                # Floor tick (grace / awkward silence / retry).
                action = self.floor.tick(now)
                if action.mark_frustrated:
                    result.frustrated = True
                    if (
                        turns
                        and turns[-1].role == "user"
                        and turns[-1].metadata
                    ):
                        turns[-1].metadata = {
                            **turns[-1].metadata,
                            "frustrated": True,
                            "grace_missed_ms": self.floor.interrupt_grace_ms,
                        }
                    elif turns and turns[-1].role == "user":
                        turns[-1].metadata = {
                            "barge_in": True,
                            "frustrated": True,
                            "grace_missed_ms": self.floor.interrupt_grace_ms,
                        }
                if action.stop_uplink:
                    await _maybe_stop_uplink()
                if action.barge_succeeded:
                    interrupted_assistant = True
                    await _finalize_assistant(interrupted=True)
                    interrupted_assistant = False
                    # Continue listening for the agent's next reply.
                    self.floor.reset_turn()
                    agent_started = False
                    trailing_silence_ms = 0.0
                    barges_this_agent_turn = 0
                    last_judged_len = 0
                    last_stt_pcm_len = 0
                if action.retry_barge and self.floor.frustrated:
                    # Ask judge once more with frustration for retry content.
                    retry = await self._judge(
                        turns=turns,
                        partial=partial_transcript or "(agent still speaking)",
                        frustrated=True,
                    )
                    if retry.should_interrupt and retry.utterance:
                        await _start_barge(retry.utterance, frustrated=True)
                    else:
                        # Fall back to a short frustrated prompt utterance.
                        await _start_barge("Sorry — go ahead.", frustrated=True)

                # Pull next agent event with a short timeout so timers keep ticking.
                if event_task is None:
                    event_task = asyncio.create_task(events.__anext__())
                try:
                    event = await asyncio.wait_for(
                        asyncio.shield(event_task), timeout=0.05
                    )
                except asyncio.TimeoutError:
                    # End-of-turn via trailing silence while agent was speaking.
                    if (
                        agent_started
                        and not self.floor.user_uplink_active
                        and trailing_silence_ms >= eot_silence
                        and not self.floor.agent_speaking
                    ):
                        await _finalize_assistant(
                            interrupted=interrupted_assistant
                        )
                        exchange_done = True
                    continue
                except StopAsyncIteration:
                    event_task = None
                    break
                event_task = None

                now = time.perf_counter()
                if event.transcript:
                    partial_transcript = event.transcript
                    connector_transcript_seen = True

                if event.audio:
                    silent = audio_utils.is_silent(event.audio)
                    frame_ms = (
                        len(event.audio) / 2 / self.sample_rate
                    ) * 1000.0
                    if not silent:
                        if not agent_started or not self.floor.agent_speaking:
                            self.floor.on_agent_speech_start(now)
                            agent_started = True
                            trailing_silence_ms = 0.0
                        if first_audio_at is None:
                            first_audio_at = now
                        agent_pcm.extend(event.audio)
                    else:
                        if self.floor.agent_speaking:
                            trailing_silence_ms += frame_ms
                            if trailing_silence_ms >= eot_silence:
                                end_action = self.floor.on_agent_speech_end(now)
                                if end_action.barge_succeeded:
                                    interrupted_assistant = True
                                    await _finalize_assistant(interrupted=True)
                                    self.floor.reset_turn()
                                    agent_started = False
                                    trailing_silence_ms = 0.0
                                    barges_this_agent_turn = 0
                                    last_judged_len = 0
                                elif (
                                    agent_started
                                    and not self.floor.user_uplink_active
                                ):
                                    await _finalize_assistant(
                                        interrupted=interrupted_assistant
                                    )
                                    exchange_done = True
                                    continue
                        agent_pcm.extend(event.audio)

                    # Streaming STT fallback when no platform transcript.
                    if (
                        not partial_transcript
                        and not connector_transcript_seen
                        and len(agent_pcm) - last_stt_pcm_len
                        > self.sample_rate * 2  # ~1s of new audio
                        and (stt_task is None or stt_task.done())
                    ):
                        last_stt_pcm_len = len(agent_pcm)
                        snap = bytes(agent_pcm)

                        async def _stt_partial(pcm=snap):
                            audio = _pcm_to_audio(pcm, self.sample_rate)
                            text, cost = await self.stt_model.a_transcribe(
                                audio
                            )
                            return text, cost

                        stt_task = asyncio.create_task(_stt_partial())

                if stt_task is not None and stt_task.done():
                    try:
                        text, cost = stt_task.result()
                        if text:
                            partial_transcript = text
                        if cost is not None:
                            result.stt_cost += cost
                    except Exception:
                        logger.exception("Partial STT failed")
                    stt_task = None

                if event.turn_complete:
                    if self.floor.user_uplink_active:
                        # Agent ended while we were barging — treat as success.
                        end_action = self.floor.on_agent_speech_end(now)
                        if end_action.barge_succeeded or True:
                            interrupted_assistant = True
                            await _finalize_assistant(interrupted=True)
                            self.floor.reset_turn()
                            agent_started = False
                            trailing_silence_ms = 0.0
                            # Keep listening for follow-up agent speech briefly.
                            continue
                    elif agent_started:
                        await _finalize_assistant(
                            interrupted=interrupted_assistant
                        )
                        exchange_done = True
                        continue

                # Post-interrupt yield if armed and agent talking outside grace hold.
                if self.floor.should_stop_user_for_agent_speech:
                    # During GRACE_WAIT/BARGING we hold the line; tick handles
                    # yield. If state drifted to LISTENING with uplink still
                    # on, cut now.
                    if self.floor.state == FloorState.LISTENING:
                        await _maybe_stop_uplink()

                # Schedule interrupt judge when listening to agent speech.
                if (
                    self.floor.can_run_judge
                    and partial_transcript
                    and judge_task is None
                    and should_poll_judge(
                        policy=self.policy,
                        partial_transcript=partial_transcript,
                        last_judged_len=last_judged_len,
                        last_judge_at=last_judge_at,
                        now=now,
                        barges_this_conversation=barges_this_conversation,
                        barges_this_agent_turn=barges_this_agent_turn,
                    )
                ):
                    last_judged_len = len(partial_transcript)
                    last_judge_at = now
                    snap_turns = list(turns)
                    snap_partial = partial_transcript
                    frustrated = self.floor.frustrated

                    async def _run_judge(
                        t=snap_turns, p=snap_partial, f=frustrated
                    ):
                        return await self._judge(
                            turns=t, partial=p, frustrated=f
                        )

                    judge_task = asyncio.create_task(_run_judge())

        finally:
            if judge_task is not None and not judge_task.done():
                judge_task.cancel()
            if event_task is not None:
                if not event_task.done():
                    event_task.cancel()
                try:
                    await event_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            if stt_task is not None:
                if not stt_task.done():
                    stt_task.cancel()
                try:
                    await stt_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Partial STT failed during cleanup")
            await _maybe_stop_uplink()
            # If we never finalized but collected audio, emit assistant turn.
            if agent_pcm and (
                not result.turns or result.turns[-1].role != "assistant"
            ):
                await _finalize_assistant(interrupted=interrupted_assistant)

        result.frustrated = self.floor.frustrated or result.frustrated
        return result

    async def _judge(
        self,
        *,
        turns: List[Turn],
        partial: str,
        frustrated: bool,
    ) -> InterruptDecision:
        prompt = SimulatorInterruptTemplate.decide_interrupt(
            golden=self.golden,
            turns=turns,
            partial_agent_transcript=partial,
            interruption_level=self.policy.level,
            language=self.language,
            frustrated=frustrated,
        )
        decision = await self.a_generate_schema(prompt, InterruptDecision)
        if decision.should_interrupt and not decision.utterance:
            decision.should_interrupt = False
        return decision
