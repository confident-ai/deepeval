"""Duplex voice exchange: LLM barge judge + floor control."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Audio, Turn
from deepeval.voice.background import BackgroundMixer, mix_background
from deepeval.voice.connectors import audio_utils
from deepeval.voice.floor_control import FloorController, FloorState
from deepeval.voice.streaming import PcmRecorder
from deepeval.voice.interruption import (
    InterruptionPolicy,
    should_poll_judge,
)
from deepeval.voice.timeline import audio_duration
from deepeval.simulator.interrupt_template import SimulatorInterruptTemplate
from deepeval.simulator.schema import InterruptDecision

if TYPE_CHECKING:
    from deepeval.models.base_model import DeepEvalBaseSTT, DeepEvalBaseTTS
    from deepeval.voice.connectors.transports.base import BaseVoiceConnector
    from deepeval.voice.config import VoiceConfig

logger = logging.getLogger(__name__)

# How long the loop blocks on the next downlink event before servicing timers.
_EVENT_POLL_S = 0.05

# Speaking rate used to judge how much of a reply the caller has heard when the
# connector hands us text ahead of the audio. Deliberately below conversational
# English (~15 chars/s) so the estimate lags the speech rather than running past
# it: showing the judge words that have not been spoken is the failure that
# matters, and hearing slightly less than was said is what interrupting is like.
_AGENT_CHARS_PER_SECOND = 13.0


@dataclass
class _Barge:
    """A barge from the moment it is decided on until it has all been said.

    Deciding to cut in and being heard cutting in are separate moments: the
    words have to be synthesized and put on the uplink in between. `turn` is
    filled in at the second moment, so it doubles as the record of whether the
    agent has heard any of this yet.
    """

    utterance: str
    frustrated: bool
    task: Optional[asyncio.Task] = None
    turn: Optional[Turn] = None


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


def _padded_for_transcription(
    pcm: bytes, sample_rate: int, pad_seconds: float
) -> Audio:
    """The same speech ending in silence, for transcription only.

    How much silence a transcriber needs to accept a clipped word as the end of
    an utterance is its own property, so the amount comes from the STT model.
    The turn keeps the unpadded audio, so its duration and placement on the call
    timeline stay true to what the caller heard.
    """
    silence = b"\x00\x00" * int(sample_rate * pad_seconds)
    return _pcm_to_audio(pcm + silence, sample_rate)


def _spoken_prefix(transcript: str, heard_seconds: float) -> str:
    """The part of `transcript` the caller can actually have heard by now.

    A connector's transcript describes the reply the agent decided to make, and
    it runs ahead of the speech — some platforms send the whole utterance with
    the first audio frame. A caller only knows what has reached their ear, so the
    text is capped at what the delivered audio can account for and rounded down
    to a whole word.
    """
    if not transcript:
        return transcript
    budget = int(heard_seconds * _AGENT_CHARS_PER_SECOND)
    if budget >= len(transcript):
        return transcript
    if budget <= 0:
        return ""
    head, sep, _ = transcript[:budget].rpartition(" ")
    return head if sep else ""


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
        barge: Optional[_Barge] = None
        pending_decision: Optional[InterruptDecision] = None
        stt_task: Optional[asyncio.Task] = None
        event_task: Optional[asyncio.Task] = None
        last_stt_pcm_len = 0
        connector_transcript_seen = False
        # Transports that close each turn explicitly are believed over silence.
        awaiting_end_signal = self.connector.signals_turn_complete
        eot_silence = getattr(self.connector, "end_of_turn_silence_ms", 800)
        max_timeout = getattr(self.connector, "max_turn_timeout_s", 30.0)
        deadline = time.perf_counter() + max_timeout
        exchange_done = False
        interrupted_assistant = False
        # Where in `turns` the agent's current utterance belongs, and which
        # uplink it is answering. Both move on once an utterance is recorded.
        utterance_index = len(turns)
        utterance_sent_at = sent_at
        last_uplink_at = sent_at

        def _heard_transcript() -> str:
            """What the caller has heard of the reply now in progress.

            The judge stands in for someone listening to the call, so it decides
            whether to cut in on this rather than on the reply's full text —
            otherwise it answers questions the agent has not asked yet.
            """
            if not connector_transcript_seen:
                # Transcribed from the audio already delivered, so it can only
                # contain speech that played.
                return partial_transcript
            heard_s = (
                (len(agent_pcm) / 2) / self.sample_rate
                if self.sample_rate
                else 0.0
            )
            return _spoken_prefix(partial_transcript, heard_s)

        async def _maybe_stop_uplink() -> None:
            """Stop sending the caller's speech, keeping whatever went out.

            The transport is asked to stop rather than having the send cancelled
            from here: a barge that was partly spoken still has to be recorded
            as what the caller said, and killing the task mid-utterance would
            throw that away.
            """
            await self.connector.stop_uplink()
            self.floor.on_user_uplink_stop()

        def _take_floor(barge: _Barge, sent_at: float) -> None:
            """Record the caller cutting in, at the moment the agent hears it."""
            nonlocal barges_this_agent_turn, barges_this_conversation
            nonlocal last_uplink_at
            meta = {"barge_in": True}
            if barge.frustrated:
                meta["frustrated"] = True
            barge.turn = Turn(
                role="user", content=barge.utterance, metadata=meta
            )
            turns.append(barge.turn)
            self.floor.on_user_barge_start(sent_at)
            # The agent cannot answer the barge until the caller finishes it,
            # which is only known once the last frame is out; until then the
            # start of the barge is the best floor for its reply's wait.
            last_uplink_at = sent_at
            barges_this_agent_turn += 1
            barges_this_conversation += 1
            result.barges += 1

        def _finish_barge(barge: _Barge, audio: Audio, sent_at: float) -> None:
            """Attach what the barge sounded like once its last frame is out."""
            nonlocal last_uplink_at, utterance_sent_at
            if barge.turn is None:
                return
            audio.start_time = max(0.0, sent_at - self.call_started_at)
            barge.turn.audio = audio
            ended = sent_at + (audio_duration(audio) or 0.0)
            if utterance_sent_at == sent_at:
                # A reply was finalized mid-barge and is waiting on this
                # utterance, so its wait starts where the caller stopped.
                utterance_sent_at = ended
            last_uplink_at = ended

        async def _speak_barge(barge: _Barge) -> None:
            """Put a barge on the uplink as it is synthesized, off the loop.

            A caller settles on what to say and then says it; nothing the agent
            utters in between changes the words or holds them back. Waiting for
            the finished clip before sending any of it put the whole TTS round
            trip between the decision and the first word, which the recording
            shows as the agent falling silent seconds before the caller cuts in
            — a pause no live call would contain. The caller takes the floor
            when the transport says the speech is going out, which for one that
            needs the utterance whole is still the end of synthesis.
            """
            # No trailing silence on a barge — we want overlap, not VAD pad.
            if not self.tts_model.supports_streaming():
                audio, tts_cost = await self.tts_model.a_synthesize(
                    barge.utterance, **self.tts_kwargs
                )
                if tts_cost is not None:
                    result.tts_cost += tts_cost
                audio = mix_background(audio, self.background_noise)
                sent_at = time.perf_counter()
                _take_floor(barge, sent_at)
                _finish_barge(barge, audio, sent_at)
                await self.connector.stream_uplink(
                    audio, trailing_silence=False
                )
                return

            mixer = BackgroundMixer(self.background_noise)
            recorder = PcmRecorder()
            sent_at: Optional[float] = None

            def _audible(at: float) -> None:
                nonlocal sent_at
                sent_at = at
                _take_floor(barge, at)

            async def _frames():
                async for chunk in self.tts_model.a_synthesize_stream(
                    barge.utterance, **self.tts_kwargs
                ):
                    mixed = mixer.mix_chunk(chunk)
                    recorder.add(mixed)
                    yield mixed

            try:
                await self.connector.stream_uplink_chunks(
                    _frames(),
                    trailing_silence=False,
                    on_first_frame=_audible,
                )
            finally:
                tts_cost = self.tts_model.synthesis_cost(barge.utterance)
                if tts_cost is not None:
                    result.tts_cost += tts_cost
                if sent_at is not None:
                    _finish_barge(barge, recorder.to_audio(), sent_at)

        async def _speak_frustrated_barge(barge: _Barge) -> None:
            retry = await self._judge(
                turns=list(turns),
                partial=_heard_transcript() or "(agent still speaking)",
                frustrated=True,
            )
            barge.utterance = (
                retry.utterance
                if retry.should_interrupt and retry.utterance
                else "Sorry — go ahead."
            )
            await _speak_barge(barge)

        def _start_barge(utterance: str, *, frustrated: bool) -> _Barge:
            barge = _Barge(utterance=utterance, frustrated=frustrated)
            barge.task = asyncio.create_task(_speak_barge(barge))
            return barge

        async def _finalize_assistant(
            *, interrupted: bool, content: Optional[str] = None
        ) -> None:
            nonlocal agent_pcm, first_audio_at, partial_transcript
            nonlocal utterance_index, utterance_sent_at
            text = content if content is not None else partial_transcript
            spoken_pcm = bytes(agent_pcm)
            has_audio = len(spoken_pcm) > 0
            audio = (
                _pcm_to_audio(spoken_pcm, self.sample_rate)
                if has_audio
                else None
            )
            if audio is not None and first_audio_at is not None:
                audio.start_time = max(
                    0.0, first_audio_at - self.call_started_at
                )
            metadata = None
            # The agent never said it had finished, so this turn ended because
            # we stopped listening — its recording can hold less than the
            # transcript describes.
            unfinished = awaiting_end_signal and not interrupted
            if has_audio and (interrupted or unfinished or not text):
                # Connectors that carry a transcript send the whole utterance,
                # including the part a barge stopped the agent from saying.
                # Transcribe the audio that actually played so `content` is
                # what the caller heard, and keep the rest as evidence of what
                # the agent was cut off from saying.
                pad_s = getattr(
                    self.stt_model, "truncated_audio_pad_seconds", 0.0
                )
                to_transcribe = audio
                if interrupted and pad_s > 0:
                    to_transcribe = _padded_for_transcription(
                        spoken_pcm, self.sample_rate, pad_s
                    )
                spoken, stt_cost = await self.stt_model.a_transcribe(
                    to_transcribe
                )
                if stt_cost is not None:
                    result.stt_cost += stt_cost
                if text and text != spoken:
                    metadata = {"intended_content": text}
                    if unfinished:
                        metadata["ended_without_agent_signal"] = True
                text = spoken
            latency_ms = (
                (first_audio_at - utterance_sent_at) * 1000.0
                if first_audio_at is not None
                else None
            )
            turn = Turn(
                role="assistant",
                content=text or "",
                audio=audio,
                latency_ms=latency_ms,
                interrupted=interrupted if interrupted else None,
                metadata=metadata,
            )
            # Place the reply where it began rather than at the end: a barge is
            # appended the moment it fires, so appending here would record the
            # caller cutting in *before* the agent they cut off.
            turns.insert(utterance_index, turn)
            result.turns.append(turn)
            # Reset buffers for a possible post-barge agent reply.
            agent_pcm = bytearray()
            partial_transcript = ""
            first_audio_at = None
            utterance_index = len(turns)
            # Anything the agent says next is a reply to the most recent
            # uplink, not to the one that opened this exchange.
            utterance_sent_at = last_uplink_at

        async def _advance_end_of_turn(elapsed_ms: float, now: float) -> bool:
            """Advance the end-of-turn timer. True when the exchange is over.

            Driven both by silent downlink frames and by stretches where no
            frame arrives at all: an agent that yields the floor to a barge
            stops sending audio rather than sending quiet audio, so silence
            measured only from received frames never elapses and a successful
            interruption is misread as the agent talking through it.

            Silence only decides the end of a turn for transports that never
            say so themselves. When one does, quiet mid-reply is the agent
            pausing, and ending the turn there abandons the rest of the reply
            while keeping the transcript of all of it. A barge is still settled
            on silence either way: it is asking whether the agent has stopped,
            and waiting for a signal that may never come would hold the caller
            mid-interruption.
            """
            nonlocal trailing_silence_ms, agent_started, interrupted_assistant
            nonlocal barges_this_agent_turn, last_judged_len, last_stt_pcm_len
            nonlocal awaiting_end_signal

            if self.floor.agent_speaking:
                trailing_silence_ms += elapsed_ms
                if trailing_silence_ms < eot_silence:
                    return False
                if awaiting_end_signal and not self.floor.barge_in_progress:
                    return False
                if self.floor.on_agent_speech_end(now).barge_succeeded:
                    await _finalize_assistant(interrupted=True)
                    interrupted_assistant = False
                    self.floor.reset_turn()
                    agent_started = False
                    trailing_silence_ms = 0.0
                    barges_this_agent_turn = 0
                    last_judged_len = 0
                    last_stt_pcm_len = 0
                    awaiting_end_signal = self.connector.signals_turn_complete
                    return False

            if (
                trailing_silence_ms >= eot_silence
                and agent_started
                and not awaiting_end_signal
                and not self.floor.agent_speaking
                and not self.floor.user_uplink_active
            ):
                await _finalize_assistant(interrupted=interrupted_assistant)
                return True
            return False

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

                # A barge takes the floor from inside its own task, the moment
                # the transport starts sending it — not when it was decided on.
                # Until then the agent holds the floor and its frames keep being
                # recorded, so the two voices overlap as they would on a live
                # call. The slot stays occupied until the barge has been said in
                # full, since a caller cannot start a second one over it.
                if barge is not None and barge.task.done():
                    try:
                        barge.task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception("Barge failed")
                    barge = None

                if (
                    pending_decision is not None
                    and pending_decision.should_interrupt
                    and self.floor.can_run_judge
                    and pending_decision.utterance
                    and barge is None
                ):
                    barge = _start_barge(
                        pending_decision.utterance,
                        frustrated=self.floor.frustrated,
                    )
                    pending_decision = None

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
                    awaiting_end_signal = self.connector.signals_turn_complete
                if (
                    action.retry_barge
                    and self.floor.frustrated
                    and barge is None
                ):
                    # Ask the judge once more, in frustration, for something to
                    # cut back in with. Off the loop for the same reason as any
                    # other barge: the agent has the floor until this is spoken.
                    barge = _Barge(utterance="", frustrated=True)
                    barge.task = asyncio.create_task(
                        _speak_frustrated_barge(barge)
                    )

                # Pull next agent event with a short timeout so timers keep ticking.
                if event_task is None:
                    event_task = asyncio.create_task(events.__anext__())
                try:
                    event = await asyncio.wait_for(
                        asyncio.shield(event_task), timeout=_EVENT_POLL_S
                    )
                except asyncio.TimeoutError:
                    # No frame arrived: count the wait as silence so an agent
                    # that simply stops streaming still ends its turn.
                    if await _advance_end_of_turn(_EVENT_POLL_S * 1000.0, now):
                        exchange_done = True
                    continue
                except StopAsyncIteration:
                    event_task = None
                    break
                event_task = None

                # Timers run on the wall clock, but anything describing when
                # the agent spoke has to use the frame's arrival time: the loop
                # blocks for seconds while a barge is synthesized, and frames
                # that landed during that block would otherwise be backdated.
                now = time.perf_counter()
                arrived_at = event.received_at or now
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
                            self.floor.on_agent_speech_start(arrived_at)
                            agent_started = True
                        # The end-of-turn timer measures silence *since the last
                        # speech*, so every spoken frame clears it. Letting it
                        # accumulate across a turn instead ends the agent
                        # mid-sentence once the pauses and scheduling gaps add up
                        # to the threshold, discarding the rest of the reply.
                        trailing_silence_ms = 0.0
                        if first_audio_at is None:
                            first_audio_at = arrived_at
                        agent_pcm.extend(event.audio)
                    else:
                        if await _advance_end_of_turn(frame_ms, arrived_at):
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
                    # The agent has spoken for itself; silence is no longer
                    # being asked to guess for this utterance.
                    awaiting_end_signal = False
                    if self.floor.user_uplink_active:
                        # Agent ended while we were barging — treat as success.
                        self.floor.on_agent_speech_end(arrived_at)
                        await _finalize_assistant(interrupted=True)
                        interrupted_assistant = False
                        self.floor.reset_turn()
                        agent_started = False
                        trailing_silence_ms = 0.0
                        barges_this_agent_turn = 0
                        last_judged_len = 0
                        last_stt_pcm_len = 0
                        awaiting_end_signal = (
                            self.connector.signals_turn_complete
                        )
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
                heard = _heard_transcript()
                if (
                    self.floor.can_run_judge
                    and heard
                    and judge_task is None
                    and should_poll_judge(
                        policy=self.policy,
                        partial_transcript=heard,
                        last_judged_len=last_judged_len,
                        last_judge_at=last_judge_at,
                        now=now,
                        barges_this_conversation=barges_this_conversation,
                        barges_this_agent_turn=barges_this_agent_turn,
                    )
                ):
                    last_judged_len = len(heard)
                    last_judge_at = now
                    snap_turns = list(turns)
                    snap_partial = heard
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
            if barge is not None:
                if barge.turn is None and not barge.task.done():
                    # The agent finished while this was still being synthesized,
                    # so there is nothing left to interrupt. Dropping it lets the
                    # simulator take the floor as an ordinary next turn.
                    barge.task.cancel()
                try:
                    # Already being spoken: let it finish, so the turn holds all
                    # of what the caller said rather than a clipped fragment.
                    await barge.task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Barge failed during cleanup")
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
