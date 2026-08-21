import asyncio
import time

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Audio, AudioChunk, Turn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.base import BaseVoiceConnector
from deepeval.voice.connectors.transports.callback import CallbackVoiceConnector
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.duplex import DuplexExchange, _spoken_prefix
from deepeval.voice.floor_control import FloorController
from deepeval.voice.interruption import interruption_policy
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.streaming import PcmRecorder, UplinkResult
from deepeval.voice.timeline import audio_duration


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


def _drain(connector) -> list:
    events = []
    while True:
        try:
            events.append(connector._events.get_nowait())
        except asyncio.QueueEmpty:
            return events


@pytest.mark.asyncio
async def test_uplink_ends_the_reply_it_talks_over():
    """A new turn must not inherit the previous reply's queued frames.

    The reply task paces frames in real time, so it keeps enqueueing long
    after an exchange has finished. Leaking them makes the next assistant turn
    start at the same instant as the user turn and carry the wrong audio.

    Dropping them is not enough on its own: every uplink starts a fresh agent
    invocation, so the abandoned utterance also has to be closed explicitly or
    the next reply accumulates into the same turn, leaving one turn whose audio
    and transcript come from different utterances.
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
        # Let the reply task queue a few frames without consuming them. The
        # first uplink had nothing to talk over, so it ends no utterance.
        await asyncio.sleep(0.1)
        queued = _drain(connector)
        assert queued
        assert AgentEvent(turn_complete=True) not in queued

        await connector.stream_uplink(probe)
        await asyncio.sleep(0.1)
        # The abandoned reply is closed before any of the next one arrives.
        assert _drain(connector)[0] == AgentEvent(turn_complete=True)


def _tone(ms: int, *, sample_rate: int = 24000) -> bytes:
    return b"\xe8\x03" * int(sample_rate * ms / 1000)


def _quiet(ms: int, *, sample_rate: int = 24000) -> bytes:
    return b"\x00\x00" * int(sample_rate * ms / 1000)


@pytest.mark.asyncio
async def test_pauses_within_a_reply_do_not_add_up_to_end_of_turn():
    """End-of-turn is silence *since the last speech*, not silence in total.

    An agent that pauses for breath several times hands us stretches of quiet
    that are each well short of the threshold. Accumulating them across the turn
    ends it mid-sentence and drops every frame still queued behind the pause,
    leaving a clipped recording next to a complete transcript.
    """
    # Each pause sits well inside "eager"'s 500ms window, but they add up to
    # 600ms of quiet across the reply.
    reply_pcm = (_tone(160) + _quiet(200)) * 3 + _tone(900)
    expected_s = len(reply_pcm) / 2 / 24000

    async def agent(_audio):
        return ConnectorTurn(
            audio=Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(reply_pcm, sample_rate=24000),
                "audio/wav",
            ),
            transcript="Agent reply",
        )

    connector = CallbackVoiceConnector(agent, turn_detection="eager")
    policy = interruption_policy("normal")
    exchange = DuplexExchange(
        connector=connector,
        tts_model=object(),
        stt_model=FailIfCalledSTT(),
        policy=policy,
        floor=FloorController(policy=policy),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=None,
        call_started_at=time.perf_counter(),
    )
    probe = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(_tone(10), sample_rate=24000),
        "audio/wav",
    )

    async with connector:
        await connector.stream_uplink(probe)
        result = await asyncio.wait_for(
            exchange.run(
                turns=[Turn(role="user", content="Hello", audio=probe)],
                sent_at=time.perf_counter(),
                # Past the barge cap, so the judge stays out of this.
                barges_this_conversation=policy.max_barges_per_conversation,
            ),
            timeout=15,
        )

    assert len(result.turns) == 1
    recorded = audio_duration(result.turns[0].audio)
    assert recorded == pytest.approx(expected_s, abs=0.1)


def _exchange(connector, **overrides):
    policy = interruption_policy("normal")
    kwargs = dict(
        connector=connector,
        tts_model=object(),
        stt_model=FailIfCalledSTT(),
        policy=policy,
        floor=FloorController(policy=policy),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=None,
        call_started_at=time.perf_counter(),
    )
    kwargs.update(overrides)
    return DuplexExchange(**kwargs)


async def _run_one_turn(exchange, connector, *, timeout: float = 15):
    probe = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(_tone(10), sample_rate=24000),
        "audio/wav",
    )
    async with connector:
        await connector.stream_uplink(probe)
        return await asyncio.wait_for(
            exchange.run(
                turns=[Turn(role="user", content="Hello", audio=probe)],
                sent_at=time.perf_counter(),
                # Past the barge cap, so the judge stays out of this.
                barges_this_conversation=(
                    exchange.policy.max_barges_per_conversation
                ),
            ),
            timeout=timeout,
        )


@pytest.mark.asyncio
async def test_a_pause_is_not_the_end_when_the_agent_says_when_it_is_done():
    """A single pause can outlast the threshold and still not be the end.

    Nothing about quiet distinguishes an agent that has finished from one
    drawing breath, so a transport that closes its turns outright is the better
    witness. Guessing from silence instead ends the turn at the first long
    pause, abandons the rest of the reply, and keeps the transcript describing
    all of it — a recording that stops mid-sentence next to text that does not.
    """
    reply_pcm = _tone(300) + _quiet(900) + _tone(800)
    expected_s = len(reply_pcm) / 2 / 24000

    async def agent(_audio):
        return ConnectorTurn(
            audio=Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(reply_pcm, sample_rate=24000),
                "audio/wav",
            ),
            transcript="Agent reply",
        )

    # The pause is nearly twice "eager"'s 500ms window.
    connector = CallbackVoiceConnector(agent, turn_detection="eager")
    assert connector.signals_turn_complete
    result = await _run_one_turn(_exchange(connector), connector)

    assert len(result.turns) == 1
    assert result.turns[0].content == "Agent reply"
    assert audio_duration(result.turns[0].audio) == pytest.approx(
        expected_s, abs=0.1
    )


class _StillTalking(BaseVoiceConnector):
    """An agent that trails off and never says it has finished.

    Its transcript promises a whole sentence; only the opening of it is ever
    sent.
    """

    protocol = VoiceProtocol.WEBSOCKET
    end_of_turn_silence_ms = 200
    max_turn_timeout_s = 1.0
    signals_turn_complete = True
    sample_rate = 24000

    @property
    def recv_sample_rate(self) -> int:
        return self.sample_rate

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def exchange_turn(self, audio):
        raise NotImplementedError

    async def stream_uplink(self, audio, *, trailing_silence: bool = True):
        return None

    async def stop_uplink(self) -> None:
        return None

    async def iter_agent_events(self):
        yield AgentEvent(
            transcript="I need your order number before I can look into this.",
            received_at=time.perf_counter(),
        )
        yield AgentEvent(audio=_tone(400), received_at=time.perf_counter())
        while True:
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_turn_we_stopped_listening_to_says_only_what_was_heard():
    """`content` is what the caller heard, never what the agent meant to say.

    We gave up on this turn — the agent never closed it — so the recording
    holds an opening and the transcript holds a sentence. Reporting the
    transcript would describe speech that was never on the call; the rest is
    kept as what the agent was left saying to nobody.
    """
    connector = _StillTalking()
    exchange = _exchange(connector, stt_model=_StubSTT())

    result = await _run_one_turn(exchange, connector, timeout=5)

    assert len(result.turns) == 1
    turn = result.turns[0]
    assert turn.content == "what the caller heard"
    assert turn.metadata == {
        "intended_content": "I need your order number before I can look "
        "into this.",
        "ended_without_agent_signal": True,
    }
    assert audio_duration(turn.audio) == pytest.approx(0.4, abs=0.05)


class _SlowTTS:
    """A TTS round trip slow enough to see on the recording."""

    delay_s = 0.6

    def supports_streaming(self) -> bool:
        return False

    async def a_synthesize(self, text, **kwargs):
        await asyncio.sleep(self.delay_s)
        return (
            Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(_tone(400), sample_rate=24000),
                "audio/wav",
            ),
            None,
        )


class _StubSTT:
    truncated_audio_pad_seconds = 0.0

    async def a_transcribe(self, audio):
        return "what the caller heard", None


@pytest.mark.asyncio
async def test_agent_holds_the_floor_until_the_barge_is_ready_to_speak():
    """Deciding to interrupt is not the same moment as interrupting.

    A caller settles on what to say and then says it, and the agent talks
    through the gap between. Silencing the agent when the judge decides instead
    leaves the whole synthesis round trip as dead air in the recording — the
    agent stops, nobody speaks, and the barge arrives seconds later.
    """

    replies = iter([_tone(6000)])

    async def agent(_audio):
        # Only the first reply is long enough to interrupt; the follow-up just
        # lets the exchange finish.
        return ConnectorTurn(
            audio=Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(
                    next(replies, _tone(200)), sample_rate=24000
                ),
                "audio/wav",
            ),
            transcript="I need your order number before I can look into any of "
            "this for you, and it will take a moment to pull up once you do.",
        )

    async def a_generate_schema(prompt, schema):
        return schema(
            should_interrupt=True,
            utterance="Just check it.",
            reason="The caller has waited long enough.",
        )

    connector = CallbackVoiceConnector(agent, turn_detection="eager")
    policy = interruption_policy("frequent")
    exchange = DuplexExchange(
        connector=connector,
        tts_model=_SlowTTS(),
        stt_model=_StubSTT(),
        policy=policy,
        floor=FloorController(policy=policy),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=a_generate_schema,
        call_started_at=time.perf_counter(),
    )
    probe = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(_tone(10), sample_rate=24000),
        "audio/wav",
    )
    turns = [Turn(role="user", content="Hello", audio=probe)]

    async with connector:
        await connector.stream_uplink(probe)
        await asyncio.wait_for(
            exchange.run(
                turns=turns,
                sent_at=time.perf_counter(),
                barges_this_conversation=0,
            ),
            timeout=20,
        )

    barge = next(
        t
        for t in turns
        if t.role == "user" and (t.metadata or {}).get("barge_in")
    )
    assistant = next(t for t in turns if t.role == "assistant")
    agent_ends_at = assistant.audio.start_time + audio_duration(assistant.audio)
    # Anything approaching the synthesis delay means the agent was muted when
    # the barge was decided rather than when it was spoken.
    assert barge.audio.start_time - agent_ends_at < _SlowTTS.delay_s / 2


class _StreamingTTS:
    """Speech that goes out in frames while the rest is still being made."""

    frame_delay = 0.3
    frames = 3

    def supports_streaming(self) -> bool:
        return True

    def synthesis_cost(self, text: str):
        return None

    async def a_synthesize_stream(self, text, **kwargs):
        for index in range(self.frames):
            await asyncio.sleep(self.frame_delay)
            yield AudioChunk.from_bytes(
                _tone(200),
                "audio/pcm",
                sampleRate=24000,
                encoding="pcm",
                final=index == self.frames - 1,
            )


class _StreamingWire(BaseVoiceConnector):
    """A transport that forwards each frame the moment it is synthesized."""

    protocol = VoiceProtocol.WEBSOCKET
    end_of_turn_silence_ms = 500
    max_turn_timeout_s = 20.0

    def __init__(self, reply_pcm: bytes, *, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.reply_pcm = reply_pcm
        self.frames_at: list = []
        self._events = asyncio.Queue()
        self._reply_task = None
        self._cancelled = False

    @property
    def audio_format(self):
        return (self.sample_rate, "pcm")

    @property
    def recv_sample_rate(self) -> int:
        return self.sample_rate

    async def connect(self) -> None:
        self._reply_task = asyncio.create_task(self._play_reply(self.reply_pcm))

    async def disconnect(self) -> None:
        await self._end_reply()

    async def exchange_turn(self, audio):
        raise NotImplementedError

    async def _play_reply(self, pcm: bytes, *, delay: float = 0.0) -> None:
        if delay:
            await asyncio.sleep(delay)
        await self._events.put(
            AgentEvent(
                transcript="I need your order number before I can look into "
                "any of this, and it will take a moment to pull up.",
                received_at=time.perf_counter(),
            )
        )
        for frame in audio_utils.iter_pcm16_frames(pcm, self.sample_rate):
            await self._events.put(
                AgentEvent(audio=frame, received_at=time.perf_counter())
            )
            await asyncio.sleep(audio_utils.DEFAULT_FRAME_MS / 1000)
        await self._events.put(AgentEvent(turn_complete=True))

    async def _end_reply(self) -> None:
        if self._reply_task is not None and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except (asyncio.CancelledError, Exception):
                pass
            await self._events.put(AgentEvent(turn_complete=True))

    async def iter_agent_events(self):
        while True:
            yield await self._events.get()

    async def stream_uplink(self, audio, *, trailing_silence: bool = True):
        self.frames_at.append(time.perf_counter())
        await self._end_reply()

    async def stream_uplink_chunks(
        self, chunks, *, trailing_silence: bool = True, on_first_frame=None
    ):
        self._cancelled = False
        recorder = PcmRecorder()
        async for chunk in chunks:
            recorder.add(chunk)
            if self._cancelled:
                continue
            first = not self.frames_at
            self.frames_at.append(time.perf_counter())
            if first:
                if on_first_frame is not None:
                    on_first_frame(self.frames_at[0])
                # Speech arriving ends the reply, as a real agent's would. It
                # happens here rather than when this call began: nothing has
                # reached the agent until a frame does.
                await self._end_reply()
                # And the agent answers what it was interrupted with, so the
                # exchange can finish rather than wait out its timeout.
                self._reply_task = asyncio.create_task(
                    self._play_reply(_tone(300), delay=0.8)
                )
        return UplinkResult(
            audio=recorder.to_audio(),
            first_frame_at=self.frames_at[0] if self.frames_at else None,
        )

    async def stop_uplink(self) -> None:
        self._cancelled = True


@pytest.mark.asyncio
async def test_a_barge_is_heard_from_its_first_frame_not_its_last():
    """The agent hears the caller cut in while the rest is still being made.

    Holding the barge back until the whole clip exists puts the entire synthesis
    round trip between the agent going quiet and the caller speaking, and places
    the caller's voice on the recording seconds after they started talking.
    """

    async def a_generate_schema(prompt, schema):
        return schema(
            should_interrupt=True,
            utterance="Just check it.",
            reason="The caller has waited long enough.",
        )

    wire = _StreamingWire(_tone(6000))
    policy = interruption_policy("frequent")
    call_started_at = time.perf_counter()
    exchange = DuplexExchange(
        connector=wire,
        tts_model=_StreamingTTS(),
        stt_model=_StubSTT(),
        policy=policy,
        floor=FloorController(policy=policy),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=a_generate_schema,
        call_started_at=call_started_at,
    )
    probe = Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(_tone(10), sample_rate=24000),
        "audio/wav",
    )
    turns = [Turn(role="user", content="Hello", audio=probe)]

    await wire.connect()
    try:
        await asyncio.wait_for(
            exchange.run(
                turns=turns,
                sent_at=time.perf_counter(),
                barges_this_conversation=0,
            ),
            timeout=20,
        )
    finally:
        await wire.disconnect()

    barge = next(
        t
        for t in turns
        if t.role == "user" and (t.metadata or {}).get("barge_in")
    )
    # Frames left over a stretch of time, so sending overlapped synthesis
    # instead of following it.
    assert len(wire.frames_at) == _StreamingTTS.frames
    spread = wire.frames_at[-1] - wire.frames_at[0]
    assert spread >= _StreamingTTS.frame_delay * 1.5
    # The clip sits where the caller started talking, and holds all they said.
    assert barge.audio is not None
    assert barge.audio.start_time == pytest.approx(
        wire.frames_at[0] - call_started_at, abs=0.05
    )
    assert audio_duration(barge.audio) == pytest.approx(0.6, abs=0.01)


class TestSpokenPrefix:
    """The judge may only see the reply as far as the caller has heard it."""

    transcript = "What is your order number? I will look it up."

    def test_text_beyond_the_delivered_audio_is_withheld(self):
        # A connector that sends the whole reply up front must not let the
        # judge answer a question the agent has not reached yet.
        assert "order number" not in _spoken_prefix(self.transcript, 0.5)

    def test_grows_as_the_audio_plays(self):
        heard = [len(_spoken_prefix(self.transcript, s)) for s in (0, 1, 2, 3)]
        assert heard == sorted(heard)

    def test_never_ends_mid_word(self):
        for tenths in range(1, 40):
            prefix = _spoken_prefix(self.transcript, tenths / 10)
            assert (
                not prefix
                or self.transcript.startswith(prefix + " ")
                or (prefix == self.transcript)
            )

    def test_whole_transcript_once_the_audio_covers_it(self):
        assert _spoken_prefix(self.transcript, 60.0) == self.transcript

    def test_silence_so_far_means_nothing_heard(self):
        assert _spoken_prefix(self.transcript, 0.0) == ""
