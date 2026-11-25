import asyncio
import time

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Audio, Turn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.base import BaseVoiceConnector
from deepeval.voice.connectors.transports.callback import CallbackVoiceConnector
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.duplex import DuplexExchange, _spoken_prefix
from deepeval.voice.floor_control import FloorController
from deepeval.voice.interruption import InterruptionSchedule
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.timeline import audio_duration


class FailIfCalledSTT:
    def supports_streaming(self) -> bool:
        return False

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
    exchange = DuplexExchange(
        connector=connector,
        tts_model=object(),
        stt_model=FailIfCalledSTT(),
        schedule=InterruptionSchedule(),
        floor=FloorController(),
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
    exchange = DuplexExchange(
        connector=connector,
        tts_model=object(),
        stt_model=FailIfCalledSTT(),
        schedule=InterruptionSchedule(),
        floor=FloorController(),
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
            ),
            timeout=15,
        )

    assert len(result.turns) == 1
    recorded = audio_duration(result.turns[0].audio)
    assert recorded == pytest.approx(expected_s, abs=0.1)


def _exchange(connector, **overrides):
    kwargs = dict(
        connector=connector,
        tts_model=object(),
        stt_model=FailIfCalledSTT(),
        schedule=InterruptionSchedule(),
        floor=FloorController(),
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


class _NeverSpeaks(_StillTalking):
    async def iter_agent_events(self):
        while True:
            yield AgentEvent(
                audio=b"\x00\x00" * 2400, received_at=time.perf_counter()
            )
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_silence_we_stopped_listening_to_is_not_an_agent_turn():
    connector = _NeverSpeaks()
    exchange = _exchange(connector, stt_model=_StubSTT())

    result = await _run_one_turn(exchange, connector, timeout=5)

    assert result.turns == []


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
    def supports_streaming(self) -> bool:
        return False

    truncated_audio_pad_seconds = 0.0

    async def a_transcribe(self, audio):
        return "what the caller heard", None


async def _write_line(prompt, schema):
    return schema(utterance="Just check it.")


class _LiveWire(BaseVoiceConnector):

    protocol = VoiceProtocol.WEBSOCKET
    end_of_turn_silence_ms = 500
    max_turn_timeout_s = 20.0
    signals_turn_complete = True

    def __init__(
        self, reply_pcm: bytes, *, yields: bool = True, sample_rate: int = 24000
    ):
        self.sample_rate = sample_rate
        self.reply_pcm = reply_pcm
        self.yields = yields
        self.frames_at: list = []
        self._events = asyncio.Queue()
        self._reply_task = None

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
                transcript="Agent reply", received_at=time.perf_counter()
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
        if not self.yields:
            return
        await self._end_reply()
        self._reply_task = asyncio.create_task(
            self._play_reply(_tone(300), delay=0.8)
        )

    async def stop_uplink(self) -> None:
        return None


def _scheduled_on_next_reply() -> InterruptionSchedule:
    schedule = InterruptionSchedule(3)
    schedule.on_agent_reply_start()
    schedule.on_agent_reply_start()
    return schedule


async def _run_scheduled(wire: _LiveWire, schedule, *, floor=None, tts=None):
    call_started_at = time.perf_counter()
    exchange = DuplexExchange(
        connector=wire,
        tts_model=tts or _SlowTTS(),
        stt_model=_StubSTT(),
        schedule=schedule,
        floor=floor or FloorController(),
        golden=ConversationalGolden(scenario="Test"),
        language="English",
        a_generate_schema=_write_line,
        call_started_at=call_started_at,
    )
    turns = [Turn(role="user", content="Hello")]
    prepared = exchange.prepare_barge(turns)
    assert prepared is not None
    await wire.connect()
    try:
        result = await asyncio.wait_for(
            exchange.run(
                turns=turns,
                sent_at=time.perf_counter(),
                prepared_barge=prepared,
            ),
            timeout=20,
        )
    finally:
        await wire.disconnect()
    return turns, result


def _barge_turns(turns):
    return [
        t
        for t in turns
        if t.role == "user" and (t.metadata or {}).get("barge_in")
    ]


@pytest.mark.asyncio
async def test_a_prepared_barge_cuts_into_the_scheduled_reply():
    schedule = _scheduled_on_next_reply()

    turns, _ = await _run_scheduled(_LiveWire(_tone(6000)), schedule)

    [barge] = _barge_turns(turns)
    cut, answer = [t for t in turns if t.role == "assistant"]
    assert barge.content == "Just check it."
    assert cut.interrupted is True
    assert 0.5 <= barge.audio.start_time - cut.audio.start_time < 1.2
    cut_ends_at = cut.audio.start_time + audio_duration(cut.audio)
    assert barge.audio.start_time - cut_ends_at < 0.3
    assert answer.interrupted is None
    assert schedule.interrupts_next_reply is False


@pytest.mark.asyncio
async def test_a_reply_over_before_the_line_is_ready_keeps_it_owed():
    schedule = _scheduled_on_next_reply()
    wire = _LiveWire(_tone(300))

    turns, _ = await _run_scheduled(wire, schedule)

    assert _barge_turns(turns) == []
    assert wire.frames_at == []
    [reply] = [t for t in turns if t.role == "assistant"]
    assert reply.interrupted is None
    assert schedule.interrupts_next_reply is True


class _SilenceEndsTheTurn(_LiveWire):
    signals_turn_complete = False

    async def _play_reply(self, pcm: bytes, *, delay: float = 0.0) -> None:
        if delay:
            await asyncio.sleep(delay)
        for frame in audio_utils.iter_pcm16_frames(pcm, self.sample_rate):
            await self._events.put(
                AgentEvent(audio=frame, received_at=time.perf_counter())
            )
            await asyncio.sleep(audio_utils.DEFAULT_FRAME_MS / 1000)


class _HearsTheCallerLate(_SilenceEndsTheTurn):

    end_of_turn_silence_ms = 800

    async def stream_uplink(self, audio, *, trailing_silence: bool = True):
        self.frames_at.append(time.perf_counter())
        asyncio.create_task(self._yield_then_answer())

    async def _yield_then_answer(self) -> None:
        await asyncio.sleep(1.4)
        await self._end_reply()
        self._reply_task = asyncio.create_task(
            self._play_reply(_tone(500), delay=2.0)
        )

    async def _end_reply(self) -> None:
        if self._reply_task is not None and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_an_agent_that_stops_within_the_grace_gives_way_to_the_caller():
    schedule = _scheduled_on_next_reply()
    floor = FloorController.from_overlap_behavior("adaptive")

    turns, _ = await _run_scheduled(
        _HearsTheCallerLate(_tone(6000)), schedule, floor=floor
    )

    cut, barge, answer = turns[1:]
    assert cut.interrupted is True
    assert barge.metadata == {"barge_in": True}
    assert answer.role == "assistant" and answer.interrupted is None


@pytest.mark.asyncio
async def test_a_reply_that_ends_before_the_cut_in_keeps_it_owed():
    schedule = _scheduled_on_next_reply()
    wire = _SilenceEndsTheTurn(_tone(500))

    turns, _ = await _run_scheduled(wire, schedule)

    assert _barge_turns(turns) == []
    [reply] = [t for t in turns if t.role == "assistant"]
    assert reply.interrupted is None
    assert schedule.interrupts_next_reply is True


@pytest.mark.asyncio
async def test_an_agent_that_talks_through_the_barge_was_not_cut_off():
    schedule = _scheduled_on_next_reply()
    wire = _LiveWire(_tone(3000), yields=False)

    turns, _ = await _run_scheduled(
        wire, schedule, floor=FloorController.from_overlap_behavior("yield")
    )

    [barge] = _barge_turns(turns)
    [reply] = [t for t in turns if t.role == "assistant"]
    assert barge.metadata.get("frustrated") is True
    assert reply.interrupted is None
    assert audio_duration(reply.audio) == pytest.approx(3.0, abs=0.15)
    assert schedule.interrupts_next_reply is False


class _AnswersOverTheCaller(_StillTalking):

    signals_turn_complete = False
    max_turn_timeout_s = 5.0

    def __init__(self, early_ms: int):
        self.early_ms = early_ms

    async def iter_agent_events(self):
        yield AgentEvent(
            audio=_tone(self.early_ms), received_at=time.perf_counter()
        )
        for _ in range(12):
            await asyncio.sleep(0.05)
            yield AgentEvent(audio=_quiet(50), received_at=time.perf_counter())
        yield AgentEvent(audio=_tone(500), received_at=time.perf_counter())
        while True:
            await asyncio.sleep(0.05)
            yield AgentEvent(audio=_quiet(50), received_at=time.perf_counter())


async def _run_over_the_caller(early_ms: int, caller_started_s: float = -1.0):
    connector = _AnswersOverTheCaller(early_ms)
    exchange = _exchange(connector, stt_model=_StubSTT())
    now = time.perf_counter()
    return await asyncio.wait_for(
        exchange.run(
            turns=[Turn(role="user", content="Hello")],
            sent_at=now + 0.5,
            caller_spoke=(now + caller_started_s, now + 0.5),
        ),
        timeout=5,
    )


@pytest.mark.asyncio
async def test_speech_the_caller_talked_over_is_an_interrupted_turn():
    result = await _run_over_the_caller(400)

    cut, answer = result.turns
    assert cut.interrupted is True
    assert answer.interrupted is None
    assert answer.audio.start_time > cut.audio.start_time


@pytest.mark.asyncio
async def test_speech_that_ended_before_the_caller_began_was_not_cut_off():
    result = await _run_over_the_caller(300, caller_started_s=0.45)

    assert result.turns[0].interrupted is None


@pytest.mark.asyncio
async def test_a_blip_the_caller_talked_over_is_dropped():
    result = await _run_over_the_caller(100)

    [answer] = result.turns
    assert answer.interrupted is None
    assert audio_duration(answer.audio) >= 0.5


class TestSpokenPrefix:

    transcript = "What is your order number? I will look it up."

    def test_text_beyond_the_delivered_audio_is_withheld(self):
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


class _EndsQuietlyWhileCallerTalks(_StillTalking):
    def __init__(self, floor: FloorController):
        self.floor = floor

    async def iter_agent_events(self):
        yield AgentEvent(audio=_quiet(300), received_at=time.perf_counter())
        self.floor.on_user_barge_start(time.perf_counter())
        yield AgentEvent(turn_complete=True, received_at=time.perf_counter())
        while True:
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_reply_ended_without_speech_still_has_a_start_time():
    floor = FloorController()
    connector = _EndsQuietlyWhileCallerTalks(floor)
    exchange = _exchange(connector, stt_model=_StubSTT(), floor=floor)

    result = await _run_one_turn(exchange, connector, timeout=5)

    assistant = next(t for t in result.turns if t.role == "assistant")
    assert assistant.audio is not None
    assert assistant.audio.start_time is not None
    assert assistant.interrupted is None
