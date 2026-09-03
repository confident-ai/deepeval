import os
import asyncio
import logging
import time
import uuid
from datetime import timedelta
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    ClassVar,
    List,
    Optional,
    Set,
    Tuple,
)

from deepeval.errors import DeepEvalError
from deepeval.utils import require_dependency
from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.protocol import VoiceProtocol
from deepeval.voice.connectors.transports.base import (
    BaseVoiceConnector,
    UplinkStream,
    iter_downlink,
)
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.turn_engine import collect_agent_turn
from deepeval.voice.streaming import (
    DEFAULT_STREAM_SAMPLE_RATE,
    PcmRecorder,
    UplinkResult,
)
from deepeval.voice.turn_detection import TurnDetection, turn_detection_timing

logger = logging.getLogger(__name__)

_INSTALL_HINT = 'Install it with `pip install "deepeval[voice]"`.'

# Agents publish their speech transcript on this text-stream topic.
# https://docs.livekit.io/agents/multimodality/text/
TRANSCRIPTION_TOPIC = "lk.transcription"
TRANSCRIPTION_FINAL_ATTRIBUTE = "lk.transcription_final"


class LiveKitConnector(BaseVoiceConnector):

    protocol: ClassVar[VoiceProtocol] = VoiceProtocol.WEBRTC

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        room: Optional[Any] = None,
        token: Optional[str] = None,
        room_name: Optional[str] = None,
        identity: str = "deepeval-test",
        agent_name: Optional[str] = None,
        agent_identity: Optional[str] = None,
        turn_detection: TurnDetection = "balanced",
        silence_threshold_rms: float = audio_utils.DEFAULT_SILENCE_RMS,
        connect_timeout_s: float = 15.0,
        input_sample_rate: int = 24000,
        livekit_sample_rate: int = 48000,
        token_ttl_s: int = 3600,
        transcript_grace_s: float = 1.5,
    ):
        self.url = url or os.getenv("LIVEKIT_URL")
        self.api_key = api_key or os.getenv("LIVEKIT_API_KEY")
        self.api_secret = api_secret or os.getenv("LIVEKIT_API_SECRET")

        # Credentials are only needed for the parts of the connection deepeval
        # is being asked to make. A caller who brings a connected room has
        # already done all of it, and one who brings a token has already done
        # the half that needs the API secret.
        self._room_arg = room
        self._token_arg = token
        if room is None and token is None:
            if not (self.url and self.api_key and self.api_secret):
                raise DeepEvalError(
                    "LiveKitConnector requires a LiveKit URL, API key and API "
                    "secret (pass url/api_key/api_secret or set LIVEKIT_URL, "
                    "LIVEKIT_API_KEY, LIVEKIT_API_SECRET), or pass a `room` / "
                    "`token` you have already created."
                )

        self._room_name_arg = room_name
        self.room_name = room_name
        self.identity = identity
        self.agent_name = agent_name
        self.agent_identity = agent_identity
        self.turn_detection = turn_detection
        timing = turn_detection_timing(turn_detection)
        self.end_of_turn_silence_ms = timing.end_of_turn_silence_ms
        self.max_turn_timeout_s = timing.max_turn_timeout_s
        self.silence_threshold_rms = silence_threshold_rms
        self.connect_timeout_s = connect_timeout_s
        self.input_sample_rate = input_sample_rate
        self.livekit_sample_rate = livekit_sample_rate
        self.token_ttl_s = token_ttl_s
        self.transcript_grace_s = transcript_grace_s
        self._frame_gap_timeout_s = max(
            1.0, self.end_of_turn_silence_ms / 1000.0 + 0.5
        )

        # Lazily populated in connect().
        self._rtc = None
        self._api = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._room = None
        self._source = None
        self._local_track = None
        self._local_publication = None
        self._agent_track = None
        self._agent_stream = None
        self._agent_participant = None
        self._drain_task: Optional[asyncio.Task] = None
        self._out_frames: Optional[asyncio.Queue] = None
        self._agent_track_ready: Optional[asyncio.Event] = None
        self._uplink: Optional[UplinkStream] = None
        self._owns_room_connection = False
        self._current_transcript: Optional[str] = None
        self._transcript_ready: Optional[asyncio.Event] = None
        self._transcript_tasks: Set[asyncio.Task] = set()

    @property
    def audio_format(self) -> Tuple[int, str]:
        return (self.input_sample_rate, "wav")

    @property
    def recv_sample_rate(self) -> int:
        return self.livekit_sample_rate

    async def connect(self) -> None:
        await self._join_room()
        await self._after_join()
        await self._await_agent_track()

    async def _join_room(self) -> None:
        self._rtc = require_dependency(
            "livekit.rtc",
            provider_label="LiveKitConnector",
            install_hint=_INSTALL_HINT,
        )
        rtc = self._rtc

        self._loop = asyncio.get_event_loop()
        self._out_frames = asyncio.Queue()
        self._agent_track_ready = asyncio.Event()
        self._transcript_ready = asyncio.Event()
        self._uplink = UplinkStream()
        self._room = (
            self._room_arg if self._room_arg is not None else rtc.Room()
        )
        self.room_name = self._resolve_room_name()

        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("participant_connected", self._on_participant_connected)
        self._register_transcript_handler()

        if not self._is_room_connected():
            if not self.url:
                raise DeepEvalError(
                    "LiveKitConnector needs a LiveKit URL to connect the room "
                    "(pass url or set LIVEKIT_URL), or pass a room that is "
                    "already connected."
                )
            await self._room.connect(
                self.url,
                self._token_arg or self._build_token(),
                rtc.RoomOptions(auto_subscribe=True),
            )
            self._owns_room_connection = True

        self._source = rtc.AudioSource(self.livekit_sample_rate, 1)
        self._local_track = rtc.LocalAudioTrack.create_audio_track(
            "deepeval-user", self._source
        )
        self._local_publication = (
            await self._room.local_participant.publish_track(
                self._local_track,
                rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_MICROPHONE
                ),
            )
        )

    async def _after_join(self) -> None:
        """Hook for subclasses that put the other party in the room themselves.

        Runs once the room is joined and the microphone published, before
        anything waits for an audio track. Dialing a phone number through a SIP
        trunk goes here: the callee does not exist until something places the
        call, so waiting for its track first would only ever time out.
        """
        return None

    async def _await_agent_track(self) -> None:
        self._adopt_existing_agent_track()

        try:
            await asyncio.wait_for(
                self._agent_track_ready.wait(), timeout=self.connect_timeout_s
            )
        except asyncio.TimeoutError:
            await self.disconnect()
            if self.agent_identity is not None:
                raise DeepEvalError(
                    f"Participant '{self.agent_identity}' published no audio in "
                    f"room '{self.room_name}' within {self.connect_timeout_s}s."
                )
            raise DeepEvalError(
                f"No LiveKit agent joined room '{self.room_name}' within "
                f"{self.connect_timeout_s}s. Is the agent worker running and "
                "dispatched to this project?"
            )

    def _resolve_room_name(self) -> str:
        # A room the caller connected already knows its own name, and that is
        # the one the agent was dispatched to.
        if self._is_room_connected():
            name = getattr(self._room, "name", None)
            if name:
                return name
        return self._room_name_arg or f"deepeval-{uuid.uuid4().hex[:12]}"

    def _is_room_connected(self) -> bool:
        is_connected = getattr(self._room, "isconnected", None)
        if not callable(is_connected):
            return False
        try:
            return bool(is_connected())
        except Exception:
            return False

    def _build_token(self) -> str:
        if not (self.api_key and self.api_secret):
            raise DeepEvalError(
                "LiveKitConnector needs a LiveKit API key and secret to mint a "
                "token (pass api_key/api_secret or set LIVEKIT_API_KEY, "
                "LIVEKIT_API_SECRET), or pass a `token` of your own."
            )
        self._api = require_dependency(
            "livekit.api",
            provider_label="LiveKitConnector",
            install_hint=_INSTALL_HINT,
        )
        api = self._api
        grants = api.VideoGrants(
            room_join=True,
            room=self.room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        builder = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(self.identity)
            .with_name(self.identity)
            .with_grants(grants)
            .with_ttl(timedelta(seconds=self.token_ttl_s))
        )
        if self.agent_name:
            builder = builder.with_room_config(
                api.RoomConfiguration(
                    agents=[api.RoomAgentDispatch(agent_name=self.agent_name)]
                )
            )
        return builder.to_jwt()

    def _is_agent_participant(self, participant) -> bool:
        """Whether this participant is the agent under test.

        A room can hold more than the caller and the agent — a SIP participant,
        a human observer, another agent — and the first audio track to arrive is
        not necessarily the one under test.

        `agent_identity` is the caller saying which participant to listen to,
        so it wins outright, ahead of the kind check. It is the only way to
        choose between two agents in one room, and the party under test is not
        always agent-kind at all — a SIP callee dialed from `_after_join` is a
        standard participant. Without it, kind is the best available guess.
        Older server or SDK builds report no kind, in which case there is
        nothing to go on and any audio track has to be accepted.
        """
        if self.agent_identity is not None:
            return participant.identity == self.agent_identity

        kind = getattr(participant, "kind", None)
        expected = getattr(
            getattr(self._rtc, "ParticipantKind", None),
            "PARTICIPANT_KIND_AGENT",
            None,
        )
        if kind is None or expected is None:
            return True
        return kind == expected

    def _adopt_existing_agent_track(self) -> None:
        if self._agent_track is not None:
            return
        rtc = self._rtc
        for participant in self._room.remote_participants.values():
            if not self._is_agent_participant(participant):
                continue
            for publication in participant.track_publications.values():
                track = publication.track
                if track is not None and track.kind == rtc.TrackKind.KIND_AUDIO:
                    self._attach_agent_track(track, participant)
                    return

    def _on_track_subscribed(self, track, publication, participant) -> None:
        rtc = self._rtc
        if self._agent_track is not None:
            return
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if not self._is_agent_participant(participant):
            return
        self._attach_agent_track(track, participant)

    def _register_transcript_handler(self) -> None:
        """Listen for the agent's own transcript of what it just said.

        The agent has already run the text through TTS, so the words exist
        upstream and arrive here for free. Taking them means a turn does not
        have to be sent back through an STT model to recover text the agent
        knew all along.
        """
        register = getattr(self._room, "register_text_stream_handler", None)
        if not callable(register):
            logger.debug(
                "This livekit build has no text stream handler; agent "
                "transcripts will fall back to STT."
            )
            return
        try:
            register(TRANSCRIPTION_TOPIC, self._on_transcript_stream)
        except Exception:
            # Raised when a handler for the topic is already registered, which
            # is the caller's and takes precedence over ours.
            logger.debug(
                "Could not register a %s handler; agent transcripts will fall "
                "back to STT.",
                TRANSCRIPTION_TOPIC,
                exc_info=True,
            )

    def _on_transcript_stream(self, reader, participant_identity: str) -> None:
        if participant_identity == self.identity:
            return  # our own speech, transcribed back to us
        task = self._loop.create_task(self._read_transcript(reader))
        self._transcript_tasks.add(task)
        task.add_done_callback(self._transcript_tasks.discard)

    async def _read_transcript(self, reader) -> None:
        try:
            text = await reader.read_all()
        except Exception:
            logger.debug("Reading a LiveKit transcript failed.", exc_info=True)
            return
        attributes = getattr(getattr(reader, "info", None), "attributes", None)
        final = (attributes or {}).get(TRANSCRIPTION_FINAL_ATTRIBUTE)
        # Interim segments are revised as the agent keeps talking, so only the
        # final one describes the turn.
        if final not in (True, "true"):
            return
        text = (text or "").strip()
        if not text:
            return
        self._current_transcript = text
        if self._transcript_ready is not None:
            self._transcript_ready.set()
        if self._out_frames is not None:
            self._out_frames.put_nowait(
                AgentEvent(transcript=text, received_at=time.perf_counter())
            )

    def _on_participant_connected(self, participant) -> None:
        if self._agent_participant is None and self._is_agent_participant(
            participant
        ):
            self._agent_participant = participant

    def _attach_agent_track(self, track, participant) -> None:
        rtc = self._rtc
        self._agent_track = track
        self._agent_participant = participant
        self._agent_stream = rtc.AudioStream(
            track, sample_rate=self.livekit_sample_rate, num_channels=1
        )
        self._drain_task = self._loop.create_task(self._drain_loop())
        self._agent_track_ready.set()

    async def _drain_loop(self) -> None:
        try:
            async for event in self._agent_stream:
                await self._out_frames.put(
                    AgentEvent(
                        audio=bytes(event.frame.data),
                        received_at=time.perf_counter(),
                    )
                )
        except asyncio.CancelledError:
            raise
        finally:
            await self._out_frames.put(AgentEvent(turn_complete=True))

    async def stream_uplink(
        self, audio: Audio, *, trailing_silence: bool = False
    ) -> None:
        """Publish user audio without waiting for agent end-of-turn.

        Unlike `exchange_turn`, this does not wait for playout before returning
        control to a duplex listen loop — frames are captured until complete
        or `stop_uplink()` cancels the stream. Trailing silence is off by
        default so barge-in audio does not pad the uplink.
        """
        if self._uplink is None or self._source is None:
            raise DeepEvalError(
                "LiveKitConnector.stream_uplink() called before connect()."
            )
        await self.stop_uplink()
        self._uplink.begin()
        frames = self._make_input_frames(audio)

        async def _stream() -> None:
            for frame in frames:
                if self._uplink.cancelled:
                    break
                await self._source.capture_frame(frame)

        self._uplink.task = asyncio.create_task(_stream())
        await self._uplink.task
        self._uplink.task = None

    async def stream_uplink_chunks(
        self,
        chunks: AsyncIterable[AudioChunk],
        *,
        trailing_silence: bool = True,
        on_first_frame: Optional[Callable[[float], None]] = None,
    ) -> UplinkResult:
        """Publish each frame of speech as soon as it has been synthesized.

        The agent's turn detection and transcription then see the opening words
        while the rest of the utterance is still being made, which is how a real
        room behaves — a microphone does not wait for a sentence to finish.
        """
        if self._uplink is None or self._source is None:
            raise DeepEvalError(
                "LiveKitConnector.stream_uplink_chunks() called before connect()."
            )
        await self.stop_uplink()
        self._uplink.begin()

        first_frame_at: Optional[float] = None
        recorder = PcmRecorder()
        # One resampler for the whole utterance: it carries filter state across
        # pushes, so per-frame resampling matches resampling the utterance whole
        # instead of leaving a seam at every frame boundary.
        resampler = None
        async for chunk in chunks:
            pcm = recorder.add(chunk)
            if self._uplink.cancelled:
                continue
            rate = chunk.sampleRate or DEFAULT_STREAM_SAMPLE_RATE
            if rate != self.livekit_sample_rate:
                if resampler is None:
                    resampler = self._rtc.AudioResampler(
                        rate, self.livekit_sample_rate, num_channels=1
                    )
                frames = resampler.push(bytearray(pcm))
            else:
                frames = self._pcm_to_frames(pcm, rate)
            for frame in frames:
                if self._uplink.cancelled:
                    break
                if first_frame_at is None:
                    first_frame_at = time.perf_counter()
                    if on_first_frame is not None:
                        on_first_frame(first_frame_at)
                await self._source.capture_frame(frame)
        if resampler is not None and not self._uplink.cancelled:
            for frame in resampler.flush():
                await self._source.capture_frame(frame)
        return UplinkResult(
            audio=recorder.to_audio(), first_frame_at=first_frame_at
        )

    async def stop_uplink(self) -> None:
        if self._uplink is not None:
            await self._uplink.stop()

    @property
    def supports_duplex(self) -> bool:
        return True

    async def iter_agent_events(self) -> AsyncIterator[AgentEvent]:
        if self._out_frames is None:
            raise DeepEvalError(
                "LiveKitConnector.iter_agent_events() called before connect()."
            )
        async for event in iter_downlink(self._out_frames):
            yield event

    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        try:
            await asyncio.wait_for(
                self._agent_track_ready.wait(), timeout=self.max_turn_timeout_s
            )
        except asyncio.TimeoutError:
            raise DeepEvalError(
                "LiveKit agent audio track never became available. Is the "
                "agent worker running and dispatched to the room?"
            )

        self._current_transcript = None
        self._transcript_ready.clear()

        input_audio_started_at = time.perf_counter()
        for frame in self._make_input_frames(audio):
            await self._source.capture_frame(frame)
        await self._source.wait_for_playout()

        sent_at = time.perf_counter()
        agent_pcm, first_audio_at = await collect_agent_turn(
            self._out_frames,
            sample_rate=self.livekit_sample_rate,
            end_of_turn_silence_ms=self.end_of_turn_silence_ms,
            frame_gap_timeout_s=self._frame_gap_timeout_s,
            max_turn_timeout_s=self.max_turn_timeout_s,
            silence_threshold_rms=self.silence_threshold_rms,
        )
        await self._await_transcript(bool(agent_pcm))

        reply_audio = self._agent_pcm_to_audio(agent_pcm)
        latency_ms = (
            (first_audio_at - sent_at) * 1000.0
            if first_audio_at is not None
            else None
        )
        return ConnectorTurn(
            audio=reply_audio,
            transcript=self._current_transcript,
            latency_ms=latency_ms,
            interrupted=False,
            input_audio_started_at=input_audio_started_at,
            input_audio_ended_at=sent_at,
            audio_started_at=first_audio_at,
        )

    async def _await_transcript(self, spoke: bool) -> None:
        """Give a transcript that is still in flight a moment to land.

        The turn ends on silence in the audio, but the text stream carrying the
        same words is a separate delivery and can trail it. Without this the
        transcript would arrive just after the turn was assembled and be read
        as part of the next one.
        """
        if not spoke or self._current_transcript is not None:
            return
        if self.transcript_grace_s <= 0:
            return
        try:
            await asyncio.wait_for(
                self._transcript_ready.wait(), timeout=self.transcript_grace_s
            )
        except asyncio.TimeoutError:
            pass  # no transcript published; the caller falls back to STT

    def drain_downlink(self) -> None:
        while not self._out_frames.empty():
            try:
                self._out_frames.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _make_input_frames(self, audio: Audio) -> List:
        rtc = self._rtc
        pcm, sample_rate, num_channels = audio_utils.wav_bytes_to_pcm16(
            audio.get_bytes()
        )
        pcm = audio_utils.downmix_to_mono(pcm, num_channels)

        if sample_rate != self.livekit_sample_rate:
            resampler = rtc.AudioResampler(
                sample_rate, self.livekit_sample_rate, num_channels=1
            )
            frames = resampler.push(bytearray(pcm))
            frames += resampler.flush()
            return frames

        return self._pcm_to_frames(pcm, sample_rate)

    def _pcm_to_frames(self, pcm: bytes, sample_rate: int) -> List:
        rtc = self._rtc
        return [
            rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk) // 2,
            )
            for chunk in audio_utils.iter_pcm16_frames(
                pcm, sample_rate, frame_ms=audio_utils.DEFAULT_FRAME_MS
            )
        ]

    def _agent_pcm_to_audio(self, pcm: bytes) -> Audio:
        rtc = self._rtc
        if self.livekit_sample_rate != self.input_sample_rate and pcm:
            resampler = rtc.AudioResampler(
                self.livekit_sample_rate, self.input_sample_rate, num_channels=1
            )
            out = resampler.push(bytearray(pcm))
            out += resampler.flush()
            pcm = b"".join(bytes(f.data) for f in out)

        wav = audio_utils.pcm16_to_wav_bytes(pcm, self.input_sample_rate, 1)
        duration = (
            (len(pcm) / 2) / self.input_sample_rate
            if self.input_sample_rate
            else None
        )
        return Audio.from_bytes(
            wav,
            "audio/wav",
            sampleRate=self.input_sample_rate,
            encoding="wav",
            duration=duration,
        )

    async def disconnect(self) -> None:
        await self.stop_uplink()
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
            self._drain_task = None

        for task in list(self._transcript_tasks):
            task.cancel()
        self._transcript_tasks.clear()

        if self._agent_stream is not None:
            try:
                await self._agent_stream.aclose()
            except Exception:
                # Older livekit builds expose no `aclose`; closing the room or
                # the track below tears the stream down either way.
                pass
            self._agent_stream = None

        if self._room is not None:
            if self._owns_room_connection:
                try:
                    await self._room.disconnect()
                except Exception:
                    pass
            else:
                # The room is the caller's and may outlive this simulation, so
                # leave it connected and take back only what we added to it.
                await self._unpublish_microphone()
            self._room = None
            self._owns_room_connection = False

        self._source = None
        self._local_track = None
        self._local_publication = None
        self._agent_track = None
        self._agent_participant = None

    async def _unpublish_microphone(self) -> None:
        sid = getattr(self._local_publication, "sid", None)
        if not sid:
            return
        try:
            await self._room.local_participant.unpublish_track(sid)
        except Exception:
            logger.debug(
                "Could not unpublish the simulated microphone track.",
                exc_info=True,
            )
