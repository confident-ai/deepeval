import asyncio
import time
import uuid
from datetime import timedelta
from typing import List, Optional, Tuple

from deepeval.errors import DeepEvalError
from deepeval.utils import require_dependency
from deepeval.test_case import Audio
from deepeval.voice.connectors.base_connector import BaseVoiceConnector
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.turn_engine import collect_agent_turn

_INSTALL_HINT = "Install it with `pip install deepeval[livekit]`."


class LiveKitConnector(BaseVoiceConnector):
    def __init__(
        self,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        room_name: Optional[str] = None,
        identity: str = "deepeval-test",
        agent_name: Optional[str] = None,
        end_of_turn_silence_ms: int = 800,
        max_turn_timeout_s: float = 30.0,
        connect_timeout_s: float = 15.0,
        input_sample_rate: int = 24000,
        livekit_sample_rate: int = 48000,
        token_ttl_s: int = 3600,
    ):
        import os

        self.url = url or os.getenv("LIVEKIT_URL")
        self.api_key = api_key or os.getenv("LIVEKIT_API_KEY")
        self.api_secret = api_secret or os.getenv("LIVEKIT_API_SECRET")
        if not (self.url and self.api_key and self.api_secret):
            raise DeepEvalError(
                "LiveKitConnector requires a LiveKit URL, API key and API secret "
                "(pass url/api_key/api_secret or set LIVEKIT_URL, "
                "LIVEKIT_API_KEY, LIVEKIT_API_SECRET)."
            )

        self._room_name_arg = room_name
        self.room_name = room_name
        self.identity = identity
        self.agent_name = agent_name
        self.end_of_turn_silence_ms = end_of_turn_silence_ms
        self.max_turn_timeout_s = max_turn_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.input_sample_rate = input_sample_rate
        self.livekit_sample_rate = livekit_sample_rate
        self.token_ttl_s = token_ttl_s
        self._frame_gap_timeout_s = max(
            1.0, end_of_turn_silence_ms / 1000.0 + 0.5
        )

        # Lazily populated in connect().
        self._rtc = None
        self._api = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._room = None
        self._source = None
        self._local_track = None
        self._agent_track = None
        self._agent_stream = None
        self._agent_participant = None
        self._drain_task: Optional[asyncio.Task] = None
        self._out_frames: Optional[asyncio.Queue] = None
        self._agent_track_ready: Optional[asyncio.Event] = None

    @property
    def audio_format(self) -> Tuple[int, str]:
        return (self.input_sample_rate, "wav")

    async def connect(self) -> None:
        self._rtc = require_dependency(
            "livekit.rtc",
            provider_label="LiveKitConnector",
            install_hint=_INSTALL_HINT,
        )
        self._api = require_dependency(
            "livekit.api",
            provider_label="LiveKitConnector",
            install_hint=_INSTALL_HINT,
        )
        rtc = self._rtc

        self.room_name = (
            self._room_name_arg or f"deepeval-{uuid.uuid4().hex[:12]}"
        )

        self._loop = asyncio.get_event_loop()
        self._out_frames = asyncio.Queue()
        self._agent_track_ready = asyncio.Event()
        self._room = rtc.Room()

        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("participant_connected", self._on_participant_connected)

        token = self._build_token()
        await self._room.connect(
            self.url, token, rtc.RoomOptions(auto_subscribe=True)
        )

        self._source = rtc.AudioSource(self.livekit_sample_rate, 1)
        self._local_track = rtc.LocalAudioTrack.create_audio_track(
            "deepeval-user", self._source
        )
        await self._room.local_participant.publish_track(
            self._local_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        self._adopt_existing_agent_track()

        try:
            await asyncio.wait_for(
                self._agent_track_ready.wait(), timeout=self.connect_timeout_s
            )
        except asyncio.TimeoutError:
            await self.disconnect()
            raise DeepEvalError(
                f"No LiveKit agent joined room '{self.room_name}' within "
                f"{self.connect_timeout_s}s. Is the agent worker running and "
                "dispatched to this project?"
            )

    def _build_token(self) -> str:
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

    def _adopt_existing_agent_track(self) -> None:
        if self._agent_track is not None:
            return
        rtc = self._rtc
        for participant in self._room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = getattr(publication, "track", None)
                if track is not None and track.kind == rtc.TrackKind.KIND_AUDIO:
                    self._attach_agent_track(track, participant)
                    return

    def _on_track_subscribed(self, track, publication, participant) -> None:
        rtc = self._rtc
        if self._agent_track is not None:
            return
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self._attach_agent_track(track, participant)

    def _on_participant_connected(self, participant) -> None:
        if self._agent_participant is None:
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
                await self._out_frames.put(bytes(event.frame.data))
        except asyncio.CancelledError:
            raise
        finally:
            await self._out_frames.put(None)

    async def send_turn(self, audio: Audio) -> ConnectorTurn:
        try:
            await asyncio.wait_for(
                self._agent_track_ready.wait(), timeout=self.max_turn_timeout_s
            )
        except asyncio.TimeoutError:
            raise DeepEvalError(
                "LiveKit agent audio track never became available. Is the "
                "agent worker running and dispatched to the room?"
            )

        self._drain_stale_frames()

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
        )

        reply_audio = self._agent_pcm_to_audio(agent_pcm)
        latency_ms = (
            (first_audio_at - sent_at) * 1000.0
            if first_audio_at is not None
            else None
        )
        return ConnectorTurn(
            audio=reply_audio,
            transcript=None,
            latency_ms=latency_ms,
            interrupted=False,
        )

    def _drain_stale_frames(self) -> None:
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

        frames = []
        for chunk in audio_utils.iter_pcm16_frames(
            pcm, sample_rate, frame_ms=audio_utils.DEFAULT_FRAME_MS
        ):
            frames.append(
                rtc.AudioFrame(
                    data=chunk,
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
            )
        return frames

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
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
            self._drain_task = None

        if self._agent_stream is not None:
            aclose = getattr(self._agent_stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
            self._agent_stream = None

        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception:
                pass
            self._room = None

        self._source = None
        self._local_track = None
        self._agent_track = None
        self._agent_participant = None
