import base64
import os
import tempfile
import time
import wave
from typing import Optional

from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils

_CHANNELS = ("user", "agent")


def _chunk_pcm(chunk, default_rate: int):
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk), default_rate
    data_base64 = getattr(chunk, "dataBase64", None)
    if not data_base64:
        return b"", default_rate
    data = base64.b64decode(data_base64)
    rate = getattr(chunk, "sampleRate", None) or default_rate
    mime_type = getattr(chunk, "mimeType", "") or ""
    encoding = getattr(chunk, "encoding", "") or ""
    if "wav" in mime_type or encoding == "wav":
        try:
            pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(data)
        except Exception:
            return b"", default_rate
        return audio_utils.downmix_to_mono(pcm, channels or 1), rate
    return data, rate


class CallRecorder:
    """A tape of one live call, spooled to disk as it happens.

    Each side of the call is appended to its own temporary PCM spool at the
    wall-clock moment it was heard, so memory holds at most one frame. The
    finished recording is a stereo WAV — caller on the left channel, agent on
    the right — whose duration, overlaps, and silences match the call exactly.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._origin: Optional[float] = None
        self._spools = {}
        for channel in _CHANNELS:
            handle, path = tempfile.mkstemp(
                prefix=f"deepeval-call-{channel}-", suffix=".pcm"
            )
            self._spools[channel] = {
                "file": os.fdopen(handle, "wb"),
                "path": path,
                "samples": 0,
            }
        self._finished: Optional[str] = None
        self._discarded = False

    def add(
        self, channel: str, pcm: bytes, sample_rate: int, at: float
    ) -> None:
        if self._finished is not None or self._discarded or not pcm:
            return
        if self._origin is None:
            self._origin = at
        if sample_rate != self.sample_rate:
            pcm = audio_utils.resample_pcm16(pcm, sample_rate, self.sample_rate)
        spool = self._spools[channel]
        target = int(max(at - self._origin, 0.0) * self.sample_rate)
        if target > spool["samples"]:
            spool["file"].write(b"\x00\x00" * (target - spool["samples"]))
            spool["samples"] = target
        spool["file"].write(pcm)
        spool["samples"] += len(pcm) // 2

    def finish(self) -> Optional[str]:
        if self._finished is not None:
            return self._finished
        if self._discarded:
            return None
        for spool in self._spools.values():
            spool["file"].close()
        total = max(spool["samples"] for spool in self._spools.values())
        if total == 0:
            self.discard()
            return None
        handle, path = tempfile.mkstemp(
            prefix="deepeval-call-recording-", suffix=".wav"
        )
        os.close(handle)
        chunk_samples = self.sample_rate
        with wave.open(path, "wb") as writer, open(
            self._spools["user"]["path"], "rb"
        ) as user, open(self._spools["agent"]["path"], "rb") as agent:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            remaining = total
            while remaining > 0:
                count = min(chunk_samples, remaining)
                left = user.read(count * 2).ljust(count * 2, b"\x00")
                right = agent.read(count * 2).ljust(count * 2, b"\x00")
                frame = bytearray(count * 4)
                frame[0::4] = left[0::2]
                frame[1::4] = left[1::2]
                frame[2::4] = right[0::2]
                frame[3::4] = right[1::2]
                writer.writeframes(bytes(frame))
                remaining -= count
        for spool in self._spools.values():
            os.unlink(spool["path"])
        self._finished = path
        return path

    def discard(self) -> None:
        if self._discarded:
            return
        self._discarded = True
        for spool in self._spools.values():
            try:
                spool["file"].close()
            except Exception:
                pass
            try:
                os.unlink(spool["path"])
            except OSError:
                pass


class RecordingConnector:
    """Wraps a connector so every frame either side produced lands on tape.

    The tap sits at the connector boundary: uplink audio is recorded for the
    time it was actually on the wire (a cancelled utterance is trimmed to what
    was sent), and downlink audio is recorded as it arrives — including
    anything past a turn cut that the simulator later discards.
    """

    def __init__(self, connector, recorder: CallRecorder):
        self._inner = connector
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def clone(self) -> "RecordingConnector":
        return RecordingConnector(self._inner.clone(), self._recorder)

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._inner.__aexit__(*args)

    def _decode(self, audio: Audio):
        try:
            pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(
                audio.get_bytes()
            )
        except Exception:
            return None, None
        return audio_utils.downmix_to_mono(pcm, channels or 1), rate

    async def exchange_turn(self, audio: Audio):
        started = time.perf_counter()
        result = await self._inner.exchange_turn(audio)
        pcm, rate = self._decode(audio)
        if pcm:
            self._recorder.add(
                "user", pcm, rate, result.input_audio_started_at or started
            )
        if result.audio is not None:
            reply_pcm, reply_rate = self._decode(result.audio)
            if reply_pcm:
                reply_at = result.audio_started_at
                if reply_at is None:
                    reply_at = (
                        result.input_audio_ended_at or time.perf_counter()
                    ) + (result.latency_ms or 0.0) / 1000.0
                self._recorder.add("agent", reply_pcm, reply_rate, reply_at)
        return result

    async def exchange_turn_stream(self, frames):
        default_rate = getattr(
            self._inner, "input_sample_rate", self._recorder.sample_rate
        )

        async def tapped():
            async for chunk in frames:
                pcm, rate = _chunk_pcm(chunk, default_rate)
                if pcm:
                    self._recorder.add("user", pcm, rate, time.perf_counter())
                yield chunk

        result = await self._inner.exchange_turn_stream(tapped())
        if result.audio is not None:
            reply_pcm, reply_rate = self._decode(result.audio)
            if reply_pcm:
                reply_at = result.audio_started_at
                if reply_at is None:
                    reply_at = (
                        result.input_audio_ended_at or time.perf_counter()
                    ) + (result.latency_ms or 0.0) / 1000.0
                self._recorder.add("agent", reply_pcm, reply_rate, reply_at)
        return result

    async def stream_uplink(self, audio: Audio, **kwargs):
        started = time.perf_counter()
        try:
            return await self._inner.stream_uplink(audio, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            pcm, rate = self._decode(audio)
            if pcm:
                sent_bytes = min(len(pcm), 2 * int(elapsed * rate))
                self._recorder.add("user", pcm[:sent_bytes], rate, started)

    async def stream_uplink_chunks(self, frames, **kwargs):
        default_rate = getattr(
            self._inner, "input_sample_rate", self._recorder.sample_rate
        )

        async def tapped():
            async for chunk in frames:
                pcm, rate = _chunk_pcm(chunk, default_rate)
                if pcm:
                    self._recorder.add("user", pcm, rate, time.perf_counter())
                yield chunk

        return await self._inner.stream_uplink_chunks(tapped(), **kwargs)

    async def iter_agent_events(self):
        rate = getattr(
            self._inner, "recv_sample_rate", self._recorder.sample_rate
        )
        async for event in self._inner.iter_agent_events():
            if event.audio:
                self._recorder.add(
                    "agent",
                    bytes(event.audio),
                    rate,
                    event.received_at or time.perf_counter(),
                )
            yield event
