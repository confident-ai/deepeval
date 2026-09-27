import os
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import BinaryIO, Dict, Optional

from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils


@dataclass
class _Spool:
    file: BinaryIO
    path: str
    samples: int = 0


class CallRecorder:
    """A stereo tape of one call, caller left, agent right, spooled to disk."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._origin: Optional[float] = None
        self._spools: Dict[str, _Spool] = {}
        for channel in ("user", "agent"):
            handle, path = tempfile.mkstemp(
                prefix=f"deepeval-call-{channel}-", suffix=".pcm"
            )
            self._spools[channel] = _Spool(os.fdopen(handle, "wb"), path)
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
        if target > spool.samples:
            spool.file.write(b"\x00\x00" * (target - spool.samples))
            spool.samples = target
        spool.file.write(pcm)
        spool.samples += len(pcm) // 2

    def finish(self) -> Optional[str]:
        if self._finished is not None or self._discarded:
            return self._finished
        for spool in self._spools.values():
            spool.file.close()
        if max(spool.samples for spool in self._spools.values()) == 0:
            self.discard()
            return None
        handle, path = tempfile.mkstemp(
            prefix="deepeval-call-recording-", suffix=".wav"
        )
        os.close(handle)
        with wave.open(path, "wb") as writer, open(
            self._spools["user"].path, "rb"
        ) as user, open(self._spools["agent"].path, "rb") as agent:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            while True:
                left = user.read(self.sample_rate * 2)
                right = agent.read(self.sample_rate * 2)
                if not left and not right:
                    break
                width = max(len(left), len(right))
                left = left.ljust(width, b"\x00")
                right = right.ljust(width, b"\x00")
                frame = bytearray(width * 2)
                frame[0::4] = left[0::2]
                frame[1::4] = left[1::2]
                frame[2::4] = right[0::2]
                frame[3::4] = right[1::2]
                writer.writeframes(bytes(frame))
        for spool in self._spools.values():
            os.unlink(spool.path)
        self._finished = path
        return path

    def discard(self) -> None:
        if self._discarded:
            return
        self._discarded = True
        for spool in self._spools.values():
            spool.file.close()
            try:
                os.unlink(spool.path)
            except OSError:
                pass


class RecordingConnector:
    """Proxies a connector, taping each frame at the moment it hit the wire."""

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

    def _tape(
        self,
        channel: str,
        audio: Audio,
        at: float,
        seconds: Optional[float] = None,
    ) -> None:
        try:
            pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(
                audio.get_bytes()
            )
        except Exception:
            return
        pcm = audio_utils.downmix_to_mono(pcm, channels or 1)
        if seconds is not None:
            pcm = pcm[: 2 * int(seconds * rate)]
        self._recorder.add(channel, pcm, rate, at)

    async def exchange_turn(self, audio: Audio):
        started = time.perf_counter()
        result = await self._inner.exchange_turn(audio)
        self._tape("user", audio, result.input_audio_started_at or started)
        if result.audio is not None:
            reply_at = result.audio_started_at
            if reply_at is None:
                reply_at = (
                    result.input_audio_ended_at or time.perf_counter()
                ) + (result.latency_ms or 0.0) / 1000.0
            self._tape("agent", result.audio, reply_at)
        return result

    async def stream_uplink(self, audio: Audio, **kwargs):
        started = time.perf_counter()
        try:
            return await self._inner.stream_uplink(audio, **kwargs)
        finally:
            self._tape("user", audio, started, time.perf_counter() - started)

    async def stream_uplink_chunks(self, frames, **kwargs):
        async def tapped():
            async for chunk in frames:
                self._recorder.add(
                    "user",
                    chunk.get_bytes(),
                    chunk.sampleRate or self._recorder.sample_rate,
                    time.perf_counter(),
                )
                yield chunk

        return await self._inner.stream_uplink_chunks(tapped(), **kwargs)

    async def iter_agent_events(self):
        rate = self._inner.recv_sample_rate
        async for event in self._inner.iter_agent_events():
            if event.audio:
                self._recorder.add(
                    "agent",
                    bytes(event.audio),
                    rate,
                    event.received_at or time.perf_counter(),
                )
            yield event
