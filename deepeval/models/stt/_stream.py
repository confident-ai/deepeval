"""Partial transcripts for providers whose batch API is all we use."""

from __future__ import annotations

from typing import AsyncGenerator, AsyncIterable, Optional

from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils

DEFAULT_STREAM_SAMPLE_RATE = 24000


class BufferedTranscribeMixin:
    """Approximates streaming transcription on top of a batch endpoint.

    `DeepEvalBaseSTT.a_transcribe_stream` raises by default, so a model without
    it cannot follow an agent as it speaks — a caller wanting a transcript
    before the utterance ends has nothing to consume. Every provider here has a
    realtime WebSocket API, but the models talk to the batch endpoint, so this
    buffers the incoming PCM and re-transcribes the accumulated audio whenever
    roughly `partial_every_seconds` of new audio has arrived, yielding the
    transcript each time it changes.

    Re-transcribing from the top is wasteful, and deliberately so: it keeps a
    provider's streaming protocol out of the models, and the audio involved is a
    single conversational turn. Mixing this in gives a model working partials
    from `a_transcribe` alone; a provider whose realtime API is worth the
    complexity can override `a_transcribe_stream` instead.

    Must be mixed in ahead of `DeepEvalBaseSTT` so it takes precedence over the
    base's `NotImplementedError`.
    """

    async def a_transcribe_stream(
        self,
        audio_stream: AsyncIterable[AudioChunk],
        *args,
        language: Optional[str] = None,
        partial_every_seconds: float = 1.0,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        pcm = bytearray()
        sample_rate = DEFAULT_STREAM_SAMPLE_RATE
        bytes_since_partial = 0
        last_text = ""

        async for chunk in audio_stream:
            if chunk.sampleRate:
                sample_rate = chunk.sampleRate
            frame = chunk.get_bytes()
            pcm.extend(frame)
            bytes_since_partial += len(frame)

            threshold = int(sample_rate * 2 * partial_every_seconds)
            if bytes_since_partial < threshold and not chunk.final:
                continue

            audio = Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(bytes(pcm), sample_rate, 1),
                "audio/wav",
                sampleRate=sample_rate,
                encoding="wav",
            )
            text, _ = await self.a_transcribe(
                audio, language=language, **kwargs
            )
            if text and text != last_text:
                last_text = text
                yield text
            bytes_since_partial = 0
            if chunk.final:
                break

    def supports_streaming(self) -> bool:
        return True
