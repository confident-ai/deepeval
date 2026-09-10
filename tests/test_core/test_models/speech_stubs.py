"""A stand-in for `SpeechTransport` that records calls instead of making them."""

from typing import Any, Dict, List, Optional

from deepeval.models.speech import SpeechHTTPError
from deepeval.test_case import Audio
from deepeval.voice.connectors import audio_utils


class FakeTransport:
    """Replaces the transport on a speech model.

    Recording the request each model builds is the whole point: the interesting
    behaviour is which endpoint, query parameters and body a provider needs, and
    that is exactly what a real HTTP call would hide.
    """

    def __init__(
        self,
        *,
        content: bytes = b"",
        json: Any = None,
        json_sequence: Optional[List[Any]] = None,
        stream_reads: Optional[List[bytes]] = None,
    ):
        self._content = content
        self._json_sequence = (
            list(json_sequence) if json_sequence is not None else None
        )
        self._json = json
        self._stream_reads = stream_reads or []
        self.calls: List[Dict[str, Any]] = []

    #
    # Recording
    #

    def _record(self, method: str, path: str, kwargs: Dict[str, Any]) -> None:
        self.calls.append({"method": method, "path": path, **kwargs})

    @property
    def last(self) -> Dict[str, Any]:
        return self.calls[-1]

    def params_of(self, index: int = -1) -> Dict[str, str]:
        """Query params as a dict, flattening the pair list the transport uses."""
        params = self.calls[index].get("params") or {}
        if isinstance(params, dict):
            return {k: str(v) for k, v in params.items()}
        return {k: str(v) for k, v in params}

    def paths(self) -> List[str]:
        return [call["path"] for call in self.calls]

    #
    # Responses
    #

    def _next_json(self) -> Any:
        if self._json_sequence is not None:
            if not self._json_sequence:
                raise SpeechHTTPError("FakeTransport ran out of responses")
            return self._json_sequence.pop(0)
        return self._json

    def request_bytes(self, method: str, path: str, **kwargs) -> bytes:
        self._record(method, path, kwargs)
        return self._content

    def request_json(self, method: str, path: str, **kwargs) -> Any:
        self._record(method, path, kwargs)
        return self._next_json()

    async def a_request_bytes(self, method: str, path: str, **kwargs) -> bytes:
        self._record(method, path, kwargs)
        return self._content

    async def a_request_json(self, method: str, path: str, **kwargs) -> Any:
        self._record(method, path, kwargs)
        return self._next_json()

    async def a_stream_bytes(self, method: str, path: str, **kwargs):
        self._record(method, path, kwargs)
        for read in self._stream_reads:
            yield read


def wav_audio(
    seconds: float = 1.0,
    sample_rate: int = 24000,
    *,
    duration: Optional[float] = None,
) -> Audio:
    """A real (silent) WAV, so duration-based cost and routing are exercised."""
    pcm = b"\x00\x00" * int(sample_rate * seconds)
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate, 1),
        "audio/wav",
        sampleRate=sample_rate,
        encoding="wav",
        duration=duration,
    )
