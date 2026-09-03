"""The part of Pipecat's `frames.proto` that a caller has to speak.

Pipecat's WebSocket transports serialize with `ProtobufFrameSerializer`, whose
schema is five messages wide and carries only what a client sends or reads:
audio, text, a transcription, a JSON transport message, and an interruption.
Encoding those directly keeps `pipecat-ai` — a server-side framework with a
large dependency tree — out of the requirements of running a simulation
against it.

https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/serializers/protobuf.py
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LEN = 2
_WIRE_32BIT = 5

# Fields of the `Frame` oneof.
_FRAME_TEXT = 1
_FRAME_AUDIO = 2
_FRAME_TRANSCRIPTION = 3
_FRAME_MESSAGE = 4
_FRAME_INTERRUPTION = 5

# Fields within each of those messages.
_TEXT_TEXT = 3
_AUDIO_AUDIO = 3
_AUDIO_SAMPLE_RATE = 4
_AUDIO_NUM_CHANNELS = 5
_TRANSCRIPTION_TEXT = 3
_MESSAGE_DATA = 1


@dataclass
class PipecatFrame:
    """One decoded frame, named by which arm of the oneof it arrived on."""

    kind: str  # text | audio | transcription | message | interruption
    audio: Optional[bytes] = None
    sample_rate: Optional[int] = None
    num_channels: Optional[int] = None
    text: Optional[str] = None
    data: Optional[str] = None  # JSON, for `message`


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint(field << 3 | wire)


def _len_field(field: int, payload: bytes) -> bytes:
    return _tag(field, _WIRE_LEN) + _varint(len(payload)) + payload


def _varint_field(field: int, value: int) -> bytes:
    return _tag(field, _WIRE_VARINT) + _varint(value)


def encode_audio_frame(
    pcm: bytes, sample_rate: int, num_channels: int = 1
) -> bytes:
    """A `Frame` carrying PCM16, which Pipecat reads as an `InputAudioRawFrame`."""
    audio = (
        _len_field(_AUDIO_AUDIO, pcm)
        + _varint_field(_AUDIO_SAMPLE_RATE, sample_rate)
        + _varint_field(_AUDIO_NUM_CHANNELS, num_channels)
    )
    return _len_field(_FRAME_AUDIO, audio)


def encode_message_frame(data: str) -> bytes:
    """A `Frame` carrying a JSON transport message, such as RTVI's."""
    return _len_field(
        _FRAME_MESSAGE, _len_field(_MESSAGE_DATA, data.encode("utf-8"))
    )


def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    value = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def _iter_fields(buf: bytes) -> Iterator[Tuple[int, int, object]]:
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 0x07
        if wire == _WIRE_VARINT:
            value, i = _read_varint(buf, i)
        elif wire == _WIRE_LEN:
            length, i = _read_varint(buf, i)
            value, i = buf[i : i + length], i + length
            if len(value) != length:
                raise ValueError("truncated length-delimited field")
        elif wire in (_WIRE_64BIT, _WIRE_32BIT):
            width = 8 if wire == _WIRE_64BIT else 4
            value, i = buf[i : i + width], i + width
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _string(value: bytes) -> str:
    return value.decode("utf-8", "replace")


def _audio_frame(payload: bytes) -> PipecatFrame:
    frame = PipecatFrame(kind="audio", audio=b"")
    for field, _, value in _iter_fields(payload):
        if field == _AUDIO_AUDIO:
            frame.audio = bytes(value)
        elif field == _AUDIO_SAMPLE_RATE:
            frame.sample_rate = value
        elif field == _AUDIO_NUM_CHANNELS:
            frame.num_channels = value
    return frame


def _one_string_frame(
    payload: bytes, kind: str, field_number: int
) -> PipecatFrame:
    frame = PipecatFrame(kind=kind, text="")
    for field, _, value in _iter_fields(payload):
        if field == field_number:
            frame.text = _string(value)
    return frame


def _message_frame(payload: bytes) -> PipecatFrame:
    frame = PipecatFrame(kind="message", data="")
    for field, _, value in _iter_fields(payload):
        if field == _MESSAGE_DATA:
            frame.data = _string(value)
    return frame


def decode_frame(raw: bytes) -> Optional[PipecatFrame]:
    """Read a serialized `Frame`, or None if it carries nothing we know.

    Raises `ValueError` on bytes that are not protobuf at all, which is how a
    pipeline serializing something else announces itself.
    """
    for field, wire, value in _iter_fields(raw):
        if wire != _WIRE_LEN:
            continue
        if field == _FRAME_AUDIO:
            return _audio_frame(value)
        if field == _FRAME_MESSAGE:
            return _message_frame(value)
        if field == _FRAME_TRANSCRIPTION:
            return _one_string_frame(
                value, "transcription", _TRANSCRIPTION_TEXT
            )
        if field == _FRAME_TEXT:
            return _one_string_frame(value, "text", _TEXT_TEXT)
        if field == _FRAME_INTERRUPTION:
            return PipecatFrame(kind="interruption")
    return None
