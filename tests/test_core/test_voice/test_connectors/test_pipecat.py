"""Speaking Pipecat's protobuf frames, and reading RTVI off the same socket."""

import json

from deepeval.voice import PipecatConnector
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.providers.pipecat import protobuf


def _connector(**kwargs) -> PipecatConnector:
    kwargs.setdefault("url", "ws://localhost:8765/ws")
    return PipecatConnector(**kwargs)


def _rtvi(message_type: str, **data) -> bytes:
    payload = {"label": "rtvi-ai", "type": message_type, "id": "1"}
    if data:
        payload["data"] = data
    return protobuf.encode_message_frame(json.dumps(payload))


def _string_frame(frame_field: int, text: str) -> bytes:
    """A `Frame` whose only content is a string in field 3, as text frames are."""
    inner = protobuf._len_field(3, text.encode("utf-8"))
    return protobuf._len_field(frame_field, inner)


def test_audio_frames_match_the_wire_schema():
    """Byte-for-byte, since the pipeline on the other end is the only reader."""
    assert protobuf.encode_audio_frame(b"\x01\x02", 16000, 1) == (
        b"\x12\x09"  # Frame.audio, 9 bytes
        b"\x1a\x02\x01\x02"  # AudioRawFrame.audio
        b"\x20\x80\x7d"  # AudioRawFrame.sample_rate = 16000
        b"\x28\x01"  # AudioRawFrame.num_channels = 1
    )


def test_an_encoded_audio_frame_decodes_back():
    frame = protobuf.decode_frame(
        protobuf.encode_audio_frame(b"\x00\x01\x02\x03", 24000, 2)
    )

    assert frame.kind == "audio"
    assert frame.audio == b"\x00\x01\x02\x03"
    assert frame.sample_rate == 24000
    assert frame.num_channels == 2


def test_unknown_frames_and_junk_are_told_apart():
    """An unread frame is None; bytes that are not protobuf at all raise."""
    assert protobuf.decode_frame(protobuf._varint_field(9, 1)) is None

    connector = _connector()
    assert connector._decode_inbound(b"\xff\xff\xff") is None
    assert connector._decode_inbound("a text message") is None


def test_audio_goes_out_at_the_rate_the_pipeline_listens_at():
    connector = _connector(agent_sample_rate=8000)

    frame = protobuf.decode_frame(connector._encode_outbound(b"\x01\x02"))

    assert frame.sample_rate == 8000


def test_a_client_ready_message_opens_the_session():
    """Pipelines commonly hold their greeting until the client says it is up."""
    frame = protobuf.decode_frame(_connector()._initial_messages()[0])

    message = json.loads(frame.data)
    assert message["label"] == "rtvi-ai"
    assert message["type"] == "client-ready"
    assert message["data"]["about"]["library"] == "deepeval"


def test_client_ready_can_be_turned_off():
    assert _connector(client_ready=False)._initial_messages() == []


def test_the_turn_end_signal_is_claimed_only_once_seen():
    """A pipeline without RTVI never sends one, and must not be waited on."""
    connector = _connector()
    assert connector.signals_turn_complete is False

    event = connector._decode_inbound(_rtvi("bot-stopped-speaking"))

    assert event.turn_complete is True
    assert connector.signals_turn_complete is True


def test_bot_transcriptions_accumulate_across_one_turn():
    """RTVI sends one per sentence, not one per turn."""
    connector = _connector()

    first = connector._decode_inbound(
        _rtvi("bot-transcription", text="Sure, one moment.")
    )
    second = connector._decode_inbound(
        _rtvi("bot-transcription", text="Your table is booked.")
    )

    assert first.transcript == "Sure, one moment."
    assert second.transcript == "Sure, one moment. Your table is booked."


def test_a_new_turn_starts_from_an_empty_transcript():
    connector = _connector()
    connector._decode_inbound(_rtvi("bot-transcription", text="First turn."))

    connector._decode_inbound(_rtvi("bot-stopped-speaking"))
    event = connector._decode_inbound(
        _rtvi("bot-transcription", text="Second turn.")
    )

    assert event.transcript == "Second turn."


def test_an_interruption_is_recorded_on_the_turn():
    connector = _connector()

    assert connector._decode_inbound(_rtvi("bot-interrupted")) is None
    assert connector._interrupted is True


def test_other_rtvi_messages_are_ignored():
    connector = _connector()

    assert connector._decode_inbound(_rtvi("user-started-speaking")) is None
    assert connector._decode_inbound(_rtvi("bot-llm-text", text="Hi")) is None


def test_pipeline_text_is_not_mistaken_for_the_agents_reply():
    """A `transcription` frame is the pipeline's STT of deepeval's own audio."""
    connector = _connector()

    assert connector._decode_inbound(_string_frame(3, "book a table")) is None
    assert connector._decode_inbound(_string_frame(1, "some text")) is None
    assert connector._current_transcript is None


def test_the_rate_the_pipeline_speaks_at_wins_over_the_default():
    connector = _connector(sample_rate=24000)

    connector._decode_inbound(
        protobuf.encode_audio_frame(b"\x01\x02", 16000, 1)
    )

    assert connector._recv_rate == 16000


def test_wav_wrapped_audio_is_unwrapped():
    """`add_wav_header=True` on the transport wraps every audio frame."""
    pcm = b"\x01\x02\x03\x04"
    wav = audio_utils.pcm16_to_wav_bytes(pcm, 16000, 1)
    connector = _connector()

    event = connector._decode_inbound(
        protobuf.encode_audio_frame(wav, 16000, 1)
    )

    assert event.audio == pcm
    assert connector._recv_rate == 16000
