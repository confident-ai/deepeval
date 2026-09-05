"""Reading Vapi's client messages off the call socket."""

import json

import pytest

from deepeval.errors import DeepEvalError
from deepeval.voice import VapiConnector


def _connector(**kwargs) -> VapiConnector:
    kwargs.setdefault("assistant_id", "assistant-1")
    kwargs.setdefault("api_key", "key-1")
    return VapiConnector(**kwargs)


def _speech_update(status: str, role: str = "assistant") -> str:
    return json.dumps({"type": "speech-update", "status": status, "role": role})


def _transcript(text: str, *, role="assistant", kind="final") -> str:
    return json.dumps(
        {
            "type": "transcript",
            "role": role,
            "transcriptType": kind,
            "transcript": text,
        }
    )


def test_call_payload_asks_for_pcm_at_the_connector_rate():
    payload = _connector(sample_rate=24000)._call_payload()

    assert payload["assistantId"] == "assistant-1"
    assert payload["transport"] == {
        "provider": "vapi.websocket",
        "audioFormat": {
            "format": "pcm_s16le",
            "container": "raw",
            "sampleRate": 24000,
        },
    }
    assert "assistantOverrides" not in payload


def test_overrides_are_passed_through_untouched():
    overrides = {"variableValues": {"customer_name": "Alice"}}

    payload = _connector(assistant_overrides=overrides)._call_payload()

    assert payload["assistantOverrides"] == overrides


def test_narrowed_client_messages_regain_the_ones_deepeval_reads():
    """Without these, turns end on silence and every turn pays for STT."""
    connector = _connector(assistant_overrides={"clientMessages": ["hang"]})

    messages = connector._call_payload()["assistantOverrides"]["clientMessages"]

    assert messages == ["hang", "speech-update", "transcript"]


def test_client_messages_are_left_alone_when_already_complete():
    connector = _connector(
        assistant_overrides={"clientMessages": ["transcript", "speech-update"]}
    )

    messages = connector._call_payload()["assistantOverrides"]["clientMessages"]

    assert messages == ["transcript", "speech-update"]


def test_binary_frames_are_agent_audio():
    event = _connector()._decode_inbound(b"\x01\x02")

    assert event.audio == b"\x01\x02"


def test_a_stopped_assistant_closes_the_turn():
    connector = _connector()

    assert connector.signals_turn_complete is True
    assert connector._decode_inbound(_speech_update("stopped")).turn_complete


def test_the_caller_speaking_does_not_close_the_agents_turn():
    connector = _connector()

    assert connector._decode_inbound(_speech_update("started")) is None
    assert connector._decode_inbound(_speech_update("stopped", "user")) is None


def test_final_transcripts_accumulate_across_one_turn():
    """Vapi sends one final per utterance, not one per turn."""
    connector = _connector()

    first = connector._decode_inbound(_transcript("Sure, one moment."))
    second = connector._decode_inbound(_transcript("Your table is booked."))

    assert first.transcript == "Sure, one moment."
    assert second.transcript == "Sure, one moment. Your table is booked."


def test_partial_and_caller_transcripts_are_ignored():
    connector = _connector()

    assert (
        connector._decode_inbound(_transcript("Sure", kind="partial")) is None
    )
    assert connector._decode_inbound(_transcript("Hi", role="user")) is None


def test_a_new_turn_starts_from_an_empty_transcript():
    connector = _connector()
    connector._decode_inbound(_transcript("First turn."))

    connector._decode_inbound(_speech_update("stopped"))
    event = connector._decode_inbound(_transcript("Second turn."))

    assert event.transcript == "Second turn."


def test_events_wrapped_in_a_message_envelope_are_understood():
    """Webhook deliveries wrap the event; the socket may do either."""
    connector = _connector()

    raw = json.dumps(
        {
            "message": {
                "type": "speech-update",
                "status": "stopped",
                "role": "assistant",
            }
        }
    )

    assert connector._decode_inbound(raw).turn_complete


def test_an_interruption_is_recorded_on_the_turn():
    connector = _connector()

    assert (
        connector._decode_inbound(json.dumps({"type": "user-interrupted"}))
        is None
    )
    assert connector._interrupted is True


def test_unknown_events_and_junk_are_ignored():
    connector = _connector()

    assert (
        connector._decode_inbound(json.dumps({"type": "voice-input"})) is None
    )
    assert connector._decode_inbound("not json") is None
    assert connector._decode_inbound(json.dumps([1, 2])) is None


def test_the_rate_vapi_settles_on_wins_over_the_one_requested():
    connector = _connector(sample_rate=24000)

    connector._apply_audio_format({"format": "pcm_s16le", "sampleRate": 16000})

    assert connector._send_rate == 16000
    assert connector._recv_rate == 16000


@pytest.mark.asyncio
async def test_a_missing_api_key_is_reported_before_any_request(monkeypatch):
    monkeypatch.delenv("VAPI_API_KEY", raising=False)
    connector = VapiConnector(assistant_id="assistant-1", api_key=None)

    with pytest.raises(DeepEvalError, match="VAPI_API_KEY"):
        await connector._open_session()
