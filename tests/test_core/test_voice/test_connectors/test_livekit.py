import asyncio
from types import SimpleNamespace

import pytest

from deepeval.voice import LiveKitConnector
from deepeval.voice.connectors.providers.livekit import (
    connector as livekit_module,
)
from tests.test_core.test_voice.helpers import install_fake_livekit


class _BlockingReader:
    info = SimpleNamespace(attributes={})

    async def read_all(self):
        await asyncio.Event().wait()


def _connector(**kwargs) -> LiveKitConnector:
    return LiveKitConnector(
        url="wss://livekit.test", api_key="key", api_secret="secret", **kwargs
    )


@pytest.mark.asyncio
async def test_hanging_up_one_call_leaves_another_calls_transcripts(
    monkeypatch,
):
    install_fake_livekit(monkeypatch, livekit_module)
    template = _connector()
    first, second = template.clone(), template.clone()
    await first._join_room()
    await second._join_room()

    first._on_transcript_stream(_BlockingReader(), "agent")
    second._on_transcript_stream(_BlockingReader(), "agent")
    await asyncio.sleep(0)
    [second_task] = list(second._transcript_tasks)

    await first.disconnect()
    await asyncio.sleep(0)

    assert not second_task.cancelled()
    assert second._transcript_tasks == {second_task}
    await second.disconnect()
    await asyncio.sleep(0)
    assert second_task.cancelled()


def test_a_connector_that_creates_its_own_rooms_supports_parallel_calls():
    assert _connector().supports_parallel_calls


@pytest.mark.parametrize(
    "fixed",
    [
        {"room_name": "support-line"},
        {"token": "caller-token"},
        {"room": SimpleNamespace()},
    ],
)
def test_a_connector_tied_to_one_room_does_not_support_parallel_calls(fixed):
    assert not _connector(**fixed).supports_parallel_calls


class _FinalReader:
    def __init__(self, text: str, timestamp: float):
        self._text = text
        self.info = SimpleNamespace(
            attributes={livekit_module.TRANSCRIPTION_FINAL_ATTRIBUTE: "true"},
            timestamp=timestamp,
        )

    async def read_all(self):
        return self._text


def _emitted_transcripts(connector: LiveKitConnector):
    transcripts = []
    while not connector._out_frames.empty():
        event = connector._out_frames.get_nowait()
        if event.transcript:
            transcripts.append(event.transcript)
    return transcripts


async def _joined(monkeypatch) -> LiveKitConnector:
    install_fake_livekit(monkeypatch, livekit_module)
    connector = _connector()
    await connector._join_room()
    return connector


@pytest.mark.asyncio
async def test_every_part_of_an_agent_reply_reaches_its_transcript(monkeypatch):
    connector = await _joined(monkeypatch)
    now = livekit_module._now_ms()

    await connector._read_transcript(_FinalReader("Welcome to Acme.", now))
    await connector._read_transcript(
        _FinalReader("This call may be recorded.", now + 5)
    )

    assert _emitted_transcripts(connector) == [
        "Welcome to Acme.",
        "Welcome to Acme. This call may be recorded.",
    ]
    assert connector._current_transcript == (
        "Welcome to Acme. This call may be recorded."
    )
    await connector.disconnect()


@pytest.mark.asyncio
async def test_a_new_caller_turn_starts_a_new_agent_transcript(monkeypatch):
    connector = await _joined(monkeypatch)
    await connector._read_transcript(
        _FinalReader("How can I help?", livekit_module._now_ms())
    )

    connector._note_caller_speech()
    await connector._read_transcript(
        _FinalReader("Sure, let me check.", livekit_module._now_ms() + 5)
    )

    assert _emitted_transcripts(connector) == [
        "How can I help?",
        "Sure, let me check.",
    ]
    await connector.disconnect()


@pytest.mark.asyncio
async def test_speech_from_before_the_callers_turn_is_left_out(monkeypatch):
    connector = await _joined(monkeypatch)
    spoken_before = livekit_module._now_ms() - 5000

    connector._note_caller_speech()
    await connector._read_transcript(
        _FinalReader("an earlier reply", spoken_before)
    )
    await connector._read_transcript(
        _FinalReader("The reply to you.", livekit_module._now_ms())
    )

    assert _emitted_transcripts(connector) == ["The reply to you."]
    await connector.disconnect()


@pytest.mark.asyncio
async def test_the_agents_transcript_of_the_caller_starts_a_caller_turn(
    monkeypatch,
):
    connector = await _joined(monkeypatch)
    now = livekit_module._now_ms()
    await connector._read_transcript(_FinalReader("Hello there.", now))

    await connector._read_transcript(
        _FinalReader("Hi, I need help.", now + 2000), heard_us=True
    )
    await connector._read_transcript(_FinalReader("Of course.", now + 4000))

    assert _emitted_transcripts(connector) == ["Hello there.", "Of course."]
    await connector.disconnect()


@pytest.mark.asyncio
async def test_a_new_agent_reply_starts_an_empty_transcript(monkeypatch):
    connector = await _joined(monkeypatch)
    now = livekit_module._now_ms()
    await connector._read_transcript(_FinalReader("Welcome to Acme.", now))

    connector.start_agent_reply()
    await connector._read_transcript(_FinalReader("How can I help?", now + 5))

    assert _emitted_transcripts(connector) == [
        "Welcome to Acme.",
        "How can I help?",
    ]
    assert not connector.transcript_covers_reply
    await connector.disconnect()
