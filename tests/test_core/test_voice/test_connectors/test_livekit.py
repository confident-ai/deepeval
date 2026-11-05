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


def _publication(name, kind=1):
    return SimpleNamespace(name=name, track=SimpleNamespace(kind=kind))


def _participant(identity, *publications):
    return SimpleNamespace(
        identity=identity,
        kind=None,
        attributes={},
        track_publications=dict(enumerate(publications)),
    )


@pytest.mark.asyncio
async def test_the_background_audio_track_is_not_taken_for_the_agent_voice(
    monkeypatch,
):
    install_fake_livekit(monkeypatch, livekit_module)
    connector = _connector()
    await connector._join_room()
    attached = []
    connector._attach_agent_track = lambda track, participant: attached.append(
        track
    )
    agent = _participant("agent")
    background = _publication("background_audio")
    voice = _publication("roomio_audio")

    connector._on_track_subscribed(background.track, background, agent)
    connector._on_track_subscribed(voice.track, voice, agent)

    assert attached == [voice.track]


@pytest.mark.asyncio
async def test_an_agent_already_in_the_room_is_heard_on_its_voice_track(
    monkeypatch,
):
    install_fake_livekit(monkeypatch, livekit_module)
    connector = _connector()
    await connector._join_room()
    attached = []
    connector._attach_agent_track = lambda track, participant: attached.append(
        track
    )
    voice = _publication("roomio_audio")
    connector._room.remote_participants = {
        "agent": _participant("agent", _publication("background_audio"), voice)
    }

    connector._adopt_existing_agent_track()

    assert attached == [voice.track]


async def _state_watching_connector(monkeypatch):
    install_fake_livekit(monkeypatch, livekit_module)
    monkeypatch.setattr(livekit_module, "_TURN_END_SETTLE_S", 0.01)
    connector = _connector()
    await connector._join_room()
    return connector


def _drain(connector):
    events = []
    while not connector._out_frames.empty():
        events.append(connector._out_frames.get_nowait())
    return events


def _turn_ends(connector):
    return [event for event in _drain(connector) if event.turn_complete]


def test_silence_decides_the_turn_until_the_agent_publishes_its_state():
    assert _connector().signals_turn_complete is False


@pytest.mark.asyncio
async def test_the_turn_ends_when_the_agent_goes_back_to_listening(
    monkeypatch,
):
    connector = await _state_watching_connector(monkeypatch)

    for state in ("listening", "speaking", "thinking", "speaking", "listening"):
        connector._on_agent_state(state)
    await asyncio.sleep(0.05)

    assert connector.signals_turn_complete is True
    assert len(_turn_ends(connector)) == 1


@pytest.mark.asyncio
async def test_a_tool_call_pause_does_not_end_the_turn(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("speaking")
    connector._on_agent_state("thinking")
    await asyncio.sleep(0.05)

    assert _turn_ends(connector) == []


@pytest.mark.asyncio
async def test_speaking_again_within_the_settle_window_keeps_the_turn_open(
    monkeypatch,
):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("speaking")
    connector._on_agent_state("listening")
    connector._on_agent_state("speaking")
    await asyncio.sleep(0.05)

    assert _turn_ends(connector) == []


@pytest.mark.asyncio
async def test_thinking_without_speaking_ends_no_turn(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)

    connector._on_agent_state("thinking")
    connector._on_agent_state("listening")
    await asyncio.sleep(0.05)

    assert _turn_ends(connector) == []


@pytest.mark.asyncio
async def test_the_room_reports_agent_state_changes(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)
    agent = _participant("agent")
    on_change = connector._room.handlers["participant_attributes_changed"]

    on_change({"lk.agent.state": "speaking"}, agent)
    on_change({"other": "listening"}, agent)
    on_change({"lk.agent.state": "listening"}, agent)
    await asyncio.sleep(0.05)

    assert len(_turn_ends(connector)) == 1


@pytest.mark.asyncio
async def test_the_state_an_agent_joined_with_is_adopted(monkeypatch):
    connector = await _state_watching_connector(monkeypatch)
    connector._agent_participant = SimpleNamespace(
        attributes={"lk.agent.state": "speaking"}
    )

    connector._adopt_agent_state()
    connector._on_agent_state("listening")
    await asyncio.sleep(0.05)

    assert connector.signals_turn_complete is True
    assert len(_turn_ends(connector)) == 1
