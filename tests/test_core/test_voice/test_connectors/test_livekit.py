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
