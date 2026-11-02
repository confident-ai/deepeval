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
