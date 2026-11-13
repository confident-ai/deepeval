import asyncio

import pytest

from deepeval.voice import SilenceTimeoutMixin
from deepeval.voice.connectors.transports import silence
from deepeval.voice.connectors.types import AgentEvent

QUIET = b"\x00\x00" * 160
VOICED = b"\xe8\x03" * 160


class _Connector:
    silence_threshold_rms = 100.0

    def __init__(self, frames=()):
        self.frames = list(frames)
        self.disconnects = 0

    async def connect(self):
        return None

    async def disconnect(self):
        self.disconnects += 1

    @property
    def call_ended(self):
        return False

    async def iter_agent_events(self):
        for frame in self.frames:
            await asyncio.sleep(0.02)
            yield AgentEvent(audio=frame)


class _SilenceConnector(SilenceTimeoutMixin, _Connector):
    pass


@pytest.fixture(autouse=True)
def fast_checks(monkeypatch):
    monkeypatch.setattr(silence, "SILENCE_CHECK_INTERVAL_S", 0.01)


@pytest.mark.asyncio
async def test_a_silent_call_is_hung_up():
    connector = _SilenceConnector(silence_timeout_s=0.05)

    await connector.connect()
    await asyncio.sleep(0.2)

    assert connector.call_ended is True
    assert connector.disconnects == 1


@pytest.mark.asyncio
async def test_no_timeout_never_hangs_up():
    connector = _SilenceConnector()

    await connector.connect()
    await asyncio.sleep(0.05)

    assert connector.call_ended is False
    assert connector.disconnects == 0


@pytest.mark.asyncio
async def test_agent_speech_keeps_the_call_open():
    connector = _SilenceConnector([VOICED] * 10, silence_timeout_s=0.08)

    await connector.connect()
    async for _ in connector.iter_agent_events():
        pass

    assert connector.call_ended is False
    await connector.disconnect()


@pytest.mark.asyncio
async def test_quiet_agent_audio_does_not_keep_the_call_open():
    connector = _SilenceConnector([QUIET] * 10, silence_timeout_s=0.08)

    await connector.connect()
    async for _ in connector.iter_agent_events():
        pass

    assert connector.call_ended is True
