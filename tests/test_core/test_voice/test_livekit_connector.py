import pytest

from deepeval.voice import LiveKitConnector


@pytest.fixture
def livekit_credentials(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")


class _Participant:
    def __init__(self, identity: str):
        self.identity = identity


def test_any_participant_is_the_agent_when_no_identity_is_named(
    livekit_credentials,
):
    connector = LiveKitConnector()

    assert connector._is_agent_participant(_Participant("support-agent"))
    assert connector._is_agent_participant(_Participant("callee"))


def test_only_the_named_participant_is_the_agent(livekit_credentials):
    connector = LiveKitConnector(agent_identity="callee")

    assert connector._is_agent_participant(_Participant("callee"))
    assert not connector._is_agent_participant(_Participant("support-agent"))


@pytest.mark.asyncio
async def test_connect_runs_the_after_join_hook_before_waiting_for_audio(
    livekit_credentials,
):
    connector = LiveKitConnector()
    order = []

    async def join_room():
        order.append("join")

    async def after_join():
        order.append("after_join")

    async def await_agent_track():
        order.append("await_track")

    connector._join_room = join_room
    connector._after_join = after_join
    connector._await_agent_track = await_agent_track

    await connector.connect()

    assert order == ["join", "after_join", "await_track"]


def test_connector_requires_credentials(monkeypatch):
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(Exception, match="LIVEKIT_URL"):
        LiveKitConnector()
