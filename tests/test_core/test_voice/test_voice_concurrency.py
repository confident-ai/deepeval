import asyncio
from dataclasses import dataclass, field
from typing import Set

import pytest

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from tests.test_core.test_simulator.helpers import StaticSimulatorModel
from tests.test_core.test_voice.helpers import EchoAgent, StubSTT, StubTTS

_CONNECT_DELAY_S = 0.05


@dataclass
class _Tracker:
    open: int = 0
    peak: int = 0
    connected_ids: Set[int] = field(default_factory=set)


class _CountingConnector(CallbackVoiceConnector):
    def __init__(self, agent, tracker: _Tracker, **kwargs):
        super().__init__(agent, **kwargs)
        self.tracker = tracker

    async def connect(self) -> None:
        self.tracker.open += 1
        self.tracker.peak = max(self.tracker.peak, self.tracker.open)
        self.tracker.connected_ids.add(id(self))
        await asyncio.sleep(_CONNECT_DELAY_S)
        await super().connect()

    async def disconnect(self) -> None:
        self.tracker.open -= 1
        await super().disconnect()


def _voice_config(connector) -> VoiceConfig:
    return VoiceConfig(
        connector=connector,
        tts_model=StubTTS(),
        stt_model=StubSTT(),
        output_dir=None,
        combine_audio_files=False,
    )


def _goldens(count: int):
    return [
        ConversationalGolden(scenario=f"Refund request {index}")
        for index in range(count)
    ]


def _simulate(connector, *, max_concurrent: int, goldens: int = 3):
    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        max_concurrent=max_concurrent,
        voice_config=_voice_config(connector),
    )
    return simulator, simulator.simulate(
        _goldens(goldens), max_user_simulations=1
    )


def test_each_conversation_gets_its_own_connector():
    tracker = _Tracker()
    template = _CountingConnector(EchoAgent(), tracker)

    _simulate(template, max_concurrent=3)

    assert len(tracker.connected_ids) == 3
    assert id(template) not in tracker.connected_ids
    assert template._events is None


def test_voice_conversations_run_concurrently_up_to_max_concurrent():
    tracker = _Tracker()

    _simulate(_CountingConnector(EchoAgent(), tracker), max_concurrent=3)

    assert tracker.peak == 3
    assert tracker.open == 0


def test_max_concurrent_still_bounds_voice_conversations():
    tracker = _Tracker()

    _simulate(_CountingConnector(EchoAgent(), tracker), max_concurrent=1)

    assert tracker.peak == 1


def test_a_connector_factory_is_called_once_per_conversation():
    tracker = _Tracker()
    built = []

    def factory():
        connector = _CountingConnector(EchoAgent(), tracker)
        built.append(connector)
        return connector

    _simulate(factory, max_concurrent=2)

    assert len(built) == 3
    assert tracker.connected_ids == {id(connector) for connector in built}


def test_call_timing_is_relative_to_each_conversations_own_call():
    tracker = _Tracker()

    _, cases = _simulate(
        _CountingConnector(EchoAgent(), tracker), max_concurrent=3
    )

    for case in cases:
        user_turn, assistant_turn = case.turns[0], case.turns[1]
        assert user_turn.audio is not None and assistant_turn.audio is not None
        assert 0.0 <= user_turn.audio.start_time < _CONNECT_DELAY_S
        assert assistant_turn.audio.start_time >= user_turn.audio.start_time


def test_the_configured_connector_is_exposed_only_when_it_is_an_instance():
    template = CallbackVoiceConnector(EchoAgent())
    by_instance = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=_voice_config(template),
    )
    by_factory = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=_voice_config(lambda: CallbackVoiceConnector(EchoAgent())),
    )

    assert by_instance.voice_connector is template
    assert by_factory.voice_connector is None


def test_voice_simulation_requires_async_mode():
    with pytest.raises(ValueError, match="async_mode=True"):
        ConversationSimulator(
            simulator_model=StaticSimulatorModel(),
            async_mode=False,
            voice_config=_voice_config(CallbackVoiceConnector(EchoAgent())),
        )


def test_voice_config_rejects_connectors_that_are_neither_instance_nor_factory():
    with pytest.raises(TypeError, match="BaseVoiceConnector"):
        _voice_config("not a connector")


def test_voice_config_rejects_a_factory_that_returns_something_else():
    config = _voice_config(lambda: object())

    with pytest.raises(TypeError, match="factory"):
        config.make_connector()
