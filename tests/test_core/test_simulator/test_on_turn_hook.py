import asyncio
from typing import List

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Turn
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_static_callback,
    static_callback,
)
from tests.test_core.test_voice.helpers import EchoAgent, StubSTT, StubTTS


def _golden(name: str = "Refund") -> ConversationalGolden:
    return ConversationalGolden(scenario=name, expected_outcome="Done.")


def _roles(snapshots: List[List[Turn]]) -> List[List[str]]:
    return [[turn.role for turn in turns] for turns in snapshots]


def test_on_turn_sees_the_conversation_grow_one_turn_at_a_time():
    snapshots: List[List[Turn]] = []
    indices: List[int] = []

    def on_turn(turns: List[Turn], index: int) -> None:
        snapshots.append(turns)
        indices.append(index)

    ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
    ).simulate([_golden()], max_user_simulations=2, on_turn=on_turn)

    assert _roles(snapshots) == [
        ["user"],
        ["user", "assistant"],
        ["user", "assistant", "user"],
        ["user", "assistant", "user", "assistant"],
    ]
    assert indices == [0, 0, 0, 0]


def test_on_turn_snapshots_are_copies():
    snapshots: List[List[Turn]] = []

    ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
    ).simulate(
        [_golden()],
        max_user_simulations=1,
        on_turn=lambda turns, _: snapshots.append(turns),
    )

    assert len(snapshots[0]) == 1
    assert len(snapshots[-1]) == 2


def test_async_on_turn_hooks_are_awaited():
    seen: List[int] = []

    async def on_turn(turns: List[Turn], index: int) -> None:
        await asyncio.sleep(0)
        seen.append(len(turns))

    ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
    ).simulate([_golden()], max_user_simulations=2, on_turn=on_turn)

    assert seen == [1, 2, 3, 4]


def test_on_turn_reports_the_golden_index_for_concurrent_conversations():
    indices: List[int] = []

    ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
        max_concurrent=2,
    ).simulate(
        [_golden("a"), _golden("b")],
        max_user_simulations=1,
        on_turn=lambda turns, index: indices.append(index),
    )

    assert sorted(indices) == [0, 0, 1, 1]


def test_a_failing_on_turn_hook_does_not_abort_the_conversation():
    def on_turn(turns: List[Turn], index: int) -> None:
        raise RuntimeError("hook exploded")

    cases = ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
    ).simulate([_golden()], max_user_simulations=2, on_turn=on_turn)

    assert [turn.role for turn in cases[0].turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_on_turn_fires_in_synchronous_simulations():
    lengths: List[int] = []

    async def on_turn(turns: List[Turn], index: int) -> None:
        lengths.append(len(turns))

    ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
    ).simulate([_golden()], max_user_simulations=2, on_turn=on_turn)

    assert lengths == [1, 2, 3, 4]


def test_on_turn_fires_for_voice_conversations():
    snapshots: List[List[Turn]] = []

    ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(EchoAgent()),
            tts_model=StubTTS(),
            stt_model=StubSTT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    ).simulate(
        [_golden()],
        max_user_simulations=2,
        on_turn=lambda turns, _: snapshots.append(turns),
    )

    assert _roles(snapshots) == [
        ["user"],
        ["user", "assistant"],
        ["user", "assistant", "user"],
        ["user", "assistant", "user", "assistant"],
    ]
    assert all(turn.audio is not None for turn in snapshots[-1])
