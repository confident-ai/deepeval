"""What the simulator does while it waits on itself."""

import asyncio
import time
from typing import List, Optional, Tuple

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Audio
from deepeval.voice import VoiceConfig
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.transports.callback import (
    CallbackVoiceConnector,
)
from deepeval.voice.connectors.types import ConnectorTurn
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_static_callback,
)

_CALL_DELAY_S = 0.2


def _wav_audio() -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(b"\xe8\x03" * 240, sample_rate=24000),
        "audio/wav",
    )


class TimedSimulatorModel(StaticSimulatorModel):
    """Records the window each schema call occupies, to see what overlaps."""

    def __init__(self):
        super().__init__()
        self.windows: List[Tuple[str, float, float]] = []

    async def a_generate(self, prompt: str, schema=None):
        name = schema.__name__ if schema is not None else "text"
        started = time.perf_counter()
        await asyncio.sleep(_CALL_DELAY_S)
        result = self.generate(prompt, schema=schema)
        self.windows.append((name, started, time.perf_counter()))
        return result

    def overlaps(self, one: str, other: str) -> bool:
        for _, a_start, a_end in [w for w in self.windows if w[0] == one]:
            for _, b_start, b_end in [w for w in self.windows if w[0] == other]:
                if a_start < b_end and b_start < a_end:
                    return True
        return False


class _Agent:
    async def __call__(self, audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=_wav_audio(), transcript="Agent reply")


class _TTS:
    async def a_synthesize(self, text: str, **kwargs):
        return _wav_audio(), None


class _STT:
    truncated_audio_pad_seconds = 0.0

    async def a_transcribe(self, audio, **kwargs):
        return "Agent reply", None


def _golden() -> ConversationalGolden:
    # The stopping check only calls the model when there is an outcome to check
    # and a conversation to check it against, so it first runs on turn two.
    return ConversationalGolden(
        scenario="Refund", expected_outcome="The refund is issued."
    )


def test_voice_generates_the_next_turn_while_the_stopping_check_runs():
    """Dead air is not free in voice mode — the agent hears it.

    An agent waiting on the simulated user may fill the silence, re-prompt, or
    hang up, so time the harness spends thinking lands in the recording and
    changes the behavior under test. The stopping check and the next turn read
    the same conversation and neither needs the other's answer.
    """
    model = TimedSimulatorModel()
    simulator = ConversationSimulator(
        simulator_model=model,
        voice_config=VoiceConfig(
            connector=CallbackVoiceConnector(_Agent()),
            tts_model=_TTS(),
            stt_model=_STT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    )

    simulator.simulate([_golden()], max_user_simulations=2)

    assert model.overlaps("ConversationCompletion", "SimulatedInput")


def test_text_mode_still_waits_for_the_stopping_check():
    """Nothing hears the pause in text mode, so speculating only wastes a call."""
    model = TimedSimulatorModel()
    simulator = ConversationSimulator(
        simulator_model=model, model_callback=async_static_callback
    )

    simulator.simulate([_golden()], max_user_simulations=2)

    assert not model.overlaps("ConversationCompletion", "SimulatedInput")
