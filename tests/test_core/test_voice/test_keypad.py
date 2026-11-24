import asyncio
import math
from array import array
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from deepeval.dataset.golden import ConversationalGolden, Persona
from deepeval.models import DeepEvalBaseLLM
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.schema import KeypadSimulatedInput
from deepeval.test_case import Audio
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors.providers.livekit.connector import (
    LiveKitConnector,
)
from deepeval.voice.connectors.types import ConnectorTurn
from deepeval.voice.keypad import (
    DTMF_FREQUENCIES,
    GAP_S,
    TONE_S,
    dtmf_tones,
)
from tests.test_core.test_voice.test_voice_simulation import (
    _StubSTT,
    _StubTTS,
    _tone,
)

_RATE = 8000


def _power(samples, freq: float) -> float:
    coefficient = 2 * math.cos(2 * math.pi * freq / _RATE)
    previous, before = 0.0, 0.0
    for sample in samples:
        previous, before = sample + coefficient * previous - before, previous
    return previous**2 + before**2 - coefficient * previous * before


def _detect(pcm: bytes) -> str:
    samples = array("h", pcm)
    window = int(_RATE * TONE_S)
    step = int(_RATE * (TONE_S + GAP_S))
    rows = sorted({low for low, _ in DTMF_FREQUENCIES.values()})
    columns = sorted({high for _, high in DTMF_FREQUENCIES.values()})
    by_pair = {pair: digit for digit, pair in DTMF_FREQUENCIES.items()}
    digits = ""
    for start in range(0, len(samples) - window + 1, step):
        chunk = samples[start : start + window]
        row = max(rows, key=lambda freq: _power(chunk, freq))
        column = max(columns, key=lambda freq: _power(chunk, freq))
        digits += by_pair[(row, column)]
    return digits


def test_tones_carry_every_digit():
    assert _detect(dtmf_tones("0123456789*#", _RATE)) == "0123456789*#"


def test_keypad_input_takes_only_keypad_digits():
    assert (
        KeypadSimulatedInput(simulated_input="", keypad="1234#").keypad
        == "1234#"
    )
    assert KeypadSimulatedInput(simulated_input="Hi").keypad is None
    with pytest.raises(ValidationError):
        KeypadSimulatedInput(simulated_input="", keypad="MP12")


class _KeypadModel(DeepEvalBaseLLM):
    def __init__(self, digits: str):
        self.digits = digits
        self.decisions = 0
        super().__init__(model="keypad-model")

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        name = schema.__name__
        if name == "SimulatedInput":
            return schema(simulated_input="It's one two three four.")
        if name == "KeypadSimulatedInput":
            self.decisions += 1
            return schema(
                simulated_input="It's one two three four.",
                keypad=self.digits if self.decisions == 1 else None,
            )
        if name == "ConversationCompletion":
            return schema(is_complete=False, reason="not done")
        if name == "EdgeChoice":
            return schema(index=None, reason="static")
        raise AssertionError(f"Unexpected schema: {name}")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "keypad-model"


class _SignallingConnector(CallbackVoiceConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent: List[str] = []

    async def send_dtmf(self, digits: str) -> bool:
        self.sent.append(digits)
        return True


def _simulate(connector, *, keypad_input: bool = True, digits: str = "1234#"):
    model = _KeypadModel(digits)
    simulator = ConversationSimulator(
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=_StubTTS(),
            stt_model=_StubSTT(),
            output_dir=None,
            keypad_input=keypad_input,
        ),
        simulator_model=model,
    )
    golden = ConversationalGolden(
        scenario="Enter 1234# on the keypad when asked",
        persona=Persona(characteristics="A caller."),
    )
    case = simulator.simulate([golden], max_user_simulations=2)[0]
    return model, case


def _heard_agent():
    heard: List[Audio] = []

    async def agent(audio: Audio) -> ConnectorTurn:
        heard.append(audio)
        return ConnectorTurn(audio=_tone(0.4), transcript="Please enter it.")

    return heard, agent


def test_a_key_press_is_a_user_turn_and_the_agent_hears_the_tones():
    heard, agent = _heard_agent()
    _, case = _simulate(CallbackVoiceConnector(agent, turn_detection="eager"))

    pressed = case.turns[0]
    assert pressed.role == "user"
    assert pressed.content == "Pressed 1234# on the keypad"
    assert pressed.metadata == {"dtmf": "1234#"}
    assert pressed.audio is not None
    rate = heard[0].sampleRate
    pcm = heard[0].get_bytes()[44:]
    assert any(abs(sample) > 1000 for sample in array("h", pcm))
    assert rate
    assert case.turns[2].content == "It's one two three four."


def test_a_signalling_connector_gets_the_digits_and_silence():
    heard, agent = _heard_agent()
    connector = _SignallingConnector(agent, turn_detection="eager")
    _, case = _simulate(connector)

    assert case.turns[0].metadata == {"dtmf": "1234#"}
    pcm = heard[0].get_bytes()[44:]
    assert all(sample == 0 for sample in array("h", pcm))


def test_no_keypad_decision_without_keypad_input():
    _, agent = _heard_agent()
    model, case = _simulate(
        CallbackVoiceConnector(agent, turn_detection="eager"),
        keypad_input=False,
    )

    assert model.decisions == 0
    assert all(not (turn.metadata or {}).get("dtmf") for turn in case.turns)


def test_livekit_publishes_each_digit(monkeypatch):
    import deepeval.voice.connectors.providers.livekit.connector as livekit

    monkeypatch.setattr(livekit, "_DTMF_DIGIT_INTERVAL_S", 0)
    connector = LiveKitConnector.__new__(LiveKitConnector)
    participant = MagicMock(publish_dtmf=AsyncMock())
    connector._room = MagicMock(local_participant=participant)

    assert asyncio.run(connector.send_dtmf("9*#"))
    assert [
        call.kwargs for call in participant.publish_dtmf.call_args_list
    ] == [
        {"code": 9, "digit": "9"},
        {"code": 10, "digit": "*"},
        {"code": 11, "digit": "#"},
    ]
