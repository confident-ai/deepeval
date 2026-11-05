"""Why each simulated conversation stopped."""

from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller.controller import end, proceed
from deepeval.simulator.conversation_simulator import (
    AGENT_ENDED_CALL_STOP_REASON,
    STOP_REASON_KEY,
    max_turns_stop_reason,
)
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


def _golden() -> ConversationalGolden:
    return ConversationalGolden(scenario="Refund")


def _stop_reason(simulator: ConversationSimulator) -> str:
    [case] = simulator.simulate([_golden()], max_user_simulations=2)
    return case.additional_metadata[STOP_REASON_KEY]


def _wav_audio() -> Audio:
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(b"\xe8\x03" * 240, sample_rate=24000),
        "audio/wav",
    )


class _Agent:
    async def __call__(self, audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=_wav_audio(), transcript="Agent reply")


class _TTS:
    def supports_streaming(self) -> bool:
        return False

    async def a_synthesize(self, text: str, **kwargs):
        return _wav_audio(), None


class _STT:
    truncated_audio_pad_seconds = 0.0

    def supports_streaming(self) -> bool:
        return False

    async def a_transcribe(self, audio, **kwargs):
        return "Agent reply", None


class _HangsUpConnector(CallbackVoiceConnector):
    """Ends the call after its first reply, for the reason it was given."""

    def __init__(self, reason):
        super().__init__(_Agent())
        self.reason = reason
        self.replied = False

    async def exchange_turn(self, audio):
        turn = await super().exchange_turn(audio)
        self.replied = True
        return turn

    @property
    def call_ended(self) -> bool:
        return self.replied

    @property
    def call_end_reason(self):
        return self.reason


def _voice_simulator(connector) -> ConversationSimulator:
    return ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        voice_config=VoiceConfig(
            connector=connector,
            tts_model=_TTS(),
            stt_model=_STT(),
            output_dir=None,
            combine_audio_files=False,
        ),
    )


def test_a_conversation_that_reaches_the_turn_limit_says_so():
    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        model_callback=async_static_callback,
    )

    assert _stop_reason(simulator) == max_turns_stop_reason(2)


def test_a_controller_that_ends_the_conversation_gives_its_reason():
    def booked(simulated_user_turns):
        return end("Booking confirmed") if simulated_user_turns else proceed()

    simulator = ConversationSimulator(
        simulator_model=StaticSimulatorModel(),
        model_callback=async_static_callback,
        stopping_controller=booked,
    )

    assert _stop_reason(simulator) == "Booking confirmed"


def test_a_call_the_agent_hangs_up_says_the_agent_ended_it():
    simulator = _voice_simulator(_HangsUpConnector(reason=None))

    assert _stop_reason(simulator) == AGENT_ENDED_CALL_STOP_REASON


def test_a_call_the_connector_hangs_up_gives_the_connectors_reason():
    simulator = _voice_simulator(
        _HangsUpConnector(reason="Reached the max call length of 1 min")
    )

    assert _stop_reason(simulator) == "Reached the max call length of 1 min"
