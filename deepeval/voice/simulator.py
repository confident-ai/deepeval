import os
import logging
from datetime import datetime
from typing import List, Optional

from deepeval.errors import DeepEvalError
from deepeval.simulator.conversation_simulator import ConversationSimulator
from deepeval.simulator.controller.controller import (
    expected_outcome_controller,
)
from deepeval.simulator.simulation_graph import SimulationNode
from deepeval.models import OpenAITTSModel, OpenAISTTModel
from deepeval.models.base_model import DeepEvalBaseTTS, DeepEvalBaseSTT
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.voice.connectors.base_connector import BaseVoiceConnector
from deepeval.voice.connectors import audio_utils

logger = logging.getLogger(__name__)

_MIME_EXT = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/opus": "opus",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/pcm": "pcm",
}


class VoiceConversationSimulator(ConversationSimulator):
    def __init__(
        self,
        connection: BaseVoiceConnector,
        *,
        tts_model: Optional[DeepEvalBaseTTS] = None,
        stt_model: Optional[DeepEvalBaseSTT] = None,
        simulator_model=None,
        simulation_graph: Optional[SimulationNode] = None,
        stopping_controller=expected_outcome_controller,
        language: str = "English",
        output_dir: Optional[str] = "voice_simulations",
        combine_audio: bool = True,
    ):
        self.connection = connection
        self.tts_model = tts_model or OpenAITTSModel()
        self.stt_model = stt_model or OpenAISTTModel()
        self.output_dir = output_dir
        self.combine_audio = combine_audio
        self._audio_buffer: dict = {}
        self._timing_buffer: dict = {}
        self._run_timestamp: Optional[str] = None
        self._num_goldens: Optional[int] = None
        self.tts_cost: float = 0.0
        self.stt_cost: float = 0.0
        super().__init__(
            model_callback=self._voice_model_callback,
            simulation_graph=simulation_graph,
            stopping_controller=stopping_controller,
            simulator_model=simulator_model,
            max_concurrent=1,
            async_mode=True,  # voice is inherently async
            language=language,
        )

    @property
    def total_cost(self) -> float:
        llm_cost = getattr(self, "simulation_cost", None) or 0.0
        return llm_cost + self.tts_cost + self.stt_cost

    def simulate(
        self,
        conversational_goldens,
        max_user_simulations: int = 10,
        on_simulation_complete=None,
    ) -> List[ConversationalTestCase]:
        self._run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._num_goldens = len(conversational_goldens)
        self.tts_cost = 0.0
        self.stt_cost = 0.0
        return super().simulate(
            conversational_goldens=conversational_goldens,
            max_user_simulations=max_user_simulations,
            on_simulation_complete=on_simulation_complete,
        )

    def _simulate_single_conversation(self, *args, **kwargs):
        raise DeepEvalError(
            "VoiceConversationSimulator only supports asynchronous simulation "
            "Please set (async_mode=True)."
        )

    async def _voice_model_callback(
        self, input: str, turns: List[Turn], thread_id: str
    ) -> Turn:
        user_audio, tts_cost = await self.tts_model.a_synthesize(input)
        if tts_cost is not None:
            self.tts_cost += tts_cost
        user_index = len(turns) - 1
        assistant_index = len(turns)

        conn_turn = await self.connection.send_turn(user_audio)
        has_audio = (
            conn_turn.audio is not None
            and len(conn_turn.audio.get_bytes()) > 44
        )
        if conn_turn.transcript:
            agent_text = conn_turn.transcript
        elif has_audio:
            agent_text, stt_cost = await self.stt_model.a_transcribe(
                conn_turn.audio
            )
            if stt_cost is not None:
                self.stt_cost += stt_cost
        else:
            agent_text = ""
            logger.warning(
                "Turn %d produced no agent transcript or audio; the agent may "
                "not be responding (check connection, credentials, and audio "
                "format).",
                assistant_index,
            )

        self._audio_buffer[user_index] = user_audio
        self._audio_buffer[assistant_index] = conn_turn.audio
        # Only latency is persisted. `interrupted` (barge-in) is a real-time
        # duplex signal that a turn-based simulation can't measure honestly —
        # providers fire spurious interruption events when the next user turn
        # arrives while the previous reply is still "playing" server-side. So
        # we leave Turn.interrupted unset (None), which makes InterruptionMetric
        # skip rather than report a misleading result. Revisit for a duplex
        # connector.
        self._timing_buffer[assistant_index] = conn_turn.latency_ms
        return Turn(role="assistant", content=agent_text)

    async def _a_simulate_single_conversation(
        self,
        golden,
        max_user_simulations,
        index=None,
        progress=None,
        pbar_id=None,
        on_simulation_complete=None,
    ) -> ConversationalTestCase:
        self._audio_buffer = {}
        self._timing_buffer = {}
        async with self.connection:
            test_case = await super()._a_simulate_single_conversation(
                golden,
                max_user_simulations,
                index,
                progress,
                pbar_id,
                on_simulation_complete=None,
            )

        for position, turn in enumerate(test_case.turns):
            if position in self._audio_buffer:
                turn.audio = self._audio_buffer[position]
            if position in self._timing_buffer:
                turn.latency_ms = self._timing_buffer[position]
        test_case.voice = any(t.audio is not None for t in test_case.turns)

        if self.output_dir:
            self._save_conversation_audio(test_case, golden, index)
        if on_simulation_complete:
            on_simulation_complete(test_case, index)
        return test_case

    def _save_conversation_audio(
        self, test_case: ConversationalTestCase, golden, index
    ) -> None:
        timestamp = self._run_timestamp or datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        run_label = f"simulation-{timestamp}"
        folder = os.path.join(self.output_dir, run_label)
        if (self._num_goldens or 1) > 1:
            conversation_id = (
                golden.name
                or f"conversation-{index if index is not None else 0}"
            )
            folder = os.path.join(folder, conversation_id)
        os.makedirs(folder, exist_ok=True)

        self._write_turn_files(test_case.turns, folder)
        if self.combine_audio:
            self._write_combined_file(test_case.turns, folder, run_label)

    @staticmethod
    def _write_turn_files(turns: List[Turn], folder: str) -> None:
        turn_number = 0
        for turn in turns:
            if turn.role == "user":
                turn_number += 1
            if turn.audio is None:
                continue
            ext = _MIME_EXT.get(turn.audio.mimeType, "wav")
            filename = f"deepeval-turn-{turn_number}-{turn.role}.{ext}"
            with open(os.path.join(folder, filename), "wb") as f:
                f.write(turn.audio.get_bytes())

    def _write_combined_file(
        self, turns: List[Turn], folder: str, conversation_id: str
    ) -> None:
        combined = self._concat_wav_turns(turns)
        if combined is None:
            logger.warning(
                "Skipping combined audio for %s: turns are not uniform 16-bit "
                "WAV.",
                conversation_id,
            )
            return
        with open(os.path.join(folder, "deepeval-conversation.wav"), "wb") as f:
            f.write(combined)

    @staticmethod
    def _concat_wav_turns(turns: List[Turn]) -> Optional[bytes]:
        pcm_parts: List[bytes] = []
        rate: Optional[int] = None
        channels: Optional[int] = None
        for turn in turns:
            if (
                turn.audio is None
                or _MIME_EXT.get(turn.audio.mimeType) != "wav"
            ):
                return None
            try:
                pcm, turn_rate, turn_channels = audio_utils.wav_bytes_to_pcm16(
                    turn.audio.get_bytes()
                )
            except ValueError:
                return None
            if rate is None:
                rate, channels = turn_rate, turn_channels
            elif (turn_rate, turn_channels) != (rate, channels):
                return None
            pcm_parts.append(pcm)

        if not pcm_parts:
            return None
        return audio_utils.pcm16_to_wav_bytes(
            b"".join(pcm_parts), rate, channels
        )
