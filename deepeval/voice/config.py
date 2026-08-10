from dataclasses import dataclass
from typing import Optional

from deepeval.models.base_model import DeepEvalBaseTTS, DeepEvalBaseSTT
from deepeval.voice.connectors.transports.base import BaseVoiceConnector


@dataclass
class VoiceConfig:
    """Voice-mode settings for `ConversationSimulator`.

    Passing a `VoiceConfig` puts the simulator in voice mode: simulated user
    turns are spoken (TTS), sent to the agent over `connector`, and the
    agent's spoken replies are transcribed (STT). `tts_model` / `stt_model`
    default to the OpenAI implementations when left as None.
    """

    connector: BaseVoiceConnector
    tts_model: Optional[DeepEvalBaseTTS] = None
    stt_model: Optional[DeepEvalBaseSTT] = None
    # Directory to write per-turn and combined audio files into.
    # Set to None to skip writing audio to disk.
    output_dir: Optional[str] = "voice_simulations"
    combine_audio: bool = True
