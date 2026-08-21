from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from deepeval.models.base_model import DeepEvalBaseTTS, DeepEvalBaseSTT
from deepeval.voice.connectors.transports.base import BaseVoiceConnector
from deepeval.voice.output import UNSET, resolve_output_dir

if TYPE_CHECKING:
    from deepeval.dataset.golden import InterruptionBehavior


@dataclass
class VoiceConfig:
    """Voice-mode settings for `ConversationSimulator`.

    Passing a `VoiceConfig` puts the simulator in voice mode: simulated user
    turns are spoken (TTS), sent to the agent over `connector`, and the
    agent's spoken replies are transcribed (STT). `tts_model` / `stt_model`
    default to the OpenAI implementations when left as None.

    `interruption_settings` is deprecated: set
    `Persona(interruption_behavior=...)` on the golden instead. When set here
    it applies to every golden that has no persona-level behavior of its own.
    """

    connector: BaseVoiceConnector
    tts_model: Optional[DeepEvalBaseTTS] = None
    stt_model: Optional[DeepEvalBaseSTT] = None
    # Directory to write per-turn and combined audio files into. Left alone,
    # this resolves through `DEEPEVAL_VOICE_FOLDER` to a default folder; pass
    # `None` to skip writing audio to disk. Resolved once, here, so that by the
    # time anything reads it the answer is a plain path or `None`.
    output_dir: Optional[str] = field(default=UNSET)
    combine_audio_files: bool = True
    interruption_settings: Optional[InterruptionBehavior] = None

    def __post_init__(self) -> None:
        self.output_dir = resolve_output_dir(self.output_dir)


def __getattr__(name: str):
    # Deprecated alias for `InterruptionBehavior`, resolved lazily because that
    # class lives with `Persona` in `deepeval.dataset`, which imports
    # `deepeval.models` and therefore this package.
    if name == "InterruptionSettings":
        from deepeval.dataset.golden import InterruptionBehavior

        return InterruptionBehavior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
