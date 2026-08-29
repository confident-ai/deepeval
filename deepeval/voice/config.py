from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING, Union

from deepeval.models.base_model import DeepEvalBaseTTS, DeepEvalBaseSTT
from deepeval.voice.connectors.utils import validate_connector
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

    `connector` is the connector the session runs over — `ElevenLabsConnector`,
    `LiveKitConnector`, `WebSocketConnector`, `CallbackVoiceConnector`, or your
    own `BaseVoiceConnector` subclass. Concurrent conversations each get their
    own session, cloned from the one passed here; pass a zero-argument callable
    instead when a connector cannot be copied, such as one holding a LiveKit
    room you connected yourself.

    `interruption_settings` is deprecated: set
    `Persona(interruption_behavior=...)` on the golden instead. When set here
    it applies to every golden that has no persona-level behavior of its own.
    """

    connector: Union[BaseVoiceConnector, Callable[[], BaseVoiceConnector]]
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
        # A factory can only be checked by calling it, which belongs in
        # `make_connector`. Anything else is checked now, while the traceback
        # still points at the user's `VoiceConfig(...)` — including the native
        # provider objects `validate_connector` redirects by name.
        if not callable(self.connector) or isinstance(
            self.connector, BaseVoiceConnector
        ):
            self.connector = validate_connector(self.connector)
        self.output_dir = resolve_output_dir(self.output_dir)

    def make_connector(self) -> BaseVoiceConnector:
        """The connector for one conversation.

        Every conversation gets its own, so that concurrent calls do not share
        a session.
        """
        if isinstance(self.connector, BaseVoiceConnector):
            return self.connector.clone()
        return validate_connector(self.connector())


def __getattr__(name: str):
    # Deprecated alias for `InterruptionBehavior`, resolved lazily because that
    # class lives with `Persona` in `deepeval.dataset`, which imports
    # `deepeval.models` and therefore this package.
    if name == "InterruptionSettings":
        from deepeval.dataset.golden import InterruptionBehavior

        return InterruptionBehavior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
