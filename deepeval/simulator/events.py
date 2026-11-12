"""Voice events: one thing the simulated caller does on one node's turn.

Events that change the caller's audio are applied in voice mode. Every event
is recorded on the turn as `Turn.metadata["events"]`, with where it fell in
the turn's audio, so metrics can check how the agent handled it.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from deepeval.test_case import Audio


@dataclass
class VoiceEvent:
    def record(self) -> Dict[str, Any]:
        fields = {
            key: value
            for key, value in asdict(self).items()
            if value is not None and not isinstance(value, (Audio, dict))
        }
        return {"type": type(self).__name__, **fields}


@dataclass
class Hold(VoiceEvent):
    """The caller goes quiet for `seconds` before saying their line, after
    telling the agent they need a moment."""

    seconds: float = 20.0


@dataclass
class DeadAir(VoiceEvent):
    """The caller says nothing for `seconds` before their line, without
    explaining why."""

    seconds: float = 10.0


@dataclass
class SlowToAnswer(VoiceEvent):
    """Something the caller says before going quiet, such as "Hang on, let
    me find it." Put it before a `Hold` or `DeadAir`."""

    text: Optional[str] = None
    audio: Optional[Audio] = None


@dataclass
class SideSpeech(VoiceEvent):
    """Something the caller says to someone else, quieter, before their
    line."""

    text: Optional[str] = None
    audio: Optional[Audio] = None
    volume: float = 0.5


@dataclass
class BackgroundLine(VoiceEvent):
    """Another person in the room speaking over the caller's line."""

    text: Optional[str] = None
    audio: Optional[Audio] = None
    volume: float = 0.5


@dataclass
class MidSentencePause(VoiceEvent):
    """Silence in the middle of the caller's line.

    Give `at_seconds` into the line, or `after_words`, which places the pause
    by how far through the line's words it falls.
    """

    seconds: float = 2.0
    after_words: Optional[int] = None
    at_seconds: Optional[float] = None


@dataclass
class NonSpeechSound(VoiceEvent):
    """A cough, sneeze or other noise in the middle of the caller's line.
    Needs a recording of the sound."""

    kind: str = "cough"
    audio: Optional[Audio] = None
    after_words: Optional[int] = None
    at_seconds: Optional[float] = None


@dataclass
class SelfCorrects(VoiceEvent):
    """Marks a line in which the caller corrects themselves."""


@dataclass
class TrailsOff(VoiceEvent):
    """Marks a line the caller leaves unfinished."""


@dataclass
class AsksToRepeat(VoiceEvent):
    """Marks a line in which the caller asks the agent to repeat itself."""


@dataclass
class SwitchesLanguage(VoiceEvent):
    """Marks the line from which the caller speaks another language."""

    language: Optional[str] = None


@dataclass
class Prosody(VoiceEvent):
    """Marks a line whose meaning is in its tone, such as sarcasm or a
    statement said as a question."""

    cue: Optional[str] = None


@dataclass
class SpeakerChange(VoiceEvent):
    """Marks a line said by a different person from the caller."""

    voice: Optional[str] = None
