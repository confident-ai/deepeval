"""ElevenLabs speech-to-text request and response shapes.

`protected_namespaces=()` is cleared because ElevenLabs names its field
`model_id` and pydantic reserves the `model_` prefix for its own methods, so
without it every import warns about a conflict that does not matter here.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

# Requests allow extras so `transcription_kwargs` can set a provider parameter
# deepeval has no field for. Responses allow them so a provider adding a field
# never breaks a run.
_REQUEST = ConfigDict(extra="allow", protected_namespaces=())
_RESPONSE = ConfigDict(extra="allow", protected_namespaces=())


class ElevenLabsSTTRequest(BaseModel):
    """Scalar multipart fields for `POST /v1/speech-to-text`.

    Scribe detects the language whenever `language_code` is absent, so omitting
    it is how automatic detection is requested.
    """

    model_config = _REQUEST

    model_id: str
    language_code: Optional[str] = None


class ElevenLabsTranscriptChannel(BaseModel):
    model_config = _RESPONSE

    text: Optional[str] = None


class ElevenLabsSTTResponse(BaseModel):
    """Response from `POST /v1/speech-to-text`.

    A single-channel request answers with `text`; a multi-channel one answers
    with one entry per channel in `transcripts` and no top-level `text`.
    """

    model_config = _RESPONSE

    text: Optional[str] = None
    audio_duration_secs: Optional[float] = None
    transcripts: Optional[List[ElevenLabsTranscriptChannel]] = None

    def transcript(self) -> Optional[str]:
        """The spoken text, or `None` when the response carries no transcript.

        `None` and `""` mean different things: `""` is silence, which is a
        perfectly good answer, while `None` means neither field was present.
        """
        if self.text is not None:
            return self.text
        if self.transcripts is not None:
            parts = [
                channel.text for channel in self.transcripts if channel.text
            ]
            return " ".join(parts).strip()
        return None
