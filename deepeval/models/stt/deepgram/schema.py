"""Deepgram speech-to-text request and response shapes.

Deepgram takes its transcription parameters in the query string rather than a
body, so the request side is a params model.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

# Requests allow extras so `transcription_kwargs` can set a provider parameter
# deepeval has no field for. Responses allow them so a provider adding a field
# never breaks a run. `protected_namespaces` is cleared because pydantic
# reserves the `model_` prefix and Deepgram's parameter is called `model`.
_REQUEST = ConfigDict(extra="allow", protected_namespaces=())
_RESPONSE = ConfigDict(extra="allow", protected_namespaces=())


class DeepgramListenParams(BaseModel):
    """Query string for `POST /v1/listen`.

    Deepgram detects the language only when asked to. An absent `language`
    means English rather than detection, so `detect_language` has to be set
    explicitly.
    """

    model_config = _REQUEST

    model: str
    smart_format: bool = True
    language: Optional[str] = None
    detect_language: Optional[bool] = None


class DeepgramAlternative(BaseModel):
    model_config = _RESPONSE

    transcript: Optional[str] = None


class DeepgramChannel(BaseModel):
    model_config = _RESPONSE

    alternatives: List[DeepgramAlternative] = []


class DeepgramResults(BaseModel):
    model_config = _RESPONSE

    channels: List[DeepgramChannel] = []


class DeepgramListenResponse(BaseModel):
    """Response from `POST /v1/listen`."""

    model_config = _RESPONSE

    results: Optional[DeepgramResults] = None

    def transcript(self) -> Optional[str]:
        """The spoken text, or `None` when the response carries no transcript.

        `None` and `""` mean different things: silence comes back as a present
        but empty `transcript`, where `None` means the nesting this reads
        through was not there at all.
        """
        if self.results is None:
            return None
        for channel in self.results.channels:
            for alternative in channel.alternatives:
                if alternative.transcript is not None:
                    return alternative.transcript
        return None
