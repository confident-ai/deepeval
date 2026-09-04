"""Cartesia speech-to-text request and response shapes."""

from typing import Optional

from pydantic import BaseModel, ConfigDict

# Requests allow extras so `transcription_kwargs` can set a provider parameter
# deepeval has no field for. Responses allow them so a provider adding a field
# never breaks a run. `protected_namespaces` is cleared because pydantic
# reserves the `model_` prefix and Cartesia's field is called `model`.
_REQUEST = ConfigDict(extra="allow", protected_namespaces=())
_RESPONSE = ConfigDict(extra="allow", protected_namespaces=())


class CartesiaSTTRequest(BaseModel):
    """Scalar multipart fields for `POST /stt`.

    Cartesia defaults to English when `language` is absent rather than
    detecting, so there is no way to ask for detection here.
    """

    model_config = _REQUEST

    model: str
    language: Optional[str] = None


class CartesiaSTTResponse(BaseModel):
    """Response from `POST /stt`."""

    model_config = _RESPONSE

    text: Optional[str] = None
    duration: Optional[float] = None
    language: Optional[str] = None

    def transcript(self) -> Optional[str]:
        """The spoken text, or `None` when the response carries no `text`.

        `None` and `""` mean different things: `""` is silence, which is a
        perfectly good answer, while `None` means the field was absent.
        """
        return self.text
