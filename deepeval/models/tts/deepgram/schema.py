"""Deepgram text-to-speech request shapes.

Deepgram is the odd one here: the text goes in the body and every other
parameter goes in the query string, which is the reverse of the other providers.
Hence a params model alongside the request body model.
"""

from pydantic import BaseModel, ConfigDict

# Requests allow extras so `generation_kwargs` can set a provider parameter
# deepeval has no field for. `protected_namespaces` is cleared because pydantic
# reserves the `model_` prefix and Deepgram's parameter is called `model`.
_REQUEST = ConfigDict(extra="allow", protected_namespaces=())


class DeepgramSpeakParams(BaseModel):
    """Query string for `POST /v1/speak`.

    `container` is `wav` for a batch call and `none` for a streamed one: a WAV
    header declares a length the first frame of a stream does not know yet.
    """

    model_config = _REQUEST

    model: str
    container: str
    sample_rate: int
    encoding: str = "linear16"


class DeepgramSpeakRequest(BaseModel):
    """Body for `POST /v1/speak`."""

    model_config = _REQUEST

    text: str
