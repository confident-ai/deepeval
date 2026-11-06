"""Cartesia text-to-speech request shapes.

Cartesia is the only provider here that nests its audio settings, in an
`output_format` object rather than flat fields, so that nesting gets a model of
its own.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict

# Requests allow extras so `generation_kwargs` can set a provider parameter
# deepeval has no field for. `protected_namespaces` is cleared because pydantic
# reserves the `model_` prefix and Cartesia's field is called `model_id`.
_REQUEST = ConfigDict(extra="allow", protected_namespaces=())


class CartesiaOutputFormat(BaseModel):
    """The `output_format` object on a TTS request.

    `container` is `wav` for a batch call and `raw` for a streamed one: a WAV
    header declares a length the first frame of a stream does not know yet.
    """

    model_config = _REQUEST

    container: str
    sample_rate: int
    encoding: str = "pcm_s16le"


class CartesiaTTSRequest(BaseModel):
    """Body for `POST /tts/bytes`.

    `voice` is a voice ID from your Cartesia account. There is no default, which
    is why the model requires one rather than falling back.
    """

    model_config = _REQUEST

    model_id: str
    transcript: str
    voice: str
    output_format: CartesiaOutputFormat
    language: Optional[str] = None
