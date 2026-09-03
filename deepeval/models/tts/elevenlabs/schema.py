"""ElevenLabs text-to-speech request shapes.

`protected_namespaces=()` is cleared because ElevenLabs names its field
`model_id` and pydantic reserves the `model_` prefix for its own methods, so
without it every import warns about a conflict that does not matter here.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

# Requests allow extras so `generation_kwargs` can set a provider parameter
# deepeval has no field for.
_REQUEST = ConfigDict(extra="allow", protected_namespaces=())


class ElevenLabsTTSParams(BaseModel):
    """Query string for `POST /v1/text-to-speech/{voice_id}`.

    `output_format` is `wav_{rate}` for a batch call and `pcm_{rate}` for a
    streamed one: a WAV header declares a length the first frame of a stream
    does not know yet.
    """

    model_config = _REQUEST

    output_format: str


class ElevenLabsTTSRequest(BaseModel):
    """Body for `POST /v1/text-to-speech/{voice_id}`."""

    model_config = _REQUEST

    text: str
    model_id: str
    language_code: Optional[str] = None
    voice_settings: Optional[Dict[str, Any]] = None
