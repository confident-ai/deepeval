"""AssemblyAI request and response shapes.

AssemblyAI has two transcription paths with different request shapes, so both
are modelled: a synchronous endpoint that takes its settings as one
JSON-encoded multipart field, and the upload-create-poll flow that takes a JSON
body.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

# Requests allow extras so `transcription_kwargs` can set a provider parameter
# deepeval has no field for. Responses allow them so a provider adding a field
# never breaks a run.
_REQUEST = ConfigDict(extra="allow", protected_namespaces=())
_RESPONSE = ConfigDict(extra="allow", protected_namespaces=())

# Statuses that end the poll loop.
TERMINAL_STATUSES = ("completed", "error")


class AssemblyAISyncConfig(BaseModel):
    """The `config` multipart field on `POST sync.assemblyai.com/transcribe`.

    Unlike the async flow, this endpoint takes the model in an `X-AAI-Model`
    header rather than in the config, so there is no model field here.
    """

    model_config = _REQUEST

    sample_rate: Optional[int] = None
    language_code: Optional[str] = None


class AssemblyAITranscriptRequest(BaseModel):
    """Body for `POST /v2/transcript`.

    `speech_models` is an ordered fallback list; the older singular
    `speech_model` field is deprecated.
    """

    model_config = _REQUEST

    audio_url: str
    speech_models: List[str]
    language_code: Optional[str] = None
    language_detection: Optional[bool] = None


class AssemblyAIUploadResponse(BaseModel):
    """Response from `POST /v2/upload`."""

    model_config = _RESPONSE

    upload_url: str


class AssemblyAITranscriptResponse(BaseModel):
    """Response from the sync endpoint and from `/v2/transcript` (create + poll).

    `status` is absent on the synchronous endpoint's response, which returns a
    finished transcript directly and has no queue to report on.
    """

    model_config = _RESPONSE

    id: Optional[str] = None
    status: Optional[str] = None
    text: Optional[str] = None
    error: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.status is None or self.status in TERMINAL_STATUSES

    @property
    def failed(self) -> bool:
        return self.status == "error"

    def transcript(self) -> Optional[str]:
        """The spoken text, or `None` when the response carries no `text`.

        `None` and `""` mean different things: `""` is silence, which is a
        perfectly good answer, while `None` means the field was absent.
        """
        return self.text
