from typing import Any, Dict, Optional, Tuple

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.models.speech import (
    DEFAULT_TIMEOUT_SECONDS,
    Multipart,
    SpeechTransport,
    dump_request,
    parse_response,
)
from deepeval.models.stt._audio import audio_duration_seconds
from deepeval.models.stt._stream import BufferedTranscribeMixin
from deepeval.models.stt.elevenlabs.schema import (
    ElevenLabsSTTRequest,
    ElevenLabsSTTResponse,
)
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.test_case import Audio

retry_elevenlabs = create_retry_decorator(PS.ELEVENLABS)

DEFAULT_BASE_URL = "https://api.elevenlabs.io"

# `scribe_v1` is deprecated in favour of v2.
DEFAULT_STT_MODEL = "scribe_v2"

_STT_PRICE_PER_HOUR = {
    "scribe_v2": 0.22,
    "scribe_v1": 0.22,
}


class ElevenLabsSTTModel(BufferedTranscribeMixin, DeepEvalBaseSTT):
    """ElevenLabs Scribe speech-to-text."""

    # Scribe completes a word clipped mid-syllable, inventing speech the caller
    # never heard, so interrupted audio gets a tail of silence to mark the end.
    truncated_audio_pad_seconds: float = 0.3

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        cost_per_hour: Optional[float] = None,
        transcription_kwargs: Optional[Dict] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_STT_MODEL)
        self.language = language
        self.transcription_kwargs = transcription_kwargs or {}
        self.cost_per_hour = (
            cost_per_hour
            if cost_per_hour is not None
            else _STT_PRICE_PER_HOUR.get(self.name)
        )
        self.api_key = (
            SecretStr(api_key)
            if api_key is not None
            else settings.ELEVENLABS_API_KEY
        )
        self.base_url = (
            str(base_url).rstrip("/")
            if base_url is not None
            else DEFAULT_BASE_URL
        )
        self.timeout = timeout
        self.model = self.load_model()

    def load_model(self) -> SpeechTransport:
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="ElevenLabs",
            env_var_name="ELEVENLABS_API_KEY",
            param_hint="`api_key` to ElevenLabsSTTModel(...)",
        )
        return SpeechTransport(
            provider_label="ElevenLabs",
            base_url=self.base_url,
            headers={"xi-api-key": api_key},
            timeout=self.timeout,
        )

    def _multipart(
        self, audio: Audio, language: Optional[str], kwargs: Dict[str, Any]
    ) -> Multipart:
        lang = language or self.language
        request = ElevenLabsSTTRequest(
            model_id=self.name,
            # Scribe detects the language whenever `language_code` is absent,
            # so "auto" means leaving it out.
            language_code=None if lang == "auto" else lang,
            **{**self.transcription_kwargs, **kwargs},
        )
        return Multipart(
            file_field="file",
            filename=audio.filename or "audio.wav",
            content=audio.get_bytes(),
            content_type=audio.mimeType or "audio/wav",
            fields=dump_request(request),
        )

    def _cost(
        self, audio: Audio, payload: ElevenLabsSTTResponse
    ) -> Optional[float]:
        if self.cost_per_hour is None:
            return None
        seconds = payload.audio_duration_secs
        if seconds is None:
            seconds = audio_duration_seconds(audio)
        if seconds is None:
            return None
        return seconds / 3600.0 * self.cost_per_hour

    def _parse(self, payload: Any) -> ElevenLabsSTTResponse:
        return parse_response(
            ElevenLabsSTTResponse, payload, provider_label="ElevenLabs"
        )

    @retry_elevenlabs
    def transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        payload = self._parse(
            self.model.request_json(
                "POST",
                "/v1/speech-to-text",
                multipart=self._multipart(audio, language, kwargs),
            )
        )
        return payload.transcript() or "", self._cost(audio, payload)

    @retry_elevenlabs
    async def a_transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        payload = self._parse(
            await self.model.a_request_json(
                "POST",
                "/v1/speech-to-text",
                multipart=self._multipart(audio, language, kwargs),
            )
        )
        return payload.transcript() or "", self._cost(audio, payload)

    def get_model_name(self) -> str:
        return self.name
