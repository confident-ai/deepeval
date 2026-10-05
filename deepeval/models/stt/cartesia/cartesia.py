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
from deepeval.models.stt.cartesia.schema import (
    CartesiaSTTRequest,
    CartesiaSTTResponse,
)
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.test_case import Audio

retry_cartesia = create_retry_decorator(PS.CARTESIA)

DEFAULT_BASE_URL = "https://api.cartesia.ai"
DEFAULT_API_VERSION = "2026-08-14"

# The batch endpoint only serves the ink-whisper family; `ink-2` is realtime
# only, so it is not a valid default here.
DEFAULT_STT_MODEL = "ink-whisper"


class CartesiaSTTModel(BufferedTranscribeMixin, DeepEvalBaseSTT):
    """Cartesia Ink speech-to-text."""

    # Whisper-derived, so it finishes a word the barge-in cut off unless the
    # clip is presented as a complete utterance.
    truncated_audio_pad_seconds: float = 0.3

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        cost_per_minute: Optional[float] = None,
        transcription_kwargs: Optional[Dict] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_STT_MODEL)
        self.language = language
        self.api_version = api_version
        self.transcription_kwargs = transcription_kwargs or {}
        # Cartesia meters in credits with no published dollar rate, so there is
        # nothing to default to. Pass `cost_per_minute` to account for spend.
        self.cost_per_minute = cost_per_minute
        self.api_key = (
            SecretStr(api_key)
            if api_key is not None
            else settings.CARTESIA_API_KEY
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
            provider_label="Cartesia",
            env_var_name="CARTESIA_API_KEY",
            param_hint="`api_key` to CartesiaSTTModel(...)",
        )
        return SpeechTransport(
            provider_label="Cartesia",
            base_url=self.base_url,
            headers={
                "X-Api-Key": api_key,
                "Cartesia-Version": self.api_version,
            },
            timeout=self.timeout,
        )

    def _multipart(
        self, audio: Audio, language: Optional[str], kwargs: Dict[str, Any]
    ) -> Multipart:
        lang = language or self.language
        request = CartesiaSTTRequest(
            model=self.name,
            # Cartesia defaults to English when `language` is absent rather
            # than detecting, so "auto" has to be dropped instead of forwarded.
            language=None if lang == "auto" else lang,
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
        self, audio: Audio, payload: CartesiaSTTResponse
    ) -> Optional[float]:
        if self.cost_per_minute is None:
            return None
        seconds = payload.duration
        if seconds is None:
            seconds = audio_duration_seconds(audio)
        if seconds is None:
            return None
        return seconds / 60.0 * self.cost_per_minute

    def _parse(self, payload: Any) -> CartesiaSTTResponse:
        return parse_response(
            CartesiaSTTResponse, payload, provider_label="Cartesia"
        )

    @retry_cartesia
    def transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        payload = self._parse(
            self.model.request_json(
                "POST",
                "/stt",
                multipart=self._multipart(audio, language, kwargs),
            )
        )
        return payload.transcript() or "", self._cost(audio, payload)

    @retry_cartesia
    async def a_transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        payload = self._parse(
            await self.model.a_request_json(
                "POST",
                "/stt",
                multipart=self._multipart(audio, language, kwargs),
            )
        )
        return payload.transcript() or "", self._cost(audio, payload)

    def get_model_name(self) -> str:
        return self.name
