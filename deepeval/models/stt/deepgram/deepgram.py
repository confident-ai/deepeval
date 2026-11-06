from typing import Any, Dict, Optional, Tuple

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.models.speech import (
    DEFAULT_TIMEOUT_SECONDS,
    SpeechTransport,
    dump_request,
    parse_response,
)
from deepeval.models.stt._audio import audio_duration_seconds
from deepeval.models.stt._stream import BufferedTranscribeMixin
from deepeval.models.stt.deepgram.schema import (
    DeepgramListenParams,
    DeepgramListenResponse,
)
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.test_case import Audio

retry_deepgram = create_retry_decorator(PS.DEEPGRAM)

DEFAULT_BASE_URL = "https://api.deepgram.com"
DEFAULT_STT_MODEL = "nova-3"

_STT_PRICE_PER_MINUTE = {
    "nova-3": 0.0043,
    "nova-3-general": 0.0043,
}


class DeepgramSTTModel(BufferedTranscribeMixin, DeepEvalBaseSTT):
    """Deepgram Nova speech-to-text."""

    # Nova is not autoregressive, so it transcribes what it hears and stops
    # rather than completing a word the barge-in cut off. No padding needed.
    truncated_audio_pad_seconds: float = 0.0

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        smart_format: bool = True,
        cost_per_minute: Optional[float] = None,
        transcription_kwargs: Optional[Dict] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_STT_MODEL)
        self.language = language
        self.smart_format = smart_format
        self.transcription_kwargs = transcription_kwargs or {}
        self.cost_per_minute = (
            cost_per_minute
            if cost_per_minute is not None
            else _STT_PRICE_PER_MINUTE.get(self.name)
        )
        self.api_key = (
            SecretStr(api_key)
            if api_key is not None
            else settings.DEEPGRAM_API_KEY
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
            provider_label="Deepgram",
            env_var_name="DEEPGRAM_API_KEY",
            param_hint="`api_key` to DeepgramSTTModel(...)",
        )
        return SpeechTransport(
            provider_label="Deepgram",
            base_url=self.base_url,
            headers={"Authorization": f"Token {api_key}"},
            timeout=self.timeout,
        )

    def _params(
        self, language: Optional[str], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        lang = language or self.language
        auto = lang == "auto"
        return dump_request(
            DeepgramListenParams(
                model=self.name,
                smart_format=self.smart_format,
                # Deepgram detects only when asked to, so "auto" turns
                # detection on instead of omitting the field.
                language=None if auto else lang,
                detect_language=True if auto else None,
                **{**self.transcription_kwargs, **kwargs},
            )
        )

    def _cost(self, audio: Audio) -> Optional[float]:
        if self.cost_per_minute is None:
            return None
        seconds = audio_duration_seconds(audio)
        if seconds is None:
            return None
        return seconds / 60.0 * self.cost_per_minute

    def _parse(self, payload: Any) -> DeepgramListenResponse:
        return parse_response(
            DeepgramListenResponse, payload, provider_label="Deepgram"
        )

    @retry_deepgram
    def transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        payload = self._parse(
            self.model.request_json(
                "POST",
                "/v1/listen",
                params=self._params(language, kwargs),
                content=audio.get_bytes(),
                headers={"Content-Type": audio.mimeType or "audio/wav"},
            )
        )
        return payload.transcript() or "", self._cost(audio)

    @retry_deepgram
    async def a_transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        payload = self._parse(
            await self.model.a_request_json(
                "POST",
                "/v1/listen",
                params=self._params(language, kwargs),
                content=audio.get_bytes(),
                headers={"Content-Type": audio.mimeType or "audio/wav"},
            )
        )
        return payload.transcript() or "", self._cost(audio)

    def get_model_name(self) -> str:
        return self.name
