from typing import Optional, Dict, Tuple
from pydantic import SecretStr
from openai import OpenAI, AsyncOpenAI

from deepeval.test_case import Audio
from deepeval.models import DeepEvalBaseTTS
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.constants import ProviderSlug as PS
from deepeval.config.settings import get_settings

retry_openai = create_retry_decorator(PS.OPENAI)

DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"

_OPENAI_TTS_SAMPLE_RATE = 24000
_FORMAT_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}
_TTS_PRICE_PER_1M_CHARS = {
    "tts-1": 15.0,
    "tts-1-hd": 30.0,
    "gpt-4o-mini-tts": 12.0,
}


class OpenAITTSModel(DeepEvalBaseTTS):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: Optional[str] = None,
        response_format: str = "wav",
        cost_per_1m_chars: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_TTS_MODEL)
        self.cost_per_1m_chars = (
            cost_per_1m_chars
            if cost_per_1m_chars is not None
            else _TTS_PRICE_PER_1M_CHARS.get(self.name)
        )
        self.api_key = (
            SecretStr(api_key)
            if api_key is not None
            else settings.OPENAI_API_KEY
        )
        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else None
        )
        self.voice = voice or DEFAULT_VOICE
        self.response_format = response_format
        self.generation_kwargs = generation_kwargs or {}
        self.kwargs = kwargs
        self._async_client: Optional[AsyncOpenAI] = None
        self.model = self.load_model()

    def load_model(self) -> OpenAI:
        return self._build_client(OpenAI)

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenAI",
            env_var_name="OPENAI_API_KEY",
            param_hint="`api_key` to OpenAITTSModel(...)",
        )
        return cls(api_key=api_key, base_url=self.base_url, **self.kwargs)

    def _async(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = self._build_client(AsyncOpenAI)
        return self._async_client

    def _to_audio(self, data: bytes, fmt: str) -> Audio:
        return Audio.from_bytes(
            data,
            _FORMAT_MIME.get(fmt, "audio/wav"),
            sampleRate=_OPENAI_TTS_SAMPLE_RATE,
            encoding=fmt,
        )

    def _calculate_cost(self, text: str) -> Optional[float]:
        if self.cost_per_1m_chars is None:
            return None
        return len(text) / 1e6 * self.cost_per_1m_chars

    @retry_openai
    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        response_format: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Audio, Optional[float]]:
        fmt = response_format or self.response_format
        response = self.model.audio.speech.create(
            model=self.name,
            voice=voice or self.voice,
            input=text,
            response_format=fmt,
            **{**self.generation_kwargs, **kwargs},
        )
        return self._to_audio(response.content, fmt), self._calculate_cost(text)

    @retry_openai
    async def a_synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        response_format: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Audio, Optional[float]]:
        fmt = response_format or self.response_format
        response = await self._async().audio.speech.create(
            model=self.name,
            voice=voice or self.voice,
            input=text,
            response_format=fmt,
            **{**self.generation_kwargs, **kwargs},
        )
        return self._to_audio(response.content, fmt), self._calculate_cost(text)

    def get_model_name(self) -> str:
        return self.name
