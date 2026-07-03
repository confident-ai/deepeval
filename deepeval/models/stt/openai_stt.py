import io
import wave
from typing import Optional, Dict, Tuple
from pydantic import SecretStr
from openai import OpenAI, AsyncOpenAI

from deepeval.test_case.audio import Audio
from deepeval.models import DeepEvalBaseSTT
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.constants import ProviderSlug as PS
from deepeval.config.settings import get_settings

retry_openai = create_retry_decorator(PS.OPENAI)

DEFAULT_STT_MODEL = "gpt-4o-transcribe"

_STT_TOKEN_PRICE = {
    "gpt-4o-transcribe": (6.0, 10.0),
    "gpt-4o-mini-transcribe": (3.0, 5.0),
}

_STT_PRICE_PER_MINUTE = {
    "whisper-1": 0.006,
}


class OpenAISTT(DeepEvalBaseSTT):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        cost_per_1m_input_tokens: Optional[float] = None,
        cost_per_1m_output_tokens: Optional[float] = None,
        cost_per_minute: Optional[float] = None,
        transcription_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_STT_MODEL)
        _in, _out = _STT_TOKEN_PRICE.get(self.name, (None, None))
        self.cost_per_1m_input_tokens = (
            cost_per_1m_input_tokens
            if cost_per_1m_input_tokens is not None
            else _in
        )
        self.cost_per_1m_output_tokens = (
            cost_per_1m_output_tokens
            if cost_per_1m_output_tokens is not None
            else _out
        )
        self.cost_per_minute = (
            cost_per_minute
            if cost_per_minute is not None
            else _STT_PRICE_PER_MINUTE.get(self.name)
        )
        self.api_key = (
            SecretStr(api_key)
            if api_key is not None
            else settings.OPENAI_API_KEY
        )
        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else None
        )
        self.language = language
        self.transcription_kwargs = transcription_kwargs or {}
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
            param_hint="`api_key` to OpenAISTT(...)",
        )
        return cls(api_key=api_key, base_url=self.base_url, **self.kwargs)

    def _async(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = self._build_client(AsyncOpenAI)
        return self._async_client

    def _file_tuple(self, audio: Audio):
        data = audio.get_bytes()
        filename = audio.filename or "audio.wav"
        return (filename, data, audio.mimeType or "audio/wav")

    def _request_kwargs(self, language: Optional[str], kwargs: Dict) -> Dict:
        merged = {**self.transcription_kwargs, **kwargs}
        lang = language or self.language
        if lang is not None and "language" not in merged:
            merged["language"] = lang
        return merged

    @staticmethod
    def _audio_duration_seconds(audio: Audio) -> Optional[float]:
        if audio.duration is not None:
            return audio.duration
        try:
            with wave.open(io.BytesIO(audio.get_bytes()), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate:
                    return frames / float(rate)
        except (wave.Error, EOFError, ValueError):
            pass
        return None

    def _calculate_cost(self, audio: Audio, response) -> Optional[float]:
        usage = getattr(response, "usage", None)
        usage_type = getattr(usage, "type", None)

        if usage_type == "tokens":
            in_p = self.cost_per_1m_input_tokens
            out_p = self.cost_per_1m_output_tokens
            if in_p is None and out_p is None:
                return None
            cost = 0.0
            if in_p is not None:
                cost += getattr(usage, "input_tokens", 0) / 1e6 * in_p
            if out_p is not None:
                cost += getattr(usage, "output_tokens", 0) / 1e6 * out_p
            return cost

        if self.cost_per_minute is not None:
            seconds = (
                getattr(usage, "seconds", None)
                if usage_type == "duration"
                else None
            )
            if seconds is None:
                seconds = self._audio_duration_seconds(audio)
            if seconds is not None:
                return seconds / 60.0 * self.cost_per_minute

        return None

    @retry_openai
    def transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        response = self.model.audio.transcriptions.create(
            model=self.name,
            file=self._file_tuple(audio),
            **self._request_kwargs(language, kwargs),
        )
        return response.text, self._calculate_cost(audio, response)

    @retry_openai
    async def a_transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        response = await self._async().audio.transcriptions.create(
            model=self.name,
            file=self._file_tuple(audio),
            **self._request_kwargs(language, kwargs),
        )
        return response.text, self._calculate_cost(audio, response)

    def get_model_name(self) -> str:
        return self.name
