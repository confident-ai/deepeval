from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.models.speech import (
    DEFAULT_TIMEOUT_SECONDS,
    SpeechTransport,
    dump_request,
)
from deepeval.models.tts._frames import frame_pcm_stream
from deepeval.models.tts.deepgram.schema import (
    DeepgramSpeakParams,
    DeepgramSpeakRequest,
)
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.test_case import Audio, AudioChunk

retry_deepgram = create_retry_decorator(PS.DEEPGRAM)

DEFAULT_BASE_URL = "https://api.deepgram.com"
DEFAULT_TTS_MODEL = "aura-2-thalia-en"
DEFAULT_SAMPLE_RATE = 24000

# Rates Deepgram accepts for linear16.
_PCM_SAMPLE_RATES = frozenset({8000, 16000, 24000, 32000, 48000})

_AURA_2_PRICE_PER_1M_CHARS = 30.0
_AURA_1_PRICE_PER_1M_CHARS = 15.0


class DeepgramTTSModel(DeepEvalBaseTTS):
    """Deepgram Aura text-to-speech.

    Aura model names carry the voice and the language in the identifier
    (`aura-2-thalia-en`), so the `voice` argument rewrites that middle segment
    rather than being sent as a separate field.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        cost_per_1m_chars: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_TTS_MODEL)
        # A voice given here is folded into the model name up front, so
        # `get_model_name()` reports what will actually be synthesized.
        if voice:
            self.name = apply_voice(self.name, voice)
        self.generation_kwargs = generation_kwargs or {}
        self.cost_per_1m_chars = (
            cost_per_1m_chars
            if cost_per_1m_chars is not None
            else _default_price(self.name)
        )

        if sample_rate not in _PCM_SAMPLE_RATES:
            raise ValueError(
                f"Deepgram cannot synthesize linear16 at {sample_rate} Hz. "
                f"Supported rates: {sorted(_PCM_SAMPLE_RATES)}."
            )
        # Instance attribute, not the class default: connectors read
        # `tts.sample_rate` off the instance to configure uplink resampling.
        self.sample_rate = sample_rate

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
            param_hint="`api_key` to DeepgramTTSModel(...)",
        )
        return SpeechTransport(
            provider_label="Deepgram",
            base_url=self.base_url,
            headers={"Authorization": f"Token {api_key}"},
            timeout=self.timeout,
        )

    def _params(
        self, voice: Optional[str], *, container: str, extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        return dump_request(
            DeepgramSpeakParams(
                model=apply_voice(self.name, voice) if voice else self.name,
                container=container,
                sample_rate=self.sample_rate,
                **{**self.generation_kwargs, **extra},
            )
        )

    def _body(self, text: str) -> Dict[str, Any]:
        return dump_request(DeepgramSpeakRequest(text=text))

    def synthesis_cost(self, text: str) -> Optional[float]:
        if self.cost_per_1m_chars is None:
            return None
        return len(text) / 1e6 * self.cost_per_1m_chars

    @retry_deepgram
    def synthesize(
        self, text: str, *, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        data = self.model.request_bytes(
            "POST",
            "/v1/speak",
            params=self._params(voice, container="wav", extra=kwargs),
            json=self._body(text),
        )
        return self._to_audio(data), self.synthesis_cost(text)

    @retry_deepgram
    async def a_synthesize(
        self, text: str, *, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        data = await self.model.a_request_bytes(
            "POST",
            "/v1/speak",
            params=self._params(voice, container="wav", extra=kwargs),
            json=self._body(text),
        )
        return self._to_audio(data), self.synthesis_cost(text)

    def _to_audio(self, data: bytes) -> Audio:
        return Audio.from_bytes(
            data,
            "audio/wav",
            sampleRate=self.sample_rate,
            encoding="wav",
        )

    async def a_synthesize_stream(
        self, text: str, *args, voice: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[AudioChunk, None]:
        source = self.model.a_stream_bytes(
            "POST",
            "/v1/speak",
            # `container=none` is what makes this headerless PCM; a WAV header
            # would arrive declaring a length the first frame cannot know.
            params=self._params(voice, container="none", extra=kwargs),
            json=self._body(text),
        )
        async for chunk in frame_pcm_stream(
            source, sample_rate=self.sample_rate
        ):
            yield chunk

    def supports_streaming(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self.name


def apply_voice(model_name: str, voice: str) -> str:
    """Swap the voice segment of an Aura model name.

    Accepts either a bare voice (`zeus`) or a full model name
    (`aura-2-zeus-en`), since a `Persona.voice` written for another provider is
    just a name while one written for Deepgram may well be the whole thing.
    """
    if voice.startswith("aura"):
        return voice

    parts = model_name.split("-")
    # aura-2-thalia-en -> [aura, 2, thalia, en]; aura-asteria-en -> [aura, asteria, en]
    if len(parts) >= 4:
        return "-".join([parts[0], parts[1], voice, *parts[3:]])
    if len(parts) == 3:
        return "-".join([parts[0], voice, parts[2]])
    return voice


def _default_price(model_name: str) -> Optional[float]:
    if model_name.startswith("aura-2"):
        return _AURA_2_PRICE_PER_1M_CHARS
    if model_name.startswith("aura"):
        return _AURA_1_PRICE_PER_1M_CHARS
    return None
