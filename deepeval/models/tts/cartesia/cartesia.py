from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseTTS
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.models.speech import (
    DEFAULT_TIMEOUT_SECONDS,
    SpeechTransport,
    dump_request,
)
from deepeval.models.tts._frames import frame_pcm_stream
from deepeval.models.tts.cartesia.schema import (
    CartesiaOutputFormat,
    CartesiaTTSRequest,
)
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.test_case import Audio, AudioChunk

retry_cartesia = create_retry_decorator(PS.CARTESIA)

DEFAULT_BASE_URL = "https://api.cartesia.ai"

# Cartesia dates its API and requires the version on every request. Pinning it
# here rather than tracking "latest" means a server-side release cannot change
# the response shape underneath us.
DEFAULT_API_VERSION = "2026-08-14"

# `sonic-2` and `sonic-turbo` are gone from the current model enum and now fail
# with model_not_found, so neither is a safe default.
DEFAULT_TTS_MODEL = "sonic-3.6"

DEFAULT_SAMPLE_RATE = 24000
_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})


class CartesiaTTSModel(DeepEvalBaseTTS):
    """Cartesia Sonic text-to-speech.

    Cartesia has no default voice, so `voice` is required: it is a voice ID from
    your Cartesia account rather than a name from a fixed cast.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        cost_per_1m_chars: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_TTS_MODEL)
        self.voice = voice
        self.language = language
        self.api_version = api_version
        self.generation_kwargs = generation_kwargs or {}
        # Cartesia meters in credits and publishes no credit-to-dollar rate, so
        # there is no price table to default to. Pass `cost_per_1m_chars` to
        # have synthesis costs accounted for.
        self.cost_per_1m_chars = cost_per_1m_chars

        if sample_rate not in _SAMPLE_RATES:
            raise ValueError(
                f"Cartesia cannot synthesize at {sample_rate} Hz. "
                f"Supported rates: {sorted(_SAMPLE_RATES)}."
            )
        # Instance attribute, not the class default: connectors read
        # `tts.sample_rate` off the instance to configure uplink resampling.
        self.sample_rate = sample_rate

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
            param_hint="`api_key` to CartesiaTTSModel(...)",
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

    def _resolve_voice(self, voice: Optional[str]) -> str:
        resolved = voice or self.voice
        if not resolved:
            raise DeepEvalError(
                "Cartesia has no default voice. Pass `voice` to "
                "CartesiaTTSModel(...) with a voice ID from your Cartesia "
                "account, or set it on the golden's `Persona`."
            )
        return resolved

    def _body(
        self,
        text: str,
        voice: Optional[str],
        *,
        container: str,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        return dump_request(
            CartesiaTTSRequest(
                model_id=self.name,
                transcript=text,
                voice=self._resolve_voice(voice),
                output_format=CartesiaOutputFormat(
                    container=container, sample_rate=self.sample_rate
                ),
                language=self.language,
                **{**self.generation_kwargs, **extra},
            )
        )

    def synthesis_cost(self, text: str) -> Optional[float]:
        if self.cost_per_1m_chars is None:
            return None
        return len(text) / 1e6 * self.cost_per_1m_chars

    @retry_cartesia
    def synthesize(
        self, text: str, *, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        data = self.model.request_bytes(
            "POST",
            "/tts/bytes",
            json=self._body(text, voice, container="wav", extra=kwargs),
        )
        return self._to_audio(data), self.synthesis_cost(text)

    @retry_cartesia
    async def a_synthesize(
        self, text: str, *, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        data = await self.model.a_request_bytes(
            "POST",
            "/tts/bytes",
            json=self._body(text, voice, container="wav", extra=kwargs),
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
            "/tts/bytes",
            # A raw container is headerless PCM, which is what makes each frame
            # playable the moment it arrives.
            json=self._body(text, voice, container="raw", extra=kwargs),
        )
        async for chunk in frame_pcm_stream(
            source, sample_rate=self.sample_rate
        ):
            yield chunk

    def supports_streaming(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self.name
