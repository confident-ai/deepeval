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
from deepeval.models.tts.elevenlabs.schema import (
    ElevenLabsTTSParams,
    ElevenLabsTTSRequest,
)
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.test_case import Audio, AudioChunk

retry_elevenlabs = create_retry_decorator(PS.ELEVENLABS)

DEFAULT_BASE_URL = "https://api.elevenlabs.io"

# Flash is the model built for realtime agents (~75 ms). The Turbo family is
# deprecated in favour of it, and the expressive v3 models cost twice as much
# and are slower, which is the wrong trade for a simulated caller.
DEFAULT_TTS_MODEL = "eleven_flash_v2_5"

# Deliberately not one of the legacy "Default" voices (Rachel, Adam, Bella...):
# those are unavailable to accounts created after March 2026 and stop resolving
# entirely on 2027-01-01. This is George, from the Voice Library.
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"

DEFAULT_SAMPLE_RATE = 24000

# Rates ElevenLabs will emit as PCM. Anything else has to be requested in a
# container format, which cannot be streamed frame by frame.
_PCM_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})

_TTS_PRICE_PER_1M_CHARS = {
    "eleven_flash_v2_5": 50.0,
    "eleven_flash_v2": 50.0,
    "eleven_v3": 100.0,
    "eleven_v3_conversational": 100.0,
    "eleven_multilingual_v2": 100.0,
    "eleven_turbo_v2_5": 50.0,
    "eleven_turbo_v2": 50.0,
}


class ElevenLabsTTSModel(DeepEvalBaseTTS):
    """ElevenLabs text-to-speech.

    Note this is unrelated to `ElevenLabsConnector`, which bridges to a voice
    agent hosted on ElevenLabs. This is the model that speaks the simulated
    user's side of the conversation, and it can be pointed at an agent on any
    platform.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language_code: Optional[str] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        cost_per_1m_chars: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_TTS_MODEL)
        self.voice = voice or DEFAULT_VOICE
        self.language_code = language_code
        self.voice_settings = voice_settings
        self.generation_kwargs = generation_kwargs or {}
        self.cost_per_1m_chars = (
            cost_per_1m_chars
            if cost_per_1m_chars is not None
            else _TTS_PRICE_PER_1M_CHARS.get(self.name)
        )

        if sample_rate not in _PCM_SAMPLE_RATES:
            raise ValueError(
                f"ElevenLabs cannot synthesize at {sample_rate} Hz. "
                f"Supported rates: {sorted(_PCM_SAMPLE_RATES)}."
            )
        # Instance attribute, not the class default: connectors read
        # `tts.sample_rate` off the instance to configure uplink resampling.
        self.sample_rate = sample_rate

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
            param_hint="`api_key` to ElevenLabsTTSModel(...)",
        )
        return SpeechTransport(
            provider_label="ElevenLabs",
            base_url=self.base_url,
            headers={"xi-api-key": api_key},
            timeout=self.timeout,
        )

    def _body(self, text: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        return dump_request(
            ElevenLabsTTSRequest(
                text=text,
                model_id=self.name,
                language_code=self.language_code,
                voice_settings=self.voice_settings,
                **{**self.generation_kwargs, **extra},
            )
        )

    def _params(self, *, stream: bool) -> Dict[str, Any]:
        prefix = "pcm" if stream else "wav"
        return dump_request(
            ElevenLabsTTSParams(output_format=f"{prefix}_{self.sample_rate}")
        )

    def _path(self, voice: Optional[str], *, stream: bool) -> str:
        voice_id = voice or self.voice
        suffix = "/stream" if stream else ""
        return f"/v1/text-to-speech/{voice_id}{suffix}"

    def synthesis_cost(self, text: str) -> Optional[float]:
        if self.cost_per_1m_chars is None:
            return None
        return len(text) / 1e6 * self.cost_per_1m_chars

    @retry_elevenlabs
    def synthesize(
        self, text: str, *, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        data = self.model.request_bytes(
            "POST",
            self._path(voice, stream=False),
            params=self._params(stream=False),
            json=self._body(text, kwargs),
        )
        return self._to_audio(data), self.synthesis_cost(text)

    @retry_elevenlabs
    async def a_synthesize(
        self, text: str, *, voice: Optional[str] = None, **kwargs
    ) -> Tuple[Audio, Optional[float]]:
        data = await self.model.a_request_bytes(
            "POST",
            self._path(voice, stream=False),
            params=self._params(stream=False),
            json=self._body(text, kwargs),
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
            self._path(voice, stream=True),
            params=self._params(stream=True),
            json=self._body(text, kwargs),
        )
        async for chunk in frame_pcm_stream(
            source, sample_rate=self.sample_rate
        ):
            yield chunk

    def supports_streaming(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self.name
