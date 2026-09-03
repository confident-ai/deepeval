import asyncio
import json as jsonlib
import time
from typing import Any, Dict, Optional, Tuple

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.models.speech import (
    DEFAULT_TIMEOUT_SECONDS,
    Multipart,
    SpeechHTTPError,
    SpeechTransport,
    dump_request,
    parse_response,
)
from deepeval.models.stt._audio import audio_duration_seconds
from deepeval.models.stt._stream import BufferedTranscribeMixin
from deepeval.models.stt.assemblyai.schema import (
    AssemblyAISyncConfig,
    AssemblyAITranscriptRequest,
    AssemblyAITranscriptResponse,
    AssemblyAIUploadResponse,
)
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.test_case import Audio

retry_assemblyai = create_retry_decorator(PS.ASSEMBLYAI)

DEFAULT_BASE_URL = "https://api.assemblyai.com"

# The synchronous endpoint lives on its own host and answers in one round trip.
SYNC_URL = "https://sync.assemblyai.com/transcribe"

# `slam-1` is deprecated in favour of this.
DEFAULT_STT_MODEL = "universal-3-5-pro"

# The synchronous endpoint's ceiling. Conversation turns sit far below it, so it
# is the normal path; anything longer falls back to upload-and-poll.
SYNC_MAX_SECONDS = 120.0

_SYNC_PRICE_PER_HOUR = 0.45
_ASYNC_PRICE_PER_HOUR = {
    "universal-3-5-pro": 0.21,
    "universal-2": 0.15,
}

DEFAULT_POLL_INTERVAL_SECONDS = 1.0


class AssemblyAISTTModel(BufferedTranscribeMixin, DeepEvalBaseSTT):
    """AssemblyAI speech-to-text.

    Prefers the synchronous endpoint, which returns a transcript in a single
    request (~134 ms p50) instead of the three-step upload, create and poll
    flow. That endpoint caps clips at 120 seconds, which no simulated
    conversation turn approaches, so the slower flow is used only when the audio
    really is longer.
    """

    # Universal does not complete a word clipped by a barge-in, so the audio is
    # transcribed exactly as it was heard.
    truncated_audio_pad_seconds: float = 0.0

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[str] = None,
        cost_per_hour: Optional[float] = None,
        transcription_kwargs: Optional[Dict] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        settings = get_settings()
        self.name = parse_model_name(model or DEFAULT_STT_MODEL)
        self.language = language
        self.transcription_kwargs = transcription_kwargs or {}
        self.cost_per_hour = cost_per_hour
        # Only consulted on the upload-and-poll path, which a conversation turn
        # never takes.
        self.poll_interval_seconds = poll_interval_seconds
        self.api_key = (
            SecretStr(api_key)
            if api_key is not None
            else settings.ASSEMBLYAI_API_KEY
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
            provider_label="AssemblyAI",
            env_var_name="ASSEMBLYAI_API_KEY",
            param_hint="`api_key` to AssemblyAISTTModel(...)",
        )
        return SpeechTransport(
            provider_label="AssemblyAI",
            base_url=self.base_url,
            # Note the bare key: AssemblyAI does not use a `Bearer` prefix.
            headers={"Authorization": api_key},
            timeout=self.timeout,
        )

    #
    # Routing
    #

    def _use_sync_endpoint(self, audio: Audio) -> bool:
        seconds = audio_duration_seconds(audio)
        # An unmeasurable duration takes the slower path: it always works,
        # where the fast one would reject audio that turns out to be too long.
        if seconds is None:
            return False
        return seconds <= SYNC_MAX_SECONDS

    def _cost(self, audio: Audio, *, used_sync: bool) -> Optional[float]:
        rate = self.cost_per_hour
        if rate is None:
            rate = (
                _SYNC_PRICE_PER_HOUR
                if used_sync
                else _ASYNC_PRICE_PER_HOUR.get(self.name)
            )
        if rate is None:
            return None
        seconds = audio_duration_seconds(audio)
        if seconds is None:
            return None
        return seconds / 3600.0 * rate

    #
    # Request building
    #

    def _sync_multipart(
        self, audio: Audio, language: Optional[str], kwargs: Dict[str, Any]
    ) -> Multipart:
        lang = language or self.language
        config = AssemblyAISyncConfig(
            sample_rate=audio.sampleRate or None,
            language_code=None if lang == "auto" else lang,
            **{**self.transcription_kwargs, **kwargs},
        )
        dumped = dump_request(config)

        fields: Dict[str, Any] = {}
        if dumped:
            # The whole config rides as one JSON-encoded form field.
            fields["config"] = jsonlib.dumps(dumped)

        return Multipart(
            file_field="audio",
            filename=audio.filename or "audio.wav",
            content=audio.get_bytes(),
            content_type=audio.mimeType or "audio/wav",
            fields=fields,
        )

    def _create_body(
        self,
        upload_url: str,
        language: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        lang = language or self.language
        auto = lang == "auto"
        return dump_request(
            AssemblyAITranscriptRequest(
                audio_url=upload_url,
                speech_models=[self.name],
                language_code=None if auto else lang,
                language_detection=True if auto else None,
                **{**self.transcription_kwargs, **kwargs},
            )
        )

    #
    # Response handling
    #

    def _parse(self, payload: Any) -> AssemblyAITranscriptResponse:
        return parse_response(
            AssemblyAITranscriptResponse,
            payload,
            provider_label="AssemblyAI",
        )

    def _raise_if_failed(self, payload: AssemblyAITranscriptResponse) -> None:
        if payload.failed:
            raise SpeechHTTPError(
                "AssemblyAI failed to transcribe the audio: "
                f"{payload.error or 'no reason given'}",
                provider_label="AssemblyAI",
            )

    def _raise_if_timed_out(self, deadline: float, transcript_id: str) -> None:
        if time.monotonic() >= deadline:
            raise SpeechHTTPError(
                "AssemblyAI did not finish transcribing within "
                f"{self.timeout:g}s (transcript {transcript_id}).",
                provider_label="AssemblyAI",
            )

    #
    # DeepEvalBaseSTT
    #

    @retry_assemblyai
    def transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        used_sync = self._use_sync_endpoint(audio)
        if used_sync:
            payload = self._parse(
                self.model.request_json(
                    "POST",
                    SYNC_URL,
                    multipart=self._sync_multipart(audio, language, kwargs),
                    headers={"X-AAI-Model": self.name},
                )
            )
        else:
            payload = self._transcribe_by_upload(audio, language, kwargs)
        return payload.transcript() or "", self._cost(
            audio, used_sync=used_sync
        )

    def _transcribe_by_upload(
        self, audio: Audio, language: Optional[str], kwargs: Dict[str, Any]
    ) -> AssemblyAITranscriptResponse:
        uploaded = parse_response(
            AssemblyAIUploadResponse,
            self.model.request_json(
                "POST",
                "/v2/upload",
                content=audio.get_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            ),
            provider_label="AssemblyAI",
        )
        payload = self._parse(
            self.model.request_json(
                "POST",
                "/v2/transcript",
                json=self._create_body(uploaded.upload_url, language, kwargs),
            )
        )
        transcript_id = payload.id or ""

        deadline = time.monotonic() + self.timeout
        while not payload.finished:
            self._raise_if_timed_out(deadline, transcript_id)
            time.sleep(self.poll_interval_seconds)
            payload = self._parse(
                self.model.request_json(
                    "GET", f"/v2/transcript/{transcript_id}"
                )
            )
        self._raise_if_failed(payload)
        return payload

    @retry_assemblyai
    async def a_transcribe(
        self, audio: Audio, *, language: Optional[str] = None, **kwargs
    ) -> Tuple[str, Optional[float]]:
        used_sync = self._use_sync_endpoint(audio)
        if used_sync:
            payload = self._parse(
                await self.model.a_request_json(
                    "POST",
                    SYNC_URL,
                    multipart=self._sync_multipart(audio, language, kwargs),
                    headers={"X-AAI-Model": self.name},
                )
            )
        else:
            payload = await self._a_transcribe_by_upload(
                audio, language, kwargs
            )
        return payload.transcript() or "", self._cost(
            audio, used_sync=used_sync
        )

    async def _a_transcribe_by_upload(
        self, audio: Audio, language: Optional[str], kwargs: Dict[str, Any]
    ) -> AssemblyAITranscriptResponse:
        uploaded = parse_response(
            AssemblyAIUploadResponse,
            await self.model.a_request_json(
                "POST",
                "/v2/upload",
                content=audio.get_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            ),
            provider_label="AssemblyAI",
        )
        payload = self._parse(
            await self.model.a_request_json(
                "POST",
                "/v2/transcript",
                json=self._create_body(uploaded.upload_url, language, kwargs),
            )
        )
        transcript_id = payload.id or ""

        deadline = time.monotonic() + self.timeout
        while not payload.finished:
            self._raise_if_timed_out(deadline, transcript_id)
            await asyncio.sleep(self.poll_interval_seconds)
            payload = self._parse(
                await self.model.a_request_json(
                    "GET", f"/v2/transcript/{transcript_id}"
                )
            )
        self._raise_if_failed(payload)
        return payload

    def get_model_name(self) -> str:
        return self.name
