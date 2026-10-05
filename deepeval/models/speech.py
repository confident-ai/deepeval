"""Transport and schema plumbing shared by the TTS and STT providers.

The models live in `deepeval.models.tts` and `deepeval.models.stt`, one
subpackage per provider. Every provider except OpenAI is reached over plain HTTP
rather than through a vendor SDK: each is a handful of REST calls, and `requests`
and `aiohttp` are already core dependencies, so `pip install deepeval` is enough
to reach any of them without a per-provider install and version constraint to
track. Rather than have nine models each grow their own session handling and
error translation, that lives here once.

Two HTTP clients are used rather than one: `requests` for the synchronous
`synthesize`/`transcribe` methods and `aiohttp` for their `a_` counterparts.
Using each in its native mode avoids driving a sync client from an event loop
(which blocks it) or bridging an async one into sync code (which needs a loop
that may already be running).
"""

from __future__ import annotations

import json as jsonlib
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import aiohttp
import requests
from pydantic import BaseModel, ValidationError

# Raised from here, defined in `deepeval.errors`, re-exported for callers.
from deepeval.errors import (
    SpeechAuthError,
    SpeechHTTPError,
    SpeechRateLimitError,
)

T = TypeVar("T", bound=BaseModel)

# Long enough that a slow batch transcription of a whole conversation does not
# trip it, short enough that a hung connection does not stall a run forever.
DEFAULT_TIMEOUT_SECONDS = 120.0

# Error bodies go into exception messages, so they are truncated: providers can
# return a wall of JSON, and the useful part is always at the front.
_MAX_ERROR_BODY_CHARS = 500


def _raise_for_status(
    status: int, body: bytes, *, provider_label: str, url: str
) -> None:
    if 200 <= status < 300:
        return

    detail = body.decode("utf-8", errors="replace").strip()
    if len(detail) > _MAX_ERROR_BODY_CHARS:
        detail = detail[:_MAX_ERROR_BODY_CHARS] + "..."
    message = f"{provider_label} request to {url} failed with HTTP {status}"
    if detail:
        message = f"{message}: {detail}"

    if status in (401, 403):
        raise SpeechAuthError(
            f"{message}. Check the API key you configured.",
            status_code=status,
            provider_label=provider_label,
        )
    if status == 429:
        raise SpeechRateLimitError(
            message, status_code=status, provider_label=provider_label
        )
    raise SpeechHTTPError(
        message, status_code=status, provider_label=provider_label
    )


@dataclass
class Multipart:
    """A multipart/form-data body: one audio file plus scalar fields.

    Every speech endpoint that takes multipart follows this exact shape, so the
    two HTTP clients' quite different form APIs can be hidden behind it.
    """

    file_field: str
    filename: str
    content: bytes
    content_type: str
    fields: Dict[str, Any] = field(default_factory=dict)

    def _scalar_fields(self) -> Dict[str, str]:
        # Both clients want strings; booleans have to go over the wire in the
        # JSON spelling ("true"), not Python's ("True").
        out: Dict[str, str] = {}
        for key, value in self.fields.items():
            if value is None:
                continue
            if isinstance(value, bool):
                out[key] = "true" if value else "false"
            elif isinstance(value, (list, tuple, dict)):
                out[key] = jsonlib.dumps(value)
            else:
                out[key] = str(value)
        return out

    def as_requests_kwargs(self) -> Dict[str, Any]:
        return {
            "files": {
                self.file_field: (
                    self.filename,
                    self.content,
                    self.content_type,
                )
            },
            "data": self._scalar_fields(),
        }

    def as_aiohttp_form(self) -> aiohttp.FormData:
        form = aiohttp.FormData()
        for key, value in self._scalar_fields().items():
            form.add_field(key, value)
        form.add_field(
            self.file_field,
            self.content,
            filename=self.filename,
            content_type=self.content_type,
        )
        return form


class SpeechTransport:
    """Issues HTTP requests to one provider, sync or async.

    Each async request opens its own `aiohttp.ClientSession`. Caching one would
    save a handshake per call, but an `aiohttp.ClientSession` is bound to the
    event loop that created it and has to be closed on that same loop — and a
    speech model has no teardown hook to close it from, since `VoiceConfig`
    holds the model for the life of the process. A cached session would
    therefore outlive its loop and be reported as an unclosed session at
    interpreter shutdown. A streaming response still holds one connection for
    its whole body, which is where reuse actually matters here.
    """

    def __init__(
        self,
        *,
        provider_label: str,
        base_url: str,
        headers: Dict[str, str],
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.provider_label = provider_label
        self.base_url = base_url.rstrip("/")
        self.headers = {k: v for k, v in headers.items() if v is not None}
        self.timeout = timeout
        self._sync_session: Optional[requests.Session] = None

    def url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _merged_headers(
        self, extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        headers = dict(self.headers)
        if extra:
            headers.update({k: v for k, v in extra.items() if v is not None})
        return headers

    #
    # Synchronous
    #

    def _requests_session(self) -> requests.Session:
        if self._sync_session is None:
            self._sync_session = requests.Session()
        return self._sync_session

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        multipart: Optional[Multipart] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, bytes]:
        url = self.url(path)
        kwargs: Dict[str, Any] = {
            "params": _clean_params(params),
            "headers": self._merged_headers(headers),
            "timeout": self.timeout,
        }
        if json is not None:
            kwargs["json"] = json
        if content is not None:
            kwargs["data"] = content
        if multipart is not None:
            kwargs.update(multipart.as_requests_kwargs())

        response = self._requests_session().request(method, url, **kwargs)
        body = response.content
        _raise_for_status(
            response.status_code,
            body,
            provider_label=self.provider_label,
            url=url,
        )
        return response.status_code, body

    def request_bytes(self, method: str, path: str, **kwargs) -> bytes:
        _, body = self._request(method, path, **kwargs)
        return body

    def request_json(self, method: str, path: str, **kwargs) -> Any:
        _, body = self._request(method, path, **kwargs)
        return _decode_json(body, provider_label=self.provider_label)

    #
    # Asynchronous
    #

    def _session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

    def _a_request_kwargs(
        self,
        *,
        params: Optional[Dict[str, Any]],
        json: Optional[Dict[str, Any]],
        content: Optional[bytes],
        multipart: Optional[Multipart],
        headers: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "params": _clean_params(params),
            "headers": self._merged_headers(headers),
        }
        if json is not None:
            kwargs["json"] = json
        if content is not None:
            kwargs["data"] = content
        if multipart is not None:
            kwargs["data"] = multipart.as_aiohttp_form()
        return kwargs

    async def _a_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        multipart: Optional[Multipart] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        url = self.url(path)
        kwargs = self._a_request_kwargs(
            params=params,
            json=json,
            content=content,
            multipart=multipart,
            headers=headers,
        )
        async with self._session() as session:
            async with session.request(method, url, **kwargs) as response:
                body = await response.read()
                _raise_for_status(
                    response.status,
                    body,
                    provider_label=self.provider_label,
                    url=url,
                )
                return body

    async def a_request_bytes(self, method: str, path: str, **kwargs) -> bytes:
        return await self._a_request(method, path, **kwargs)

    async def a_request_json(self, method: str, path: str, **kwargs) -> Any:
        body = await self._a_request(method, path, **kwargs)
        return _decode_json(body, provider_label=self.provider_label)

    async def a_stream_bytes(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        multipart: Optional[Multipart] = None,
        headers: Optional[Dict[str, str]] = None,
        chunk_size: int = 4096,
    ) -> AsyncGenerator[bytes, None]:
        """Yield the response body as it arrives.

        An error response is read in full before raising, since the body is the
        only place the provider explains what went wrong.
        """
        url = self.url(path)
        kwargs = self._a_request_kwargs(
            params=params,
            json=json,
            content=content,
            multipart=multipart,
            headers=headers,
        )
        async with self._session() as session:
            async with session.request(method, url, **kwargs) as response:
                if response.status < 200 or response.status >= 300:
                    _raise_for_status(
                        response.status,
                        await response.read(),
                        provider_label=self.provider_label,
                        url=url,
                    )
                async for data in response.content.iter_chunked(chunk_size):
                    if data:
                        yield data

    def close(self) -> None:
        if self._sync_session is not None:
            self._sync_session.close()
            self._sync_session = None


def _clean_params(
    params: Optional[Dict[str, Any]],
) -> Optional[List[Tuple[str, str]]]:
    """Drop unset params and spell values the way query strings expect.

    Returns pairs rather than a dict for two reasons: providers spell a
    repeated param (Deepgram's `keyterm`) as the same key several times, which a
    dict cannot hold, and `aiohttp` rejects a dict whose values are lists while
    accepting pairs. Booleans are normalized here too, since `?smart_format=True`
    is not the `true` these APIs look for.
    """
    if not params:
        return None
    out: List[Tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            out.extend(
                (key, _param_value(item)) for item in value if item is not None
            )
        else:
            out.append((key, _param_value(value)))
    return out or None


def _param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _decode_json(body: bytes, *, provider_label: str) -> Any:
    try:
        return jsonlib.loads(body)
    except ValueError as exc:
        snippet = body.decode("utf-8", errors="replace")[:200]
        raise SpeechHTTPError(
            f"{provider_label} returned a response that is not valid JSON: "
            f"{snippet}",
            provider_label=provider_label,
        ) from exc


#
# Schema
#
# Each provider's `schema.py` declares the request and response shapes deepeval
# relies on. These two helpers are the only places those models cross the wire,
# so serialization rules and validation failures are handled the same way
# everywhere.
#


def dump_request(model: BaseModel) -> Dict[str, Any]:
    """Serialize a request model for the wire.

    `exclude_none` is what lets a schema declare every optional parameter a
    provider accepts without sending the ones nobody set — an explicit `null`
    is not the same as an absent field to these APIs, and several read it as an
    error. Extras survive, which is how `generation_kwargs` reaches a provider
    parameter that has no field of its own yet.
    """
    return model.model_dump(exclude_none=True, by_alias=True)


def parse_response(schema: Type[T], payload: Any, *, provider_label: str) -> T:
    """Validate a provider response, or fail with something readable.

    Response models allow unknown fields, so this only rejects a payload whose
    declared fields are the wrong shape entirely. When that happens the useful
    thing to report is that the provider's response no longer matches what
    deepeval expects — a raw `ValidationError` names pydantic models the reader
    has never heard of and says nothing about which provider broke.
    """
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise SpeechHTTPError(
            f"{provider_label} returned a response deepeval could not read. "
            f"This usually means the provider changed its response format: "
            f"{exc.error_count()} field(s) did not match "
            f"{schema.__name__} ({exc.errors(include_url=False)}).",
            provider_label=provider_label,
        ) from exc
