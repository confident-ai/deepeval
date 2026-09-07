from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from deepeval.tracing.integrations import Provider


@dataclass(frozen=True)
class Gateway:
    """How to label and mine a response from one recognized gateway.

    `metadata_key` namespaces the gateway's extras on the span so they can never
    collide with metadata the user set via `update_current_span(metadata=...)`.
    """
    provider: str
    metadata_key: str
    extract_metadata: Callable[[Any], Optional[Dict[str, Any]]]


def _model_extra(obj: Any) -> Dict[str, Any]:
    extra = getattr(obj, "model_extra", None)
    return extra if isinstance(extra, dict) else {}


def _get(obj: Any, name: str) -> Any:
    """Read a field whether it's declared on the model or arrived as an extra."""
    value = getattr(obj, name, None)
    if value is None:
        value = _model_extra(obj).get(name)
    # The openrouter SDK marks absent nullable fields with an UNSET sentinel
    # that is not None and stringifies to "Unset".
    if value is not None and type(value).__name__ == "Unset":
        return None
    return value


def _dump(value: Any) -> Any:
    """Best-effort plain-data rendering, so span metadata stays JSON-safe."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return str(value)


def extract_openrouter_metadata(response: Any) -> Optional[Dict[str, Any]]:
    """Pull OpenRouter's non-standard response fields into a metadata dict.

    Captures what OpenRouter knows that a plain OpenAI response does not: which
    upstream provider served the request, what it cost, and how the router got
    there. Returns None when nothing OpenRouter-specific is present.
    """
    try:
        metadata: Dict[str, Any] = {}

        generation_id = _get(response, "id")
        if generation_id:
            metadata["generation_id"] = generation_id

        upstream = _get(response, "provider")
        if upstream:
            metadata["upstream_provider"] = upstream

        usage = _get(response, "usage")
        if usage is not None:
            cost = _get(usage, "cost")
            if cost is not None:
                metadata["cost"] = cost

            cost_details = _get(usage, "cost_details")
            if cost_details is not None:
                metadata["cost_details"] = _dump(cost_details)

            is_byok = _get(usage, "is_byok")
            if is_byok is not None:
                metadata["is_byok"] = is_byok

            # Chat Completions calls these prompt/completion; the Responses API
            # calls the same things input/output.
            prompt_details = _get(usage, "prompt_tokens_details") or _get(
                usage, "input_tokens_details"
            )
            if prompt_details is not None:
                for field in ("cached_tokens", "cache_write_tokens"):
                    value = _get(prompt_details, field)
                    if value:
                        metadata[field] = value

            completion_details = _get(
                usage, "completion_tokens_details"
            ) or _get(usage, "output_tokens_details")
            if completion_details is not None:
                reasoning_tokens = _get(completion_details, "reasoning_tokens")
                if reasoning_tokens:
                    metadata["reasoning_tokens"] = reasoning_tokens

        # Routing detail, present only on the native SDK's ChatResult.
        router = _get(response, "openrouter_metadata")
        if router is not None:
            routing: Dict[str, Any] = {}
            for field in ("strategy", "summary", "attempt", "region"):
                value = _get(router, field)
                if value is not None:
                    routing[field] = _dump(value)
            if routing:
                metadata["routing"] = routing
            # `is_byok` also lives here; only fall back to it if usage lacked one.
            if "is_byok" not in metadata:
                is_byok = _get(router, "is_byok")
                if is_byok is not None:
                    metadata["is_byok"] = is_byok

        return metadata or None
    except Exception:
        # Metadata is strictly additive — never let it break a traced call.
        return None


OPENROUTER = Gateway(
    provider=Provider.OPEN_ROUTER.value,
    metadata_key="openrouter",
    extract_metadata=extract_openrouter_metadata,
)

_GATEWAYS_BY_HOST: Dict[str, Gateway] = {
    "openrouter.ai": OPENROUTER,
}


def detect_gateway(base_url: Any) -> Optional[Gateway]:
    if not base_url:
        return None
    try:
        host = getattr(base_url, "host", None)
        if not host:
            host = str(base_url).split("//")[-1].split("/")[0].split(":")[0]
        host = host.lower()
    except Exception:
        return None

    for known_host, gateway in _GATEWAYS_BY_HOST.items():
        # Suffix match so regional/vanity subdomains resolve too.
        if host == known_host or host.endswith("." + known_host):
            return gateway
    return None
