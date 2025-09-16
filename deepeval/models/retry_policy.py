"""Generic retry policy helpers for provider SDKs.

This module lets models define *what is transient* vs *non-retryable* (permanent) failure
without coupling to a specific SDK. You provide an `ErrorPolicy` describing
exception classes and special “non-retryable” error codes, such as quota-exhausted from OpenAI,
and get back a Tenacity predicate suitable for `retry_if_exception`.

Typical use:

    # Import dependencies
    from tenacity import retry, before_sleep_log
    from deepeval.models.retry_policy import (
        OPENAI_ERROR_POLICY, dynamic_wait, dynamic_stop, retry_predicate
    )

    # Define retry rule keywords
    _retry_kw = dict(
        wait=dynamic_wait(),
        stop=dynamic_stop(),
        retry=retry_predicate(OPENAI_ERROR_POLICY),
        before_sleep=before_sleep_log(logger, logging.INFO), # <- Optional: logs only on retries
    )

    # Apply retry rule keywords where desired
    @retry(**_retry_kw)
    def call_openai(...):
        ...
"""

from __future__ import annotations

import logging

from deepeval.utils import read_env_int, read_env_float
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple
from collections.abc import Mapping as ABCMapping
from tenacity import (
    RetryCallState,
    retry,
    wait_exponential_jitter,
    stop_after_attempt,
    retry_if_exception,
    before_sleep_log,
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base

from deepeval.config.settings import get_settings


logger = logging.getLogger(__name__)

# --------------------------
# Policy description
# --------------------------


@dataclass(frozen=True)
class ErrorPolicy:
    """Describe exception classes & rules for retry classification.

    Attributes:
        auth_excs: Exceptions that indicate authentication/authorization problems.
                   These are treated as non-retryable.
        rate_limit_excs: Exceptions representing rate limiting (HTTP 429).
        network_excs: Exceptions for timeouts / connection issues (transient).
        http_excs: Exceptions carrying an integer `status_code` (4xx, 5xx)
        non_retryable_codes: Error “code” strings that should be considered permanent,
                             such as "insufficient_quota". Used to refine rate-limit handling.
        retry_5xx: Whether to retry provider 5xx responses (defaults to True).
    """

    auth_excs: Tuple[type[Exception], ...]
    rate_limit_excs: Tuple[type[Exception], ...]
    network_excs: Tuple[type[Exception], ...]
    http_excs: Tuple[type[Exception], ...]
    non_retryable_codes: frozenset[str] = field(default_factory=frozenset)
    retry_5xx: bool = True
    message_markers: Mapping[str, Iterable[str]] = field(default_factory=dict)


# --------------------------
# Extraction helpers
# --------------------------


def extract_error_code(
    e: Exception,
    *,
    response_attr: str = "response",
    body_attr: str = "body",
    code_path: Sequence[str] = ("error", "code"),
    message_markers: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """Best effort extraction of an error 'code' for SDK compatibility.

    Order of attempts:
      1. Structured JSON via `e.response.json()` (typical HTTP error payload).
      2. A dict stored on `e.body` (some gateways/proxies use this).
      3. Message sniffing fallback, using `message_markers`.

    Args:
        e: The exception raised by the SDK/provider client.
        response_attr: Attribute name that holds an HTTP response object.
        body_attr: Attribute name that may hold a parsed payload (dict).
        code_path: Path of keys to traverse to the code (e.g., ["error", "code"]).
        message_markers: Mapping from canonical code -> substrings to search for.

    Returns:
        The code string if found, else "".
    """
    # 0. gRPC: use e.code() -> grpc.StatusCode
    code_fn = getattr(e, "code", None)
    if callable(code_fn):
        try:
            sc = code_fn()
            name = getattr(sc, "name", None) or str(sc)
            if isinstance(name, str):
                return name.lower()
        except Exception:
            pass

    # 1. Structured JSON in e.response.json()
    resp = getattr(e, response_attr, None)
    if resp is not None:

        if isinstance(resp, ABCMapping):
            # Structured mapping directly on response
            cur = resp
            for k in ("Error", "Code"):  # <- AWS boto style Error / Code
                if not isinstance(cur, ABCMapping):
                    cur = {}
                    break
                cur = cur.get(k, {})
            if isinstance(cur, (str, int)):
                return str(cur)

        else:
            try:
                cur = resp.json()
                for k in code_path:
                    if not isinstance(cur, ABCMapping):
                        cur = {}
                        break
                    cur = cur.get(k, {})
                if isinstance(cur, (str, int)):
                    return str(cur)
            except Exception:
                # if response.json() raises, ignore and fall through
                pass

    # 2. SDK provided dict body
    body = getattr(e, body_attr, None)
    if isinstance(body, ABCMapping):
        cur = body
        for k in code_path:
            if not isinstance(cur, ABCMapping):
                cur = {}
                break
            cur = cur.get(k, {})
        if isinstance(cur, (str, int)):
            return str(cur)

    # 3. Message sniff (hopefully this helps catch message codes that slip past the previous 2 parsers)
    msg = str(e).lower()
    markers = message_markers or {}
    for code_key, needles in markers.items():
        if any(n in msg for n in needles):
            return code_key

    return ""


# --------------------------
# Predicate factory
# --------------------------


def make_is_transient(
    policy: ErrorPolicy,
    *,
    message_markers: Mapping[str, Iterable[str]] | None = None,
    extra_non_retryable_codes: Iterable[str] = (),
) -> Callable[[Exception], bool]:
    """Create a Tenacity predicate: True = retry, False = surface immediately.

    Semantics:
        - Auth errors: non-retryable.
        - Rate limit errors: retry unless the extracted code is in the non-retryable set
        - Network/timeout errors: retry.
        - HTTP errors with a `status_code`: retry 5xx if `policy.retry_5xx` is True.
        - Everything else: treated as non-retryable.

    Args:
        policy: An ErrorPolicy describing error classes and rules.
        message_markers: Optional override/extension for code inference via message text.
        extra_non_retryable_codes: Additional code strings to treat as non-retryable.

    Returns:
        A callable `predicate(e) -> bool` suitable for `retry_if_exception`.
    """
    non_retryable = frozenset(policy.non_retryable_codes) | frozenset(
        extra_non_retryable_codes
    )

    def _pred(e: Exception) -> bool:
        if isinstance(e, policy.auth_excs):
            return False

        if isinstance(e, policy.rate_limit_excs):
            code = extract_error_code(
                e, message_markers=(message_markers or policy.message_markers)
            )
            code = (code or "").lower()
            return code not in non_retryable

        if isinstance(e, policy.network_excs):
            return True

        if isinstance(e, policy.http_excs):
            try:
                sc = int(getattr(e, "status_code", 0))
            except Exception:
                sc = 0
            return policy.retry_5xx and 500 <= sc < 600

        return False

    return _pred


# --------------------------
# Tenacity convenience
# --------------------------


class StopFromEnv(stop_base):
    def __call__(self, retry_state):
        attempts = read_env_int("DEEPEVAL_RETRY_MAX_ATTEMPTS", 2, min_value=1)
        return stop_after_attempt(attempts)(retry_state)


class WaitFromEnv(wait_base):
    def __call__(self, retry_state):
        initial = read_env_float(
            "DEEPEVAL_RETRY_INITIAL_SECONDS", 1.0, min_value=0.0
        )
        exp_base = read_env_float("DEEPEVAL_RETRY_EXP_BASE", 2.0, min_value=1.0)
        jitter = read_env_float("DEEPEVAL_RETRY_JITTER", 2.0, min_value=0.0)
        cap = read_env_float("DEEPEVAL_RETRY_CAP_SECONDS", 5.0, min_value=0.0)
        return wait_exponential_jitter(
            initial=initial, exp_base=exp_base, jitter=jitter, max=cap
        )(retry_state)


def dynamic_stop():
    return StopFromEnv()


def dynamic_wait():
    return WaitFromEnv()


def retry_predicate(policy: ErrorPolicy, **kw):
    """Build a Tenacity `retry=` argument from a policy.

    Example:
        retry=retry_predicate(OPENAI_ERROR_POLICY, extra_non_retryable_codes=["some_code"])
    """
    return retry_if_exception(make_is_transient(policy, **kw))


# --------------------------
# Built-in policies
# --------------------------

##################
# Open AI Policy #
##################

OPENAI_MESSAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "insufficient_quota": (
        "insufficient_quota",
        "insufficient quota",
        "exceeded your current quota",
        "requestquotaexceeded",
    ),
}

try:
    from openai import (
        AuthenticationError,
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
    )

    OPENAI_ERROR_POLICY = ErrorPolicy(
        auth_excs=(AuthenticationError,),
        rate_limit_excs=(RateLimitError,),
        network_excs=(APIConnectionError, APITimeoutError),
        http_excs=(APIStatusError,),
        non_retryable_codes=frozenset({"insufficient_quota"}),
        message_markers=OPENAI_MESSAGE_MARKERS,
    )
except Exception:  # pragma: no cover - OpenAI may not be installed in some envs
    OPENAI_ERROR_POLICY = None


##########################
# Models that use OpenAI #
##########################
AZURE_OPENAI_ERROR_POLICY = OPENAI_ERROR_POLICY
DEEPSEEK_ERROR_POLICY = OPENAI_ERROR_POLICY
KIMI_ERROR_POLICY = OPENAI_ERROR_POLICY
LOCAL_ERROR_POLICY = OPENAI_ERROR_POLICY

######################
# AWS Bedrock Policy #
######################

try:
    from botocore.exceptions import (
        ClientError,
        EndpointConnectionError,
        ConnectTimeoutError,
        ReadTimeoutError,
        ConnectionClosedError,
    )

    # Map common AWS error messages to keys via substring match (lowercased)
    # Update as we encounter new error messages from the sdk
    # These messages are heuristics, we don't have a list of exact error messages
    BEDROCK_MESSAGE_MARKERS = {
        # retryable throttling / transient
        "throttlingexception": (
            "throttlingexception",
            "too many requests",
            "rate exceeded",
        ),
        "serviceunavailableexception": (
            "serviceunavailableexception",
            "service unavailable",
        ),
        "internalserverexception": (
            "internalserverexception",
            "internal server error",
        ),
        "modeltimeoutexception": ("modeltimeoutexception", "model timeout"),
        # clear non-retryables
        "accessdeniedexception": ("accessdeniedexception",),
        "validationexception": ("validationexception",),
        "resourcenotfoundexception": ("resourcenotfoundexception",),
    }

    BEDROCK_ERROR_POLICY = ErrorPolicy(
        auth_excs=(),
        rate_limit_excs=(
            ClientError,
        ),  # classify by code extracted from message
        network_excs=(
            EndpointConnectionError,
            ConnectTimeoutError,
            ReadTimeoutError,
            ConnectionClosedError,
        ),
        http_excs=(),  # no status_code attributes. We will rely on ClientError + markers
        non_retryable_codes=frozenset(
            {
                "accessdeniedexception",
                "validationexception",
                "resourcenotfoundexception",
            }
        ),
        message_markers=BEDROCK_MESSAGE_MARKERS,
    )
except Exception:  # botocore not present (aiobotocore optional)
    BEDROCK_ERROR_POLICY = None


####################
# Anthropic Policy #
####################

try:
    from anthropic import (
        AuthenticationError,
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
    )

    ANTHROPIC_ERROR_POLICY = ErrorPolicy(
        auth_excs=(AuthenticationError,),
        rate_limit_excs=(RateLimitError,),
        network_excs=(APIConnectionError, APITimeoutError),
        http_excs=(APIStatusError,),
        non_retryable_codes=frozenset(),  # update if we learn of hard quota codes
        message_markers={},
    )
except Exception:  # Anthropic optional
    ANTHROPIC_ERROR_POLICY = None


#####################
# Google/Gemini Policy
#####################
# The google genai SDK raises google.genai.errors.*. Public docs and issues show:
# - errors.ClientError for 4xx like 400/401/403/404/422/429
# - errors.ServerError for 5xx
# - errors.APIError is a common base that exposes `.code` and message text
# The SDK doesn’t guarantee a `.status_code` attribute, but it commonly exposes `.code`,
# so we treat ServerError as transient (network-like) to get 5xx retries.
# For rate limiting (429 Resource Exhausted), we treat *ClientError* as rate limit class
# and gate retries using message markers (code sniffing).
# See: https://github.com/googleapis/python-genai?tab=readme-ov-file#error-handling
try:
    from google.genai import errors as gerrors

    try:
        # httpx is an indirect dependency
        import httpx

        _HTTPX_EXCS_NAMES = (
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "TimeoutException",
            "PoolTimeout",
        )
        _HTTPX_EXCS = tuple(
            getattr(httpx, name)
            for name in _HTTPX_EXCS_NAMES
            if hasattr(httpx, name)
        )
    except Exception:
        _HTTPX_EXCS = ()

    GOOGLE_MESSAGE_MARKERS = {
        # retryable rate limit
        "429": ("429", "resource_exhausted", "rate limit"),
        # clearly non-retryable client codes
        "401": ("401", "unauthorized", "api key"),
        "403": ("403", "permission denied", "forbidden"),
        "404": ("404", "not found"),
        "400": ("400", "invalid argument", "bad request"),
        "422": ("422", "failed_precondition", "unprocessable"),
    }

    GOOGLE_ERROR_POLICY = ErrorPolicy(
        auth_excs=(),  # we will classify 401/403 via markers below (see non-retryable codes)
        rate_limit_excs=(
            gerrors.ClientError,
        ),  # includes 429; markers decide retry vs not
        network_excs=(gerrors.ServerError,)
        + _HTTPX_EXCS,  # treat 5xx as transient
        http_excs=(),  # no reliable .status_code on exceptions; handled above
        # Non-retryable codes for *ClientError*. Anything else is retried.
        non_retryable_codes=frozenset({"400", "401", "403", "404", "422"}),
        message_markers=GOOGLE_MESSAGE_MARKERS,
    )
except Exception:
    GOOGLE_ERROR_POLICY = None

#################
# Grok Policy   #
#################
# The xAI Python SDK (xai-sdk) uses gRPC. Errors raised are grpc.RpcError (sync)
# and grpc.aio.AioRpcError (async). The SDK retries UNAVAILABLE by default with
# backoff; you can disable via channel option ("grpc.enable_retries", 0) or
# customize via "grpc.service_config". See xai-sdk docs.
# Refs:
# - https://github.com/xai-org/xai-sdk-python/blob/main/README.md#retries
# - https://github.com/xai-org/xai-sdk-python/blob/main/README.md#error-codes
try:
    import grpc

    try:
        from grpc import aio as grpc_aio

        _AioRpcError = getattr(grpc_aio, "AioRpcError", None)
    except Exception:
        _AioRpcError = None

    _GRPC_EXCS = tuple(
        c for c in (getattr(grpc, "RpcError", None), _AioRpcError) if c
    )

    # rely on extract_error_code reading e.code().name (lowercased).
    GROK_ERROR_POLICY = ErrorPolicy(
        auth_excs=(),  # handled via code() mapping below
        rate_limit_excs=_GRPC_EXCS,  # gated by code() value
        network_excs=(),  # gRPC code handles transience
        http_excs=(),  # no .status_code on gRPC errors
        non_retryable_codes=frozenset(
            {
                "invalid_argument",
                "unauthenticated",
                "permission_denied",
                "not_found",
                "resource_exhausted",
                "failed_precondition",
                "out_of_range",
                "unimplemented",
                "data_loss",
            }
        ),
        message_markers={},
    )
except Exception:  # xai-sdk/grpc not present
    GROK_ERROR_POLICY = None


############
# Lite LLM #
############
LITELLM_ERROR_POLICY = None  # TODO: LiteLLM is going to take some extra care. I will return to this task last


#########################
# Ollama (local server) #
#########################

try:
    import httpx

    # Catch transport + timeout issues via base classes
    _HTTPX_NET_EXCS = tuple(
        exc_cls
        for exc_cls in (
            getattr(httpx, "RequestError", None),
            getattr(httpx, "TimeoutException", None),
        )
        if exc_cls is not None
    )

    OLLAMA_ERROR_POLICY = ErrorPolicy(
        auth_excs=(),
        rate_limit_excs=(),  # no rate limiting semantics locally
        network_excs=_HTTPX_NET_EXCS,  # retry network/timeouts
        http_excs=(),  # optionally add httpx.HTTPStatusError if you call raise_for_status()
        non_retryable_codes=frozenset(),
        message_markers={},
    )
except Exception:
    OLLAMA_ERROR_POLICY = None


###########
# Helpers #
###########
# Convenience helpers

# Map provider slugs to their policy objects. It's OK if some are None
# (optional deps not installed), we'll treat that as "no Tenacity".
_POLICY_BY_SLUG: dict[str, Optional[ErrorPolicy]] = {
    "openai": OPENAI_ERROR_POLICY,
    "azure": AZURE_OPENAI_ERROR_POLICY,
    "bedrock": BEDROCK_ERROR_POLICY,
    "anthropic": ANTHROPIC_ERROR_POLICY,
    "deepseek": DEEPSEEK_ERROR_POLICY,
    "google": GOOGLE_ERROR_POLICY,
    "grok": GROK_ERROR_POLICY,
    "kimi": KIMI_ERROR_POLICY,
    "litellm": LITELLM_ERROR_POLICY,
    "local": LOCAL_ERROR_POLICY,
    "ollama": OLLAMA_ERROR_POLICY,
}


def sdk_retries_for(provider_slug: str) -> bool:
    """True if this provider should delegate retries to the SDK (per settings)."""
    chosen = get_settings().DEEPEVAL_SDK_RETRY_PROVIDERS or []
    slug = provider_slug.lower()
    return "*" in chosen or slug in chosen


def get_retry_policy_for(provider_slug: str) -> Optional[ErrorPolicy]:
    """
    Return the ErrorPolicy for a given provider slug, or None when:
      - the user requested SDK-managed retries for this provider, OR
      - we have no usable policy (optional dependency missing).
    """
    if sdk_retries_for(provider_slug):
        return None
    return _POLICY_BY_SLUG.get(provider_slug.lower()) or None


_STATIC_PRED_BY_SLUG: dict[str, Optional[Callable[[Exception], bool]]] = {
    "openai": (
        make_is_transient(OPENAI_ERROR_POLICY) if OPENAI_ERROR_POLICY else None
    ),
    "azure": (
        make_is_transient(AZURE_OPENAI_ERROR_POLICY)
        if AZURE_OPENAI_ERROR_POLICY
        else None
    ),
    "bedrock": (
        make_is_transient(BEDROCK_ERROR_POLICY)
        if BEDROCK_ERROR_POLICY
        else None
    ),
    "anthropic": (
        make_is_transient(ANTHROPIC_ERROR_POLICY)
        if ANTHROPIC_ERROR_POLICY
        else None
    ),
    "deepseek": (
        make_is_transient(DEEPSEEK_ERROR_POLICY)
        if DEEPSEEK_ERROR_POLICY
        else None
    ),
    "google": (
        make_is_transient(GOOGLE_ERROR_POLICY) if GOOGLE_ERROR_POLICY else None
    ),
    "grok": (
        make_is_transient(GROK_ERROR_POLICY) if GROK_ERROR_POLICY else None
    ),
    "kimi": (
        make_is_transient(KIMI_ERROR_POLICY) if KIMI_ERROR_POLICY else None
    ),
    "litellm": (
        make_is_transient(LITELLM_ERROR_POLICY)
        if LITELLM_ERROR_POLICY
        else None
    ),
    "local": (
        make_is_transient(LOCAL_ERROR_POLICY) if LOCAL_ERROR_POLICY else None
    ),
    "ollama": (
        make_is_transient(OLLAMA_ERROR_POLICY) if OLLAMA_ERROR_POLICY else None
    ),
}


def dynamic_retry(slug: str):
    """
    Tenacity retry= argument that checks settings at *call time*.
    If SDK retries are chosen (or no policy available), it never retries.
    """
    static_pred = _STATIC_PRED_BY_SLUG.get(slug)

    def _pred(e: Exception) -> bool:
        if sdk_retries_for(slug):
            return False  # hand off to SDK
        if static_pred is None:
            return False  # no policy -> no Tenacity retries
        return static_pred(e)  # use prebuilt predicate

    return retry_if_exception(_pred)


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    if exception is None:
        return
    logger.error(
        f"{exception} Retrying: {retry_state.attempt_number} time(s)..."
    )


def create_retry_decorator(provider_slug: str):
    """
    Build a Tenacity @retry decorator wired to our dynamic retry policy
    for the given provider slug.
    """
    slug = provider_slug.lower()
    _logger = logging.getLogger(f"deepeval.retry.{slug}")
    return retry(
        wait=dynamic_wait(),
        stop=dynamic_stop(),
        retry=dynamic_retry(slug),
        before_sleep=before_sleep_log(_logger, logging.INFO),
        after=log_retry_error,
    )


__all__ = [
    "ErrorPolicy",
    "get_retry_policy_for",
    "create_retry_decorator",
    "dynamic_retry",
    "extract_error_code",
    "make_is_transient",
    "dynamic_stop",
    "dynamic_wait",
    "retry_predicate",
    "sdk_retries_for",
    "OPENAI_MESSAGE_MARKERS",
    "OPENAI_ERROR_POLICY",
    "AZURE_OPENAI_ERROR_POLICY",
    "BEDROCK_ERROR_POLICY",
    "BEDROCK_MESSAGE_MARKERS",
    "ANTHROPIC_ERROR_POLICY",
    "DEEPSEEK_ERROR_POLICY",
    "GOOGLE_ERROR_POLICY",
    "GROK_ERROR_POLICY",
    "LOCAL_ERROR_POLICY",
]
