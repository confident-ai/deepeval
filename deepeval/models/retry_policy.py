"""Generic retry policy helpers for provider SDKs.

This module lets models define *what is transient* vs *non-retryable* (permanent) failure
without coupling to a specific SDK. You provide an `ErrorPolicy` describing
exception classes and special “non-retryable” error codes, such as quota-exhausted from OpenAI,
and get back a Tenacity predicate suitable for `retry_if_exception`.

Typical use:

    # Import dependencies
    from tenacity import retry, before_sleep_log
    from deepeval.models.retry_policy import (
        OPENAI_ERROR_POLICY, default_wait, default_stop, retry_predicate
    )

    # Define retry rule keywords
    _retry_kw = dict(
        wait=default_wait(),
        stop=default_stop(),
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
from typing import Iterable, Mapping, Callable, Sequence, Tuple
from collections.abc import Mapping as ABCMapping
from tenacity import (
    wait_exponential_jitter,
    stop_after_attempt,
    retry_if_exception,
)


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
      1) Structured JSON via `e.response.json()` (typical HTTP error payload).
      2) A dict stored on `e.body` (some gateways/proxies use this).
      3) Message sniffing fallback, using `message_markers`.

    Args:
        e: The exception raised by the SDK/provider client.
        response_attr: Attribute name that holds an HTTP response object.
        body_attr: Attribute name that may hold a parsed payload (dict).
        code_path: Path of keys to traverse to the code (e.g., ["error", "code"]).
        message_markers: Mapping from canonical code -> substrings to search for.

    Returns:
        The code string if found, else "".
    """
    # 1) Structured JSON in e.response.json()
    resp = getattr(e, response_attr, None)
    if resp is not None:
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
            # response.json() can raise; ignore and fall through
            pass

    # 2) SDK provided dict body
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

    # 3) Message sniff (hopefully this helps catch message codes that slip past the previous 2 parsers)
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


def default_wait():
    """Default backoff: exponential with jitter, capped.
    Overridable via env:
      - DEEPEVAL_RETRY_INITIAL_SECONDS (>=0)
      - DEEPEVAL_RETRY_EXP_BASE      (>=1)
      - DEEPEVAL_RETRY_JITTER        (>=0)
      - DEEPEVAL_RETRY_CAP_SECONDS   (>=0)
    """
    initial = read_env_float(
        "DEEPEVAL_RETRY_INITIAL_SECONDS", 1.0, min_value=0.0
    )
    exp_base = read_env_float("DEEPEVAL_RETRY_EXP_BASE", 2.0, min_value=1.0)
    jitter = read_env_float("DEEPEVAL_RETRY_JITTER", 2.0, min_value=0.0)
    cap = read_env_float("DEEPEVAL_RETRY_CAP_SECONDS", 5.0, min_value=0.0)
    return wait_exponential_jitter(
        initial=initial, exp_base=exp_base, jitter=jitter, max=cap
    )


def default_stop():
    """Default stop condition: at most N attempts (N-1 retries).
    Overridable via env:
      - DEEPEVAL_RETRY_MAX_ATTEMPTS (>=1)
    """
    attempts = read_env_int("DEEPEVAL_RETRY_MAX_ATTEMPTS", 2, min_value=1)
    return stop_after_attempt(attempts)


def retry_predicate(policy: ErrorPolicy, **kw):
    """Build a Tenacity `retry=` argument from a policy.

    Example:
        retry=retry_predicate(OPENAI_ERROR_POLICY, extra_non_retryable_codes=["some_code"])
    """
    return retry_if_exception(make_is_transient(policy, **kw))


# --------------------------
# Built-in policies
# --------------------------
OPENAI_MESSAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "insufficient_quota": ("insufficient_quota", "exceeded your current quota"),
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


__all__ = [
    "ErrorPolicy",
    "extract_error_code",
    "make_is_transient",
    "default_wait",
    "default_stop",
    "retry_predicate",
    "OPENAI_MESSAGE_MARKERS",
    "OPENAI_ERROR_POLICY",
]
