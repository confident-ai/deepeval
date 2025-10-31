"""Generic retry policy helpers for provider SDKs.

This module lets models define *what is transient* vs *non-retryable* (permanent) failure
without coupling to a specific SDK. You provide an `ErrorPolicy` describing exception classes
and special “non-retryable” error codes (quota-exhausted), and get back Tenacity components:
a predicate suitable for `retry_if_exception`, plus convenience helpers for wait/stop/backoff.
You can also use `create_retry_decorator(slug)` to wire Tenacity with dynamic policy + logging.

Notes:
- `extract_error_code` best-effort parses codes from response JSON, `e.body`, botocore-style maps,
  gRPC `e.code().name`, or message markers.
- `dynamic_retry(slug)` consults settings at call time: if SDK retries are enabled for the slug,
  Tenacity will not retry.
- Logging callbacks (`before_sleep`, `after`) read log levels dynamically and log to
  the `deepeval.retry.<slug>` logger.

Configuration
-------------
Retry backoff (env):
  DEEPEVAL_RETRY_MAX_ATTEMPTS       int   (default 2, >=1)
  DEEPEVAL_RETRY_INITIAL_SECONDS    float (default 1.0, >=0)
  DEEPEVAL_RETRY_EXP_BASE           float (default 2.0, >=1)
  DEEPEVAL_RETRY_JITTER             float (default 2.0, >=0)
  DEEPEVAL_RETRY_CAP_SECONDS        float (default 5.0, >=0)

SDK-managed retries (settings):
  settings.DEEPEVAL_SDK_RETRY_PROVIDERS  list[str]  # e.g. ["azure"] or ["*"] for all

Retry logging (settings; read at call time):
  settings.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL  int/name  (default INFO)
  settings.DEEPEVAL_RETRY_AFTER_LOG_LEVEL   int/name  (default ERROR)
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import functools
import threading
import logging
import time

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union
from collections.abc import Mapping as ABCMapping
from tenacity import (
    RetryCallState,
    retry,
    wait_exponential_jitter,
    stop_after_attempt,
    retry_if_exception,
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from contextvars import ContextVar, copy_context

from deepeval.constants import (
    ProviderSlug as PS,
    slugify,
)
from deepeval.config.settings import get_settings


logger = logging.getLogger(__name__)
Provider = Union[str, PS]
_MAX_TIMEOUT_THREADS = get_settings().DEEPEVAL_TIMEOUT_THREAD_LIMIT
_TIMEOUT_SEMA = threading.BoundedSemaphore(_MAX_TIMEOUT_THREADS)
_WORKER_ID = itertools.count(1)
_OUTER_DEADLINE = ContextVar("deepeval_outer_deadline", default=None)


def set_outer_deadline(seconds: float | None):
    """Set (or clear) the outer task time budget.

    Stores a deadline in a local context variable so nested code
    can cooperatively respect a shared budget. Always pair this with
    `reset_outer_deadline(token)` in a `finally` block.

    Args:
        seconds: Number of seconds from now to set as the deadline. If `None`,
            `0`, or a non-positive value is provided, the deadline is cleared.

    Returns:
        contextvars.Token: The token returned by the underlying ContextVar `.set()`
        call, which must be passed to `reset_outer_deadline` to restore the
        previous value.
    """
    if seconds and seconds > 0:
        return _OUTER_DEADLINE.set(time.monotonic() + seconds)
    return _OUTER_DEADLINE.set(None)


def reset_outer_deadline(token):
    """Restore the previous outer deadline set by `set_outer_deadline`.

    This should be called in a `finally` block to ensure the deadline
    is restored even if an exception occurs.

    Args:
        token: The `contextvars.Token` returned by `set_outer_deadline`.
    """
    if token is not None:
        _OUTER_DEADLINE.reset(token)


def _remaining_budget() -> float | None:
    dl = _OUTER_DEADLINE.get()
    if dl is None:
        return None
    return max(0.0, dl - time.monotonic())


def _is_budget_spent() -> bool:
    rem = _remaining_budget()
    return rem is not None and rem <= 0.0


def resolve_effective_attempt_timeout():
    """Resolve the timeout to use for a single provider attempt.

    Combines the configured per-attempt timeout with any remaining outer budget:
    - If `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS` is `0` or `None`, returns `0`
      callers should skip `asyncio.wait_for` in this case and rely on the outer cap.
    - If positive and an outer deadline is present, returns
      `min(per_attempt, remaining_budget)`.
    - If positive and no outer deadline is present, returns `per_attempt`.

    Returns:
        float: Seconds to use for the inner per-attempt timeout. `0` means
        disable inner timeout and rely on the outer budget instead.
    """
    per_attempt = float(
        get_settings().DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0
    )
    # 0 or None disable inner wait_for. That means rely on outer task cap for timeouts instead.
    if per_attempt <= 0:
        return 0
    # If we do have a positive per-attempt, use up to remaining outer budget.
    rem = _remaining_budget()
    if rem is not None:
        return max(0.0, min(per_attempt, rem))
    return per_attempt


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

_BUILTIN_TIMEOUT_EXCS = (
    (TimeoutError,)
    if asyncio.TimeoutError is TimeoutError
    else (TimeoutError, asyncio.TimeoutError)
)


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
        if isinstance(e, _BUILTIN_TIMEOUT_EXCS):
            return True

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
        settings = get_settings()
        attempts = (
            settings.DEEPEVAL_RETRY_MAX_ATTEMPTS
        )  # TODO: add constraints in settings
        return stop_after_attempt(attempts)(retry_state)


class WaitFromEnv(wait_base):
    def __call__(self, retry_state):
        settings = get_settings()
        initial = settings.DEEPEVAL_RETRY_INITIAL_SECONDS
        exp_base = settings.DEEPEVAL_RETRY_EXP_BASE
        jitter = settings.DEEPEVAL_RETRY_JITTER
        cap = settings.DEEPEVAL_RETRY_CAP_SECONDS

        if cap == 0:  # <- 0 means no backoff sleeps or jitter
            return 0
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


###########
# Helpers #
###########
# Convenience helpers


def sdk_retries_for(provider: Provider) -> bool:
    """True if this provider should delegate retries to the SDK (per settings)."""
    chosen = get_settings().DEEPEVAL_SDK_RETRY_PROVIDERS or []
    slug = slugify(provider)
    return "*" in chosen or slug in chosen


def get_retry_policy_for(provider: Provider) -> Optional[ErrorPolicy]:
    """
    Return the ErrorPolicy for a given provider slug, or None when:
      - the user requested SDK-managed retries for this provider, OR
      - we have no usable policy (optional dependency missing).
    """
    if sdk_retries_for(provider):
        return None
    slug = slugify(provider)
    return _POLICY_BY_SLUG.get(slug) or None


def dynamic_retry(provider: Provider):
    """
    Tenacity retry= argument that checks settings at *call time*.
    If SDK retries are chosen (or no policy available), it never retries.
    """
    slug = slugify(provider)
    static_pred = _STATIC_PRED_BY_SLUG.get(slug)

    def _pred(e: Exception) -> bool:
        if sdk_retries_for(slug):
            return False  # hand off to SDK
        if static_pred is None:
            return False  # no policy -> no Tenacity retries
        return static_pred(e)  # use prebuilt predicate

    return retry_if_exception(_pred)


def _retry_log_levels():
    s = get_settings()
    base_level = s.LOG_LEVEL if s.LOG_LEVEL is not None else logging.INFO
    before_level = s.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL
    after_level = s.DEEPEVAL_RETRY_AFTER_LOG_LEVEL
    return (
        before_level if before_level is not None else base_level,
        after_level if after_level is not None else logging.ERROR,
    )


def make_before_sleep_log(slug: str):
    """
    Tenacity 'before_sleep' callback: runs before Tenacity sleeps for the next retry.
    Read the level dynamically each time.
    """
    _logger = logging.getLogger(f"deepeval.retry.{slug}")

    def _before_sleep(retry_state: RetryCallState) -> None:
        before_level, _ = _retry_log_levels()
        if not _logger.isEnabledFor(before_level):
            return

        exc = retry_state.outcome.exception()
        sleep = getattr(
            getattr(retry_state, "next_action", None), "sleep", None
        )

        _logger.log(
            before_level,
            "Retrying in %s s (attempt %s) after %r",
            sleep,
            retry_state.attempt_number,
            exc,
        )

    return _before_sleep


def make_after_log(slug: str):
    """
    Tenacity 'after' callback: runs after each attempt. We log only when the
    attempt raised, and we look up the level dynamically so changes to settings
    take effect immediately.
    """
    _logger = logging.getLogger(f"deepeval.retry.{slug}")

    def _after(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception()
        if exc is None:
            return

        _, after_level = _retry_log_levels()
        if not _logger.isEnabledFor(after_level):
            return

        show_trace = bool(get_settings().DEEPEVAL_LOG_STACK_TRACES)
        exc_info = (
            (type(exc), exc, getattr(exc, "__traceback__", None))
            if show_trace
            else None
        )

        _logger.log(
            after_level,
            "%s Retrying: %s time(s)...",
            exc,
            retry_state.attempt_number,
            exc_info=exc_info,
        )

    return _after


def _make_timeout_error(timeout_seconds: float) -> asyncio.TimeoutError:
    settings = get_settings()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "retry config: per_attempt=%s s, max_attempts=%s, per_task_budget=%s s",
            timeout_seconds,
            settings.DEEPEVAL_RETRY_MAX_ATTEMPTS,
            settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS,
        )
    msg = (
        f"call timed out after {timeout_seconds:g}s (per attempt). "
        "Increase DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE (None disables) or reduce work per attempt."
    )
    return asyncio.TimeoutError(msg)


def run_sync_with_timeout(func, timeout_seconds, *args, **kwargs):
    """
    Run a synchronous callable with a soft timeout enforced by a helper thread,
    with a global cap on concurrent timeout-workers.

    How it works
    ------------
    - A module-level BoundedSemaphore (size = settings.DEEPEVAL_TIMEOUT_THREAD_LIMIT)
      gates creation of timeout worker threads. If no permit is available, this call
      blocks until a slot frees up. If settings.DEEPEVAL_TIMEOUT_SEMAPHORE_WARN_AFTER_SECONDS
      > 0 and acquisition takes longer than that, a warning is logged before continuing
      to wait.
    - Once a permit is acquired, a daemon thread executes `func(*args, **kwargs)`.
    - We wait up to `timeout_seconds` for completion. If the timeout elapses, we raise
      `TimeoutError`. The worker thread is not killed, it continues and releases the semaphore when it eventually finishes.
    - If the worker finishes in time, we return its result or re-raise its exception
      (with original traceback).

    Cancellation semantics
    ----------------------
    This is a soft timeout: Python threads cannot be forcibly terminated. When timeouts
    are rare this is fine. If timeouts are common, consider moving to:
      - a shared ThreadPoolExecutor (caps threads and amortizes creation), or
      - worker process (supports killing in-flight processes)

    Concurrency control & logging
    -----------------------------
    - Concurrency is bounded by `DEEPEVAL_TIMEOUT_THREAD_LIMIT`.
    - If acquisition exceeds `DEEPEVAL_TIMEOUT_SEMAPHORE_WARN_AFTER_SECONDS`, we log a
      warning and then block until a slot is available.
    - On timeout, if DEBUG is enabled and `DEEPEVAL_VERBOSE_MODE` is True, we log a short
      thread sample to help diagnose pressure.

    Args:
        func: Synchronous callable to execute.
        timeout_seconds: Float seconds for the soft timeout (0/None disables).
        *args, **kwargs: Passed through to `func`.

    Returns:
        Whatever `func` returns.

    Raises:
        TimeoutError: If `timeout_seconds` elapse before completion.
        BaseException: If `func` raises, the same exception is re-raised with its
                       original traceback.
    """
    if not timeout_seconds or timeout_seconds <= 0:
        return func(*args, **kwargs)

    # try to respect the global cap on concurrent timeout workers
    warn_after = float(
        get_settings().DEEPEVAL_TIMEOUT_SEMAPHORE_WARN_AFTER_SECONDS or 0.0
    )
    if warn_after > 0:
        acquired = _TIMEOUT_SEMA.acquire(timeout=warn_after)
        if not acquired:
            logger.warning(
                "timeout thread limit reached (%d); waiting for a slot...",
                _MAX_TIMEOUT_THREADS,
            )
            _TIMEOUT_SEMA.acquire()
    else:
        _TIMEOUT_SEMA.acquire()

    done = threading.Event()
    result = {"value": None, "exc": None}

    context = copy_context()

    def target():
        try:
            result["value"] = context.run(func, *args, **kwargs)
        except BaseException as e:
            result["exc"] = e
        finally:
            done.set()
            _TIMEOUT_SEMA.release()

    t = threading.Thread(
        target=target,
        daemon=True,
        name=f"deepeval-timeout-worker-{next(_WORKER_ID)}",
    )

    try:
        t.start()
    except BaseException:
        _TIMEOUT_SEMA.release()
        raise

    finished = done.wait(timeout_seconds)
    if not finished:
        if (
            logger.isEnabledFor(logging.DEBUG)
            and get_settings().DEEPEVAL_VERBOSE_MODE
        ):
            names = [th.name for th in threading.enumerate()[:10]]
            logger.debug(
                "timeout after %.3fs (active_threads=%d, sample=%s)",
                timeout_seconds,
                threading.active_count(),
                names,
            )
        raise _make_timeout_error(timeout_seconds)

    # Completed within time: return or raise
    if result["exc"] is not None:
        exc = result["exc"]
        raise exc.with_traceback(getattr(exc, "__traceback__", None))
    return result["value"]


def create_retry_decorator(provider: Provider):
    """
    Build a Tenacity @retry decorator wired to our dynamic retry policy
    for the given provider slug.
    """
    slug = slugify(provider)
    base_retry = retry(
        wait=dynamic_wait(),
        stop=dynamic_stop(),
        retry=dynamic_retry(slug),
        before_sleep=make_before_sleep_log(slug),
        after=make_after_log(slug),
        reraise=False,
    )

    def _decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def attempt(*args, **kwargs):
                if _is_budget_spent():
                    raise _make_timeout_error(0)

                per_attempt_timeout = resolve_effective_attempt_timeout()

                coro = func(*args, **kwargs)
                if per_attempt_timeout > 0:
                    try:
                        return await asyncio.wait_for(coro, per_attempt_timeout)
                    except (asyncio.TimeoutError, TimeoutError) as e:
                        if (
                            logger.isEnabledFor(logging.DEBUG)
                            and get_settings().DEEPEVAL_VERBOSE_MODE is True
                        ):
                            logger.debug(
                                "async timeout after %.3fs (active_threads=%d, tasks=%d)",
                                per_attempt_timeout,
                                threading.active_count(),
                                len(asyncio.all_tasks()),
                            )
                        raise _make_timeout_error(per_attempt_timeout) from e
                return await coro

            return base_retry(attempt)

        @functools.wraps(func)
        def attempt(*args, **kwargs):
            if _is_budget_spent():
                raise _make_timeout_error(0)

            per_attempt_timeout = resolve_effective_attempt_timeout()
            if per_attempt_timeout > 0:
                return run_sync_with_timeout(
                    func, per_attempt_timeout, *args, **kwargs
                )
            return func(*args, **kwargs)

        return base_retry(attempt)

    return _decorator


def _httpx_net_excs() -> tuple[type, ...]:
    try:
        import httpx
    except Exception:
        return ()
    names = (
        "RequestError",  # base for transport errors
        "TimeoutException",  # base for timeouts
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    )
    return tuple(getattr(httpx, n) for n in names if hasattr(httpx, n))


def _requests_net_excs() -> tuple[type, ...]:
    try:
        import requests
    except Exception:
        return ()
    names = (
        "RequestException",
        "Timeout",
        "ConnectionError",
        "ReadTimeout",
        "SSLError",
        "ChunkedEncodingError",
    )
    return tuple(
        getattr(requests.exceptions, n)
        for n in names
        if hasattr(requests.exceptions, n)
    )


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

    _HTTPX_NET_EXCS = _httpx_net_excs()
    _REQUESTS_EXCS = _requests_net_excs()

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
        + _HTTPX_NET_EXCS
        + _REQUESTS_EXCS,  # treat 5xx as transient
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
    # Catch transport + timeout issues via base classes
    _HTTPX_NET_EXCS = _httpx_net_excs()
    _REQUESTS_EXCS = _requests_net_excs()

    OLLAMA_ERROR_POLICY = ErrorPolicy(
        auth_excs=(),
        rate_limit_excs=(),  # no rate limiting semantics locally
        network_excs=_HTTPX_NET_EXCS + _REQUESTS_EXCS,  # retry network/timeouts
        http_excs=(),  # optionally add httpx.HTTPStatusError if you call raise_for_status()
        non_retryable_codes=frozenset(),
        message_markers={},
    )
except Exception:
    OLLAMA_ERROR_POLICY = None


# Map provider slugs to their policy objects.
# It is OK if some are None, we'll treat that as no Error Policy / Tenacity
_POLICY_BY_SLUG: dict[str, Optional[ErrorPolicy]] = {
    PS.OPENAI.value: OPENAI_ERROR_POLICY,
    PS.AZURE.value: AZURE_OPENAI_ERROR_POLICY,
    PS.BEDROCK.value: BEDROCK_ERROR_POLICY,
    PS.ANTHROPIC.value: ANTHROPIC_ERROR_POLICY,
    PS.DEEPSEEK.value: DEEPSEEK_ERROR_POLICY,
    PS.GOOGLE.value: GOOGLE_ERROR_POLICY,
    PS.GROK.value: GROK_ERROR_POLICY,
    PS.KIMI.value: KIMI_ERROR_POLICY,
    PS.LITELLM.value: LITELLM_ERROR_POLICY,
    PS.LOCAL.value: LOCAL_ERROR_POLICY,
    PS.OLLAMA.value: OLLAMA_ERROR_POLICY,
}


def _opt_pred(
    policy: Optional[ErrorPolicy],
) -> Optional[Callable[[Exception], bool]]:
    return make_is_transient(policy) if policy else None


_STATIC_PRED_BY_SLUG: dict[str, Optional[Callable[[Exception], bool]]] = {
    PS.OPENAI.value: _opt_pred(OPENAI_ERROR_POLICY),
    PS.AZURE.value: _opt_pred(AZURE_OPENAI_ERROR_POLICY),
    PS.BEDROCK.value: _opt_pred(BEDROCK_ERROR_POLICY),
    PS.ANTHROPIC.value: _opt_pred(ANTHROPIC_ERROR_POLICY),
    PS.DEEPSEEK.value: _opt_pred(DEEPSEEK_ERROR_POLICY),
    PS.GOOGLE.value: _opt_pred(GOOGLE_ERROR_POLICY),
    PS.GROK.value: _opt_pred(GROK_ERROR_POLICY),
    PS.KIMI.value: _opt_pred(KIMI_ERROR_POLICY),
    PS.LITELLM.value: _opt_pred(LITELLM_ERROR_POLICY),
    PS.LOCAL.value: _opt_pred(LOCAL_ERROR_POLICY),
    PS.OLLAMA.value: _opt_pred(OLLAMA_ERROR_POLICY),
}


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
