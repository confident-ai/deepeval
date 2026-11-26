"""The telemetry backend seam and the one function that emits events.

What we send -- an anonymous id, an event name, and a flat property dict -- goes
as plain JSON to deepeval's own telemetry host. Where it is analysed is decided
on that server, so changing analytics vendor needs no SDK release. `HttpBackend`
is the only place in the repo that knows the host.
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Protocol, Set

import requests

from deepeval._version import __version__ as DEEPEVAL_VERSION
from deepeval.config.settings import get_settings
from deepeval.telemetry.events import TELEMETRY_SCHEMA_VERSION, Event
from deepeval.telemetry.identity import get_identity, is_logged_in
from deepeval.telemetry.properties import (
    EventProperties,
    Language,
    PropValue,
)
from deepeval.telemetry.runtime import detect_runtime

logger = logging.getLogger(__name__)

_TELEMETRY_URL = "https://telemetry.deepeval.com/v1/sdk"

# The server accepts at most 100 events per request.
_MAX_BATCH_SIZE = 100
# Events beyond this are dropped rather than growing without bound offline.
_MAX_QUEUE_SIZE = 1000
_REQUEST_TIMEOUT_SECONDS = 3.0
# How long `flush` may hold up a process that is trying to exit.
_FLUSH_BUDGET_SECONDS = 3.0


def telemetry_opt_out() -> bool:
    return bool(get_settings().DEEPEVAL_TELEMETRY_OPT_OUT)


class TelemetryBackend(Protocol):
    def capture(
        self,
        anonymous_id: str,
        event: Event,
        properties: Dict[str, PropValue],
    ) -> None: ...

    def flush(self) -> None: ...


class NoopBackend:
    """Used when telemetry is opted out, so call sites need no branching."""

    def capture(
        self,
        anonymous_id: str,
        event: Event,
        properties: Dict[str, PropValue],
    ) -> None:
        return None

    def flush(self) -> None:
        return None


class HttpBackend:
    """Posts events to deepeval's telemetry host, off the caller's thread.

    `capture` only queues and a daemon thread sends the queue in batches.
    `flush` waits for that thread, or sends from the calling thread when one
    can no longer be started (interpreter shutdown). Nothing here raises or
    logs above debug level: a user never sees a telemetry failure.
    """

    def __init__(self, url: str = _TELEMETRY_URL) -> None:
        self._url = url
        self._pending: Deque[Dict[str, object]] = deque()
        self._in_flight = 0
        self._changed = threading.Condition()
        self._worker: Optional[threading.Thread] = None

    def capture(
        self,
        anonymous_id: str,
        event: Event,
        properties: Dict[str, PropValue],
    ) -> None:
        with self._changed:
            if len(self._pending) >= _MAX_QUEUE_SIZE:
                return
            self._pending.append(
                {
                    "event": event.value,
                    "anonymousId": anonymous_id,
                    "properties": properties,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._changed.notify_all()
        self._ensure_worker()

    def flush(self) -> None:
        deadline = time.monotonic() + _FLUSH_BUDGET_SECONDS
        if self._ensure_worker():
            # Wait on the worker rather than sending here, so a host that
            # hangs can never hold the caller past the budget.
            with self._changed:
                self._changed.wait_for(
                    lambda: not self._pending and self._in_flight == 0,
                    timeout=_FLUSH_BUDGET_SECONDS,
                )
            return
        while time.monotonic() < deadline:
            batch = self._take_batch()
            if not batch:
                return
            self._send(batch, timeout=max(deadline - time.monotonic(), 0.1))

    def _ensure_worker(self) -> bool:
        """Start the sending thread if needed. False when it cannot run."""
        # Also restarts after a fork, which does not carry threads over.
        if self._worker is not None and self._worker.is_alive():
            return True
        worker = threading.Thread(
            target=self._run, name="deepeval-telemetry", daemon=True
        )
        try:
            worker.start()
        except RuntimeError:
            # Interpreter is shutting down; `flush` sends what is queued.
            return False
        self._worker = worker
        return True

    def _run(self) -> None:
        while True:
            with self._changed:
                self._changed.wait_for(lambda: bool(self._pending))
            batch = self._take_batch()
            if batch:
                self._send(batch, timeout=_REQUEST_TIMEOUT_SECONDS)

    def _take_batch(self) -> List[Dict[str, object]]:
        with self._changed:
            size = min(len(self._pending), _MAX_BATCH_SIZE)
            batch = [self._pending.popleft() for _ in range(size)]
            self._in_flight += len(batch)
            return batch

    def _send(self, batch: List[Dict[str, object]], timeout: float) -> None:
        try:
            requests.post(
                self._url,
                data=json.dumps({"events": batch}, default=str),
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
        except Exception:
            # Telemetry is best effort: a failed batch is dropped, not retried.
            logger.debug("Failed to send telemetry", exc_info=True)
        finally:
            with self._changed:
                self._in_flight -= len(batch)
                self._changed.notify_all()


_backend: Optional[TelemetryBackend] = None


def get_backend() -> TelemetryBackend:
    global _backend
    if _backend is None:
        if telemetry_opt_out():
            _backend = NoopBackend()
        else:
            try:
                _backend = HttpBackend()
            except Exception:
                logger.debug("Telemetry backend unavailable", exc_info=True)
                _backend = NoopBackend()
    return _backend


def set_backend(backend: TelemetryBackend) -> None:
    """Swap the backend. Used by tests."""
    global _backend
    _backend = backend


# Populated by `capture_tracing_integration`, read by `base_properties`.
_installed_integrations: Set[str] = set()


def register_integration(name: str) -> bool:
    """Record an integration. Returns True the first time each is seen."""
    if name in _installed_integrations:
        return False
    _installed_integrations.add(name)
    return True


def installed_integrations() -> List[str]:
    return sorted(_installed_integrations)


def base_properties() -> EventProperties:
    """Stamped on every event."""
    identity = get_identity()
    active = installed_integrations()
    return EventProperties(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        sdk_language=Language.PYTHON,
        sdk_version=DEEPEVAL_VERSION,
        runtime=detect_runtime(),
        user_status=identity.status,
        user_id=identity.anonymous_id,
        logged_in=is_logged_in(),
        integrations=active,
        integrations_count=len(active),
        integrations_primary=active[0] if active else "none",
    )


def capture(event: Event, properties: EventProperties) -> None:
    """The only function that hands an event to a backend.

    Taking `Event` rather than `str` is what makes a dynamic event name
    impossible to reintroduce.
    """
    if telemetry_opt_out():
        return
    try:
        payload = base_properties().merged_with(properties)
        get_backend().capture(get_identity().anonymous_id, event, payload)
    except Exception:
        # Telemetry must never break a user's evaluation.
        logger.debug("Failed to capture %s", event, exc_info=True)


def flush() -> None:
    try:
        get_backend().flush()
    except Exception:
        logger.debug("Failed to flush telemetry", exc_info=True)
