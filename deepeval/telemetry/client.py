"""The analytics backend seam and the one function that emits events.

What we send -- an anonymous id, an event name, and a flat property dict -- is
the shape every product-analytics vendor takes. `PostHogBackend` is the only
module in the repo that may import `posthog`, so swapping vendors is one new
class rather than a search-and-replace.
"""

import logging
from typing import Dict, List, Optional, Protocol, Set

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

_POSTHOG_PROJECT_API_KEY = "phc_IXvGRcscJJoIb049PtjIZ65JnXQguOUZ5B5MncunFdB"
_POSTHOG_HOST = "https://us.i.posthog.com"


# Our PostHog client's SDK log lines go here instead of the SDK's own logger.
# Only errors pass; the SDK's warnings (e.g. "flush ran out of budget" when a
# short CLI command exits before the upload finishes) are noise to a user.
_SDK_LOGGER_NAME = "deepeval.telemetry.posthog"


def _quiet_sdk_logging(client: object) -> None:
    """Route one client's SDK log lines to our own logger.

    The `posthog` SDK logs through a single ``logging.getLogger("posthog")``
    shared by every client in the process, so muting that logger would also
    mute a user's own PostHog client running alongside deepeval. Instead,
    shadow the class-level ``log`` attribute on *our* instance and its lanes,
    which is all the SDK consults when it logs. Best effort: if the SDK's
    internals change, the SDK just keeps logging as before.
    """
    quiet = logging.getLogger(_SDK_LOGGER_NAME)
    quiet.setLevel(logging.ERROR)
    try:
        setattr(client, "log", quiet)
        for lane in getattr(client, "_lanes", None) or ():
            setattr(lane, "log", quiet)
    except Exception:
        pass


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


class PostHogBackend:
    """The only place `posthog` is imported.

    Maps our vendor-neutral `anonymous_id` onto PostHog's `distinct_id`.
    """

    def __init__(self) -> None:
        from posthog import Posthog

        self._client = Posthog(
            project_api_key=_POSTHOG_PROJECT_API_KEY,
            host=_POSTHOG_HOST,
        )
        _quiet_sdk_logging(self._client)

    def capture(
        self,
        anonymous_id: str,
        event: Event,
        properties: Dict[str, PropValue],
    ) -> None:
        self._client.capture(
            distinct_id=anonymous_id,
            event=event.value,
            properties=properties,
        )

    def flush(self) -> None:
        self._client.flush()  # type: ignore[no-untyped-call]


_backend: Optional[TelemetryBackend] = None


def get_backend() -> TelemetryBackend:
    global _backend
    if _backend is None:
        if telemetry_opt_out():
            _backend = NoopBackend()
        else:
            try:
                _backend = PostHogBackend()
            except Exception:
                logger.debug("Telemetry backend unavailable", exc_info=True)
                _backend = NoopBackend()
    return _backend


def set_backend(backend: TelemetryBackend) -> None:
    """Swap the backend. Used by tests and by any future vendor migration."""
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
