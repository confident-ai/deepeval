import time
import uuid
from types import SimpleNamespace
from datetime import datetime, timezone

from deepeval.tracing.api import TraceApi, TraceSpanApiStatus
from tests.test_core.stubs import RecordingPortalockerLock


def ts_iso8601_utc(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def make_trace_api(
    *,
    uuid_str: str | None = None,
    status: TraceSpanApiStatus = TraceSpanApiStatus.SUCCESS,
) -> TraceApi:
    now = time.time()
    return TraceApi(
        uuid=uuid_str or str(uuid.uuid4()),
        name="test-trace",
        status=status,
        error=None,
        input=None,
        output=None,
        expectedOutput=None,
        context=None,
        retrievalContext=None,
        # give these concrete lists to avoid calling append on None
        agentSpans=[],
        llmSpans=[],
        retrieverSpans=[],
        toolSpans=[],
        baseSpans=[],
        metricsData=[],
        startTime=ts_iso8601_utc(now),
        endTime=ts_iso8601_utc(now),
    )


def teardown_settings_singleton():
    import deepeval.config.settings as settings_mod

    settings_mod._settings_singleton = None


def reset_settings_env(monkeypatch, *, skip_keys: set[str] = set()):
    # reset singleton
    teardown_settings_singleton()

    # drop env vars that map to Settings fields
    from deepeval.config.settings import Settings

    for k in Settings.model_fields.keys():
        if k not in skip_keys:
            monkeypatch.delenv(k, raising=False)

    # don’t carry default save across tests, keep things clean
    monkeypatch.delenv("DEEPEVAL_DEFAULT_SAVE", raising=False)


def _make_fake_portalocker():
    """
    Minimal portalocker replacement for tests that need to inspect file writes.
    """
    return SimpleNamespace(
        Lock=RecordingPortalockerLock,
        LOCK_EX=1,
        LOCK_SH=2,
        LOCK_NB=4,
        exceptions=SimpleNamespace(LockException=RuntimeError),
    )
