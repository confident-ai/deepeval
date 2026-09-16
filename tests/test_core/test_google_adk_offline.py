"""Regression test for offline (no Confident API key) framework instrumentation.

``instrument_google_adk`` delegates to ``instrument_openinference``, which
accepts ``api_key=None`` and captures spans locally.  The docs advertise that
tracing runs fully offline, so requiring ``CONFIDENT_API_KEY`` up front is a
bug (see confident-ai/deepeval#3005).  The sibling framework helpers
(``instrument_openinference``/``instrument_agentcore``/``instrument_strands``)
never hard-gate on the key, so ``google_adk`` should behave the same way.
"""

import deepeval.integrations.google_adk.otel as gadk_otel
import deepeval.integrations.openinference as oi


class _DummyInstrumentor:
    instrumented = False

    def instrument(self):
        type(self).instrumented = True


def test_instrument_google_adk_runs_without_confident_api_key(monkeypatch):
    _DummyInstrumentor.instrumented = False
    # No Confident key configured anywhere.
    monkeypatch.setattr(gadk_otel, "get_confident_api_key", lambda: None)
    monkeypatch.setattr(
        gadk_otel,
        "_require_google_adk_instrumentor",
        lambda: _DummyInstrumentor,
    )

    captured = {}
    monkeypatch.setattr(
        oi, "instrument_openinference", lambda **kw: captured.update(kw)
    )

    # Must not raise ValueError about CONFIDENT_API_KEY.
    gadk_otel.instrument_google_adk()

    assert _DummyInstrumentor.instrumented is True
    # The missing key is forwarded as-is so downstream stays in local-only mode.
    assert captured["api_key"] is None
    assert captured["integration"] == gadk_otel.Integration.GOOGLE_ADK.value


def test_instrument_google_adk_forwards_explicit_key(monkeypatch):
    _DummyInstrumentor.instrumented = False
    monkeypatch.setattr(
        gadk_otel,
        "get_confident_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("must not be consulted")),
    )
    monkeypatch.setattr(
        gadk_otel,
        "_require_google_adk_instrumentor",
        lambda: _DummyInstrumentor,
    )

    captured = {}
    monkeypatch.setattr(
        oi, "instrument_openinference", lambda **kw: captured.update(kw)
    )

    gadk_otel.instrument_google_adk(api_key="explicit-key")

    assert captured["api_key"] == "explicit-key"
