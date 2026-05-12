from opentelemetry import trace

from deepeval.dataset import test_run_tracer


class FakeOTLPSpanExporter:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def export(self, spans):
        return None

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        return None


def test_test_run_tracer_uses_local_provider_without_replacing_global(
    monkeypatch,
):
    original_provider = trace.get_tracer_provider()

    def fail_if_global_provider_is_replaced(provider):
        raise AssertionError(
            "test-run telemetry must not replace the host OpenTelemetry provider"
        )

    monkeypatch.setattr(
        trace, "set_tracer_provider", fail_if_global_provider_is_replaced
    )
    monkeypatch.setattr(test_run_tracer, "is_opentelemetry_installed", True)
    monkeypatch.setattr(
        test_run_tracer,
        "get_confident_api_key",
        lambda: "confident_test_key",
    )
    monkeypatch.setattr(
        test_run_tracer,
        "OTLPSpanExporter",
        FakeOTLPSpanExporter,
        raising=False,
    )

    provider, tracer = test_run_tracer.init_global_test_run_tracer()

    try:
        assert provider is test_run_tracer.GLOBAL_TEST_RUN_TRACER_PROVIDER
        assert tracer is test_run_tracer.GLOBAL_TEST_RUN_TRACER
        assert trace.get_tracer_provider() is original_provider
    finally:
        provider.shutdown()
        # `init_global_test_run_tracer` mutates module globals via `global`,
        # which monkeypatch can't undo. Reset them so later tests don't see
        # this test's fake-backed provider.
        test_run_tracer.GLOBAL_TEST_RUN_TRACER_PROVIDER = None
        test_run_tracer.GLOBAL_TEST_RUN_TRACER = None
