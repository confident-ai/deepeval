try:
    import anthropic  # noqa: F401
except ImportError:
    raise ModuleNotFoundError(
        "Please install anthropic to use this feature: 'pip install anthropic'"
    )

try:
    from anthropic import Anthropic, AsyncAnthropic  # noqa: F401
except ImportError:
    Anthropic = None  # type: ignore
    AsyncAnthropic = None  # type: ignore

if Anthropic or AsyncAnthropic:
    from deepeval.telemetry import capture_tracing_integration
    from deepeval.tracing.integrations import Integration
    from deepeval.tracing.otel.setup import ensure_confident_trace

    with capture_tracing_integration(Integration.ANTHROPIC):
        ensure_confident_trace(("anthropic",))
