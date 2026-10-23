try:
    import openai  # noqa: F401
except ImportError:
    raise ModuleNotFoundError(
        "Please install OpenAI to use this feature: 'pip install openai'"
    )


try:
    from openai import OpenAI, AsyncOpenAI  # noqa: F401
except ImportError:
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


if OpenAI or AsyncOpenAI:
    from deepeval.telemetry import capture_tracing_integration
    from deepeval.tracing.integrations import Integration
    from deepeval.tracing.otel.setup import ensure_confident_trace

    with capture_tracing_integration(Integration.OPEN_AI):
        ensure_confident_trace(("openai",))
