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
    from deepeval.anthropic.patch import patch_anthropic_classes
    from deepeval.telemetry import capture_tracing_integration

    with capture_tracing_integration("anthropic"):
        patch_anthropic_classes()
