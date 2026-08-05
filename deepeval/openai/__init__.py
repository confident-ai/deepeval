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
    from deepeval.openai.patch import patch_openai_classes
    from deepeval.telemetry import capture_tracing_integration
    from deepeval.tracing.integrations import Integration

    with capture_tracing_integration(Integration.OPEN_AI):
        patch_openai_classes()
