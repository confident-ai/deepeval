try:
    from openrouter import OpenRouter
except ImportError:
    raise ModuleNotFoundError(
        "Please install openrouter to use this feature: 'pip install openrouter'"
    )

from deepeval.openrouter.patch import patch_openrouter_classes
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.integrations import Integration

with capture_tracing_integration(Integration.OPEN_ROUTER):
    patch_openrouter_classes()

__all__ = ["OpenRouter"]
