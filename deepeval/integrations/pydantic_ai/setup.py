from typing import Optional
from .patch import safe_patch_agent_iter_method, safe_patch_agent_run_method
import deepeval
from deepeval.tracing.otel import ConfidentSpanExporter
from deepeval.telemetry import capture_tracing_integration

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import set_tracer_provider

    opentelemetry_installed = True
except:
    opentelemetry_installed = False


def is_opentelemetry_available():
    if not opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. Please install it with `pip install opentelemetry-sdk`."
        )
    return True


def instrument_pydantic_ai(api_key: Optional[str] = None):
    with capture_tracing_integration("pydantic_ai"):
        is_opentelemetry_available()
        # safe_patch_agent_iter_method()
        safe_patch_agent_run_method()

        if api_key:
            deepeval.login(api_key)

        created_new_provider = False
        if not isinstance(trace.get_tracer_provider(), TracerProvider):
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)
            created_new_provider = True
        else:
            tracer_provider = trace.get_tracer_provider()

        exporter = ConfidentSpanExporter()
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(span_processor)

        # Only set tracer provider if we created a new one, not if we're reusing existing
        if created_new_provider:
            set_tracer_provider(tracer_provider)
