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

        # create a new tracer provider
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(
            ConfidentSpanExporter(api_key=api_key)
        ))
        trace.set_tracer_provider(tracer_provider)

        # create an instrumented exporter
        from pydantic_ai.models.instrumented import InstrumentationSettings
        from pydantic_ai import Agent
        instrumentation_settings = InstrumentationSettings(
            tracer_provider=tracer_provider
        )

    
        # instrument all agents
        Agent.instrument_all(instrument=instrumentation_settings)