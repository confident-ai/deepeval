import time
import json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


from deepeval.tracing.otel.exporter import ConfidentSpanExporter

# Set up tracer provider
tracer_provider = trace.get_tracer_provider()
if not isinstance(tracer_provider, TracerProvider):
    trace.set_tracer_provider(TracerProvider())

# Add confident span exporter wrapped in batch span processor
tracer_provider.add_span_processor(BatchSpanProcessor(ConfidentSpanExporter()))

# Get tracer
tracer = trace.get_tracer("deepeval_tracer")


def llm(input: str) -> str:
    with tracer.start_as_current_span("llm_span") as span:
        span.set_attribute("confident.span.type", "llm")
        span.set_attribute("confident.llm.model", "gpt-3.5-turbo")
        span.set_attribute(
            "confident.span.input",
            [
                json.dumps(
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    }
                ),
                json.dumps({"role": "user", "content": input}),
            ],
        )
        time.sleep(0.5)
        span.set_attribute("confident.span.output", "Hello world")

    return "Hello world"


def my_app(input: str):
    with tracer.start_as_current_span("my_app") as span:
        span.set_attribute("confident.span.input", input)
        res = llm(input)
        span.set_attribute("confident.span.output", res)


my_app("Hi")

time.sleep(10)
