import logging
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace.status import Status, StatusCode
from deepeval.tracing.otel.exporter import DeepEvalSpanExporter

# Configure logging to see the output
logging.basicConfig(level=logging.INFO)

# Setup OpenTelemetry tracer - handle existing TracerProvider
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
else:
    tracer_provider = trace.get_tracer_provider()

# Create our exporter
exporter = DeepEvalSpanExporter()
span_processor = SimpleSpanProcessor(exporter)
tracer_provider.add_span_processor(span_processor)

# Get a tracer
tracer = trace.get_tracer("test_tracer")

# Create a trace
with tracer.start_as_current_span("test_trace") as parent_span:
    # Set trace-level attributes
    parent_span.set_attribute("gen_ai.operation.name", "chat")
    parent_span.set_attribute("gen_ai.request.model", "gpt-4")
    parent_span.set_attribute("gen_ai.system.message", "You are a helpful assistant.")
    parent_span.set_attribute("gen_ai.user.message", "What is the capital of France?")
    parent_span.set_attribute("gen_ai.usage.input_tokens", 50)
    parent_span.set_attribute("gen_ai.usage.output_tokens", 150)
    
    time.sleep(0.5)
    parent_span.set_attribute("gen_ai.assistant.message", "The capital of France is Paris.")
    # Set status code to OK
    parent_span.set_status(Status(StatusCode.OK))
    
    # # Add a child LLM span
    # with tracer.start_as_current_span("llm_call") as llm_span:
    #     llm_span.set_attribute("gen_ai.request.model", "gpt-4")
    #     llm_span.set_attribute("gen_ai.user.message", "What is the capital of France?")
    #     llm_span.set_attribute("gen_ai.assistant.message", "The capital of France is Paris.")
    #     llm_span.set_attribute("gen_ai.usage.input_tokens", 10)
    #     llm_span.set_attribute("gen_ai.usage.output_tokens", 8)
        
    # # Add another child span with an error
    # with tracer.start_as_current_span("error_span") as error_span:
    #     error_span.set_status(Status(StatusCode.ERROR, "Something went wrong"))
    #     error_span.set_attribute("gen_ai.request.model", "gpt-4")
    #     error_span.set_attribute("gen_ai.user.message", "Generate an image")
    #     error_span.set_attribute("gen_ai.usage.input_tokens", 3)

# Force the exporter to process all spans
tracer_provider.force_flush()

print("Test completed - check logs above for trace details")
