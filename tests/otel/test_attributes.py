import time
from deepeval.tracing.otel.exporter import ConfidentSpanExporter

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openai import OpenAI
from time import sleep
import os
import json

from dotenv import load_dotenv

load_dotenv()

os.environ["CONFIDENT_TRACE_FLUSH"] = "YES"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup OpenTelemetry
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
else:
    tracer_provider = trace.get_tracer_provider()

exporter = ConfidentSpanExporter()
span_processor = BatchSpanProcessor(span_exporter=exporter)
tracer_provider.add_span_processor(span_processor)
tracer = trace.get_tracer("deepeval_tracer")


def tool_span(input: str):
    time.sleep(3)
    with tracer.start_as_current_span("tool_span") as span:
        span.set_attribute("confident.span.type", "tool")
        span.set_attribute("confident.tool.name", "tool name")
        span.set_attribute("confident.tool.description", "tool description")

        span.set_attribute(
            "confident.tool.attributes.input_parameters",
            json.dumps({"input": input}),
        )
        span.set_attribute(
            "confident.tool.attributes.output", json.dumps({"output": input})
        )


def retriever_span(input: str):
    time.sleep(2.5)
    with tracer.start_as_current_span("retriever_span") as span:
        span.set_attribute("confident.span.type", "retriever")
        span.set_attribute("confident.retriever.embedder", "embedder")

        span.set_attribute(
            "confident.retriever.attributes.embedding_input", input
        )
        span.set_attribute(
            "confident.retriever.attributes.retrieval_context", ["asd", "asd"]
        )

        span.set_attribute("confident.retriever.attributes.top_k", 10)
        span.set_attribute("confident.retriever.attributes.chunk_size", 10)

        tool_span(input)


def agent_span(input: str):
    time.sleep(2)
    with tracer.start_as_current_span("agent_span") as span:
        span.set_attribute("confident.span.type", "agent")
        span.set_attribute("confident.agent.name", "agent_span")
        span.set_attribute(
            "confident.agent.available_tools",
            ["llm_agent", "retriever_span", "tool_span"],
        )
        span.set_attribute(
            "confident.agent.agent_handoffs",
            ["llm_agent", "retriever_span", "tool_span"],
        )
        span.set_attribute(
            "confident.agent.attributes.input", json.dumps({"input": input})
        )
        span.set_attribute(
            "confident.agent.attributes.output", json.dumps({"output": input})
        )

        # trace attributes
        span.set_attribute(
            "confident.trace.metadata", json.dumps({"test_key": "test_value"})
        )

        retriever_span(input)


def llm_agent(input: str):
    time.sleep(1.5)
    with tracer.start_as_current_span("llm_span") as span:
        span.set_attribute("confident.span.type", "llm")
        span.set_attribute("confident.llm.model", "gpt-3.5-turbo")
        span.set_attribute("confident.llm.cost_per_input_token", 0.01)
        span.set_attribute("confident.llm.cost_per_output_token", 0.02)

        span.set_attribute(
            "confident.llm.attributes.input",
            [json.dumps({"role": "user", "content": input})],
        )
        span.set_attribute(
            "confident.llm.attributes.output",
            json.dumps({"role": "assistant", "content": input}),
        )
        span.set_attribute("confident.llm.attributes.output_token_count", 10)
        span.set_attribute("confident.llm.attributes.input_token_count", 10)

        # trace attributes
        span.set_attribute("confident.trace.thread_id", "123")
        span.set_attribute("confident.trace.user_id", "456")

        span.set_attribute(
            "confident.span.metadata", json.dumps({"test_key": "test_value"})
        )

        agent_span(input)


def meta_agent(input: str):
    with tracer.start_as_current_span("custom_span") as span:
        time.sleep(1)

        span.set_attribute("confident.span.input", input)
        span.set_attribute("confident.span.output", input)
        span.set_attribute("confident.span.error", "Error")
        span.set_attribute(
            "confident.span.feedback",
            json.dumps(
                {
                    "rating": 1,
                    "expected_output": "Expected output",
                    "explanation": "Explanation",
                }
            ),
        )
        span.set_attribute("confident.span.metric_collection", "My metrics")

        # trace attributes
        span.set_attribute("confident.trace.name", "test_trace")
        span.set_attribute("confident.trace.tags", ["tag1", "tag2"])

        span.set_attribute("confident.trace.input", input)
        span.set_attribute("confident.trace.output", input)

        span.set_attribute("confident.trace.llm_test_case.input", "test_input")
        span.set_attribute(
            "confident.trace.llm_test_case.actual_output", "test_actual_output"
        )
        span.set_attribute(
            "confident.trace.llm_test_case.expected_output",
            "test_expected_output",
        )
        span.set_attribute(
            "confident.trace.metric_collection", "test_collection_1"
        )

        llm_agent(input)


from deepeval.dataset import Golden

goldens = [
    Golden(input="What's the weather like in SF?"),
    # Golden(input="Tell me about Elon Musk."),
]

for golden in goldens:
    meta_agent(golden.input)

time.sleep(10)
