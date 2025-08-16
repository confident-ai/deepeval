import time
from deepeval.tracing.otel.exporter import ConfidentSpanExporter

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import json

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
    with tracer.start_as_current_span("tool_span") as span:
        # tool span attributes
        span.set_attribute("confident.span.type", "tool")
        span.set_attribute("confident.tool.name", "tool name")
        span.set_attribute("confident.tool.description", "tool description")

        # input and output
        span.set_attribute("confident.span.input", json.dumps({"input": input}))
        span.set_attribute(
            "confident.span.output", json.dumps({"output": input})
        )


def retriever_span(input: str):
    with tracer.start_as_current_span("retriever_span") as span:
        # retriever span attributes
        span.set_attribute("confident.span.type", "retriever")
        span.set_attribute("confident.retriever.embedder", "embedder")
        span.set_attribute("confident.retriever.top_k", 10)
        span.set_attribute("confident.retriever.chunk_size", 10)

        # output
        span.set_attribute(
            "confident.retriever.retrieval_context", ["context1", "context2"]
        )

        # input and output
        span.set_attribute("confident.span.input", input)
        tool_span(input)


def agent_span(input: str):
    with tracer.start_as_current_span("agent_span") as span:
        # agent span attributes
        span.set_attribute("confident.span.type", "agent")
        span.set_attribute("confident.agent.name", "agent name")
        span.set_attribute(
            "confident.agent.available_tools",
            ["llm_agent", "retriever_span", "tool_span"],
        )
        span.set_attribute(
            "confident.agent.agent_handoffs",
            ["llm_agent", "retriever_span", "tool_span"],
        )

        # input and output
        span.set_attribute("confident.span.input", json.dumps({"input": input}))
        span.set_attribute(
            "confident.span.output", json.dumps({"output": input})
        )

        retriever_span(input)


def llm_agent(input: str):
    with tracer.start_as_current_span("llm_span") as span:
        # llm span attributes
        span.set_attribute("confident.span.type", "llm")
        span.set_attribute("confident.llm.model", "gpt-3.5-turbo")
        span.set_attribute("confident.llm.cost_per_input_token", 0.01)
        span.set_attribute("confident.llm.cost_per_output_token", 0.02)
        span.set_attribute("confident.llm.output_token_count", 10)
        span.set_attribute("confident.llm.input_token_count", 10)

        # input and output
        span.set_attribute(
            "confident.span.input",
            [json.dumps({"role": "user", "content": input})],
        )
        span.set_attribute(
            "confident.span.output",
            json.dumps({"role": "assistant", "content": input}),
        )

        agent_span(input)


def meta_agent(input: str):
    with tracer.start_as_current_span("custom_span") as span:

        # span attributes
        span.set_attribute("confident.span.name", "custom_span")
        span.set_attribute(
            "confident.span.metric_collection", "test_collection_1"
        )

        span.set_attribute("confident.span.input", "test_input")
        span.set_attribute("confident.span.output", "test_actual_output")
        span.set_attribute(
            "confident.span.expected_output", "test_expected_output"
        )

        span.set_attribute("confident.span.context", ["context1", "context2"])
        span.set_attribute(
            "confident.span.retrieval_context", ["context1", "context2"]
        )

        span.set_attribute(
            "confident.span.tools_called",
            [
                json.dumps(
                    {
                        "name": "tool1",
                        "description": "tool1 description",
                        "reasoning": "tool1 reasoning",
                        "output": "tool1 output",
                        "input_parameters": {"input": "tool1 input"},
                    }
                )
            ],
        )
        span.set_attribute(
            "confident.span.expected_tools",
            [
                json.dumps(
                    {
                        "name": "tool2",
                        "description": "tool2 description",
                        "reasoning": "tool2 reasoning",
                        "output": "tool2 output",
                        "input_parameters": {"input": "tool2 input"},
                    }
                )
            ],
        )

        span.set_attribute(
            "confident.span.metadata", json.dumps({"key": "value"})
        )

        # trace attributes
        span.set_attribute("confident.trace.name", "test_trace")
        span.set_attribute(
            "confident.trace.metric_collection", "test_collection_1"
        )

        span.set_attribute("confident.trace.input", "test_input")
        span.set_attribute("confident.trace.output", "test_actual_output")
        span.set_attribute("confident.trace.expected_output", "Paris")
        span.set_attribute("confident.trace.context", ["context1", "context2"])
        span.set_attribute(
            "confident.trace.retrieval_context", ["context1", "context2"]
        )
        span.set_attribute(
            "confident.trace.tools_called",
            [
                json.dumps(
                    {
                        "name": "tool1",
                        "description": "tool1 description",
                        "reasoning": "tool1 reasoning",
                        "output": "tool1 output",
                        "input_parameters": {"input": "tool1 input"},
                    }
                )
            ],
        )
        span.set_attribute(
            "confident.trace.expected_tools",
            [
                json.dumps(
                    {
                        "name": "tool2",
                        "description": "tool2 description",
                        "reasoning": "tool2 reasoning",
                        "output": "tool2 output",
                        "input_parameters": {"input": "tool2 input"},
                    }
                )
            ],
        )

        span.set_attribute("confident.trace.tags", ["tag1", "tag2"])
        span.set_attribute(
            "confident.trace.metadata", json.dumps({"key": "value"})
        )
        span.set_attribute("confident.trace.thread_id", "123")
        span.set_attribute("confident.trace.user_id", "456")

        llm_agent(input)


from deepeval.dataset import Golden

goldens = [
    Golden(input="What's the weather like in SF?"),
    # Golden(input="Tell me about Elon Musk."),
]

for golden in goldens:
    meta_agent(golden.input)

time.sleep(10)
