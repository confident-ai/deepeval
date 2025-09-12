import os
import time
import json

from opentelemetry.trace import NoOpTracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)

from deepeval.dataset import EvaluationDataset, Golden

KEY_1 = "1st key"
KEY_2 = "2nd key"


def tool_span(tracer: NoOpTracer, input: str):
    with tracer.start_as_current_span("tool_span") as span:
        span.set_attribute("confident.span.type", "tool")
        span.set_attribute("confident.tool.description", "tool description")
        span.set_attribute("confident.span.input", json.dumps({"input": input}))
        span.set_attribute(
            "confident.span.output", json.dumps({"output": input})
        )
        time.sleep(1)


def retriever_span(tracer: NoOpTracer, input: str):
    with tracer.start_as_current_span("retriever_span") as span:
        span.set_attribute("confident.span.type", "retriever")
        span.set_attribute("confident.retriever.embedder", "embedder")
        span.set_attribute("confident.retriever.top_k", 10)
        span.set_attribute("confident.retriever.chunk_size", 10)
        span.set_attribute("confident.span.input", input)
        span.set_attribute("confident.span.retrieval_context", ["asd", "asd"])
        time.sleep(1)
        tool_span(tracer, input)


def agent_span(tracer: NoOpTracer, input: str):
    with tracer.start_as_current_span("agent_span") as span:
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
        span.set_attribute("confident.span.input", json.dumps({"input": input}))
        span.set_attribute(
            "confident.span.output", json.dumps({"output": input})
        )
        span.set_attribute(
            "confident.trace.metadata", json.dumps({"test_key": "test_value"})
        )
        time.sleep(1)
        retriever_span(tracer=tracer, input=input)


def llm_agent(tracer: NoOpTracer, input: str):
    with tracer.start_as_current_span("llm_span") as span:
        span.set_attribute("confident.span.type", "llm")
        span.set_attribute(
            "confident.span.input",
            [json.dumps({"role": "user", "content": input})],
        )
        span.set_attribute(
            "confident.span.output",
            json.dumps({"role": "assistant", "content": input}),
        )
        span.set_attribute("confident.llm.model", "gpt-3.5-turbo")
        span.set_attribute("confident.llm.cost_per_input_token", 0.01)
        span.set_attribute("confident.llm.cost_per_output_token", 0.02)
        span.set_attribute("confident.llm.output_token_count", 10)
        span.set_attribute("confident.llm.input_token_count", 10)
        span.set_attribute("confident.trace.thread_id", "123")
        span.set_attribute("confident.trace.user_id", "456")
        time.sleep(1)
        agent_span(tracer=tracer, input=input)


def meta_agent(tracer: NoOpTracer, input: str):
    with tracer.start_as_current_span("custom_span") as span:
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
        span.set_attribute(
            "confident.span.metric_collection", "test_collection_1"
        )

        span.set_attribute("confident.trace.name", "test_trace")
        span.set_attribute("confident.trace.input", "test_input")
        span.set_attribute("confident.trace.output", "test_actual_output")
        span.set_attribute(
            "confident.trace.metric_collection", "test_collection_1"
        )
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
        time.sleep(1)
        llm_agent(tracer=tracer, input=input)


def build_dataset():
    goldens = [
        Golden(input="What's the weather like in SF?"),
        Golden(input="Tell me about Elon Musk."),
    ]
    return EvaluationDataset(goldens=goldens)


from deepeval.dataset.test_run_tracer import init_global_test_run_tracer


def run_with_otlp(api_key: str):
    provider, tracer = init_global_test_run_tracer(api_key)
    dataset = build_dataset()
    for golden in dataset.evals_iterator(run_otel=True):
        meta_agent(tracer, golden.input)
    provider.force_flush()


if __name__ == "__main__":
    # OTEL runs with both keys
    run_with_otlp(KEY_1)
    # run_with_otlp(KEY_2)
