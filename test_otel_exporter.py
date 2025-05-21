import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.status import Status, StatusCode
import deepeval
from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from keyy import CONFIDENT_API_KEY

# Setup OpenTelemetry tracer - handle existing TracerProvider
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
else:
    tracer_provider = trace.get_tracer_provider()

deepeval.login_with_confident_api_key(CONFIDENT_API_KEY)

# Create confident span exporter
exporter = ConfidentSpanExporter()

span_processor = BatchSpanProcessor(exporter)
tracer_provider.add_span_processor(span_processor)
tracer = trace.get_tracer("test_tracer")

with tracer.start_as_current_span("chat gpt-4o") as parent_span:
    parent_span.set_attribute(
        "gen_ai.operation.name", "chat"
    )  # can be chat, text_completion, generate_content
    parent_span.set_attribute("gen_ai.request.model", "gpt-4o")
    parent_span.set_attribute("gen_ai.usage.input_tokens", 10)
    parent_span.set_attribute("gen_ai.usage.output_tokens", 20)
    parent_span.set_attribute("confident.llm.cost_per_input_token", 0.0001)
    parent_span.set_attribute("confident.llm.cost_per_output_token", 0.0002)

    # input messages will be events in the span
    # format should remain same
    parent_span.add_event(
        "gen_ai.system.message", {"content": "you are a helpful assistant"}
    )
    parent_span.add_event(
        "gen_ai.user.message", {"content": "What is the capital of France?"}
    )

    # output should be a single event per llmspan
    parent_span.add_event(
        "gen_ai.assistant.message",
        {"content": "The capital of France is Paris."},
    )  # can be gen_ai.assistant.message, gen_ai.choice, gen_ai.tool.message

    with tracer.start_as_current_span("execute_tool example_tool") as tool_span:
        tool_span.set_attribute("gen_ai.operation.name", "execute_tool")
        tool_span.set_attribute(
            "gen_ai.tool.description", "example_tool_description"
        )

        # input and output must be a single event per toolspan
        tool_span.add_event(
            "confident.tool.input", {"temp": "example_tool_input"}
        )
        tool_span.add_event(
            "confident.tool.output", {"temp": "example_tool_output"}
        )

        with tracer.start_as_current_span(
            "create_agent example_agent"
        ) as agent_span:
            agent_span.set_attribute(
                "gen_ai.operation.name", "create_agent"
            )  # can be create_agent or invoke_agent
            agent_span.set_attribute(
                "gen_ai.agent.description", "example_agent_description"
            )

            agent_span.set_attribute(
                "confident.agent.available_tools", ["example_tool"]
            )  # should be list of tool names
            agent_span.set_attribute(
                "confident.agent.agent_handoffs", ["example_handoff"]
            )  # should be list of handoff names

            agent_span.add_event(
                "confident.agent.input", {"input": "example_input"}
            )
            agent_span.add_event(
                "confident.agent.output", {"output": "example_output"}
            )

            with tracer.start_as_current_span(
                "embeddings openai_ada"
            ) as retriever_span:
                retriever_span.set_attribute(
                    "gen_ai.operation.name", "embeddings"
                )
                retriever_span.set_attribute(
                    "gen_ai.request.model", "openai_ada"
                )
                retriever_span.set_attribute("confident.retriever.top_k", 10)
                retriever_span.set_attribute(
                    "confident.retriever.chunk_size", 100
                )

                # input and output must be a single event per retriever span
                retriever_span.add_event(
                    "confident.retriever.input", {"input": "example_input"}
                )
                retriever_span.add_event(
                    "confident.retriever.output", {"output": "example_output"}
                )

                retriever_span.set_status(Status(StatusCode.OK))
            agent_span.set_status(Status(StatusCode.OK))
        tool_span.set_status(Status(StatusCode.OK))
    parent_span.set_status(Status(StatusCode.OK))

time.sleep(10)
