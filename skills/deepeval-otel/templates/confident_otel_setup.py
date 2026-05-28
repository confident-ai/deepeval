"""Raw OpenTelemetry -> Confident AI Observatory: minimal setup + example trace.

Requires:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
    export CONFIDENT_API_KEY="<your Confident AI API key>"

This template wires an OTLP/HTTP span exporter to Confident AI and emits one
example trace that demonstrates the `confident.*` attribute and data-type
contract. Run it directly to smoke-test the connection:

    python confident_otel_setup.py

PLACEHOLDER: replace the example span/trace attribute values below with values
from the real application before using this as production instrumentation.
"""

import json
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.trace import Status, StatusCode


def pick_endpoint(api_key: str) -> str:
    """Select the Confident AI OTLP endpoint from the API key region prefix.

    Only `confident_eu_...` keys use the EU endpoint; every other key
    (`confident_us_...`, `confident_au_...`, or anything else) uses the default.
    """
    if api_key.startswith("confident_eu_"):
        return "https://eu.otel.confident-ai.com"
    return "https://otel.confident-ai.com"


def configure_tracing() -> trace.Tracer:
    """Wire a TracerProvider that exports to Confident AI over OTLP/HTTP."""
    api_key = os.environ.get("CONFIDENT_API_KEY")
    if not api_key:
        raise SystemExit(
            "CONFIDENT_API_KEY is not set. Export it before running:\n"
            '    export CONFIDENT_API_KEY="<your Confident AI API key>"'
        )

    endpoint = pick_endpoint(api_key)
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                # The exporter endpoint MUST include the /v1/traces suffix.
                # Confident AI accepts OTLP/HTTP only -- never gRPC.
                endpoint=f"{endpoint}/v1/traces",
                headers={"x-confident-api-key": api_key},
            )
        )
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


def run_example(tracer: trace.Tracer) -> None:
    """Emit one example trace: an agent span wrapping a child LLM span."""
    # Root span. Trace-level attributes (confident.trace.*) can be set on any
    # span; the root is the natural place. Child spans nest automatically
    # because they open inside this `with` block (native OTel span context).
    with tracer.start_as_current_span("support-agent") as root:
        root.set_attribute("confident.span.type", "agent")
        root.set_attribute("confident.agent.name", "support-agent")
        root.set_attribute("confident.span.input", "Where is my order?")

        # Trace-level attributes.
        root.set_attribute("confident.trace.name", "support-chat")
        root.set_attribute("confident.trace.input", "Where is my order?")
        # String lists are native OTLP arrays.
        root.set_attribute("confident.trace.tags", ["support", "example"])
        # Dicts/metadata MUST be JSON-encoded strings (OTLP has no map type).
        root.set_attribute(
            "confident.trace.metadata",
            json.dumps({"app_version": "1.0.0", "route": "order_status"}),
        )

        # Child LLM span.
        with tracer.start_as_current_span("chat-completion") as llm:
            llm.set_attribute("confident.span.type", "llm")
            llm.set_attribute("confident.llm.model", "gpt-4o")
            llm.set_attribute("confident.llm.input_token_count", 42)
            llm.set_attribute("confident.llm.output_token_count", 18)
            llm.set_attribute(
                "confident.span.metadata",
                json.dumps({"temperature": 0.2}),
            )

            try:
                answer = "Your order ships tomorrow."
                llm.set_attribute("confident.span.output", answer)
            except Exception as exc:  # pragma: no cover - example only
                # Span errors use native OTel Status, not a confident.* attr.
                llm.set_status(Status(StatusCode.ERROR), str(exc))
                llm.record_exception(exc)
                raise

        root.set_attribute("confident.span.output", answer)
        root.set_attribute("confident.trace.output", answer)


if __name__ == "__main__":
    tracer = configure_tracing()
    run_example(tracer)
    # Flush so the batch processor exports before the process exits.
    trace.get_tracer_provider().shutdown()
    print("Trace exported. Check the Confident AI Observatory.")
