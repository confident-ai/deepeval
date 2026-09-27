# Endpoint and Exporter

Where to export OpenTelemetry traces and how to authenticate so they land in
Confident AI's Observatory.

## Endpoints

Confident AI exposes one OTLP/HTTP traces endpoint per region. There are
exactly two.

| Region | Base endpoint | Traces are POSTed to |
| --- | --- | --- |
| Default (US/AU) | `https://otel.confident-ai.com` | `https://otel.confident-ai.com/v1/traces` |
| EU | `https://eu.otel.confident-ai.com` | `https://eu.otel.confident-ai.com/v1/traces` |

When configuring an OTLP/HTTP span exporter directly, the `endpoint` value must
include the `/v1/traces` suffix. When configuring through the standard
`OTEL_EXPORTER_OTLP_ENDPOINT` environment variable, supply only the base
endpoint — the SDK appends `/v1/traces` itself.

## Choosing the Endpoint by API Key Region

Confident AI API keys are region-prefixed. Pick the endpoint from the prefix of
the project's `CONFIDENT_API_KEY`:

| API key prefix | Endpoint |
| --- | --- |
| `confident_eu_…` | `https://eu.otel.confident-ai.com` |
| `confident_us_…` | `https://otel.confident-ai.com` |
| anything else | `https://otel.confident-ai.com` |

Only `confident_eu_…` keys use the EU endpoint. When in doubt, ask the user for project region 
or use the default endpoint.

## Authentication

Every request must carry the Confident AI API key in an HTTP header:

```
x-confident-api-key: <CONFIDENT_API_KEY>
```

Set this as a header on the OTLP exporter. Read the key from the
`CONFIDENT_API_KEY` environment variable — never hardcode it into source.

## Transport: HTTP Only

Confident AI's OTLP endpoint accepts **OTLP/HTTP only — never gRPC.**

- Python: use the HTTP exporter `OTLPSpanExporter` from
  `opentelemetry.exporter.otlp.proto.http.trace_exporter` (package
  `opentelemetry-exporter-otlp-proto-http`). Do not use the
  `opentelemetry.exporter.otlp.proto.grpc` variant.
- OpenTelemetry Collector: use the `otlphttp` exporter, not `otlp` (gRPC).
- Other SDKs: choose the OTLP/HTTP exporter (`proto-http`, `HttpProtobuf`, or
  the language's equivalent).

## Standard OTel Environment Variables

The exporter honors standard OpenTelemetry environment variables, so the
endpoint and headers can be configured without code changes:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otel.confident-ai.com"
export OTEL_EXPORTER_OTLP_HEADERS="x-confident-api-key=<CONFIDENT_API_KEY>"
```

With env-var configuration the base endpoint is given; the SDK appends
`/v1/traces` automatically.

## Exporter Wiring (Python)

The minimal path:

1. Create a `TracerProvider`.
2. Attach a `BatchSpanProcessor` wrapping an OTLP/HTTP `OTLPSpanExporter`
   pointed at `<endpoint>/v1/traces` with the `x-confident-api-key` header.
3. Register the provider as the global tracer provider.
4. Get a tracer, start spans, and set `confident.*` attributes on them.

```python
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

api_key = os.environ["CONFIDENT_API_KEY"]
endpoint = (
    "https://eu.otel.confident-ai.com"
    if api_key.startswith("confident_eu_")
    else "https://otel.confident-ai.com"
)

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces",
            headers={"x-confident-api-key": api_key},
        )
    )
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("my-llm-app") as span:
    span.set_attribute("confident.span.type", "agent")
    span.set_attribute("confident.trace.name", "my-llm-app")
    # ... see trace-attributes.md and span-attributes.md
```

See `templates/confident_otel_setup.py` for the full runnable version.

## Other Languages

The wiring shape is identical in any OpenTelemetry SDK — only the class and
package names differ. In every language:

1. Construct an OTLP/HTTP span exporter.
2. Set its `endpoint`/`url` to `<region endpoint>/v1/traces`.
3. Add the `x-confident-api-key` header.
4. Register it with a batch span processor on the tracer provider.

Then emit spans as usual and set the `confident.*` attributes documented in
`references/trace-attributes.md` and `references/span-attributes.md`. The
attribute keys are the entire contract — they are the same regardless of
language.

## Export Only AI Spans (Isolate From Other Instrumentation)

Most real applications run **more OpenTelemetry instrumentation than just the
AI code**. Auto-instrumentation libraries and APM agents (Datadog, New Relic,
Grafana, OpenTelemetry auto-instrumentation, etc.) emit spans for HTTP
requests, database queries, cache calls, outbound network calls, and
framework internals. This happens in any runtime — Node.js, Python, Java, Go,
and others — not just one.

If the Confident AI exporter shares a tracer provider or processor pipeline
with that instrumentation, **all of those unrelated spans get shipped to
Confident AI's Observatory**, where they bury the AI trace in non-AI noise.
Confident AI's Observatory is for AI behavior; only AI spans belong there.

**Rule: the Confident AI export pipeline must carry AI spans only.** Use one of
two approaches.

### Approach 1 — Dedicated pipeline (preferred when feasible)

Register the Confident AI exporter on a tracer provider / processor used
**only** by the AI instrumentation, separate from the global provider that
auto-instrumentation and APM agents feed. Create AI spans with that dedicated
provider's tracer. Non-AI spans never reach the Confident AI exporter because
they were never on its pipeline.

### Approach 2 — Filter the pipeline

When AI spans and other spans unavoidably share a provider (common when an AI
framework emits its spans onto the global provider), wrap the Confident
AI-bound processor or exporter in a **filter that forwards only AI spans and
drops everything else**.

Identify an AI span by any of:

- it has a `confident.span.type` attribute set; or
- it carries `gen_ai.*` semantic-convention attributes; or
- its span name matches a known AI-framework prefix (for example, the Vercel
  AI SDK emits spans named `ai.*`).

Implement the filter as either:

- a **span processor** that no-ops `onStart`/`onEnd` for non-AI spans, so only
  AI spans are handed to the underlying Confident AI processor; or
- an **exporter wrapper** that removes non-AI spans from each batch before
  calling the real OTLP exporter.

A working reference implementation is the deepeval TypeScript SDK's
`DeepEvalBatchFilterProcessor` (a name-prefix span-processor filter) and
`DeepEvalExporterWrapper` (an exporter wrapper) in
`deepeval.ts/src/integrations/ai-sdk/index.ts`. Mirror that shape in whatever
language and SDK the app uses.

**Caveat — preserve span nesting when filtering.** Dropping an intermediate
non-AI span can orphan its AI child spans (their `parentSpanId` now points at a
span that was never exported). When filtering, re-parent orphaned AI spans onto
the nearest exported ancestor, or strip the dangling parent reference so the
child becomes a clean root. The `DeepEvalExporterWrapper` above does exactly
this for root spans.
