# `deepeval.integrations`

Contributor reference for the framework integrations. Each integration plugs deepeval's tracing / evaluation into a third-party framework using one of four mechanisms.

> Note: `deepeval.openai`, `deepeval.anthropic`, and `deepeval.openai_agents` live at the top level of the `deepeval` package, not under this folder. They're listed here so the matrix is complete.

## Integration matrix

The "Mixes with `@observe`?" column tracks whether the integration's spans flow into deepeval's native trace context — i.e. whether `@observe`, `with trace(...)`, `update_current_trace(...)`, and `update_current_span(...)` work transparently for that integration's spans, anywhere in the call stack (Langfuse-style).

- **Good** — trace metadata flows from `update_current_trace(...)`, span metadata flows from `update_current_span(...)`, and spans land in `trace_manager` so pytest tracing evals + `evals_iterator` work. Single REST POST per trace.
- **Bad** — only the trace UUID syncs between transports. Per-call metadata is locked at `instrument_*()` time; `update_current_span(...)` from inside a tool / callback is a no-op. Dual-posts (REST from `@observe` + OTLP from the integration) reconciled server-side by UUID.

| Integration     | Mode                              | Entry point                                       | Transport                                    | Mixes with `@observe`?                                      | Source                               |
| --------------- | --------------------------------- | ------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------- | ------------------------------------ |
| OpenAI          | Native client wrapper             | `from deepeval.openai import OpenAI`              | REST                                         | Good                                                        | `deepeval/openai/`                   |
| Anthropic       | Native client wrapper             | `from deepeval.anthropic import Anthropic`        | REST                                         | Good                                                        | `deepeval/anthropic/`                |
| LangChain       | Callback handler                  | `CallbackHandler()`                               | REST                                         | Good                                                        | `deepeval/integrations/langchain/`   |
| LangGraph       | Callback handler (LangChain's)    | `CallbackHandler()`                               | REST                                         | Good                                                        | `deepeval/integrations/langchain/`   |
| LlamaIndex      | Event handler                     | `instrument_llama_index()`                        | REST                                         | Good                                                        | `deepeval/integrations/llama_index/` |
| CrewAI          | Event listener + wrapper classes  | `instrument_crewai()`                             | REST                                         | Good                                                        | `deepeval/integrations/crewai/`      |
| OpenAI Agents   | Trace processor + agent wrapper   | `add_trace_processor(DeepEvalTracingProcessor())` | REST                                         | Good                                                        | `deepeval/openai_agents/`            |
| AgentCore       | OpenTelemetry                     | `instrument_agentcore()`                          | OTLP                                         | Bad &mdash; needs the Pydantic AI POC pattern applied[^poc] | `deepeval/integrations/agentcore/`   |
| Google ADK      | OpenTelemetry (via OpenInference) | `instrument_google_adk()`                         | OTLP                                         | Bad &mdash; needs the Pydantic AI POC pattern applied[^poc] | `deepeval/integrations/google_adk/`  |
| **Pydantic AI** | **OpenTelemetry**                 | **`DeepEvalInstrumentationSettings(...)`**        | **REST when context active, OTLP otherwise** | **Good (POC)**                                              | `deepeval/integrations/pydantic_ai/` |

[^poc]: The Pydantic AI integration is the reference for OTel-mode "Good" behavior. Its `SpanInterceptor` reads trace-level metadata from `current_trace_context` per span (instead of baking it at `instrument_*()` time) and pushes a `BaseSpan` placeholder onto `current_span_context` for each OTel span's lifetime so `update_current_span(...)` from anywhere lands in `confident.span.*` attributes at `on_end`. The `ContextAwareSpanProcessor` (`deepeval/tracing/otel/context_aware_processor.py`) routes spans to REST when a deepeval trace context is active or an evaluation is running, OTLP otherwise. Apply the same pattern to `agentcore/` and `openinference/` (which Google ADK delegates to) to flip those rows to Good.

## Mode reference

- **Native client wrapper** — drop-in replacement for the vendor SDK's client class (e.g. `deepeval.openai.OpenAI` instead of `openai.OpenAI`). Spans are built directly via `trace_manager`. Lowest friction, but only covers calls that go through that client.
- **Callback handler / event listener** — registers with the framework's own callback or event API (LangChain `BaseCallbackHandler`, LlamaIndex `BaseEventHandler`, CrewAI `BaseEventListener`, etc.). Spans are built directly via `trace_manager`. Covers all calls the framework dispatches through that surface — no need to swap clients.
- **Trace processor** — for frameworks that already have their own tracing pipeline (OpenAI Agents SDK), we plug into it as a processor and translate events into deepeval spans.
- **OpenTelemetry** — registers an OTel `SpanProcessor` against the global `TracerProvider`. The framework (or a community-maintained instrumentor like `openinference-instrumentation-google-adk`) emits OTel spans; deepeval translates them into Confident span attributes and ships them via OTLP.

## Transport reference

- **REST** — `trace_manager` posts the full trace to `api.confident-ai.com/v1/traces` once per trace.
- **OTLP** — `BatchSpanProcessor` flushes OTel spans to `otel.confident-ai.com/v1/traces` on a timer / queue threshold.

## OpenInference (shared OTel backend)

All three OTel-mode integrations sit on top of `deepeval/integrations/openinference/`, which sets up the `TracerProvider`, registers the `OpenInferenceSpanInterceptor` (translates OpenInference / gen_ai semconv attributes into `confident.span.*`), and wires the OTLP exporter. It is also exposed at the top level as `deepeval.instrument(...)` so users can pair it with any OpenInference instrumentor directly:

```python
import deepeval
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

deepeval.instrument(name="my-app", environment="development")
GoogleADKInstrumentor().instrument()
```

`instrument_google_adk(...)` is just a convenience wrapper that calls `GoogleADKInstrumentor().instrument()` then `deepeval.instrument(...)` for you.

## Mixing OTel-mode with `@observe`

When an OTel-mode integration runs inside an active `@observe` / `with trace(...)` context, the OTel span interceptor synchronizes the trace UUID (`current_trace_context.uuid = OTel trace_id`) so both transports land on the same trace server-side.

For integrations marked **Good** above, `ContextAwareSpanProcessor` automatically routes the OTel spans through `ConfidentSpanExporter` (REST) when a deepeval trace context is active or an evaluation is running — so a mixed trace produces a single REST POST and `update_current_trace(...)` / `update_current_span(...)` from anywhere in the call stack land on the right span. Pydantic AI is the reference implementation; apply the same pattern (see footnote in the matrix) to migrate AgentCore and OpenInference (Google ADK).

For integrations marked **Bad**, the trace UUID syncs but per-call metadata is locked at `instrument_*()` time and the integration produces dual posts (one REST POST from the `@observe` trace flush + one or more OTLP POSTs from the integration's spans) that the backend reconciles by UUID.
