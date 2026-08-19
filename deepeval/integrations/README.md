# `deepeval.integrations`

Contributor reference for the framework integrations. Each integration plugs deepeval's tracing / evaluation into a third-party framework using one of four mechanisms.

> Note: `deepeval.openai`, `deepeval.anthropic`, `deepeval.openrouter`, and `deepeval.openai_agents` live at the top level of the `deepeval` package, not under this folder. They're listed here so the matrix is complete.

## Integration matrix

Capability columns:

- **Bare** — calling the framework directly without an enclosing `@observe` / `with trace(...)` produces a trace in Confident AI. Each integration auto-creates a trace on first activity (callback fire, OTel root span, internal `@observe` wrap on the native client, etc.).
- **`@observe` / `with trace(...)`** — when wrapped, the integration's spans flow into deepeval's native trace context: `update_current_trace(...)` / `update_current_span(...)` work anywhere in the call stack, single REST POST per trace, no UUID-reconciliation needed.
- **`evals_iterator`** — works inside `dataset.evals_iterator(...)`, both end-to-end (`metrics=[...]` on the iterator) and component-level (`@observe(metrics=[...])` on a span). For OTel-mode integrations, `ContextAwareSpanProcessor` flips to REST routing automatically when `trace_manager.is_evaluating` is True so spans flow through `trace_manager` instead of OTLP.
- **`deepeval test run`** — works under the pytest tracing-eval entry point (`@assert_test`, `@generate_trace_json`, `@assert_trace_json`).

| Integration   | Mode                              | Entry point                                       | Bare | `@observe` / `with trace()` | `evals_iterator` | `deepeval test run` | Source                               |
| ------------- | --------------------------------- | ------------------------------------------------- | :--: | :-------------------------: | :--------------: | :-----------------: | ------------------------------------ |
| OpenAI        | Native client wrapper             | `from deepeval.openai import OpenAI`              | Yes  | Yes                         | Yes              | Yes                 | `deepeval/openai/`                   |
| Anthropic     | Native client wrapper             | `from deepeval.anthropic import Anthropic`        | Yes  | Yes                         | Yes              | Yes                 | `deepeval/anthropic/`                |
| OpenRouter    | Native client wrapper             | `from deepeval.openrouter import OpenRouter`      | Yes  | Yes                         | Yes              | Yes                 | `deepeval/openrouter/`               |
| LangChain     | Callback handler                  | `CallbackHandler()`                               | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/langchain/`   |
| LangGraph     | Callback handler (LangChain's)    | `CallbackHandler()`                               | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/langchain/`   |
| LlamaIndex    | Event handler                     | `instrument_llama_index()`                        | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/llama_index/` |
| CrewAI        | Event listener + wrapper classes  | `instrument_crewai()`                             | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/crewai/`      |
| OpenAI Agents | Trace processor + agent wrapper   | `add_trace_processor(DeepEvalTracingProcessor())` | Yes  | Yes                         | Yes              | Yes                 | `deepeval/openai_agents/`            |
| AgentCore     | OpenTelemetry                     | `instrument_agentcore()`                          | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/agentcore/`   |
| Strands       | OpenTelemetry                     | `instrument_strands()`                            | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/strands/`     |
| Google ADK    | OpenTelemetry (via OpenInference) | `instrument_google_adk()`                         | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/google_adk/`  |
| Pydantic AI   | OpenTelemetry                     | `DeepEvalInstrumentationSettings(...)`            | Yes  | Yes                         | Yes              | Yes                 | `deepeval/integrations/pydantic_ai/` |

> Every cell is Yes because of the recent OTel POC migrations: native-client / callback-handler / event-listener / trace-processor integrations were already feature-complete via direct `trace_manager` access, and the four OTel-mode integrations (Pydantic AI, AgentCore, Google ADK, Strands) now follow the same SpanInterceptor + `ContextAwareSpanProcessor` pattern[^poc] so their spans behave identically across all four surfaces. New integrations should target the same parity.

[^poc]: Each OTel-mode `SpanInterceptor` reads trace-level metadata from `current_trace_context` per span (instead of baking it at `instrument_*()` time) and pushes a `BaseSpan` placeholder onto `current_span_context` for each OTel span's lifetime so `update_current_span(...)` from anywhere lands in `confident.span.*` attributes at `on_end`. The `ContextAwareSpanProcessor` (`deepeval/tracing/otel/context_aware_processor.py`) routes spans to REST when a deepeval trace context is active or an evaluation is running, OTLP otherwise.

## Mode reference

- **Native client wrapper** — drop-in replacement for the vendor SDK's client class (e.g. `deepeval.openai.OpenAI` instead of `openai.OpenAI`). Spans are built directly via `trace_manager`. Lowest friction, but only covers calls that go through that client.

## Provider vs. integration on LLM spans

Two distinct fields, and they are not interchangeable:

- **`integration`** — the SDK deepeval instrumented (`Integration`).
- **`provider`** — who actually served the request (`Provider`).

They diverge whenever an SDK is pointed at a gateway. The OpenAI SDK is the
standard way to reach OpenAI-compatible gateways, so the class being patched
says nothing about the provider — only the client's base URL does.
`deepeval.openai` therefore resolves the provider per call, in this order:

1. An explicit `LlmSpanContext(provider=...)` set by the user.
2. A recognized gateway host, from the table in
   `deepeval/model_integrations/utils.py` (`openrouter.ai` today).
3. `Provider.OPEN_AI`.

So a call to `anthropic/claude-sonnet-4.5` routed through OpenRouter records
`integration="OpenAI"`, `provider="OpenRouter"`. Teach `deepeval.openai` about
another gateway by adding a host to that table; users can always label an
unrecognized one themselves via `LlmSpanContext(provider=...)`.

Gateway-specific response detail that has no first-class span field is merged
into the span's `metadata` dict under a provider-owned key rather than
assigned, so it never clobbers metadata the user set. OpenRouter writes
`metadata["openrouter"]` with the generation id, cost, cached/reasoning token
counts and (native SDK only) routing detail.
- **Callback handler / event listener** — registers with the framework's own callback or event API (LangChain `BaseCallbackHandler`, LlamaIndex `BaseEventHandler`, CrewAI `BaseEventListener`, etc.). Spans are built directly via `trace_manager`. Covers all calls the framework dispatches through that surface — no need to swap clients.
- **Trace processor** — for frameworks that already have their own tracing pipeline (OpenAI Agents SDK), we plug into it as a processor and translate events into deepeval spans.
- **OpenTelemetry** — registers an OTel `SpanProcessor` against the global `TracerProvider`. The framework (or a community-maintained instrumentor like `openinference-instrumentation-google-adk`) emits OTel spans; deepeval translates them into Confident span attributes and ships them via OTLP.

## Transport reference

- **REST** — `trace_manager` posts the full trace to `api.confident-ai.com/v1/traces` once per trace.
- **OTLP** — `BatchSpanProcessor` flushes OTel spans to `otel.confident-ai.com/v1/traces` on a timer / queue threshold.

## OpenInference (generic OTel backend for community instrumentors)

`deepeval/integrations/openinference/` is the SpanInterceptor + processor wiring shared by Google ADK and any other community-maintained OpenInference instrumentor. It sets up the `TracerProvider`, registers `OpenInferenceSpanInterceptor` (translates OpenInference semantic-convention attributes — `openinference.span.kind`, `llm.input_messages.{idx}`, `llm.output_messages.{idx}`, `tool.name`, `llm.token_count.*` — into `confident.span.*`), and routes spans through `ContextAwareSpanProcessor` (REST or OTLP).

It is exposed at the top level as `deepeval.instrument(...)` so users can pair it with any OpenInference instrumentor directly:

```python
import deepeval
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

deepeval.instrument(name="my-app", environment="development")
GoogleADKInstrumentor().instrument()
```

`instrument_google_adk(...)` is just a convenience wrapper that calls `GoogleADKInstrumentor().instrument()` then `deepeval.instrument(...)` for you.

AgentCore, Strands, and Pydantic AI do NOT delegate here — they have their own SpanInterceptors (`AgentCoreSpanInterceptor`, `StrandsSpanInterceptor`, `PydanticAISpanInterceptor`). AgentCore and Strands both read OTel GenAI semconv (`gen_ai.*`) attributes — Strands emits these natively, and AgentCore picks them up from the Strands runtime AWS Bedrock typically runs under, plus Traceloop / AWS Bedrock fallbacks; Pydantic AI uses its own logfire-shaped attrs. All four interceptors share the same processor wiring and the same `ContextAwareSpanProcessor` for routing.

## Mixing OTel-mode with `@observe`

When an OTel-mode integration runs inside an active `@observe` / `with trace(...)` context, the OTel span interceptor synchronizes the trace UUID (`current_trace_context.uuid = OTel trace_id`) so both transports land on the same trace server-side.

For all OTel-mode integrations above, `ContextAwareSpanProcessor` automatically routes the OTel spans through `ConfidentSpanExporter` (REST) when a deepeval trace context is active or an evaluation is running — so a mixed trace produces a single REST POST and `update_current_trace(...)` / `update_current_span(...)` from anywhere in the call stack land on the right span. Pydantic AI is the reference implementation; AgentCore, Strands, and Google ADK (the latter via the shared `openinference/` backend) follow the same pattern.
