# `deepeval.integrations`

Contributor reference for the framework integrations.

> Note: `deepeval.openai`, `deepeval.anthropic`, and `deepeval.openai_agents` live at the top level of the `deepeval` package, not under this folder. They're listed here so the matrix is complete.

## How an integration traces

Every integration runs on confident-trace. Its entry point calls `ensure_confident_trace(...)` (`deepeval/tracing/otel/setup.py`) with the confident-trace instrumentations it needs. Those instrumentations, or the framework's own OpenTelemetry telemetry that they switch on, produce the spans.

`DeepEvalSpanBridge` (`deepeval/tracing/otel/bridge.py`) collects the spans of every trace deepeval consumes: one inside `@observe` / `with trace(...)`, in `evals_iterator` or `evaluate`, or under a trace-shape test. It converts them into deepeval spans (`converter.py`), so metrics, `update_current_span(...)`, `update_current_trace(...)` and `next_*_span(...)` work the same as with `@observe`. confident-trace exports every other span to Confident AI.

| Integration   | Entry point                                       | confident-trace instrumentations                             | Adapter         |
| ------------- | ------------------------------------------------- | ------------------------------------------------------------ | --------------- |
| OpenAI        | `from deepeval.openai import OpenAI`              | `openai`                                                     | —               |
| Anthropic     | `from deepeval.anthropic import Anthropic`        | `anthropic`                                                  | —               |
| LangChain     | `CallbackHandler()`                               | `langchain`, `langgraph`                                     | —               |
| LangGraph     | `CallbackHandler()`                               | `langchain`, `langgraph`                                     | —               |
| LlamaIndex    | `instrument_llama_index()`                        | `llamaindex` and the model providers                         | `llamaindex`    |
| CrewAI        | `instrument_crewai()`                             | `crewai` and the model providers                             | —               |
| OpenAI Agents | `add_trace_processor(DeepEvalTracingProcessor())` | `openai_agents` (OpenInference), `openai`                    | `openinference` |
| Pydantic AI   | `DeepEvalInstrumentationSettings(...)`            | `pydantic_ai` (Pydantic AI's own telemetry)                  | `pydantic_ai`   |
| Strands       | `instrument_strands()`                            | `strands` (Strands' own telemetry)                           | `strands`       |
| AgentCore     | `instrument_agentcore()`                          | `agentcore`, `strands`                                       | `strands`       |
| Google ADK    | `instrument_google_adk()`                         | `google_adk` (ADK's own telemetry)                           | `google_adk`    |
| OpenInference | `instrument_openinference()` / `deepeval.instrument()` | none: pair it with any `openinference-instrumentation-*` | `openinference` |

## What an integration adds

confident-trace records what a framework does. An integration adds only what deepeval knows and the framework doesn't:

- **Trace settings** such as `name`, `thread_id` and `tags`. An integration that sees a run start stages them with `stage_next_trace(...)` (the LangChain handler). One that only configures instrumentation stamps them on its instrumentation scope's spans with `set_trace_defaults(...)` (Strands, ADK, OpenInference), or on one agent's tracer (Pydantic AI).
- **Span settings** such as metrics. For a span the user can't reach from inside, stage them before it opens with `stage_next_span(...)` / `staged_next_span(...)`, which a user's `next_*_span(...)` overrides (CrewAI's `Crew` / `Agent` / `LLM`, OpenAI Agents' `Agent`). Inside a span's body, such as a tool's, call `update_current_span(...)`.
- **Span type**, for a span only the integration can classify: `stage_next_span_type(...)` (LangChain types the invoked run as the agent).

## Writing an adapter

Most spans convert by the GenAI semantic conventions and `confident.*` attributes alone. When a framework records content elsewhere, add an `Adapter` (`deepeval/tracing/otel/adapters/base.py`) and register it in `ADAPTERS`, framework adapters ahead of the generic `openinference` one.

- `detect(span)` claims the framework's spans, usually by instrumentation scope.
- `start_span_type(span)` / `span_type(span)` classify spans the conventions don't. A declared `confident.span.type` still wins, and an adapter wins over the GenAI operation name.
- `span_fields(...)` and `trace_fields(...)` only fill fields the span didn't set.
- `self_providers` lists names a framework reports as the provider in place of the model's vendor.

Map fields the way the Confident AI backend does for the same framework, so a trace looks the same locally and on the platform. The deliberate difference is content that evaluation reads: model and agent output becomes its answer text, and agent input the latest user message.
