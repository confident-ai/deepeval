# `deepeval/integrations` (TypeScript)

Contributor reference for the framework integrations.

> Note: `deepeval/openai` lives at `src/openai/`, not under this folder. It's listed here so the table is complete.

## How an integration traces

Every integration runs on confident-trace's manual adapters: no preload or automatic instrumentation is involved. Its entry point calls `ensureConfidentTrace()` (`src/tracing/otel/setup.ts`), which starts confident-trace once with deepeval's `DeepEvalSpanBridge` as an extra span processor, then attaches the adapter for its framework.

`observe(...)` runs on confident-trace too: each observed function is a confident-trace span.

The bridge (`src/tracing/otel/bridge.ts`) collects the spans of every trace deepeval consumes: one inside `observe(...)` or `setTracingContext(...)`, in `evalsIterator(...)` or `toPass(...)`, or under a trace-capture sink. It converts them into deepeval spans (`converter.ts`), so metrics, `updateCurrentSpan(...)`, `updateCurrentTrace(...)` and `next*Span(...)` work the same as with `observe`. confident-trace exports every trace to Confident AI, except one an evaluation or a trace-capture sink consumes locally.

| Integration   | Entry point                                                                      | confident-trace adapter                            |
| ------------- | -------------------------------------------------------------------------------- | -------------------------------------------------- |
| OpenAI        | `instrumentOpenAI(client)`                                                       | `confident-trace/openai`                           |
| LangChain     | `new DeepEvalCallbackHandler()` in `callbacks`                                   | `confident-trace/langchain`                        |
| LangGraph     | `new DeepEvalCallbackHandler()` in `callbacks`                                   | `confident-trace/langchain`                        |
| OpenAI Agents | `addTraceProcessor(new DeepEvalTracingProcessor())`                              | `confident-trace/openai-agents`                    |
| AI SDK 7      | `new OpenTelemetry({ tracer: configureAiSdkTracing() })`                         | `confident-trace/vercel-ai`                        |
| Mastra        | `new DeepEvalExporter()` in Mastra's observability exporters                     | `confident-trace/mastra`                           |
| OpenInference | `instrumentOpenInference()`, then any `@arizeai/openinference-instrumentation-*` | none: `src/tracing/otel/adapters/openinference.ts` |

## What an integration adds

confident-trace records what a framework does. An integration adds only what deepeval knows and the framework doesn't:

- **Trace settings** such as `name`, `threadId` and `tags`. An integration that sees a run start stages them with `withStaged({ trace })` around the adapter call that opens the run's span (LangChain, OpenAI Agents). One that only configures instrumentation stamps them on its spans with `setTraceDefaults(...)` (AI SDK, OpenInference).
- **Span settings** such as a model's metrics. confident-trace's callback adapters open their spans synchronously, so `withStaged({ span, spanKind })` around the adapter call reaches that span; a user's `next*Span(...)` still wins.
- **Spans that only arrive once they end** (Mastra): `collectFinishedTrace(...)` builds the deepeval trace from a whole finished trace.

## Writing an adapter

Most spans convert by the GenAI semantic conventions and `confident.*` attributes alone. When a framework records content elsewhere, add an `Adapter` (`src/tracing/otel/adapters/base.ts`) and register it in `ADAPTERS`.

- `detect(span)` claims the framework's spans.
- `spanType(span, atStart)` classifies spans the conventions don't. A declared `confident.span.type` still wins, and an adapter wins over the GenAI operation name.
- `spanFields(...)` and `traceFields(...)` only fill fields the span didn't set.

Map fields the way the Confident AI backend does for the same framework, so a trace looks the same locally and on the platform. The deliberate difference is content that evaluation reads: model and agent output becomes its answer text, and agent input the latest user message.
