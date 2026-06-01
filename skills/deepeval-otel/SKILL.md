---
name: deepeval-otel
description: >
  Export raw OpenTelemetry traces from an AI application to Confident AI's
  Observatory. TRIGGER when the user wants to send OpenTelemetry or OTLP
  traces/spans from an LLM app, agent, RAG pipeline, or chatbot to Confident
  AI; configure the Confident AI OTLP endpoint; set confident.span.* or
  confident.trace.* attributes; export AI-app traces to Confident AI without
  the deepeval Python package; wire an OTLPSpanExporter, OpenTelemetry
  Collector, or vendor-neutral OTel SDK to Confident AI; or pick the US vs EU
  Confident AI OTLP endpoint. Language-agnostic — the mechanism is OTLP
  attribute keys plus an exporter endpoint. DO NOT TRIGGER for building
  DeepEval pytest eval suites, datasets, goldens, metrics, or deepeval test run
  (use the `deepeval` skill); for instrumenting with the DeepEval SDK's
  @observe decorator or framework integrations (use the `deepeval-tracing`
  skill); or for instrumenting non-AI software such as web servers, CRUD
  backends, or infrastructure — the confident.* attributes describe AI
  components (agents, LLM calls, retrievers, tools) and apply to AI
  applications only.
license: Apache-2.0
metadata:
  author: Confident AI
  version: "1.0.0"
  category: observability
  tags: "opentelemetry, otel, otlp, tracing, confident-ai, observatory, spans"
  compatibility: "Works with any OpenTelemetry SDK in any language. Requires a Confident AI account and `CONFIDENT_API_KEY`. Confident AI's OTLP endpoint is HTTP only — use OTLP/HTTP, not gRPC. Python examples assume `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http`."
---

# DeepEval OpenTelemetry Export

Use this skill to instrument an **AI application** — an LLM app, agent, RAG
pipeline, or chatbot — with **raw OpenTelemetry** so its traces land in
**Confident AI's Observatory**. No `deepeval` package is needed — it works with
any OTLP-capable OpenTelemetry SDK. The job is exactly two things: export to
the correct Confident AI OTLP endpoint, and set the `confident.*` attributes
Confident AI reads off each span.

## Scope: AI Applications Only

This skill instruments **AI applications only**. The `confident.*` attributes
and span types — `agent`, `llm`, `retriever`, `tool` — describe AI components,
and Confident AI's Observatory is built to evaluate and monitor AI behavior.

Instrument only the AI parts of the system: agent loops and planning, LLM
calls, retrieval / vector search, and tool calls. Do **not** apply `confident.*`
attributes to non-AI software (web servers, CRUD backends, database layers,
infrastructure) or to non-AI spans inside an otherwise-AI app — that data does
not belong in Confident AI and will not render meaningfully. If the target has
no LLM, agent, retrieval, or tool-calling component, this skill does not apply.

## When to Use vs the `deepeval` Skill

Use **this skill** for vendor-neutral OTLP export to Confident AI — pointing an
OpenTelemetry exporter at Confident AI and setting `confident.*` attributes.

Use the **`deepeval` skill** when the user wants to build a Python pytest eval
suite, generate datasets or goldens, write metrics, run `deepeval test run`, or
instrument with the `deepeval` SDK's `@observe` decorator. The two skills are
complementary, not alternatives.

## Prerequisites

- A Confident AI account and a `CONFIDENT_API_KEY`.
- An OpenTelemetry SDK for the application's language. For Python:
  `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http`.
- The Confident AI OTLP endpoint accepts **HTTP only** — never gRPC.

## How It Works

Confident AI exposes an OTLP/HTTP traces endpoint. Point any OpenTelemetry span
exporter at it with the `x-confident-api-key` header. Confident AI's exporter
then reads `confident.*` attributes off each span to build the trace and span
structure. Parent/child nesting comes from native OpenTelemetry span context,
not from any attribute.

## Workflow

1. Confirm the target is an AI application (it has LLM calls, an agent loop,
   retrieval, or tool calls). If it has none of these, stop — this skill does
   not apply. Then inspect for an existing OpenTelemetry setup (a
   `TracerProvider`, span exporters, or an OpenTelemetry Collector) and prefer
   repointing what exists over adding a parallel pipeline.
2. Choose the endpoint from the API key's region prefix. Read
   `references/endpoint-and-exporter.md`.
3. Wire (or repoint) an OTLP/HTTP span exporter with the `x-confident-api-key`
   header. For Python, start from `templates/confident_otel_setup.py`.
4. If the process runs other OpenTelemetry instrumentation or an APM agent
   (auto-instrumentation for HTTP/DB, Datadog, etc.), isolate the Confident AI
   export so only AI spans reach it — a dedicated pipeline or a span filter.
   Read "Export Only AI Spans" in `references/endpoint-and-exporter.md`.
5. Set `confident.span.*` attributes on spans; set `confident.trace.*` for
   trace-wide fields. Read `references/span-attributes.md` and
   `references/trace-attributes.md`.
6. Honor the OTLP data-type rules: JSON-encode dicts/metadata, use native
   arrays for string lists. See the Data-Type Rules in `span-attributes.md`.
7. If the app already emits OpenTelemetry GenAI semantic conventions, read
   `references/gen-ai-fallbacks.md` before adding redundant attributes.
8. Verify traces appear in the Confident AI Observatory.

## Core Principles

1. Instrument AI components only — agent, LLM, retriever, and tool spans.
   Never apply `confident.*` attributes to non-AI software or non-AI spans.
2. Export only AI spans. If the process has other OpenTelemetry
   instrumentation or an APM agent, isolate the Confident AI pipeline (a
   dedicated provider or a span filter) so non-AI spans — HTTP requests, DB
   queries, infra — are never exported to Confident AI.
3. Prefer repointing an existing OTLP exporter over adding a parallel one.
4. The `confident.*` attribute keys are the entire contract — they are the
   same in every language, so language choice is irrelevant.
5. Always use OTLP/HTTP. Confident AI's endpoint does not accept gRPC.
6. Honor OTLP data-type rules: attribute values must be primitives or
   homogeneous primitive lists; JSON-encode dicts and metadata.
7. Set `confident.span.type` explicitly when it is known; rely on `gen_ai.*`
   inference only as a fallback.
8. Never put secrets, credentials, or raw sensitive data into span attributes.

## References

| Topic | File |
| --- | --- |
| Endpoints, region selection, auth, exporter wiring | `references/endpoint-and-exporter.md` |
| Trace-level `confident.trace.*` attributes | `references/trace-attributes.md` |
| Span-level `confident.span.*` attributes and data-type rules | `references/span-attributes.md` |
| Standard OTel `gen_ai.*` fallback behavior | `references/gen-ai-fallbacks.md` |

## Templates

| Purpose | Template |
| --- | --- |
| Minimal Python OTLP exporter setup + example trace | `templates/confident_otel_setup.py` |
