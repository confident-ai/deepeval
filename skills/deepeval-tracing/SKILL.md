---
name: deepeval-tracing
description: >
  Instrument an AI application with DeepEval's native tracing so its behavior
  is visible in Confident AI. TRIGGER when the user wants to add DeepEval
  tracing, @observe (Python), or observe() (TypeScript) to an LLM app, agent,
  RAG pipeline, or chatbot; wire a framework, model-provider, or
  vector-database integration (LangGraph, LangChain, OpenAI Agents,
  LlamaIndex, Pydantic AI, CrewAI, Vercel AI SDK, Mastra, and others); choose
  between a native integration and manual instrumentation; set span types,
  tags, or metadata; or send DeepEval-SDK traces to Confident AI's
  Observatory. DO NOT TRIGGER for building DeepEval eval suites, datasets,
  goldens, metrics, or deepeval test run (use the `deepeval` skill), or for
  raw OpenTelemetry / OTLP export without the deepeval package (use the
  `deepeval-otel` skill). This skill is purely DeepEval-SDK instrumentation —
  producing well-formed traces, not running evals.
license: Apache-2.0
metadata:
  author: Confident AI
  version: "1.1.0"
  category: observability
  tags: "deepeval, tracing, observe, instrumentation, integrations, spans, confident-ai"
  compatibility: "Python 3.9+ with `pip install deepeval`, or TypeScript/JavaScript on Node.js 20+ with `npm install deepeval`. Instrumentation uses the DeepEval SDK (`deepeval.tracing` in Python, `deepeval/tracing` in TypeScript). Sending traces to Confident AI requires `deepeval login` (Python) / `npx deepeval login` (TypeScript) or an exported `CONFIDENT_API_KEY`."
---

# DeepEval Tracing

Use this skill to instrument an **AI application** — an LLM app, agent, RAG
pipeline, or chatbot — with **DeepEval's native tracing** so its execution is
visible span by span in **Confident AI's Observatory**. The work is: pick a
supported integration when one exists, fall back to manual instrumentation
otherwise — `@observe` in Python, `observe()` in TypeScript — give each span a
meaningful type, and add tags and metadata.

This skill stops at producing well-formed traces. Attaching evaluation metrics
and running evals is the `deepeval` skill's job.

## Scope: AI Applications Only

Instrument only the AI parts of the system — agent loops and planning, LLM
calls, retrieval / vector search, and tool calls. The span types (`llm`,
`retriever`, `tool`, `agent`) describe AI components. Do not trace non-AI
software (web servers, CRUD backends, infrastructure). If the target has no
LLM, agent, retrieval, or tool-calling component, this skill does not apply.

## When to Use vs the `deepeval` and `deepeval-otel` Skills

- **This skill (`deepeval-tracing`)** — instrument an app with the DeepEval SDK
  (`@observe` in Python, `observe()` in TypeScript, framework integrations) so
  traces reach Confident AI. Supports Python (`pip install deepeval`) and
  TypeScript/JavaScript (`npm install deepeval`).
- **`deepeval` skill** — build eval suites: datasets, metrics, traced evals,
  `deepeval test run`, iteration. It runs evals *against* an app this skill
  instrumented.
- **`deepeval-otel` skill** — instrument with the vendor-neutral OpenTelemetry
  SDK instead of the DeepEval SDK (raw OTLP export).

The three are complementary. If unsure between this skill and `deepeval-otel`:
use this one when the app is Python or TypeScript and you want the DeepEval
SDK; use `deepeval-otel` when you want raw OpenTelemetry, the app already has
an OTel pipeline you must keep, or the app is in a language DeepEval has no
SDK for.

## Prerequisites

- An AI application in Python (3.9+) with `pip install deepeval`, or in
  TypeScript/JavaScript (Node.js 20+) with `npm install deepeval`.
- For traces to reach Confident AI: `deepeval login` (Python) /
  `npx deepeval login` (TypeScript), or an exported `CONFIDENT_API_KEY`
  (preferred for CI and non-interactive runs).

## Workflow

1. Confirm the target is an AI application (it has LLM calls, an agent loop,
   retrieval, or tool calls). If it has none of these, stop — this skill does
   not apply.
2. Detect the language: Python (`requirements.txt`, `pyproject.toml`) or
   TypeScript/JavaScript (`package.json`, `tsconfig.json`). Both SDKs are the
   `deepeval` package on their respective package managers.
3. Detect the framework, model provider, agent SDK, and vector database in use.
4. Read `references/integrations.md` and the exact integration doc for what was
   detected — integration availability differs by language, and the index
   marks which docs cover which. Prefer a native integration over manual
   instrumentation.
5. If no native integration fits, instrument manually. Read
   `references/tracing-python.md` (Python) or
   `references/tracing-typescript.md` (TypeScript).
6. Give each span a meaningful `type` (`llm`, `retriever`, `tool`, `agent`) and
   capture inputs/outputs.
7. Add trace-level tags and metadata where they help diagnose failure patterns.
   Never trace secrets, credentials, or raw sensitive data.
8. Confirm login or `CONFIDENT_API_KEY`, then verify traces appear in the
   Confident AI Observatory. In short-lived TypeScript processes (serverless,
   scripts), call `flushTraces()` before exit.

## Core Principles

1. Instrument AI components only — `llm`, `retriever`, `tool`, `agent` spans.
   Never trace non-AI software.
2. Prefer a supported integration over manual instrumentation. Manual tracing
   is the fallback for unsupported frameworks and app-owned wrapper
   boundaries.
3. Read the exact integration doc before writing tracing code, and check it
   covers the app's language.
4. Give spans meaningful types. In Python, let names default to function
   names unless there is a strong reason to override. In TypeScript, pass
   `name` explicitly — inline `fn:` arrows carry no usable function name.
5. Never trace secrets, credentials, API keys, or raw sensitive user data.
6. Producing traces is the scope. Attaching metrics and running evals belong to
   the `deepeval` skill; raw OpenTelemetry export belongs to `deepeval-otel`.

## References

| Topic | File |
| --- | --- |
| Manual instrumentation in Python: `@observe`, span types, tags, metadata | `references/tracing-python.md` |
| Manual instrumentation in TypeScript: `observe()`, span types, tags, metadata, `flushTraces()` | `references/tracing-typescript.md` |
| Integration selection rule and framework / model / vector-DB doc index, with per-language availability | `references/integrations.md` |
