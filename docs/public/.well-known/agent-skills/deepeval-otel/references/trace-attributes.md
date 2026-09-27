# Trace-Level Attributes

Trace-level attributes describe the **whole trace**, not a single span. Set them
as `confident.trace.*` attributes on any span in the trace — most naturally on
the root span. Confident AI aggregates them up to the trace.

A trace is one end-to-end execution; its spans are the components. Span-level
fields are documented separately in `span-attributes.md`.

## Attribute Table

| Attribute key | Type | Notes |
| --- | --- | --- |
| `confident.trace.name` | string | Human-readable trace name. |
| `confident.trace.input` | string | Trace input. Passthrough; JSON-encode if not already a string. |
| `confident.trace.output` | string | Trace output. Passthrough; JSON-encode if not already a string. |
| `confident.trace.user_id` | string | End user / customer identifier. |
| `confident.trace.thread_id` | string | Conversation or session thread identifier. |
| `confident.trace.tags` | list of strings | Grouping labels. Native OTLP string array, or a JSON array string. |
| `confident.trace.metadata` | JSON string | Arbitrary key-value context. Must be a JSON-encoded object string (OTLP has no map type). |
| `confident.trace.environment` | string | Deployment environment. Defaults to `"production"`. See Environment Resolution below. |
| `confident.trace.retrieval_context` | list of strings | Retrieved chunks/documents for the trace. Native OTLP string array, or a JSON array string. |
| `confident.trace.context` | list of strings | Ground-truth context for the trace. Native OTLP string array, or a JSON array string. |
| `confident.trace.tools_called` | list of strings | Tools called during the trace. Native OTLP list where each element is a JSON-serialized `ToolCall`. |
| `confident.trace.expected_tools` | list of strings | Tools that should have been called. Native OTLP list of JSON-serialized `ToolCall` strings. |
| `confident.trace.test_case_id` | string | Reference to a test case ID. |
| `confident.trace.turn_id` | string | Turn identifier for multi-turn conversations. |
| `confident.trace.metric_collection` | string | Name of a Confident AI metric collection to run online (server-side) evals against this trace. |

All attributes are optional. Set only what is meaningful for the app.

## Environment Resolution

`confident.trace.environment` accepts a deployment environment string (commonly
`"production"`, `"staging"`, `"development"`, or `"testing"`). The default is
`"production"`.

It can be set in two places, and the **Resource attribute wins** over the span
attribute:

- As a span attribute: `confident.trace.environment` on a span.
- As an OpenTelemetry **Resource** attribute `confident.trace.environment` on
  the `TracerProvider`'s `Resource`. This is the recommended way to stamp the
  environment once for an entire process.

## Data-Type Rules

The encoding rules for objects (JSON strings) versus lists (native OTLP arrays)
versus `ToolCall` lists apply identically to trace-level and span-level
attributes. They are detailed once, in `span-attributes.md` under
**Data-Type Rules** — read that section before encoding `tags`, `metadata`,
`context`, `retrieval_context`, `tools_called`, or `expected_tools`.
