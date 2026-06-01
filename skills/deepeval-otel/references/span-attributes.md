# Span-Level Attributes

Span-level attributes describe **one span** — a single component of the trace.
Set them as `confident.span.*` (and the per-type `confident.llm.*`,
`confident.agent.*`, `confident.retriever.*`, `confident.tool.*`) attributes on
that span.

`confident.span.type` selects which per-type keys are meaningful. Set it first.

The span types — `llm`, `tool`, `agent`, `retriever`, and the generic `base` —
describe **AI-application components**. Apply `confident.*` attributes only to
the AI parts of the system (LLM calls, agent loops, retrieval, tool calls), not
to non-AI spans such as HTTP handlers, database queries, or infrastructure.

## Common Span Attributes

Valid on every span type:

| Attribute key | Type | Notes |
| --- | --- | --- |
| `confident.span.type` | string | One of `llm`, `tool`, `agent`, `retriever`, `base`. If absent, inferred from `gen_ai.*` attributes — see `gen-ai-fallbacks.md`. |
| `confident.span.name` | string | Display name; overrides the native OTel span name. |
| `confident.span.input` | string | Span input. Passthrough; JSON-encode if not already a string. |
| `confident.span.output` | string | Span output. Passthrough; JSON-encode if not already a string. |
| `confident.span.metadata` | JSON string | Component facts that help diagnose failures. Must be a JSON-encoded object string. |
| `confident.span.context` | list of strings | Ground-truth context for the span. |
| `confident.span.retrieval_context` | list of strings | Retrieved chunks for the span. |
| `confident.span.tools_called` | list of strings | Native OTLP list of JSON-serialized `ToolCall` strings. |
| `confident.span.expected_tools` | list of strings | Native OTLP list of JSON-serialized `ToolCall` strings. |
| `confident.span.metric_collection` | string | Name of a Confident AI metric collection to run online (server-side) evals against this span. |

The allowed values for `confident.span.type` are exactly `llm`, `tool`,
`agent`, `retriever`, and `base`. `base` is a generic span with no extra typed
fields.

## Span Errors

Span errors are **not** a `confident.*` attribute. Use the native OpenTelemetry
span `Status`:

```python
from opentelemetry.trace import Status, StatusCode

try:
    ...
except Exception as e:
    span.set_status(Status(StatusCode.ERROR), str(e))
    span.record_exception(e)
```

A span with `StatusCode.ERROR` is rendered as errored in the Observatory; if it
is the root span, the whole trace is marked errored.

## LLM Spans

Set `confident.span.type` to `llm`.

| Attribute key | Type | Notes |
| --- | --- | --- |
| `confident.llm.model` | string | Model name (e.g. `gpt-4o`). Fallback: `gen_ai.request.model`. |
| `confident.span.provider` | string | LLM provider (e.g. `openai`, `anthropic`). Optional — inferred from the model name when omitted. |
| `confident.llm.input_token_count` | int | Input/prompt tokens. Fallback: `gen_ai.usage.input_tokens`. |
| `confident.llm.output_token_count` | int | Output/completion tokens. Fallback: `gen_ai.usage.output_tokens`. |
| `confident.llm.cost_per_input_token` | float | Cost per input token, for cost rollups. |
| `confident.llm.cost_per_output_token` | float | Cost per output token, for cost rollups. |

If the span references a prompt managed in Confident AI, set the discrete
prompt fields:

| Attribute key | Type | Notes |
| --- | --- | --- |
| `confident.span.prompt_alias` | string | Prompt alias/name. |
| `confident.span.prompt_version` | string | Prompt version identifier. |
| `confident.span.prompt_commit_hash` | string | Prompt commit hash. |
| `confident.span.prompt_label` | string | Prompt label. |

Set only the prompt fields that apply.

## Agent Spans

Set `confident.span.type` to `agent`.

| Attribute key | Type | Notes |
| --- | --- | --- |
| `confident.agent.name` | string | Agent name/identifier. |
| `confident.agent.available_tools` | list of strings | Tools available to the agent. |
| `confident.agent.agent_handoffs` | list of strings | Other agents this agent can hand off to. |

## Retriever Spans

Set `confident.span.type` to `retriever`.

| Attribute key | Type | Notes |
| --- | --- | --- |
| `confident.retriever.embedder` | string | Embedding model name (e.g. `text-embedding-3-small`). |
| `confident.retriever.top_k` | int | Number of results retrieved. |
| `confident.retriever.chunk_size` | int | Document chunk size. |

Put the retrieved chunks on `confident.span.retrieval_context`.

## Tool Spans

Set `confident.span.type` to `tool`.

| Attribute key | Type | Notes |
| --- | --- | --- |
| `confident.tool.name` | string | Tool / function name. Fallback: `gen_ai.tool.name`. |
| `confident.tool.description` | string | Human-readable tool description. |

Put the tool's arguments on `confident.span.input` and its result on
`confident.span.output`.

## Data-Type Rules

OpenTelemetry attribute values must be **primitives** (string, bool, int,
float) or **homogeneous lists of primitives**. There is no map/object attribute
type. Encode accordingly:

- **Objects / dicts** (`confident.span.metadata`, `confident.trace.metadata`)
  must be **JSON-encoded strings** — `json.dumps(...)`.
- **String lists** (`tags`, `context`, `retrieval_context`,
  `available_tools`, `agent_handoffs`) can be **native OTLP string arrays**
  (a Python `list`/`tuple` of `str`). A JSON array string is also accepted.
- **`ToolCall` lists** (`tools_called`, `expected_tools`) must be a **native
  OTLP list where each element is a JSON-serialized `ToolCall` string** — i.e.
  a list of JSON strings, not one JSON string of a list.
- **`input` / `output`** are passthrough. If the value is not already a
  string, JSON-encode it before setting.
- **Numbers** (`top_k`, `chunk_size`, token counts, costs) are set as native
  int/float, not strings.

## Span Nesting

Parent/child relationships come from **native OpenTelemetry span context** —
start a child span inside the parent span's context. There is no `confident.*`
attribute for parenthood. With `tracer.start_as_current_span(...)`, spans
opened inside the `with` block are automatically nested under it.
