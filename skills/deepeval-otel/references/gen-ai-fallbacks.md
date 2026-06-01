# GenAI Semantic-Convention Fallbacks

When a `confident.*` attribute is absent, Confident AI's exporter falls back to
the standard OpenTelemetry **GenAI semantic-convention** attributes
(`gen_ai.*`). This matters when the app is already instrumented by a
GenAI-aware library that emits `gen_ai.*` spans — those spans carry useful data
into Confident AI without any extra `confident.*` attributes.

Read this file only when the app already produces `gen_ai.*` spans. For new
instrumentation, set `confident.*` attributes directly (see
`span-attributes.md`).

## Span-Type Inference

If `confident.span.type` is not set, the span type is inferred:

| Condition | Inferred `confident.span.type` |
| --- | --- |
| `gen_ai.operation.name` is `chat`, `generate_content`, or `text_completion` | `llm` |
| `gen_ai.tool.name` is present | `tool` |
| otherwise | `base` |

## Attribute Fallback Table

When the `confident.*` key is absent, the exporter reads the `gen_ai.*` key:

| `confident.*` attribute | Falls back to `gen_ai.*` |
| --- | --- |
| `confident.llm.model` | `gen_ai.request.model` |
| `confident.llm.input_token_count` | `gen_ai.usage.input_tokens` |
| `confident.llm.output_token_count` | `gen_ai.usage.output_tokens` |
| `confident.tool.name` | `gen_ai.tool.name` |

## Guidance

- `confident.*` attributes always **win** when both are present.
- For new instrumentation, prefer explicit `confident.*` attributes — they map
  directly and unambiguously.
- Rely on fallbacks only to avoid duplicating attributes an existing GenAI
  integration already emits. Do not add `confident.*` copies of `gen_ai.*`
  attributes the app already sets unless you need to override them.
