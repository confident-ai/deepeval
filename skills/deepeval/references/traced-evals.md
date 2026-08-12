# Traced Evals

Tracing is the default single-turn eval path when the app can produce traces
through a DeepEval integration or manual instrumentation. The trace is the
end-to-end execution and spans are the components; component-level metrics are
attached to specific spans inside the same single-turn tracing eval, not split
into a separate test shape.

This reference covers the **eval-coupled** side of tracing: attaching metrics
to spans and the test/script shapes for traced evals, in Python and
TypeScript. To **instrument** the app — add `@observe` (Python) / `observe()`
(TypeScript), wire framework integrations, set span types, tags, and
metadata — use the `deepeval-tracing` skill.

## Component / Span Metrics

When metrics belong to a specific component, keep them in the single-turn
tracing eval and attach them to the exact span they evaluate.

If a supported integration creates the span, stage metrics for the next span of
that type:

```python
from deepeval.tracing import next_retriever_span

from metrics import RETRIEVER_SPAN_METRICS


with next_retriever_span(metrics=RETRIEVER_SPAN_METRICS):
    run_ai_app_with_integration_tracing(golden.input)
```

```typescript
import { nextRetrieverSpan } from "deepeval/tracing";

import { RETRIEVER_SPAN_METRICS } from "./metrics";

await nextRetrieverSpan({ metrics: RETRIEVER_SPAN_METRICS }, () =>
  runAiAppWithIntegrationTracing(golden.input),
);
```

The staging helpers are one-shot in both languages: only the first span of
that type picks up the metrics.

If manual instrumentation or the integration supports observed component spans,
attach metrics directly to `@observe` (Python) / `observe()` (TypeScript):

```python
from deepeval.tracing import observe

from metrics import GENERATOR_LLM_SPAN_METRICS


@observe(type="llm", metrics=GENERATOR_LLM_SPAN_METRICS)
def call_model(messages):
    ...
```

```typescript
import { observe } from "deepeval/tracing";

import { GENERATOR_LLM_SPAN_METRICS } from "./metrics";

const callModel = observe({
  type: "llm",
  name: "callModel",
  metrics: GENERATOR_LLM_SPAN_METRICS,
  fn: async (messages) => {
    // ...
  },
});
```

Name span metric lists after the component, such as
`RETRIEVER_SPAN_METRICS`, `GENERATOR_LLM_SPAN_METRICS`, or
`ORDER_LOOKUP_TOOL_SPAN_METRICS`. Do not create one global component metric
list for the app. Use `next_agent_span`, `next_llm_span`, `next_tool_span`, or
`next_retriever_span` (Python) / `nextAgentSpan`, `nextLlmSpan`,
`nextToolSpan`, or `nextRetrieverSpan` (TypeScript) to match the span type the
integration creates.

## Test vs Script Shapes

For CI/CD, prefer the pytest/Vitest shape shown in each integration doc — pass
the `Golden` directly through the traced app and assert:

```python
@pytest.mark.parametrize("golden", dataset.goldens)
def test_agent(golden: Golden):
    run_ai_app_with_integration_tracing(golden.input)
    assert_test(golden=golden, metrics=TRACE_METRICS)
```

```typescript
it.each(dataset.goldens as Golden[])("agent: $input", async (golden) => {
  await expect(golden).toPass(TRACE_METRICS, {
    task: (g) => runAiAppWithIntegrationTracing(g.input),
  });
});
```

For scripts or iteration loops, use the evals iterator and pass the `Golden`
through the traced app:

```python
for golden in dataset.evals_iterator(metrics=TRACE_METRICS):
    run_ai_app_with_integration_tracing(golden.input)
```

```typescript
for await (const golden of dataset.evalsIterator({ metrics: TRACE_METRICS })) {
  await runAiAppWithIntegrationTracing(golden.input);
}
```

Do not convert a traced single-turn eval into a hand-built `LLMTestCase` unless
the user explicitly chooses no tracing.

## Confident AI

If the user chooses Confident AI results, confirm either `deepeval login`
(Python) / `npx deepeval login` (TypeScript) has been run or
`CONFIDENT_API_KEY` is exported. Prefer `CONFIDENT_API_KEY` for CI and other
non-interactive runs. After evals, use `deepeval view` / `npx deepeval view`
to open the latest hosted report when appropriate.
