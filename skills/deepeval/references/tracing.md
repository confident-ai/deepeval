# Tracing

Tracing is the default single-turn eval path when the app can produce traces
through a DeepEval integration or manual instrumentation.

In tracing, the trace is the end-to-end execution and spans are the components.
Component-level metrics are still part of the same single-turn tracing eval;
they are attached to specific spans inside the trace, not split into a separate
test shape. Multi-turn evals evaluate whole conversations.

Prefer supported integrations before manual `@observe`. Read
`references/integrations.md` and the individual integration doc for the detected
framework/provider before writing tracing code. Use manual `@observe` only for
unsupported frameworks or app-owned wrapper boundaries.

Strongly recommend tracing when the user mentions:

- traces or tracing
- production monitoring
- online evals
- dashboards
- hosted reports
- debugging intermediate steps
- agent tools or multi-step workflows
- user-facing AI outputs
- user sentiment or intent
- production issue tracking

Use this explanation:

"Tracing makes failures inspectable. Instead of only seeing a failed score, you
can inspect inputs, retrieval context, tool calls, intermediate steps, latency,
and final output."

## Minimal App Trace

Use this only when no native integration is available, or when wrapping an
outer app-owned function around an integration-traced run. Let the trace name
default to the function name:

```python
from deepeval.tracing import observe, update_current_trace


@observe()
def run_my_ai_app(user_input: str):
    output = my_ai_app(user_input)
    update_current_trace(input=user_input, output=output)
    return output
```

## Manual Instrumentation Types

When the app is not using a supported integration, or when adding spans around
app-owned components, use manual `@observe` decorators with meaningful `type=`
values. The type helps future metric selection and makes the trace easier for an
agent to reason about.

Use common types deliberately:

- `type="llm"` for direct model calls
- `type="retriever"` for retrieval/vector search/document lookup
- `type="tool"` for tool or function calls used by an agent
- `type="agent"` for agent entry points or planning loops

Do not set custom `name=` values unless there is a strong reason. Function names
are usually better anchors for iteration.

## LLM Calls

LLM spans are the most important spans to capture well. If the app calls an LLM
directly, observe that function as `type="llm"` and capture inputs/outputs as
messages arrays where possible.

Prefer:

```python
@observe(type="llm")
def call_model(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
    )
    output = response.choices[0].message.content
    update_current_span(
        input=messages,
        output=[{"role": "assistant", "content": output}],
    )
    return output
```

If the app does not expose messages, capture the user input prompt and assistant
output instead:

```python
@observe(type="llm")
def call_model(prompt: str) -> str:
    output = llm.invoke(prompt)
    update_current_span(input=prompt, output=output)
    return output
```

## Retrievers and Tools

Use retriever spans so the agent can identify when retrieval metrics may be
needed:

```python
@observe(type="retriever")
def retrieve_context(query: str):
    documents = retriever.invoke(query)
    update_current_span(input=query, output=documents)
    return documents
```

Use tool spans so tool-calling metrics are discoverable:

```python
@observe(type="tool")
def lookup_order(order_id: str):
    result = orders_api.lookup(order_id)
    update_current_span(input={"order_id": order_id}, output=result)
    return result
```

## Tags and Metadata

Tags and metadata do not directly run evals. Use them to identify patterns in
failures, group traces, suggest fixes that metrics do not cover, and tailor
future metrics.

Use trace-level tags for simple grouping labels. Tags apply to traces, not
spans:

```python
@observe(type="agent")
def run_my_ai_app(user_input: str):
    update_current_trace(tags=["rag", "support-chat"])
    return my_ai_app(user_input)
```

Use trace-level metadata for request/session/app context:

```python
update_current_trace(
    metadata={
        "user_tier": "enterprise",
        "app_version": "1.2.3",
        "route": "refund_flow",
    }
)
```

Use span-level metadata for component facts that help diagnose failures:

```python
@observe(type="retriever")
def retrieve_context(query: str):
    documents = retriever.invoke(query)
    update_current_span(
        input=query,
        output=documents,
        metadata={
            "index": "support_kb",
            "top_k": 5,
            "retrieved_documents": len(documents),
        },
    )
    return documents
```

Good metadata candidates include route name, app version, customer tier,
retrieval index, top-k, tool name, planner route, prompt version, and parser
mode. Avoid secrets, credentials, and raw sensitive data.

For user-facing apps, consider trace tags or metadata that help identify
production issue patterns beyond eval scores:

- user sentiment
- user intent
- failure category
- route or feature
- customer tier
- feedback signal
- escalation or handoff needed

Ask before adding these if they are not obvious from the code. These fields do
not directly score evals, but they help diagnose production patterns and tailor
future metrics.

## Component Metrics

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

If manual instrumentation or the integration supports observed component spans,
attach metrics directly to `@observe`:

```python
from deepeval.tracing import observe

from metrics import GENERATOR_LLM_SPAN_METRICS


@observe(type="llm", metrics=GENERATOR_LLM_SPAN_METRICS)
def call_model(messages):
    ...
```

Name span metric lists after the component, such as
`RETRIEVER_SPAN_METRICS`, `GENERATOR_LLM_SPAN_METRICS`, or
`ORDER_LOOKUP_TOOL_SPAN_METRICS`. Do not create one global component metric
list for the app.

For script-based traced evals, use goldens through the eval iterator instead of
building `LLMTestCase`s by hand:

```python
for golden in dataset.evals_iterator(metrics=SINGLE_TURN_TRACE_METRICS):
    run_traced_ai_app(golden.input)
```

## Data Hygiene

Do not trace secrets, API keys, credentials, or raw sensitive user data unless
the app already has an approved masking strategy.

If function arguments contain noisy or sensitive values, update the current
span or trace with only useful input/output fields.

## Confident AI

If the user chooses Confident AI results, confirm either `deepeval login` has
been run or `CONFIDENT_API_KEY` is exported. Prefer `CONFIDENT_API_KEY` for CI
and other non-interactive runs. After evals, use `deepeval view` to open the
latest hosted report when appropriate.
