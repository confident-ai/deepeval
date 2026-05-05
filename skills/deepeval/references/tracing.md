# Tracing

Tracing is for visibility and component-level diagnostics. It is not the default
end-to-end pytest pattern.

In tracing, the trace is the end-to-end execution and spans are the components.
Component-level testing evaluates spans inside the trace; it is therefore a
superset/add-on to an E2E trace, not a replacement for E2E. Multi-turn evals do
not have component-level tests in this template set because they evaluate whole
conversations.

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

Use this when the user wants traces but not component-level metrics yet. Let the
trace name default to the function name:

```python
from deepeval.tracing import observe, update_current_trace


@observe()
def chat_response(user_input: str) -> str:
    response = TARGET_APP_ENTRYPOINT(user_input)
    update_current_trace(input=user_input, output=response)
    return response
```

## Manual Instrumentation Types

When the app is not using a supported integration, add manual `@observe`
decorators with meaningful `type=` values. The type helps future metric
selection and makes the trace easier for an agent to reason about.

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
def answer_question(query: str):
    update_current_trace(tags=["rag", "support-chat"])
    return TARGET_APP_ENTRYPOINT(query)
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

When metrics belong to a specific component, use
`references/pytest-component-evals.md` and
`templates/test_single_turn_component.py`.

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
