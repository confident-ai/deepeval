# Integrations — TypeScript port overview

Status snapshot of the TS integrations module (`typescript/src/integrations/`), how it
differs from Python (`deepeval/integrations/`, whose own README carries the Python-side
matrix), and what's missing. Intended as raw material for the docs and as the backlog
for closing parity.

## Status at a glance

- **Six modules, covering seven framework rows**: LangChain (LangGraph rides on the
  same handler), OpenAI Agents, Mastra, AI SDK, OpenInference, plus `src/openai/` at
  the top level mirroring Python's `deepeval.openai`. Two of them — Mastra and AI SDK
  — are TS-only with no Python counterpart, so only four rows are true ports.
- **Seven of Python's rows have no TS equivalent**, listed under
  [Not ported](#not-ported).
- **No integration is at full Python parity.** Every one of them is missing
  component-level metric staging, and the two OTel-mode ones only reach the local
  trace tree behind a manual flag. See [Cross-cutting gaps](#cross-cutting-gaps).
- Python's fourth capability column, `deepeval test run`, has **no TS equivalent at
  all** — there is no `assertTest` and no test-runner plugin, so it is omitted from
  the matrix below rather than filled with No.

## Integration matrix

Capability columns (same meaning as the Python README, minus `deepeval test run`):

- **Bare** — calling the framework directly, with no enclosing `observe(...)`, produces
  a trace in Confident AI.
- **`observe()` nesting** — when wrapped, the integration's spans join the enclosing
  deepeval trace instead of starting a detached one.
- **`evalsIterator`** — spans reach `traceManager` during
  `dataset.evalsIterator(...)`, so trace/span metrics can score them.

| Integration   | Mode                  | Entry point                             | Bare | `observe()` nesting | `evalsIterator`      | Source                              |
| ------------- | --------------------- | --------------------------------------- | :--: | :-----------------: | :------------------: | ----------------------------------- |
| OpenAI        | Native client wrapper | `instrumentOpenAI(client)`              | Yes  | Yes                 | Yes                  | `src/openai/`                       |
| LangChain     | Callback handler      | `new DeepEvalCallbackHandler({})`       | Yes  | **No**              | Yes                  | `src/integrations/langchain/`       |
| LangGraph     | LangChain's handler   | `new DeepEvalCallbackHandler({})`       | Yes  | **No**              | Yes                  | `src/integrations/langchain/`       |
| OpenAI Agents | Trace processor       | `new DeepEvalTracingProcessor()`        | Yes  | Yes                 | Yes                  | `src/integrations/openai-agents/`   |
| Mastra        | Exporter              | `new DeepEvalExporter(config)`          | Yes  | **No**              | Yes (sink conflict)   | `src/integrations/mastra/`          |
| AI SDK        | OpenTelemetry         | `configureAiSdkTracing(options)`        | Yes  | **`isTestMode` only** | **`isTestMode` only** | `src/integrations/ai-sdk/`        |
| OpenInference | OpenTelemetry         | `instrumentOpenInference(options)`      | Yes  | **`isTestMode` only** | **`isTestMode` only** | `src/integrations/openinference/` |

### Not ported

Python has these; TS has no equivalent: **Anthropic**, **LlamaIndex**, **CrewAI**,
**AgentCore**, **Strands**, **Google ADK**, **Pydantic AI**. Python also ships a
`hugging_face` module with no TS counterpart.

Four of those seven (AgentCore, Strands, Google ADK, Pydantic AI) are OTel-mode in
Python and share one `SpanInterceptor` + `ContextAwareSpanProcessor` pattern, so
[gap 2](#2-no-eval-session-concept--otel-integrations-need-a-manual-flag) is a
prerequisite for porting any of them well.

## Cross-cutting gaps

Ordered by how much they block. These apply across integrations, so fixing one
generally unblocks several docs pages at once.

### 1. No `next_*_span` staging helpers — component-level evals

Python's `deepeval/tracing/context.py` exports `next_agent_span`, `next_llm_span`,
`next_tool_span`, and `next_retriever_span`. Each is a context manager that stages
metrics (and every field `update_current_span` takes) onto the **next** span of that
type the integration opens:

```python
with next_llm_span(metrics=[AnswerRelevancyMetric()]):
    agent.invoke(...)   # first LLM span the callback opens picks the metric up
```

There is no TS equivalent anywhere in `src/`. This is the single largest gap: it is
the only way to attach a metric to a span the *framework* creates (as opposed to one
you wrapped in `observe` yourself), so **component-level evals are impossible for
every integration**. On the LangChain docs page alone it blocks three subsections,
an advanced pattern, a bullet, and an FAQ answer.

Needs: an async-context-local staging slot (the ALS store in `tracing/tracing.ts` is
the natural home), a drain point in each integration's span-open path, and one-shot
semantics matching Python (only the first matching span consumes the staged config).

### 2. No eval-session concept — OTel integrations need a manual flag

Python's `trace_manager.is_evaluating` flips on when `evals_iterator` opens an
`EvalSession`, and `ContextAwareSpanProcessor` reads it to route ended OTel spans to
the REST path (through `trace_manager`) instead of OTLP. That is what makes OTel-mode
integrations work inside the iterator, automatically.

TS has no `EvalSession` and no `isEvaluating`. `evalsIterator` instead installs a
capture sink (`traceManager.setTraceCaptureSink`) that only fires from
`traceManager.endTrace`, so an integration is visible to the iterator **only if it
builds spans through `traceManager`**. For AI SDK and OpenInference that path is
gated behind a manual option:

```ts
// src/integrations/ai-sdk/processor.ts (same shape in openinference/processor.ts)
if (this.options.isTestMode) {
  traceManager.addSpan(deepEvalSpan);
  traceManager.addSpanToTrace(deepEvalSpan);
}
```

Two consequences. Users must remember to pass `isTestMode: true` to evaluate, and
even then both processors stay registered, so spans go to `traceManager` **and** get
batch-exported to OTLP — Python deliberately routes to exactly one delegate to avoid
the double export.

Needs: an eval-session flag on `traceManager` that `evalsIterator` sets, `isTestMode`
defaulting to it, and mutually exclusive routing.

### 3. `observe()` nesting is broken for LangChain and Mastra

The docs promise that LangChain spans nest under an enclosing observed span. They
cannot. The handler always passes `traceUuidOverride` into `enterCurrentContext`, and
that parameter exists specifically to bypass async-local storage:

```ts
// src/integrations/langchain/utils.ts
if (traceUuidOverride) {
  // Caller-driven placement (keyed off run_id/parent_run_id). Do not consult
  // AsyncLocalStorage: it is unreliable under the LangGraph server.
  traceUuid = traceUuidOverride;
  parentUuid = parentUuidOverride;
}
```

The override comes from `RunHierarchyTracker.ensureTrace()`, which never consults
`getCurrentTrace()` — with no parent run it unconditionally calls
`traceManager.startNewTrace()`. So a LangChain run inside `observe()` lands in its own
detached trace with no parent.

The ALS bypass is correct for the LangGraph-server case it was written for. What's
missing is the fallback: `ensureTrace()` should adopt an ambient trace when one
exists, and the root chain span should parent onto an ambient span.

`DeepEvalExporter` (Mastra) has the same problem for a simpler reason — it never calls
`getCurrentTrace()` at all, only `traceManager.startNewTrace()`, so every exported
Mastra run is a detached trace. `DeepEvalTracingProcessor` (OpenAI Agents) is the
reference for what these two should do: it checks `getCurrentTrace()` first and only
starts a new trace when there isn't one.

### 4. Trace-level test cases are built from `trace.input`, not the golden

`evaluateTrace` builds the trace scope's test case straight off the trace:

```ts
// src/evaluate/trace-eval.ts
function scopeToTestCase(scope: BaseSpan | Trace): LLMTestCase {
  return new LLMTestCase({
    input: asString(scope.input),
    actualOutput: asString(scope.output),
```

Python's iterator paths use `golden.input` and fall back to `golden.expected_output`.
Because integrations set `trace.input` to the framework's raw payload, a trace-level
metric in TS grades `JSON.stringify({messages: [...]})` rather than the user's
question, and a golden's `expectedOutput` is dropped. This is silent — scores come
back plausible and wrong.

Affects `evalsIterator({ metrics })` generally, not just integrations. Needs the
golden plumbed into `evaluateTrace`.

### 5. No `assertTest` / test-runner integration

Python's `assert_test()` plus the pytest plugin and `deepeval test run` have no TS
counterpart; the CLI (`src/cli/`) only implements `deepeval gate`. CI users have to
hand-roll Jest assertions around `evalsIterator` / `evaluate`. Treat as permanent
unless a test-runner integration is scoped deliberately.

### 6. `evalsIterator` and Mastra fight over one capture sink

`setTraceCaptureSink` writes a single global slot. `evalsIterator` sets it on entry
and clears it to `undefined` on exit, while `DeepEvalExporter` sets it from
`config.traceCaptureSink`. Whichever runs last wins, and the iterator's cleanup will
also silently drop Mastra's sink. Needs a subscriber list rather than one slot.

## LangChain / LangGraph specifics

Beyond the cross-cutting items, the LangChain integration diverges from Python in
ways that make parts of `/integrations/frameworks/langchain` unwritable for TS.

- **No agent spans, and no spans for nested chains.** `handleChainStart` creates a
  span only when `parentUuid === undefined`, and types it `SpanType.CUSTOM`. Python
  creates a span for *every* chain ("important for LangGraph where there are nested
  chains") and the docs' trace diagram shows an `Agent:` root with nested children.
  TS produces a flatter tree under a generic custom root.
- **The tool decorator does not exist.** `patch-tool.ts` is entirely commented out and
  `index.ts` exports only `DeepEvalCallbackHandler`, yet the docs advertise a
  deepeval tool decorator in two places.
- **Constructor kwargs are at parity.** `name`, `tags`, `metadata`, `threadId`,
  `userId`, `testCaseId`, `turnId`, `metrics`, `metricCollection` are all accepted and
  applied — only the casing differs from Python (`thread_id` → `threadId`). The docs'
  API-reference table is Python-spelled and needs a `<Switch>`, but that is a docs
  fix, not an SDK one.
- **`metrics` / `metricCollection` are trace-level**, matching Python: both are
  assigned to the trace on the root chain, guarded so an enclosing
  `updateCurrentTrace(...)` isn't clobbered. Prefer
  `evalsIterator({ metrics })` in eval scripts; the constructor kwarg is for online
  evals on live traffic.

## Suggested order of work

1. **`next_*_span`** (gap 1) — unblocks component-level evals for every integration and
   the largest share of the docs.
2. **Golden-based trace test cases** (gap 4) — small change, stops silently wrong scores.
3. **Eval-session routing** (gap 2) — makes AI SDK / OpenInference work in the iterator
   without a manual flag, and kills the double export.
4. **`observe()` nesting for LangChain and Mastra** (gap 3) — makes an existing docs
   section true; copy the `getCurrentTrace()` check from OpenAI Agents.
5. **LangChain trace shape** — agent spans + nested chain spans, so the documented
   trace tree matches reality.
6. **Tool decorator** — finish or delete `patch-tool.ts`, and align the docs either way.
7. **Capture-sink subscriber list** (gap 6) — small correctness fix.
