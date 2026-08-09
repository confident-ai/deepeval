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
- **No integration is at full Python parity.** None can attach a *metric* to a
  framework-created span — the staging contexts exist and carry everything else, but
  `metrics` is commented out in four places
  ([gap 1](#1-component-level-metrics-metrics-is-commented-out-of-the-staging-contexts)) —
  and the two OTel-mode ones only reach the local trace tree behind a manual flag.
  See [Cross-cutting gaps](#cross-cutting-gaps).
- **No framework page documents TypeScript today.** `frameworks/langchain`,
  `frameworks/openai` and `frameworks/openai-agents` were bilingual and have been
  taken back to `languages: [python]`; their `<Switch>` blocks were collapsed to
  the Python case and the TypeScript examples deleted, because they were written
  against the intended surface rather than read off this module. All 12 framework
  pages are now Python-only and the whole Frameworks section is hidden from a
  TypeScript reader. [Per-framework docs readiness](#per-framework-docs-readiness)
  is the starting point for rewriting them.
- Python's fourth capability column, `deepeval test run`, now **has a TS equivalent**
  — a Vitest integration with an `expect(...).toPass()` matcher — but it is young and
  not yet exercised per-integration, so it is omitted from the matrix below rather
  than filled with guesses. Its own parity backlog lives in
  [`src/evaluate/test-run/README.md`](../evaluate/test-run/README.md).

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

### 1. Component-level metrics: `metrics` is commented out of the staging contexts

Python's `deepeval/tracing/context.py` exports `next_agent_span`, `next_llm_span`,
`next_tool_span`, and `next_retriever_span`. Each is a context manager that stages
metrics (and every field `update_current_span` takes) onto the **next** span of that
type the integration opens:

```python
with next_llm_span(metrics=[AnswerRelevancyMetric()]):
    agent.invoke(...)   # first LLM span the callback opens picks the metric up
```

There are no functions by these names in TS, but **the staging mechanism itself
already exists** — the gap is narrower than "no equivalent", and this is the single
most important thing to know before rewriting any framework page.

`src/tracing/trace-context.ts` defines `LlmSpanContext` and `AgentSpanContext`,
backed by two `AsyncLocalStorage` stores, staged by `setTracingContext(opts, fn)`
(exported from `deepeval/tracing`) and drained by three integrations —
`src/openai/patch.ts`, `src/integrations/openai-agents/callback.ts`, and
`src/integrations/ai-sdk/processor.ts`. It is ambient-scope rather than Python's
one-shot-per-span-type, so it is closer to Python's own `LlmSpanContext` than to
`next_*_span`.

What it carries today, verified against the drain sites:

| Field | OpenAI | OpenAI Agents | AI SDK |
| --- | :--: | :--: | :--: |
| `expectedOutput`, `expectedTools`, `context`, `retrievalContext` | Yes | No | No |
| `prompt` | Yes | Yes (LLM spans) | — |
| `metricCollection` | Yes | Yes (agent / LLM / tool) | — |
| `metrics` | **commented out** | **commented out** | **commented out** |

**`metrics` is the whole gap.** It is commented out in four places rather than
absent by design — the field on both context types, the trace assignment in
`setTracingContext`, and the consumption site in the OpenAI patch:

```ts
// src/tracing/trace-context.ts
export type LlmSpanContext = {
  prompt?: Prompt;
  //   metrics?: BaseMetric[];

// src/openai/patch.ts
    return await observe({
      type: "llm",
      //   metrics: llmContext?.metrics,
      metricCollection: llmContext?.metricCollection,
```

So component-level *metric collections* (the Confident AI online-eval path) work
today for OpenAI and OpenAI Agents, while component-level *metrics* (the local
eval path every docs page uses) do not. Uncommenting and wiring the four sites is
plausibly the highest-leverage change in this module.

**LangChain has a separate, unrelated hatch.** It does not read the ALS contexts at
all; `handleLLMStart` instead pulls `metrics`, `metricCollection` and `prompt` off
LangChain's per-call `metadata` and assigns them straight to the LLM span:

```ts
// src/integrations/langchain/callback-handler.ts
const metrics = metadata?.["metrics"];
llmSpan.metrics = metrics;
```

`config: { metadata: { metrics: [...] } }` therefore reaches a LangChain LLM span
today with nothing commented out in the way. `handleToolStart` and
`handleRetrieverStart` take the same `metadata` argument and ignore it, and there is
no agent span to attach to, so it covers one span type on one integration. Not
exercised end to end — verify before documenting.

Needs, to reach Python parity: uncomment `metrics` through the four sites above, add
a drain point in the LangChain and Mastra span-open paths (neither reads the ALS
contexts), and decide whether to keep ambient scoping or add Python's one-shot
semantics where only the first matching span consumes the staged config.

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

### 5. Test-runner integration exists, but no integration is verified against it

`deepeval test run` now drives Vitest, and `expect(golden).toPass()` scores the trace
a test just produced — so the pattern an integration user would reach for in CI is
available. What hasn't been checked is whether each integration's spans actually land
in that trace, since the per-test capture uses the same sink that gaps 2 and 6 cover.
The runner's own parity backlog is in
[`src/evaluate/test-run/README.md`](../evaluate/test-run/README.md).

### 6. `evalsIterator`, Mastra and the test runner fight over one capture sink

`setTraceCaptureSink` writes a single global slot with three consumers now.
`evalsIterator` sets it on entry and clears it to `undefined` on exit,
`DeepEvalExporter` sets it from `config.traceCaptureSink`, and the Vitest
`beforeEach` sets it per test and clears it in `afterEach`. Whichever runs last
wins, and either cleanup silently drops the others' sink. Needs a subscriber list
rather than one slot.

### 7. OpenInference is unreachable from the published package

`src/integrations/openinference/` is complete and exports `instrumentOpenInference`,
`createOpenInferenceProcessors` and `OpenInferenceSpanProcessor`, but there is no
`./integrations/openinference` key in either `exports` or `typesVersions` in
`typescript/package.json`. An `exports` map is exhaustive, so the specifier throws
`ERR_PACKAGE_PATH_NOT_EXPORTED` at runtime and TS2307 under `nodenext`. It cannot be
documented until the key is added.

`./integrations` is exported and mapped, but `src/integrations/index.ts` is an empty
file, so that specifier resolves to a module with no exports. Either barrel the four
public entry points through it or drop the key.

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

## Per-framework docs readiness

What a TypeScript half of each framework page could honestly contain today. The
cross-cutting gaps above apply to all of them; this section is what each page's
author hits in practice, and which sections of the existing Python page survive.

Docs conventions (`<Switch>`, `<Only>`, `languages` frontmatter) are in
[`.cursor/rules/docs-languages.mdc`](../../../.cursor/rules/docs-languages.mdc);
migration status is in [`docs/LANGUAGES.md`](../../../docs/LANGUAGES.md). Pages live
in `docs/content/integrations/frameworks/`.

**Start with OpenAI and OpenAI Agents.** They are the only two where all three
capability columns are Yes, and OpenAI Agents is the reference implementation for
ambient-trace adoption that LangChain and Mastra still need.

### OpenAI — `frameworks/openai`

Entry point `instrumentOpenAI(client)` from `deepeval/openai`, which patches the
client in place rather than re-exporting a subclass — so the Python page's "drop-in
replacement, change your import" framing does not carry over and the TS case needs
its own prose.

The `LlmSpanContext` section translates better than anywhere else: TS has the same
concept, staged as `setTracingContext({ llmSpanContext: {...} }, fn)`, and the
OpenAI patch is the one drain site that consumes every field
([gap 1](#1-component-level-metrics-metrics-is-commented-out-of-the-staging-contexts)). `expectedOutput`,
`expectedTools`, `context`, `retrievalContext`, `prompt` and `metricCollection` all
land on the LLM span; only `metrics` is commented out. The pytest /
`deepeval test run` sections become Vitest + `expect(golden).toPass()`.

Two footguns to verify before writing the quickstart, both from reading
`src/openai/`:

- `instrumentOpenAI` guards on a module-level `let registered = false` and returns
  early, so only the **first** client ever passed to it is patched. A second client
  is silently un-instrumented.
- `updateAllAttributes(...)` runs only inside `if (llmContext && …)`, and
  `getLlmContext()` returns `undefined` unless `setTracingContext` staged one. On
  the bare path the LLM span may therefore get no input, output or `toolsCalled` —
  the `else` branch just logs `getLlmContext() returned undefined`. The matrix above
  says Bare = Yes for OpenAI on the strength of the trace being created; whether the
  span is *populated* bare is untested.

### OpenAI Agents — `frameworks/openai-agents`

Entry point `new DeepEvalTracingProcessor()` via `addTraceProcessor`. Nesting and
`evalsIterator` both work, and this is the reference implementation for adopting an
ambient trace — it checks `getCurrentTrace()` before starting a new one, which is
exactly what LangChain and Mastra fail to do.

Python's `Agent` / `function_tool` shims — `agent_metrics`, `llm_metrics`,
`metrics=`, `confident_prompt` — have no TS equivalent;
`src/integrations/openai-agents/` exports only the processor, so you use the SDK's
own `Agent` and `tool`. The staging contexts partly cover the difference:
`onSpanStart` reads `metricCollection` per span type (agent context for agent spans,
LLM context for LLM spans, `toolsMetricCollection` for tool spans) and `onSpanEnd`
copies `llmSpanContext.prompt` onto LLM spans with its alias, hash, version and
label. So the prompt-binding section has a real TS counterpart, and the
component sections can be written against *metric collections* — but not against
`metrics`, which is commented out.

### LangChain — `frameworks/langchain`

The most heavily gapped of the four, in rough order of how much each costs:

1. **Trace-level metrics grade the wrong text** ([gap 4](#4-trace-level-test-cases-are-built-from-traceinput-not-the-golden)).
   `handleChainStart` sets `trace.input` to LangChain's raw payload and
   `scopeToTestCase` stringifies it, so `evalsIterator({ metrics })` scores
   `{"messages":[{"role":"user","content":"…"}]}` rather than the question, and the
   golden's `expectedOutput` is dropped. This hits the page's *first* example.
   Metrics with `requiresTrace` read the nested span dict and are less exposed.
2. **No agent spans, no nested-chain spans.** Only the root chain gets one, typed
   `SpanType.CUSTOM` and named `"Langchain Chain Run"` unless `runName` is set. The
   Python page's `Agent: math_agent` trace diagram is wrong for TS. LLM, tool and
   retriever spans do exist and nest correctly through `RunHierarchyTracker`.
3. **Component metrics**: LLM spans only, via LangChain's `config.metadata` — see
   the escape hatch in [gap 1](#1-component-level-metrics-metrics-is-commented-out-of-the-staging-contexts).
   Tool, retriever and agent subsections stay unwritable.
4. **`observe()` nesting is broken** ([gap 3](#3-observe-nesting-is-broken-for-langchain-and-mastra)),
   so the "Wrap a LangChain run in `@observe`" pattern is false.
5. **No tool decorator** — `patch-tool.ts` is commented out end to end, yet the page
   advertises one in two places.

Not gaps: constructor kwargs are at full parity (only `thread_id` → `threadId`
casing differs), so the API-reference table just needs a `<Switch>`.

### LangGraph — `frameworks/langgraph`

Rides on `DeepEvalCallbackHandler`, so it inherits every LangChain item above.
`RunHierarchyTracker` exists specifically for the LangGraph-server case where ALS is
lost across callbacks, which is why the ALS bypass in
[gap 3](#3-observe-nesting-is-broken-for-langchain-and-mastra) cannot simply be
deleted — it needs an ambient-trace fallback, not a removal. This page was never
bilingual, so it is a first write rather than a rewrite.

### Mastra — no page yet

TS-only, `new DeepEvalExporter(config)`. Needs a page from scratch. Two blockers
first: `observe()` nesting is broken for the simpler reason that the exporter never
calls `getCurrentTrace()`, and its `config.traceCaptureSink` collides with
`evalsIterator` and the Vitest runner over the single global sink slot
([gap 6](#6-evalsiterator-mastra-and-the-test-runner-fight-over-one-capture-sink)).

### Vercel AI SDK (tracing) — no page yet

TS-only, `configureAiSdkTracing(options)`. Distinct from
`integrations/models/ai-sdk`, which covers `AISDKModel` as a judge model and is
already bilingual — pick a URL that does not imply the two are the same integration.
Reaching the iterator needs `isTestMode: true` and double-exports while it does
([gap 2](#2-no-eval-session-concept--otel-integrations-need-a-manual-flag)).

### OpenInference — no page, and not shippable

Blocked outright on the missing subpath export
([gap 7](#7-openinference-is-unreachable-from-the-published-package)). Same
`isTestMode` caveat as AI SDK on top of that.

## Suggested order of work

1. **Uncomment `metrics` on the staging contexts** (gap 1) — four commented lines,
   and it unblocks component-level evals for OpenAI and OpenAI Agents immediately.
   Cheapest item here by a wide margin relative to what it opens up in the docs.
2. **Golden-based trace test cases** (gap 4) — small change, stops silently wrong scores.
3. **Eval-session routing** (gap 2) — makes AI SDK / OpenInference work in the iterator
   without a manual flag, and kills the double export.
4. **`observe()` nesting for LangChain and Mastra** (gap 3) — makes an existing docs
   section true; copy the `getCurrentTrace()` check from OpenAI Agents.
5. **LangChain trace shape** — agent spans + nested chain spans, so the documented
   trace tree matches reality.
6. **Tool decorator** — finish or delete `patch-tool.ts`, and align the docs either way.
7. **Capture-sink subscriber list** (gap 6) — small correctness fix.
8. **Drain the staging contexts in LangChain and Mastra** (gap 1) — neither reads
   `getLlmContext()` / `getAgentContext()`, so they miss whatever step 1 unlocks.
9. **Export OpenInference** (gap 7) — a `package.json` key, not code.
