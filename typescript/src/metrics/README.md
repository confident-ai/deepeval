# Metrics — TypeScript port overview

Status snapshot of the TS metrics module (`typescript/src/metrics/`), how the API
differs from Python, and what's missing. Intended as raw material for the docs.

## Status at a glance

- **45 of 47 metrics ported.** Only the **2 DAG metrics** (`dag`,
  `conversational_dag`) are not ported — deliberately out of scope (they need the
  decision-graph engine).
- Each metric lives in its own folder: `metrics/<name>/{<name>.ts, schema.ts, index.ts}`,
  mirroring the Python `deepeval/metrics/<name>/` layout.
- All metrics reuse the shared, language-agnostic prompt templates
  (`templates/metrics/templates.json`, rendered through the Nunjucks resolver) — the
  same JSON the Python package compiles. Only the **orchestration** (`measure()`) and
  the **output schema** (`schema.py` → zod) are re-implemented per metric.
- Verified = renders + runs end-to-end on sample cases.

## The API in one breath

```ts
import { AnswerRelevancyMetric } from "deepeval/metrics";
import { LLMTestCase } from "deepeval/test-case";

const testCase = new LLMTestCase({ input: "...", actualOutput: "..." });
const metric = new AnswerRelevancyMetric({ threshold: 0.7, model: "gpt-4.1" });
const score = await metric.measure(testCase);
console.log(metric.score, metric.reason, metric.success, metric.evaluationCost);
```

### Key API facts (and how they differ from Python)

- **`measure()` is the only entry point, and it is always `async`.** Python exposes
  both a sync `measure()` and an async `a_measure()`; the TS port has **only
  `async measure(testCase): Promise<number>`** — there is no `aMeasure`/sync split.
  Everything underneath (model calls, the runner, the dataset iterator) is async.
- **Three base classes**, all extending `BaseMetricCore` (`base-metrics.ts`):
  - `BaseMetric` — single-turn, `measure(LLMTestCase)`, `requiredParams: SingleTurnParams[]`.
  - `BaseConversationalMetric` — multi-turn, `measure(ConversationalTestCase)`,
    `requiredParams: MultiTurnParams[]`.
  - `BaseArenaMetric` — `measure(ArenaTestCase)` returns the **winner's name (string)**,
    not a numeric score.
- **Constructor takes an options object**, not positional args:
  `{ threshold, model?, strictMode?, verboseMode?, includeReason?, showIndicator? }`.
  (Python's `async_mode` option is **not** exposed in TS — everything is already async.
  The indicator line still prints `async_mode=True` as a hardcoded value for console
  parity with Python.)
- **Result is read off the instance after `measure()`**: `score`, `reason`, `success`,
  `error`, `evaluationCost`, `verboseLogs`, `skipped`, plus `scoreBreakdown` on some.
- **Model resolution** (`initializeModel`): a metric's `model` may be a
  `DeepEvalBaseLLM` instance, a model-name string (→ `new OpenAIModel({ model })`), or
  omitted (→ `new OpenAIModel()`, default `gpt-4.1`). Every TS model is "native"
  (returns `{ output, cost }`), so cost is always accrued when the model reports it.
- **`requiresTrace`** flag (default `false`) marks the trace metrics; the runner/dataset
  iterator attaches `testCase._traceDict` for those.

## Ported metrics

### Single-turn / LLM (20)

| Metric | Class | Direction / notes |
|---|---|---|
| Answer Relevancy | `AnswerRelevancyMetric` | higher-is-better |
| Faithfulness | `FaithfulnessMetric` | higher |
| Contextual Precision | `ContextualPrecisionMetric` | higher |
| Contextual Recall | `ContextualRecallMetric` | higher |
| Contextual Relevancy | `ContextualRelevancyMetric` | higher |
| Bias | `BiasMetric` | lower-is-better |
| Toxicity | `ToxicityMetric` | lower |
| PII Leakage | `PIILeakageMetric` | higher |
| Non-Advice | `NonAdviceMetric` | higher; needs `adviceTypes` |
| Misuse | `MisuseMetric` | lower; needs `domain` |
| Role Violation | `RoleViolationMetric` | higher, binary; needs `role` |
| Hallucination | `HallucinationMetric` | lower; uses `context` |
| Prompt Alignment | `PromptAlignmentMetric` | higher; needs `promptInstructions` |
| Summarization | `SummarizationMetric` | `min(alignment, coverage)`; `n`/`assessmentQuestions` |
| GEval | `GEval` | criteria→steps→score; `evaluationParams`/`rubric`; name shown as `<name> [GEval]` |
| JSON Correctness | `JsonCorrectnessMetric` | deterministic zod-validate; needs `expectedSchema` (zod) |
| Exact Match | `ExactMatchMetric` | deterministic, **no model**; adds `precision`/`recall`/`f1` |
| Pattern Match | `PatternMatchMetric` | deterministic regex full-match, **no model** |
| Tool Correctness | `ToolCorrectnessMetric` | deterministic tool-match + optional LLM tool-selection (`availableTools`) |
| Argument Correctness | `ArgumentCorrectnessMetric` | LLM verdict over `toolsCalled` |

### Single-turn, trace-based — `requiresTrace=true` (4)

Run via `dataset.evalsIterator(...)` over `observe`-wrapped agents (they read
`testCase._traceDict`). See `tracing/` + `evaluate/trace-eval.ts`.

| Metric | Class |
|---|---|
| Task Completion | `TaskCompletionMetric` |
| Plan Adherence | `PlanAdherenceMetric` |
| Plan Quality | `PlanQualityMetric` |
| Step Efficiency | `StepEfficiencyMetric` |

### Multimodal / vision (5)

Images embed as slugs (`[DEEPEVAL:IMAGE|PDF:{id}]`) inside normal string fields; the
model layer splits them into text+image parts. Require a vision-capable model.

| Metric | Class |
|---|---|
| Image Coherence | `ImageCoherenceMetric` |
| Image Helpfulness | `ImageHelpfulnessMetric` |
| Image Reference | `ImageReferenceMetric` |
| Text→Image | `TextToImageMetric` |
| Image Editing | `ImageEditingMetric` |

### Conversational / multi-turn (12)

`measure(ConversationalTestCase)`.

| Metric | Class | Notes |
|---|---|---|
| Turn Relevancy | `TurnRelevancyMetric` | sliding-window |
| Turn Faithfulness | `TurnFaithfulnessMetric` | RAG, sliding-window mean |
| Turn Contextual Precision | `TurnContextualPrecisionMetric` | RAG |
| Turn Contextual Recall | `TurnContextualRecallMetric` | RAG |
| Turn Contextual Relevancy | `TurnContextualRelevancyMetric` | RAG |
| Conversation Completeness | `ConversationCompletenessMetric` | whole-conversation |
| Knowledge Retention | `KnowledgeRetentionMetric` | whole-conversation |
| Role Adherence | `RoleAdherenceMetric` | needs `chatbotRole` |
| Topic Adherence | `TopicAdherenceMetric` | needs `relevantTopics` |
| Goal Accuracy | `GoalAccuracyMetric` | free-text final reason |
| Conversational GEval | `ConversationalGEval` | name shown as `<name> [Conversational GEval]` |
| Tool Use | `ToolUseMetric` | turn-based; needs `availableTools` |

### MCP (3)

| Metric | Class | Kind |
|---|---|---|
| MCP Use | `MCPUseMetric` | single-turn; needs `mcpServers` |
| MCP Task Completion | `MCPTaskCompletionMetric` | conversational; needs `mcpServers` |
| Multi-turn MCP Use | `MultiTurnMCPUseMetric` | conversational; needs `mcpServers` |

### Arena (1)

| Metric | Class | Notes |
|---|---|---|
| Arena GEval | `ArenaGEval` | `measure(ArenaTestCase)` → winning contestant **name** (string), not a score |

## Gaps & discrepancies vs Python

- **DAG metrics not ported** (`dag`, `conversational_dag`) — no decision-graph engine
  in TS. (The `.txt` templates still ship in the bundle, just unused by TS.)
- **No sync `measure` / no `a_measure`** — TS is async-only. A Python user porting
  `metric.measure(tc)` must `await metric.measure(tc)` in TS.
- **No score-parity guarantee** — prompts are cross-engine parity-checked; numeric
  scores were spot-verified for sanity, not asserted equal to Python.
- **GEval is reduced**:
  - No **log-prob-weighted scoring** — TS models don't expose top-logprobs, so GEval
    uses the structured `(score, reason)` path (Python's fallback when log-probs are
    unavailable).
  - No **`upload()` / `pull()`** (Confident-AI metric push/pull) — not ported.
- **Single-turn & trace metrics are text-only** (no multimodal) — only the 5 explicit
  multimodal metrics handle images.
- **Trace-eval naming quirk**: per-trace result names restart at index 0; named spans
  label correctly, but the unnamed trace scope shows `test_case_0`.
- **Metadata field naming**: conversational `metadata` is exposed as the TS-convention
  `additionalMetadata`.
- **Latent Python template bugs intentionally diverged** (TS does the correct thing):
  `ArgumentCorrectnessMetric` passes the stringified tools (template expects
  `stringified_tools_called`, Python passes the wrong var); `ToolUseMetric` uses
  `get_tool_selection_final_reason` (Python's `get_tool_argument_final_reason` is an
  unused/latent template).

## Usage examples

### Direct `measure()`

```ts
const m = new GEval({
  threshold: 0.5,
  name: "Correctness",
  criteria: "Does the output match the expected answer?",
  evaluationParams: ["input", "actualOutput", "expectedOutput"],
});
await m.measure(new LLMTestCase({ input, actualOutput, expectedOutput }));
```

### Through the batch runner

```ts
import { evaluate } from "deepeval";

await evaluate({
  testCases: [tc1, tc2],
  metrics: [new AnswerRelevancyMetric({ threshold: 0.7 }), new FaithfulnessMetric({ threshold: 0.7 })],
});
// prints per-case + aggregate tables; posts to Confident AI if CONFIDENT_API_KEY is set.
```

### Conversational

```ts
import { ConversationalTestCase, Turn } from "deepeval/test-case";

const convo = new ConversationalTestCase({ turns: [new Turn({ role: "user", content: "..." }), /* ... */] });
await new TurnRelevancyMetric({ threshold: 0.5 }).measure(convo);
```

### Arena

```ts
import { compare } from "deepeval/evaluate";
const wins = await compare({ arenaTestCases, metric: new ArenaGEval({ /* ... */ }) });
```

### Trace metrics (agentic)

```ts
for await (const golden of dataset.evalsIterator({ metrics: [new TaskCompletionMetric({ model: "gpt-4.1" })] })) {
  await myObserveWrappedAgent(golden.input); // spans captured + evaluated on resume
}
```
