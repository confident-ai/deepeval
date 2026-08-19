# Metrics — TypeScript port overview

Status snapshot of the TS metrics module (`typescript/src/metrics/`), how the API
differs from Python, and what's missing. Intended as raw material for the docs.

## Status at a glance

- **47 of 49 metrics ported.** Missing: `AgentLoopDetectionMetric` and
  `ToolPermissionMetric`. (The DAG metrics, previously out of scope, **are** now
  ported — see the DAG section below.)
- Each metric lives in its own folder: `metrics/<name>/{<name>.ts, schema.ts, index.ts}`,
  mirroring the Python `deepeval/metrics/<name>/` layout.
- All metrics reuse the shared, language-agnostic prompt templates
  (`templates/metrics/templates.json`, rendered through the Nunjucks resolver) — the
  same JSON the Python package compiles. Only the **orchestration** (`measure()`) and
  the **output schema** (`schema.py` → zod) are re-implemented per metric.
- Verified = renders + runs end-to-end on sample cases.

## The API in one breath

```ts
import { metrics } from "deepeval";
import { LLMTestCase } from "deepeval/test-case";

const testCase = new LLMTestCase({ input: "...", actualOutput: "..." });
const metric = new metrics.AnswerRelevancyMetric({
  threshold: 0.7,
  model: "gpt-4.1",
});
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
  `{ threshold, model?, strictMode?, verboseMode?, includeReason?, showIndicator?,
flaky?, evaluationTemplate? }`.
  (Python's `async_mode` option is **not** exposed in TS — everything is already async.
  The indicator line still prints `async_mode=True` as a hardcoded value for console
  parity with Python.)
- **`threshold` is nullable (score-only mode).** `threshold: null` computes a score and
  reason but forms no verdict: `isSuccessful()` returns `undefined` and reports render
  the status as `NONE`. `evaluate()` / `toPass()` require at least one non-flaky metric
  with a threshold, so every test case can still pass or fail.
- **`flaky` metrics still score but never decide pass/fail.** Aggregation skips their
  verdicts and reports show `(flaky=N)` sub-counts. `LLMTestCase` /
  `ConversationalTestCase` take `flaky` too.
- **`evaluationTemplate` overrides prompts** — a partial object keyed by camelCase
  template method, e.g. `{ generateStatements: ({ actualOutput }) => "..." }`. Each
  override receives the render context plus a `renderDefault(vars)` second argument for
  extending the shipped prompt instead of replacing it. See "Prompt overrides" below.
- **Result is read off the instance after `measure()`**: `score`, `reason`, `success`,
  `error`, `evaluationCost`, `verboseLogs`, `skipped`, plus `scoreBreakdown` on some.
  `success` is `undefined` (not `false`) in score-only mode.
- **Model resolution** (`initializeModel`): a metric's `model` may be a
  `DeepEvalBaseLLM` instance, a model-name string, or omitted. A string or an omitted
  model goes to whichever provider `selectProvider()` picks from the `USE_*` settings,
  falling back to `OpenAIModel`. The judge it lands on is generated from Python's
  `DEFAULT_MODELS` (see `src/models/README.md`), so both SDKs default to the same model.
  Every TS model is "native" (returns `{ output, cost }`), so cost is always accrued when
  the model reports it.
- **`requiresTrace`** flag (default `false`) marks the trace metrics; the runner/dataset
  iterator attaches `testCase._traceDict` for those.

## Ported metrics

### Single-turn / LLM (20)

| Metric               | Class                       | Direction / notes                                                                 |
| -------------------- | --------------------------- | --------------------------------------------------------------------------------- |
| Answer Relevancy     | `AnswerRelevancyMetric`     | higher-is-better                                                                  |
| Faithfulness         | `FaithfulnessMetric`        | higher                                                                            |
| Contextual Precision | `ContextualPrecisionMetric` | higher                                                                            |
| Contextual Recall    | `ContextualRecallMetric`    | higher                                                                            |
| Contextual Relevancy | `ContextualRelevancyMetric` | higher                                                                            |
| Bias                 | `BiasMetric`                | lower-is-better                                                                   |
| Toxicity             | `ToxicityMetric`            | lower                                                                             |
| PII Leakage          | `PIILeakageMetric`          | higher                                                                            |
| Non-Advice           | `NonAdviceMetric`           | higher; needs `adviceTypes`                                                       |
| Misuse               | `MisuseMetric`              | lower; needs `domain`                                                             |
| Role Violation       | `RoleViolationMetric`       | higher, binary; needs `role`                                                      |
| Hallucination        | `HallucinationMetric`       | lower; uses `context`                                                             |
| Prompt Alignment     | `PromptAlignmentMetric`     | higher; needs `promptInstructions`                                                |
| Summarization        | `SummarizationMetric`       | `min(alignment, coverage)`; `n`/`assessmentQuestions`                             |
| GEval                | `GEval`                     | criteria→steps→score; `evaluationParams`/`rubric`; name shown as `<name> [GEval]` |
| JSON Correctness     | `JsonCorrectnessMetric`     | deterministic zod-validate; needs `expectedSchema` (zod)                          |
| Exact Match          | `ExactMatchMetric`          | deterministic, **no model**; adds `precision`/`recall`/`f1`                       |
| Pattern Match        | `PatternMatchMetric`        | deterministic regex full-match, **no model**                                      |
| Tool Correctness     | `ToolCorrectnessMetric`     | deterministic tool-match + optional LLM tool-selection (`availableTools`)         |
| Argument Correctness | `ArgumentCorrectnessMetric` | LLM verdict over `toolsCalled`                                                    |

### Single-turn, trace-based — `requiresTrace=true` (4)

Run via `dataset.evalsIterator(...)` over `observe`-wrapped agents (they read
`testCase._traceDict`). See `tracing/` + `evaluate/trace-eval.ts`.

| Metric          | Class                  |
| --------------- | ---------------------- |
| Task Completion | `TaskCompletionMetric` |
| Plan Adherence  | `PlanAdherenceMetric`  |
| Plan Quality    | `PlanQualityMetric`    |
| Step Efficiency | `StepEfficiencyMetric` |

### Multimodal / vision (5)

Images embed as slugs (`[DEEPEVAL:IMAGE|PDF:{id}]`) inside normal string fields; the
model layer splits them into text+image parts. Require a vision-capable model.

| Metric            | Class                    |
| ----------------- | ------------------------ |
| Image Coherence   | `ImageCoherenceMetric`   |
| Image Helpfulness | `ImageHelpfulnessMetric` |
| Image Reference   | `ImageReferenceMetric`   |
| Text→Image        | `TextToImageMetric`      |
| Image Editing     | `ImageEditingMetric`     |

### Conversational / multi-turn (12)

`measure(ConversationalTestCase)`.

| Metric                    | Class                            | Notes                                         |
| ------------------------- | -------------------------------- | --------------------------------------------- |
| Turn Relevancy            | `TurnRelevancyMetric`            | sliding-window                                |
| Turn Faithfulness         | `TurnFaithfulnessMetric`         | RAG, sliding-window mean                      |
| Turn Contextual Precision | `TurnContextualPrecisionMetric`  | RAG                                           |
| Turn Contextual Recall    | `TurnContextualRecallMetric`     | RAG                                           |
| Turn Contextual Relevancy | `TurnContextualRelevancyMetric`  | RAG                                           |
| Conversation Completeness | `ConversationCompletenessMetric` | whole-conversation                            |
| Knowledge Retention       | `KnowledgeRetentionMetric`       | whole-conversation                            |
| Role Adherence            | `RoleAdherenceMetric`            | needs `chatbotRole`                           |
| Topic Adherence           | `TopicAdherenceMetric`           | needs `relevantTopics`                        |
| Goal Accuracy             | `GoalAccuracyMetric`             | free-text final reason                        |
| Conversational GEval      | `ConversationalGEval`            | name shown as `<name> [Conversational GEval]` |
| Tool Use                  | `ToolUseMetric`                  | turn-based; needs `availableTools`            |

### MCP (3)

| Metric              | Class                     | Kind                               |
| ------------------- | ------------------------- | ---------------------------------- |
| MCP Use             | `MCPUseMetric`            | single-turn; needs `mcpServers`    |
| MCP Task Completion | `MCPTaskCompletionMetric` | conversational; needs `mcpServers` |
| Multi-turn MCP Use  | `MultiTurnMCPUseMetric`   | conversational; needs `mcpServers` |

### Arena (1)

| Metric      | Class        | Notes                                                                        |
| ----------- | ------------ | ---------------------------------------------------------------------------- |
| Arena GEval | `ArenaGEval` | `measure(ArenaTestCase)` → winning contestant **name** (string), not a score |

### DAG — decision graphs (2)

Prompts live on the **nodes** (`dag/nodes.ts`, `conversational-dag/nodes.ts`), not the
metric, so these two take no `evaluationTemplate`.

| Metric             | Class                     | Kind           |
| ------------------ | ------------------------- | -------------- |
| DAG                | `DAGMetric`               | single-turn    |
| Conversational DAG | `ConversationalDAGMetric` | conversational |

## Gaps & discrepancies vs Python

- **2 metrics not ported**: `AgentLoopDetectionMetric`, `ToolPermissionMetric`.
- **No sync `measure` / no `a_measure`** — TS is async-only. A Python user porting
  `metric.measure(tc)` must `await metric.measure(tc)` in TS.
- **No score-parity guarantee** — prompts are cross-engine parity-checked; numeric
  scores were spot-verified for sanity, not asserted equal to Python.
- **GEval has no `upload()` / `pull()`** (Confident-AI metric push/pull) — not ported.
  Log-prob-weighted scoring **is** ported; see the GEval section below.
- **Trace-eval naming quirk**: per-trace result names restart at index 0; named spans
  label correctly, but the unnamed trace scope shows `test_case_0`.
- **Metadata field naming**: conversational `metadata` is exposed as the TS-convention
  `additionalMetadata`.
- **Latent Python template bugs intentionally diverged** (TS does the correct thing):
  `ArgumentCorrectnessMetric` passes the stringified tools (template expects
  `stringified_tools_called`, Python passes the wrong var); `ToolUseMetric` uses
  `get_tool_selection_final_reason` (Python's `get_tool_argument_final_reason` is an
  unused/latent template).

## Multimodal (images & PDFs)

Images embed as slugs (`[DEEPEVAL:IMAGE|PDF:{id}]`) inside ordinary string fields.
Two independent things then happen:

1. **Delivery** — `models/multimodal.ts` splits any slug-bearing prompt into
   provider-native multipart content, so the image reaches the model. This applies to
   every metric and needs no opt-in.
2. **Prompt guidance** — the templates carry `{% if multimodal %}` branches that tell
   the judge how to treat images. These only render for metrics that opt in.

The flag is plumbed centrally rather than threaded through each private method (which
is what Python does):

- `checkSingleTurnParams` / `checkConversationalTestCaseParams` copy
  `testCase.multimodal` onto the metric before any prompt is built.
- `getPrompt` passes `multimodalAware && multimodal` as the `multimodal` template
  variable. An explicit `multimodal` in the call's vars still wins.
- A metric opts in with `this.multimodalAware = true`.

**26 metrics opt in**, mirroring exactly the Python metrics that thread
`test_case.multimodal` into `_get_prompt` (`ArenaGEval` makes 27 — it passes the flag
itself and skips the param-check seam). The rest stay text-only **on purpose**, matching
Python: `SummarizationMetric` hardcodes `multimodal=False` even on its borrowed
faithfulness prompts, and the 5 dedicated image metrics never set the flag because their
templates are already image-specific.

A multimodal test case is **refused** when the evaluation model can't see images.
`checkMultimodalSupport` (called from both param-check seams) throws unless
`model.supportsMultimodal()` is `true`, so an unknown capability counts as "cannot" —
matching Python, and preventing the silent failure where a text-only provider reads an
image slug as literal text and scores it anyway. The error names vision-capable models
from the same provider, read out of the generated registry. Deterministic metrics
(`ExactMatchMetric`, `PatternMatchMetric`) have no model and therefore reject multimodal
test cases too, as they do in Python.

Metrics whose prompts need more than the boolean use helpers in
`retrieval-context-display.ts`, ported 1:1 from Python: `idRetrievalContext` (labels
context nodes `Node N:` so verdicts can cite images positionally),
`contextualRecallVerdictVars`, and `contextualRelevancyVerdictVars`. Faithfulness's
`multimodal_instruction` and the contextual-precision `multimodal_note` are computed
locally in those metrics.

One resolver subtlety this relies on: Python renders a list of context nodes with
`repr`, and `MLLMImage.__repr__` returns the bare slug. So `pyRepr` renders any object
defining its own `toString` bare and unquoted, rather than as a dict of its fields.

## GEval log-prob weighting

`GEval` and `ConversationalGEval` refine the judge's integer score into the
probability-weighted expectation over the score tokens it considered — the G-Eval
paper's scoring, ported from Python's `calculate_weighted_summed_score`. Without it,
scores quantize onto the rubric grid (0, 0.1, 0.2, …).

Both metrics call `evaluateGEvalPrompt` (`g-eval/utils.ts`), which prefers
`model.generateRaw(prompt, { topLogprobs })` and falls back to the plain structured
`(score, reason)` call whenever the raw path is missing or throws — a gateway can
advertise log-probs and still reject the parameter. `topLogprobs` defaults to `20` as in
Python and is capped per-model by `maxLogProbs()`. Candidate tokens below 1% linear
probability, and any non-decimal token, are discarded; if everything is discarded the
raw score stands. Strict mode skips weighting entirely.

Only the OpenAI-compatible providers implement `generateRaw`, so Anthropic, Gemini,
Bedrock, and Ollama take the fallback — which matches Python, where those providers have
no `generate_raw_response` either and the registry marks none of their models
logprob-capable.

## Prompt overrides (`evaluationTemplate`)

Python's `evaluation_template` takes a subclass of a generated `<Metric>Template`
class and relies on `inspect.signature` to let an override declare only the template
variables it uses. Neither idea ports, so the TS shape is a **partial object of
functions** instead — see `templates/override.ts`.

```ts
new AnswerRelevancyMetric({
  evaluationTemplate: {
    generateStatements: ({ actualOutput }) => `...${actualOutput}...`,
    // Extend the shipped prompt rather than replacing it:
    generateVerdicts: (vars, renderDefault) =>
      `${renderDefault(vars)}\n\nExtra rule.`,
  },
});
```

- **Method names and variables are camelCase**, mapped at the boundary from the
  bundle's snake_case (`generate_statements` → `generateStatements`, `actual_output` →
  `actualOutput`). Leading underscores are dropped: `_additional_context` →
  `additionalContext`.
- **Valid method names are derived from `templates.json` at the type level**, so a new
  metric gets a correct override surface for free and a typo is a compile error.
  Template _variables_ are typed `unknown` (they interpolate fine in template literals;
  other uses need a cast).
- **Each metric exports its own alias**, e.g. `AnswerRelevancyTemplateOverride`, all
  re-exported from the metrics barrel.
- **42 of the 47 metrics expose it.** Excluded, all for structural reasons:
  `ExactMatchMetric` / `PatternMatchMetric` (deterministic, no prompts), `DAGMetric` /
  `ConversationalDAGMetric` (prompts belong to the nodes), and `MultiTurnMCPUseMetric`
  (borrows every prompt from `MCPTaskCompletionMetric`; Python likewise gives it no
  `evaluation_template`).
- **Borrowed prompts are never overridable.** A metric rendering another's template
  passes an explicit `templateClass` to `getPrompt`, which bypasses the override — so an
  override can't silently rewrite a prompt the caller never named. The type surface
  enforces this too, since it is built from each metric's own bundle keys.

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
  metrics: [
    new AnswerRelevancyMetric({ threshold: 0.7 }),
    new FaithfulnessMetric({ threshold: 0.7 }),
  ],
});
// prints per-case + aggregate tables; posts to Confident AI if CONFIDENT_API_KEY is set.
```

### Conversational

```ts
import { ConversationalTestCase, Turn } from "deepeval/test-case";

const convo = new ConversationalTestCase({
  turns: [new Turn({ role: "user", content: "..." }) /* ... */],
});
await new TurnRelevancyMetric({ threshold: 0.5 }).measure(convo);
```

### Arena

```ts
import { compare } from "deepeval";
const wins = await compare({
  arenaTestCases,
  metric: new ArenaGEval({
    /* ... */
  }),
});
```

### Trace metrics (agentic)

```ts
for await (const golden of dataset.evalsIterator({
  metrics: [new TaskCompletionMetric({ model: "gpt-4.1" })],
})) {
  await myObserveWrappedAgent(golden.input); // spans captured + evaluated on resume
}
```
