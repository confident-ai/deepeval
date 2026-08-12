# Vitest End-to-End Evals (TypeScript)

Use this for the default CI/CD path in TypeScript/JavaScript projects.
End-to-end Vitest evals run one golden through the real app per test. The
`deepeval` npm package plugs into Vitest through the `toPass` matcher and the
`npx deepeval test run` command. Every eval test file must contain
`import "deepeval/vitest"`, which registers the matcher and its types.

If tracing or a supported integration is available, make the golden the
`expect` subject and let a `task` callback run the traced app:

```typescript
await expect(golden).toPass(METRICS, { task: (g) => runTracedAiApp(g.input) });
```

Use `templates/single_turn_tracing.test.ts` for the default single-turn path.
Use `templates/single_turn_no_tracing.test.ts` only when the user explicitly
declines tracing or no tracing path is viable.

## Default Shape

Use an integration callback/instrumentation hook when one exists. If no native
integration exists, wrap the app entry point with `observe()` and update the
trace output (see the `deepeval-tracing` skill).

```typescript
import { expect, it } from "vitest";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import "deepeval/vitest";

import { SINGLE_TURN_TRACE_METRICS } from "./metrics";
import { runTracedAiApp } from "./ai-app";

const dataset = new EvaluationDataset();
await dataset.addGoldensFromJSON({ filePath: "tests/evals/.dataset.json" });

it.each(dataset.goldens as Golden[])("llm app: $input", async (golden) => {
  await expect(golden).toPass(SINGLE_TURN_TRACE_METRICS, {
    task: (g) => runTracedAiApp(g.input),
  });
});
```

Run with:

```bash
npx deepeval test run tests/evals/<app>.test.ts
```

Do not default to the raw `npx vitest` command — `npx deepeval test run` is
what accumulates results into a test run and posts it to Confident AI.

## Integration-First Rule

Instrument the app for tracing using the `deepeval-tracing` skill — it covers
the integration selection rule and manual `observe()`. A native framework
integration should be the first implementation path; manual `observe()` is the
fallback. This eval suite runs against the app once it is traced.

For integration-backed Vitest evals, the shape is still:

```typescript
it.each(dataset.goldens as Golden[])("agent: $input", async (golden) => {
  await expect(golden).toPass(SINGLE_TURN_TRACE_METRICS, {
    task: (g) => runAiAppWithIntegrationTracing(g.input),
  });
});
```

Do not translate these traced runs into hand-built `LLMTestCase`s.

## Span Metrics In The Same Eval

See `references/traced-evals.md` for the full span-metric reference.

Component-level metrics are part of the single-turn tracing eval. Do not create
a separate component test file. Attach span metrics at the component boundary
and keep `expect(golden).toPass(...)` at the trace level.

Use `next*Span({ metrics }, callback)` when an integration creates the
component span. Unlike Python's context manager, the TypeScript helper takes
the traced call as a callback:

```typescript
import { nextRetrieverSpan } from "deepeval/tracing";

import { RETRIEVER_SPAN_METRICS } from "./metrics";

it.each(dataset.goldens as Golden[])("agent: $input", async (golden) => {
  await expect(golden).toPass(SINGLE_TURN_TRACE_METRICS, {
    task: (g) =>
      nextRetrieverSpan({ metrics: RETRIEVER_SPAN_METRICS }, () =>
        runAiAppWithIntegrationTracing(g.input),
      ),
  });
});
```

Use `observe({ metrics: [...] })` when manually instrumenting the component or
when the integration supports observed component spans.

## No-Tracing Fallback

Only use the no-tracing template when tracing is intentionally out of scope. In
that case, a small wrapper around the AI app call is acceptable because this
path constructs the minimal `LLMTestCase` from AI app output and golden
reference fields before calling `expect(testCase).toPass(metrics)`:

```typescript
const testCase = new LLMTestCase({
  input: golden.input,
  actualOutput: await runAiApp(golden.input),
  expectedOutput: golden.expectedOutput,
  retrievalContext: golden.retrievalContext,
});
await expect(testCase).toPass(SINGLE_TURN_NO_TRACING_METRICS);
```

## Useful `npx deepeval test run` Flags

Check available flags when unsure:

```bash
npx deepeval test run --help
```

Use these frequently:

| Flag | Use when |
| --- | --- |
| `--identifier`, `-i` | Label the run with useful context, for example `iterating-on-retrieval-round-1`. Note: in the TypeScript CLI `-i` is `--identifier`, not `--ignore-errors` as in Python. |
| `--ignore-errors` | Continue the run when individual DeepEval evaluation errors occur. Useful for large datasets. No short form. |
| `--skip-on-missing-params`, `-s` | Skip test cases missing fields required by a metric instead of failing the whole run. Useful when datasets are large or partly incomplete. |
| `--max-concurrent <n>` | Cap how many metrics evaluate at once. Lower it for rate-limited model providers. |
| `--display`, `-d` | Control which test cases are displayed (`all`, `passing`, `failing`). Use when output is too noisy. |
| `--use-cache`, `-c` | Reuse cached metric results where possible across reruns. |

There is no `--num-processes` flag — Vitest parallelizes test files by
default. Use `--max-concurrent` to bound metric concurrency instead.

For first runs on non-trivial datasets, a good starting command is:

```bash
npx deepeval test run tests/evals/<app>.test.ts \
  --identifier "iterating-on-<purpose>-round-1" \
  --ignore-errors \
  --skip-on-missing-params
```

Use purpose-based identifiers because they are easier to scan locally and look
better in Confident AI reports. Keep them short and kebab-case.

## Conversation E2E

For chatbot / multi-turn agent use cases, use `templates/multi_turn_e2e.test.ts`.
It must simulate conversational test cases after loading the dataset, then
parametrize over the simulated test cases. Vitest test files are ES modules, so
top-level `await` on `simulate(...)` works.

Multi-turn end-to-end evals must use multi-turn conversational metrics such as
`ConversationCompletenessMetric`, `RoleAdherenceMetric`, `TurnRelevancyMetric`,
or `ConversationalGEval`. Do not use single-turn `LLMTestCase` metrics for
multi-turn evals.

The minimal shape is:

```typescript
import { expect, it } from "vitest";
import { ConversationSimulator } from "deepeval";
import { ConversationalGolden, EvaluationDataset } from "deepeval/dataset";
import "deepeval/vitest";

import { MULTI_TURN_METRICS } from "./metrics";
import { chatbotCallback } from "./ai-app";

const MAX_TURNS = 10;

const dataset = new EvaluationDataset();
await dataset.addGoldensFromJSON({ filePath: "tests/evals/.dataset.json" });

const simulator = new ConversationSimulator({ modelCallback: chatbotCallback });
const testCases = await simulator.simulate({
  conversationalGoldens: dataset.goldens as ConversationalGolden[],
  maxUserSimulations: MAX_TURNS,
});

it.each(testCases)("conversation: $scenario", async (testCase) => {
  await expect(testCase).toPass(MULTI_TURN_METRICS);
});
```

The `modelCallback` receives the simulated user's next message, the turns so
far, and a stable `threadId`, and must return a `Turn` whose `role` is
`assistant`.

## Script Fallback

Only create a standalone script if the user pushes back on Vitest. Explain that
Vitest is preferred because it leaves a durable eval suite the user can rerun in
CI. For traced single-turn scripts, use `evalsIterator` with goldens:

```typescript
for await (const golden of dataset.evalsIterator({
  metrics: SINGLE_TURN_TRACE_METRICS,
})) {
  await runAiAppWithIntegrationTracing(golden.input);
}
```

Use `evaluate()` only when it is a better fit for an already-built list of test
cases. Never call `evaluate()` inside a file executed by
`npx deepeval test run` — it posts a second, separate test run to Confident AI
instead of joining the CLI's run.
