# DeepEval for TypeScript

> **Status:** Initial version shipping **June 5th**.

DeepEval for TypeScript brings the DeepEval workflow into the JavaScript and TypeScript ecosystem, starting with the Confident AI platform features teams already use to manage datasets, prompts, and evaluation reporting.

This package is designed for TypeScript teams that want first-class access to DeepEval workflows that integrate with Confident AI from the same language they use to build their applications.

## What TypeScript Supports

The TypeScript SDK includes:

- `assertTest()` — run metrics against a single test case and assert it passes
- `evaluate()` — run metrics across multiple test cases and collect results
- `TestRunManager` — in-memory accumulator that records every `assertTest()` call
- Pushing and pulling datasets
- Running and reporting evaluations through Confident AI
- Reading/writing prompts and prompt versions
- Other Confident AI platform interactions

Local execution features, such as LLM-as-a-judge metrics, NLP models, and fully local evaluation, currently remain in the Python package while we expand TypeScript support.

---

## assertTest()

`assertTest()` is the TypeScript equivalent of Python's `assert_test`. It runs one or more metrics against a single test case and **throws an `AssertTestError` if any metric fails**.

Unlike the Python version (which integrates with pytest), `assertTest()` is a plain `async` function that works with any runtime, test framework, or plain script.

### Signature

```ts
async function assertTest(
  testCase: LLMTestCase | ConversationalTestCase,
  metrics: (BaseMetric | BaseConversationalMetric)[],
  options?: AssertTestOptions,
): Promise<void>
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `testCase` | `LLMTestCase \| ConversationalTestCase` | The test case to evaluate |
| `metrics` | `(BaseMetric \| BaseConversationalMetric)[]` | One or more metrics to run concurrently |
| `options.errorConfig` | `ErrorConfig` | Optional error-handling policy (defaults match Python: no ignoring, no skipping) |
| `options.showIndicator` | `boolean` | Show per-metric progress indicators (default: `true`) |

### Behaviour

- All metrics run **concurrently** (`Promise.all`), matching Python's `run_async=True` default.
- The outcome (pass or fail) is recorded in the shared `testRunManager` singleton.
- Throws `AssertTestError` — a subclass of `Error` — when one or more metrics fail. `err.failedMetrics` gives programmatic access to each failure.
- Does **not** depend on Jest, Vitest, or any other test framework.

### Example

```ts
import { assertTest, LLMTestCase, AssertTestError } from "deepeval";
// Metrics come from your metric package — this is a placeholder illustration
import { AnswerRelevancyMetric } from "deepeval/metrics";

const metric = new AnswerRelevancyMetric({ threshold: 0.7 });

const testCase = new LLMTestCase({
  input: "What is the capital of France?",
  actualOutput: "Paris",
});

try {
  await assertTest(testCase, [metric]);
  console.log("Test passed ✅");
} catch (err) {
  if (err instanceof AssertTestError) {
    console.error("Test failed. Failing metrics:", err.failedMetrics);
  }
}
```

### Using with Jest or Vitest

Because `assertTest()` throws a plain `Error`, it integrates with any test framework without any adapter:

```ts
// Jest / Vitest example
it("answer should be relevant", async () => {
  await assertTest(testCase, [new AnswerRelevancyMetric({ threshold: 0.7 })]);
  // assertTest throws AssertTestError → Jest/Vitest marks the test as failed
});
```

### Error policy

| Option | Default | Effect |
|---|---|---|
| `ignoreErrors: false` | default | Metric execution errors propagate as-is |
| `ignoreErrors: true` | opt-in | Errors are captured in `MetricData.error`; the metric is treated as failed |
| `skipOnMissingParams: false` | default | `MissingTestCaseParamsError` propagates |
| `skipOnMissingParams: true` | opt-in | Missing-params errors cause the metric to be **skipped** (not failed) |

---

## TestRunManager

`TestRunManager` is an **in-memory accumulator** that records every `assertTest()` result. It mirrors Python's `global_test_run_manager` but is intentionally scoped to the simplest useful interface: record, query, reset.

### Singleton

A pre-constructed singleton is exported as `testRunManager`. You rarely need to instantiate `TestRunManager` yourself.

```ts
import { testRunManager } from "deepeval";
```

### API

```ts
class TestRunManager {
  /** Append a TestResult (called automatically by assertTest). */
  record(result: TestResult): void;

  /** Shallow-copy snapshot of all recorded results. */
  getResults(): readonly TestResult[];

  /** Aggregate statistics: total, passed, failed, evaluationCost. */
  getSummary(): TestRunSummary;

  /** Clear all results and reset internal state. */
  reset(): void;
}
```

### Example

```ts
import { assertTest, testRunManager, LLMTestCase } from "deepeval";

// Run a series of evaluations
await assertTest(tc1, [metric]).catch(() => {});
await assertTest(tc2, [metric]).catch(() => {});
await assertTest(tc3, [metric]).catch(() => {});

const summary = testRunManager.getSummary();
console.log(`${summary.passed}/${summary.total} passed`);
// e.g. "2/3 passed"

// Reset between test suites
testRunManager.reset();
```

### TestRunSummary shape

```ts
interface TestRunSummary {
  total: number;           // total assertTest() calls
  passed: number;          // calls where all metrics passed
  failed: number;          // calls where at least one metric failed
  evaluationCost: number | undefined;  // cumulative cost (undefined if no metric reported cost)
  results: readonly TestResult[];
}
```

---

## Roadmap

Our next milestone is to reach **80% feature parity** across the Confident AI integration surface by the **end of July**. This includes:

- **Shared prompt templates** — one source of truth for prompt templates, consumed by both Python and TypeScript so the implementations stay aligned.
- **TypeScript-native APIs** — equivalents for the relevant Python functions and classes, shaped to feel natural in TypeScript while staying familiar to DeepEval users.
- **Dedicated TypeScript docs** — TypeScript examples and guides alongside the existing Python documentation.

## Python vs TypeScript

Python remains DeepEval's most complete implementation and the first place new local evaluation capabilities will land. TypeScript complements that foundation by making DeepEval workflows that integrate with Confident AI available to JavaScript and TypeScript teams, with a clear path toward broader feature coverage.
