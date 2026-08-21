/**
 * Component-level evals inside a test — `expect(golden).toPass(..., { task })`.
 *
 *   npx deepeval test run examples/component-evals/matcher.test.ts
 *
 * The CLI wrapper is what sets up the test-run scope and posts results. Plain
 * `vitest run` also executes the file, but scores nothing beyond the local table.
 */
import { expect, test } from "vitest";
import "deepeval/vitest"; // registers the `toPass` matcher

import { metrics } from "deepeval";
import { Golden } from "deepeval/dataset";
import { nextLlmSpan } from "deepeval/tracing";

import { ask } from "./weather-agent";

test("weather agent answers with the tool's data", async () => {
  const golden = new Golden({ input: "What's the weather in Tokyo?" });

  // The golden is the subject; `task` produces the trace judged against it.
  // Metrics passed here are TRACE-level; the staged LLM metric and the tool
  // metric are evaluated too.
  await expect(golden).toPass([new metrics.TaskCompletionMetric()], {
    task: (g) =>
      nextLlmSpan({ metrics: [new metrics.AnswerRelevancyMetric()] }, () =>
        ask(g.input),
      ),
  });
}, 120_000);

/* ---------------------------------------------------------------------------
 * Other shapes.
 *
 * Keeping the response for ordinary assertions — capture it inside `task`:
 *
 *   let answer = "";
 *   await expect(golden).toPass([new metrics.TaskCompletionMetric()], {
 *     task: async (g) => {
 *       answer = (await ask(g.input)).text;
 *     },
 *   });
 *   expect(answer.toLowerCase()).toContain("tokyo");
 *
 * Span metrics only — empty metrics array, still pass `task`:
 *
 *   await expect(golden).toPass([], { task: (g) => ask(g.input) });
 *
 * A plain test case, with no app run involved:
 *
 *   const tc = new LLMTestCase({ input: "q", actualOutput: "a" });
 *   await expect(tc).toPass([new metrics.AnswerRelevancyMetric()]);
 *
 * Rejected on purpose, each with a message telling you the fix:
 *   expect(ask(input))          — the call already started, so its trace is missed
 *   expect(golden).toPass([...]) — missing `task`; no way to know which trace
 * ------------------------------------------------------------------------- */
