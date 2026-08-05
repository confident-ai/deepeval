/**
 * Component-level evals inside a test — `expect(callback).toPass(...)`.
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

  // The call goes INSIDE expect, so `toPass` evaluates exactly the trace it
  // produced — no ordering to get wrong. Metrics passed here are TRACE-level;
  // the staged LLM metric and the tool metric are evaluated too.
  // `{ golden }` is optional: supply it for expected values / dataset linkage.
  await expect(() =>
    nextLlmSpan({ metrics: [new metrics.AnswerRelevancyMetric()] }, () =>
      ask(golden.input),
    ),
  ).toPass([new metrics.TaskCompletionMetric()], { golden });
}, 120_000);

/* ---------------------------------------------------------------------------
 * Other shapes.
 *
 * Async callback, and keeping the response for ordinary assertions — `toPass`
 * returns the verdict, not your value, so capture it inside the callback:
 *
 *   let answer = "";
 *   await expect(async () => {
 *     answer = (await ask(golden.input)).text;
 *   }).toPass([new metrics.TaskCompletionMetric()], { golden });
 *   expect(answer.toLowerCase()).toContain("tokyo");
 *
 * Span metrics only — no trace-level metric, so no arguments at all:
 *
 *   await expect(() => ask("What's the weather in Tokyo?")).toPass();
 *
 * A plain test case, with no app run involved:
 *
 *   const tc = new LLMTestCase({ input: "q", actualOutput: "a" });
 *   await expect(tc).toPass([new metrics.AnswerRelevancyMetric()]);
 *
 * Rejected on purpose, each with a message telling you the fix:
 *   expect(ask(input))   — the call already started, so its trace is missed
 *   expect(golden)       — ambiguous: there is no way to know which trace
 * ------------------------------------------------------------------------- */
