import { it, expect } from "vitest";
import "deepeval/vitest";
import { Golden } from "deepeval";
import { ragApp } from "../fixtures/rag-app";
import { correctness } from "../fixtures/metrics";

const goldens = [
  new Golden({ input: "What is the capital of France?", expectedOutput: "Paris" }),
  new Golden({
    input: "Who wrote Romeo and Juliet?",
    expectedOutput: "William Shakespeare",
  }),
];

// Passing a callback (rather than a test case) runs the app inside the assertion
// and evaluates the trace it produced, so the metrics come from the spans
// themselves — no metrics argument needed.
it.each(goldens)("traced app passes span metrics: $input", async (golden) => {
  await expect(() => ragApp(golden.input)).toPass();
});

it("traced app also accepts an explicit trace-level metric", async () => {
  const golden = new Golden({
    input: "What is the capital of France?",
    expectedOutput: "Paris",
  });
  // The golden is the subject; `task` produces the trace judged against it.
  await expect(golden).toPass([correctness()], {
    task: (g) => ragApp(g.input),
  });
});
