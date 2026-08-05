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

// Passing a Golden (rather than a test case) evaluates the trace the app just
// produced, so the metrics come from the spans themselves.
it.each(goldens)("traced app passes span metrics: $input", async (golden) => {
  await ragApp(golden.input); // produces a trace; span metrics are attached
  await expect(golden).toPass();
});

it("traced app also accepts an explicit trace-level metric", async () => {
  const golden = new Golden({
    input: "What is the capital of France?",
    expectedOutput: "Paris",
  });
  await ragApp(golden.input);
  await expect(golden).toPass([correctness()]);
});
