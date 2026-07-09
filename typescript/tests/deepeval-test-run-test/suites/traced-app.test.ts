import { it } from "vitest";
import { assertTest, Golden } from "deepeval";
import { ragApp } from "../fixtures/rag-app";
import { correctness } from "../fixtures/metrics";

const goldens = [
  new Golden({ input: "What is the capital of France?", expectedOutput: "Paris" }),
  new Golden({
    input: "Who wrote Romeo and Juliet?",
    expectedOutput: "William Shakespeare",
  }),
];

it.each(goldens)("traced app passes span metrics: $input", async (golden) => {
  await ragApp(golden.input); // produces a trace; span metrics are attached
  await assertTest({ golden }); // evaluates the captured trace
});

it("traced app also accepts an explicit trace-level metric", async () => {
  const golden = new Golden({
    input: "What is the capital of France?",
    expectedOutput: "Paris",
  });
  await ragApp(golden.input);
  await assertTest({ golden, metrics: [correctness()] });
});
