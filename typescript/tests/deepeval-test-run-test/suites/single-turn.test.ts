import { it } from "vitest";
import { assertTest, LLMTestCase } from "deepeval";
import {
  answerRelevancy,
  faithfulness,
  correctness,
} from "../fixtures/metrics";

const cases = [
  {
    input: "What is the capital of France?",
    actualOutput: "The capital of France is Paris.",
    expectedOutput: "Paris",
    retrievalContext: ["France is a country in Europe. Its capital is Paris."],
  },
  {
    input: "Who wrote Romeo and Juliet?",
    actualOutput: "Romeo and Juliet was written by William Shakespeare.",
    expectedOutput: "William Shakespeare",
    retrievalContext: [
      "Romeo and Juliet is a tragedy written by William Shakespeare.",
    ],
  },
];

it.each(cases)("single-turn passes all metrics: $input", async (c) => {
  const testCase = new LLMTestCase({
    input: c.input,
    actualOutput: c.actualOutput,
    expectedOutput: c.expectedOutput,
    retrievalContext: c.retrievalContext,
  });

  await assertTest(testCase, [
    answerRelevancy(),
    faithfulness(),
    correctness(),
  ]);
});
