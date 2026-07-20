import { it, expect } from "vitest";
import "deepeval/vitest";
import { LLMTestCase } from "deepeval";
import { answerRelevancy, faithfulness } from "../fixtures/metrics";

it("toPassMetric passes for a relevant answer", async () => {
  const testCase = new LLMTestCase({
    input: "What is the capital of France?",
    actualOutput: "The capital of France is Paris.",
  });
  await expect(testCase).toPassMetric(answerRelevancy());
});

it("toPassAll passes for a grounded, relevant answer", async () => {
  const testCase = new LLMTestCase({
    input: "Who wrote Romeo and Juliet?",
    actualOutput: "Romeo and Juliet was written by William Shakespeare.",
    retrievalContext: [
      "Romeo and Juliet is a tragedy written by William Shakespeare.",
    ],
  });
  await expect(testCase).toPassAll([answerRelevancy(), faithfulness()]);
});
