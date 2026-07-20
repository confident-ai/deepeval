import { it } from "vitest";
import { assertTest, LLMTestCase } from "deepeval";
import { answerRelevancy } from "../fixtures/metrics";

it("fails when the answer is irrelevant", async () => {
  const testCase = new LLMTestCase({
    input: "What is the capital of France?",
    actualOutput: "I really enjoy hiking and the weather has been lovely lately.",
  });
  await assertTest(testCase, [answerRelevancy()]);
});
