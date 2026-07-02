import { config } from "dotenv";

import { evaluate } from "../../src/confident/evaluate";
import { LLMTestCase } from "../../src/test-case";

config();

describe("evaluate function", () => {
  it("should send LLMTestCase with hyperparameters and identifier", async () => {
    // Create a simple LLM test case
    const testCase = new LLMTestCase({
      input: "What is the capital of France?",
      actualOutput: "The capital of France is Paris.",
      expectedOutput: "Paris is the capital of France.",
      name: "Test Case 1",
    });

    // Define hyperparameters and identifier
    const hyperparameters = {
      model: "deepseek-chat",
      promptVersion: "v1.0",
    };

    const identifier = "test-evaluation-123";

    // Call the evaluate function
    await evaluate({
      metricCollection: "test_collection_1",
      llmTestCases: [testCase],
      hyperparameters: hyperparameters,
      identifier: identifier,
    });

    // Note: Since we're not mocking, this will make an actual API call
    // The test will pass if no error is thrown
  }, 60000); // 60 second timeout for API call
});
