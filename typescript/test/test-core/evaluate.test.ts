import { evaluate } from "../../src/evaluate";
import { LLMTestCase } from "../../src/test-case";
import { ExactMatchMetric } from "../../src/metrics";

describe("evaluate function", () => {
  it("should pass exact match metric for identical output", async () => {
    const testCase = new LLMTestCase({
      input: "What is the capital of France?",
      actualOutput: "Paris",
      expectedOutput: "Paris",
      name: "Test Case 1",
    });

    const metric = new ExactMatchMetric({ threshold: 1 });

    const result = await evaluate([testCase], [metric], {
      displayConfig: { showIndicator: false, printResults: false },
    });

    expect(result.testResults).toHaveLength(1);
    expect(result.testResults[0].success).toBe(true);
    expect(result.testResults[0].metricsData?.[0]?.score).toBe(1);
  });

  it("should fail exact match metric for mismatched output", async () => {
    const testCase = new LLMTestCase({
      input: "What is the capital of France?",
      actualOutput: "London",
      expectedOutput: "Paris",
      name: "Test Case 2",
    });

    const metric = new ExactMatchMetric({ threshold: 1 });

    const result = await evaluate([testCase], [metric], {
      displayConfig: { showIndicator: false, printResults: false },
    });

    expect(result.testResults).toHaveLength(1);
    expect(result.testResults[0].success).toBe(false);
    expect(result.testResults[0].metricsData?.[0]?.score).toBe(0);
  });
});
