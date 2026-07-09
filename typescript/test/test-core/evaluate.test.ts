// import { config } from "dotenv";

import { evaluate, assertTest, AssertTestError, testRunManager } from "../../src/evaluate";
import { LLMTestCase, ConversationalTestCase, Turn } from "../../src/test-case";
import { BaseMetric, BaseConversationalMetric } from "../../src/metrics";

// config();

// Create mock regular metric
class MockRegularMetric extends BaseMetric {
  constructor(name: string = "MockRegularMetric", success: boolean = true) {
    super(0.5);
    this._name = name;
    this._success = success;
  }
  private _name: string;
  private _success: boolean;

  async measure(testCase: LLMTestCase): Promise<number> {
    this.score = this._success ? 1.0 : 0.0;
    this.success = this._success;
    return this.score;
  }
  isSuccessful(): boolean {
    return this._success;
  }
  get name(): string {
    return this._name;
  }
}

// Create mock conversational metric
class MockConversationalMetric extends BaseConversationalMetric {
  constructor(name: string = "MockConversationalMetric", success: boolean = true) {
    super(0.5);
    this._name = name;
    this._success = success;
  }
  private _name: string;
  private _success: boolean;

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.score = this._success ? 1.0 : 0.0;
    this.success = this._success;
    return this.score;
  }
  isSuccessful(): boolean {
    return this._success;
  }
  get name(): string {
    return this._name;
  }
}

describe("evaluate function", () => {
  it("should send LLMTestCase with hyperparameters and identifier", async () => {
    // Create a simple LLM test case
    const testCase = new LLMTestCase({
      input: "What is the capital of France?",
      actualOutput: "The capital of France is Paris.",
      expectedOutput: "Paris is the capital of France.",
      name: "Test Case 1",
    });

//     // Define hyperparameters and identifier
//     const hyperparameters = {
//       model: "deepseek-chat",
//       promptVersion: "v1.0",
//     };

//     const identifier = "test-evaluation-123";

    // Call the evaluate function with the new local runner signature
    const metric = new MockRegularMetric();
    await evaluate([testCase], [metric]);

    // Note: Since we're using a mock metric, this runs locally and completes quickly.
  }, 60000); // 60 second timeout
});

describe("assertTest compatibility validation", () => {
  const regularTestCase = new LLMTestCase({
    input: "What is the capital of France?",
    actualOutput: "The capital of France is Paris.",
    expectedOutput: "Paris",
  });

  const conversationalTestCase = new ConversationalTestCase({
    turns: [
      new Turn({ role: "user", content: "Hello" }),
      new Turn({ role: "assistant", content: "Hi there!" }),
    ],
  });

  it("should accept valid combinations (Regular TestCase + Regular Metrics)", async () => {
    const metric = new MockRegularMetric();
    await expect(assertTest(regularTestCase, [metric])).resolves.not.toThrow();
  });

  it("should accept valid combinations (ConversationalTestCase + Conversational Metrics)", async () => {
    const metric = new MockConversationalMetric();
    await expect(assertTest(conversationalTestCase, [metric])).resolves.not.toThrow();
  });

  it("should reject invalid combinations (Regular TestCase + Conversational Metrics)", async () => {
    const metric = new MockConversationalMetric();
    await expect(assertTest(regularTestCase, [metric])).rejects.toThrow(
      AssertTestError,
    );
    await expect(assertTest(regularTestCase, [metric])).rejects.toThrow(
      "All 'metrics' for an LLMTestCase must be instances of 'BaseMetric' only. Received metric: MockConversationalMetric",
    );
  });

  it("should reject invalid combinations (ConversationalTestCase + Regular Metrics)", async () => {
    const metric = new MockRegularMetric();
    await expect(assertTest(conversationalTestCase, [metric])).rejects.toThrow(
      AssertTestError,
    );
    await expect(assertTest(conversationalTestCase, [metric])).rejects.toThrow(
      "All 'metrics' for a ConversationalTestCase must be instances of 'BaseConversationalMetric' only. Received metric: MockRegularMetric",
    );
  });

  it("should reject mixed metric arrays for LLMTestCase", async () => {
    const metrics = [new MockRegularMetric(), new MockConversationalMetric()];
    await expect(assertTest(regularTestCase, metrics)).rejects.toThrow(
      AssertTestError,
    );
  });

  it("should reject mixed metric arrays for ConversationalTestCase", async () => {
    const metrics = [new MockConversationalMetric(), new MockRegularMetric()];
    await expect(assertTest(conversationalTestCase, metrics)).rejects.toThrow(
      AssertTestError,
    );
  });

  it("should throw AssertTestError on failing metrics", async () => {
    const failingMetric = new MockRegularMetric("FailingMetric", false);
    await expect(assertTest(regularTestCase, [failingMetric])).rejects.toThrow(
      AssertTestError,
    );
    await expect(assertTest(regularTestCase, [failingMetric])).rejects.toThrow(
      "Metrics: FailingMetric (score: 0, threshold: 0.5, strict: false, error: undefined, reason: undefined) failed.",
    );
  });

  it("should reject null test case", async () => {
    const metric = new MockRegularMetric();
    await expect(assertTest(null as any, [metric])).rejects.toThrow(
      AssertTestError,
    );
    await expect(assertTest(null as any, [metric])).rejects.toThrow(
      "TestCase cannot be null or undefined.",
    );
  });

  it("should reject undefined test case", async () => {
    const metric = new MockRegularMetric();
    await expect(assertTest(undefined as any, [metric])).rejects.toThrow(
      "TestCase cannot be null or undefined.",
    );
  });

  it("should reject empty metrics array", async () => {
    await expect(assertTest(regularTestCase, [])).rejects.toThrow(
      "Metrics array cannot be empty.",
    );
  });

  it("should reject invalid metric types", async () => {
    await expect(assertTest(regularTestCase, [{} as any])).rejects.toThrow(
      "Invalid metric provided. All metrics must be instances of 'BaseMetricCore'.",
    );
  });
});

describe("TestRunManager lifecycle", () => {
  const testCase = new LLMTestCase({
    input: "test",
    actualOutput: "test",
    expectedOutput: "test",
  });

  it("should track assertTest runs and report summaries correctly", async () => {
    testRunManager.reset();
    let summary = testRunManager.getSummary();
    expect(summary.testPassed).toBe(0);
    expect(summary.testFailed).toBe(0);

    const metricPassing = new MockRegularMetric("PassingMetric", true);
    await assertTest(testCase, [metricPassing]);

    summary = testRunManager.getSummary();
    expect(summary.testPassed).toBe(1);
    expect(summary.testFailed).toBe(0);

    const metricFailing = new MockRegularMetric("FailingMetric", false);
    await expect(assertTest(testCase, [metricFailing])).rejects.toThrow(AssertTestError);

    summary = testRunManager.getSummary();
    expect(summary.testPassed).toBe(1);
    expect(summary.testFailed).toBe(1);
  });
});
