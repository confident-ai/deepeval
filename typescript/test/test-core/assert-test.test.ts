import {
  assertTest,
  AssertionFailedError,
  globalResultCollector,
} from "../../src/evaluate/assert-test";
import { LLMTestCase } from "../../src/test-case";
import { BaseMetric, BaseConversationalMetric } from "../../src/metrics";
import { DeepEvalError, MissingTestCaseParamsError } from "../../src/errors";
import { Golden } from "../../src/dataset";
import { getIsRunningDeepEval, setIsRunningDeepEval } from "../../src/utils";

// A deterministic single-turn metric. `impl` mutates the metric's result state
// the way a real `measure()` would (runMetric reads score/success afterward).
class FakeMetric extends BaseMetric {
  private readonly label: string;
  private readonly impl: (self: FakeMetric) => number;

  constructor(
    label: string,
    threshold: number,
    impl: (self: FakeMetric) => number,
  ) {
    super(threshold);
    this.label = label;
    this.impl = impl;
  }

  get name(): string {
    return this.label;
  }

  isSuccessful(): boolean {
    return this.success ?? false;
  }

  measure(): number {
    return this.impl(this);
  }
}

class FakeConversationalMetric extends BaseConversationalMetric {
  constructor() {
    super(0.5);
  }

  get name(): string {
    return "FakeConversationalMetric";
  }

  isSuccessful(): boolean {
    return this.success ?? false;
  }

  measure(): number {
    this.score = 1;
    this.success = true;
    return 1;
  }
}

const passing = () =>
  new FakeMetric("PassMetric", 0.5, (self) => {
    self.score = 0.9;
    self.success = true;
    return 0.9;
  });

const failing = () =>
  new FakeMetric("FailMetric", 0.5, (self) => {
    self.score = 0.2;
    self.success = false;
    return 0.2;
  });

const throwing = () =>
  new FakeMetric("ThrowMetric", 0.5, () => {
    throw new Error("boom");
  });

const missingParams = () =>
  new FakeMetric("MissingMetric", 0.5, () => {
    throw new MissingTestCaseParamsError("missing input");
  });

const llmTestCase = () =>
  new LLMTestCase({ input: "What is 2+2?", actualOutput: "4" });

describe("assertTest — explicit test case shape", () => {
  it("resolves when the metric passes", async () => {
    await expect(
      assertTest(llmTestCase(), [passing()]),
    ).resolves.toBeUndefined();
  });

  it("throws AssertionFailedError naming the failing metric", async () => {
    await expect(assertTest(llmTestCase(), [failing()])).rejects.toThrow(
      AssertionFailedError,
    );
    await expect(assertTest(llmTestCase(), [failing()])).rejects.toThrow(
      /FailMetric.*score: 0\.2.*threshold: 0\.5/,
    );
  });

  it("rejects when a single metric's type doesn't match (no silent filtering)", async () => {
    // Even alongside a valid passing metric, one mismatched metric is fatal.
    await expect(
      assertTest(llmTestCase(), [passing(), new FakeConversationalMetric()]),
    ).rejects.toThrow(DeepEvalError);
    await expect(
      assertTest(llmTestCase(), [passing(), new FakeConversationalMetric()]),
    ).rejects.toThrow(/single-turn metrics only/);
  });

  it("rejects when every metric is the wrong type", async () => {
    await expect(
      assertTest(llmTestCase(), [new FakeConversationalMetric()]),
    ).rejects.toThrow(DeepEvalError);
  });

  it("throws when no metrics are provided", async () => {
    await expect(assertTest(llmTestCase(), [])).rejects.toThrow(DeepEvalError);
  });

  it("propagates a metric error under strict config", async () => {
    await expect(assertTest(llmTestCase(), [throwing()])).rejects.toThrow(
      "boom",
    );
  });

  it("propagates MissingTestCaseParamsError (does not silently pass)", async () => {
    await expect(assertTest(llmTestCase(), [missingParams()])).rejects.toThrow(
      MissingTestCaseParamsError,
    );
  });
});

describe("assertTest — trace-scoped shape", () => {
  it("throws when there is no active trace", async () => {
    await expect(
      assertTest({ golden: new Golden({ input: "hi" }), metrics: [passing()] }),
    ).rejects.toThrow(DeepEvalError);
  });
});

describe("assertTest — result collection (CLI-gated)", () => {
  const wasRunning = getIsRunningDeepEval();

  beforeEach(() => {
    globalResultCollector.reset();
  });

  afterEach(() => {
    setIsRunningDeepEval(wasRunning);
    globalResultCollector.reset();
  });

  it("does NOT collect when not running via the CLI", async () => {
    setIsRunningDeepEval(false);
    await assertTest(llmTestCase(), [passing()]);
    expect(globalResultCollector.size).toBe(0);
  });

  it("collects the evaluated case when running via the CLI", async () => {
    setIsRunningDeepEval(true);
    await assertTest(llmTestCase(), [passing()]);
    expect(globalResultCollector.size).toBe(1);
    expect(globalResultCollector.getCases()[0].metricsData[0].name).toBe(
      "PassMetric",
    );
  });

  it("collects failing cases too (records before it throws)", async () => {
    setIsRunningDeepEval(true);
    await expect(assertTest(llmTestCase(), [failing()])).rejects.toThrow(
      AssertionFailedError,
    );
    expect(globalResultCollector.size).toBe(1);
    expect(globalResultCollector.getCases()[0].metricsData[0].success).toBe(
      false,
    );
  });
});
