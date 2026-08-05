import {
  runMetrics,
  globalResultCollector,
} from "../../src/evaluate/test-run";
import { LLMTestCase } from "../../src/test-case";
import { BaseMetric, BaseConversationalMetric } from "../../src/metrics";
import { DeepEvalError, MissingTestCaseParamsError } from "../../src/errors";
import { Golden } from "../../src/dataset";
import {
  getIsRunningDeepEval,
  setIsRunningDeepEval,
} from "../../src/utils";

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

describe("toPass — explicit test case shape", () => {
  it("passes when the metric passes", async () => {
    await expect(runMetrics(llmTestCase(), [passing()])).resolves.toMatchObject({
      pass: true,
    });
  });

  it("fails with a message naming the failing metric", async () => {
    const outcome = await runMetrics(llmTestCase(), [failing()]);
    expect(outcome.pass).toBe(false);
    expect(outcome.failureMessage).toMatch(
      /FailMetric.*score: 0\.2.*threshold: 0\.5/,
    );
  });

  it("fails when any metric in the list fails", async () => {
    const outcome = await runMetrics(llmTestCase(), [passing(), failing()]);
    expect(outcome.pass).toBe(false);
    expect(outcome.failureMessage).toContain("FailMetric");
    expect(outcome.failureMessage).not.toContain("PassMetric");
  });

  it("rejects when a single metric's type doesn't match (no silent filtering)", async () => {
    // Even alongside a valid passing metric, one mismatched metric is fatal.
    await expect(
      runMetrics(llmTestCase(), [passing(), new FakeConversationalMetric()]),
    ).rejects.toThrow(DeepEvalError);
    await expect(
      runMetrics(llmTestCase(), [passing(), new FakeConversationalMetric()]),
    ).rejects.toThrow(/single-turn metrics only/);
  });

  it("rejects when every metric is the wrong type", async () => {
    await expect(
      runMetrics(llmTestCase(), [new FakeConversationalMetric()]),
    ).rejects.toThrow(DeepEvalError);
  });

  it("throws when no metrics are provided", async () => {
    await expect(runMetrics(llmTestCase(), [])).rejects.toThrow(DeepEvalError);
    await expect(runMetrics(llmTestCase())).rejects.toThrow(DeepEvalError);
  });

  it("propagates a metric error under strict config", async () => {
    await expect(runMetrics(llmTestCase(), [throwing()])).rejects.toThrow(
      "boom",
    );
  });

  it("propagates MissingTestCaseParamsError (does not silently pass)", async () => {
    await expect(runMetrics(llmTestCase(), [missingParams()])).rejects.toThrow(
      MissingTestCaseParamsError,
    );
  });
});

describe("toPass — callback (trace-scoped) shape", () => {
  it("throws when the callback produces no trace", async () => {
    await expect(runMetrics(() => undefined, [passing()])).rejects.toThrow(
      /no trace was produced/,
    );
  });

  it("throws when the callback produces no trace and there are no metrics", async () => {
    await expect(runMetrics(() => undefined)).rejects.toThrow(DeepEvalError);
  });

  it("rejects multi-turn metrics at the trace level", async () => {
    await expect(
      runMetrics(() => undefined, [new FakeConversationalMetric()]),
    ).rejects.toThrow(/trace-level metrics must be single-turn/);
  });

  it("awaits an async callback before evaluating", async () => {
    let ran = false;
    await expect(
      runMetrics(async () => {
        await new Promise((r) => setTimeout(r, 5));
        ran = true;
      }),
    ).rejects.toThrow(/no trace was produced/);
    // The rejection is about the missing trace, not about the callback: it ran,
    // and it was awaited to completion first.
    expect(ran).toBe(true);
  });

  it("rejects a promise receiver, which would run outside the capture window", async () => {
    await expect(
      runMetrics(Promise.resolve("already running") as never),
    ).rejects.toThrow(/received a promise/);
  });

  it("rejects a golden receiver and points at the callback form", async () => {
    await expect(
      runMetrics(new Golden({ input: "hi" }) as never, [passing()]),
    ).rejects.toThrow(/expect\(golden\)\.toPass\(\) is not supported/);
  });
});

describe("toPass — result collection (CLI-gated)", () => {
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
    await runMetrics(llmTestCase(), [passing()]);
    expect(globalResultCollector.size).toBe(0);
  });

  it("collects the evaluated case when running via the CLI", async () => {
    setIsRunningDeepEval(true);
    await runMetrics(llmTestCase(), [passing()]);
    expect(globalResultCollector.size).toBe(1);
    expect(globalResultCollector.getCases()[0].metricsData[0].name).toBe(
      "PassMetric",
    );
  });

  it("collects failing cases too, not just passing ones", async () => {
    setIsRunningDeepEval(true);
    const outcome = await runMetrics(llmTestCase(), [failing()]);
    expect(outcome.pass).toBe(false);
    expect(globalResultCollector.size).toBe(1);
    expect(globalResultCollector.getCases()[0].metricsData[0].success).toBe(
      false,
    );
  });
});
