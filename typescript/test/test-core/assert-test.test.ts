/**
 * Unit tests for assertTest() and TestRunManager.
 *
 * These tests are framework-agnostic at the implementation level — they use
 * Jest only as the test runner (the project's existing choice) and do not
 * rely on any Jest-specific integration in the production code.
 *
 * All metric stubs are plain TypeScript objects; no HTTP calls are made.
 */

import { assertTest, AssertTestError } from "../../src/evaluate/assert-test";
import {
  TestRunManager,
  testRunManager,
} from "../../src/evaluate/test-run-manager";
import { LLMTestCase } from "../../src/test-case/llm-test-case";
import { ConversationalTestCase, Turn } from "../../src/test-case/conversational-test-case";
import { BaseMetric } from "../../src/metrics/base-metrics";
import { BaseConversationalMetric } from "../../src/metrics/base-conversational-metric";
import { MissingTestCaseParamsError } from "../../src/errors";
import { SingleTurnParams } from "../../src/test-case/llm-test-case";
import { MultiTurnParams } from "../../src/test-case/conversational-test-case";

// ---------------------------------------------------------------------------
// Minimal stub helpers
// ---------------------------------------------------------------------------

/**
 * Create a minimal `LLMTestCase` with only the required fields.
 */
function makeLLMTestCase(name?: string): LLMTestCase {
  return new LLMTestCase({
    input: "What is 2 + 2?",
    actualOutput: "4",
    name,
  });
}

/**
 * Create a metric stub whose `measure()` resolves to `score` after an
 * optional async `delayMs`.
 */
function makeMetric(opts: {
  name?: string;
  score: number;
  threshold?: number;
  delayMs?: number;
  /** When set, `measure()` throws this error instead of resolving. */
  throwError?: Error;
}): BaseMetric {
  const threshold = opts.threshold ?? 0.5;
  const metricName = opts.name ?? "StubMetric";

  return new (class extends BaseMetric {
    get name(): string {
      return metricName;
    }

    requiredParams: Array<SingleTurnParams> = [];

    async measure(_tc: LLMTestCase): Promise<number> {
      if (opts.delayMs) {
        await new Promise((r) => setTimeout(r, opts.delayMs));
      }
      if (opts.throwError) {
        throw opts.throwError;
      }
      this.score = opts.score;
      this.success = opts.score >= this.threshold;
      this.reason = `score ${opts.score} vs threshold ${this.threshold}`;
      return opts.score;
    }

    isSuccessful(): boolean {
      return (this.score ?? 0) >= this.threshold;
    }
  })(threshold, { showIndicator: false });
}

/**
 * Create a conversational metric stub.
 */
function makeConversationalMetric(opts: {
  name?: string;
  score: number;
  threshold?: number;
}): BaseConversationalMetric {
  const threshold = opts.threshold ?? 0.5;
  const metricName = opts.name ?? "StubConversationalMetric";

  return new (class extends BaseConversationalMetric {
    get name(): string {
      return metricName;
    }

    requiredParams: Array<MultiTurnParams> = [];

    async measure(_tc: ConversationalTestCase): Promise<number> {
      this.score = opts.score;
      this.success = opts.score >= this.threshold;
      this.reason = `score ${opts.score} vs threshold ${this.threshold}`;
      return opts.score;
    }

    isSuccessful(): boolean {
      return (this.score ?? 0) >= this.threshold;
    }
  })(threshold, { showIndicator: false });
}

// ---------------------------------------------------------------------------
// Test helpers for isolation
// ---------------------------------------------------------------------------

/**
 * Reset the global testRunManager before each test so results don't bleed
 * across test cases.
 */
beforeEach(() => {
  testRunManager.reset();
});

// ===========================================================================
// assertTest() — core behaviour
// ===========================================================================

describe("assertTest()", () => {
  // -------------------------------------------------------------------------
  // Passing metrics
  // -------------------------------------------------------------------------

  describe("passing metrics", () => {
    it("resolves without throwing when a single metric passes", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.9, threshold: 0.5 });

      await expect(assertTest(tc, [metric])).resolves.toBeUndefined();
    });

    it("resolves without throwing when all metrics pass", async () => {
      const tc = makeLLMTestCase();
      const metrics = [
        makeMetric({ name: "M1", score: 0.8, threshold: 0.5 }),
        makeMetric({ name: "M2", score: 1.0, threshold: 0.9 }),
        makeMetric({ name: "M3", score: 0.6, threshold: 0.6 }),
      ];

      await expect(assertTest(tc, metrics)).resolves.toBeUndefined();
    });

    it("resolves when metric score exactly equals threshold", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.5, threshold: 0.5 });

      await expect(assertTest(tc, [metric])).resolves.toBeUndefined();
    });
  });

  // -------------------------------------------------------------------------
  // Failing metrics
  // -------------------------------------------------------------------------

  describe("failing metrics", () => {
    it("throws AssertTestError when a single metric fails", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.3, threshold: 0.5 });

      await expect(assertTest(tc, [metric])).rejects.toThrow(AssertTestError);
    });

    it("throws AssertTestError containing metric details in the message", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ name: "Relevancy", score: 0.2, threshold: 0.7 });

      const err = await assertTest(tc, [metric]).catch((e) => e);
      expect(err).toBeInstanceOf(AssertTestError);
      expect(err.message).toContain("Relevancy");
      expect(err.message).toContain("0.2");
      expect(err.message).toContain("0.7");
    });

    it("includes only failing metrics in AssertTestError.failedMetrics", async () => {
      const tc = makeLLMTestCase();
      const metrics = [
        makeMetric({ name: "Pass", score: 0.9, threshold: 0.5 }),
        makeMetric({ name: "Fail1", score: 0.1, threshold: 0.5 }),
        makeMetric({ name: "Fail2", score: 0.3, threshold: 0.8 }),
      ];

      const err = await assertTest(tc, metrics).catch((e) => e);
      expect(err).toBeInstanceOf(AssertTestError);
      expect(err.failedMetrics).toHaveLength(2);
      expect(err.failedMetrics.map((m: { name: string }) => m.name)).toEqual(
        expect.arrayContaining(["Fail1", "Fail2"]),
      );
      expect(err.failedMetrics.map((m: { name: string }) => m.name)).not.toContain("Pass");
    });

    it("throws when metric score is just below threshold", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.499, threshold: 0.5 });

      await expect(assertTest(tc, [metric])).rejects.toThrow(AssertTestError);
    });
  });

  // -------------------------------------------------------------------------
  // Async metrics
  // -------------------------------------------------------------------------

  describe("async metrics", () => {
    it("awaits async metric measurement before asserting", async () => {
      const tc = makeLLMTestCase();
      const slowMetric = makeMetric({ score: 0.9, threshold: 0.5, delayMs: 30 });

      await expect(assertTest(tc, [slowMetric])).resolves.toBeUndefined();
    });

    it("runs multiple metrics concurrently", async () => {
      const tc = makeLLMTestCase();
      const start = Date.now();
      const metrics = [
        makeMetric({ name: "M1", score: 0.9, threshold: 0.5, delayMs: 50 }),
        makeMetric({ name: "M2", score: 0.9, threshold: 0.5, delayMs: 50 }),
        makeMetric({ name: "M3", score: 0.9, threshold: 0.5, delayMs: 50 }),
      ];

      await assertTest(tc, metrics);
      const elapsed = Date.now() - start;
      // If run concurrently, total time should be < 2× the individual delay.
      // Allow generous headroom for CI variability.
      expect(elapsed).toBeLessThan(130);
    });

    it("collects errors from multiple failing async metrics", async () => {
      const tc = makeLLMTestCase();
      const metrics = [
        makeMetric({ name: "A", score: 0.1, threshold: 0.5, delayMs: 20 }),
        makeMetric({ name: "B", score: 0.2, threshold: 0.5, delayMs: 10 }),
      ];

      const err = await assertTest(tc, metrics).catch((e) => e);
      expect(err).toBeInstanceOf(AssertTestError);
      expect(err.failedMetrics).toHaveLength(2);
    });
  });

  // -------------------------------------------------------------------------
  // Error handling
  // -------------------------------------------------------------------------

  describe("error handling", () => {
    it("re-throws metric errors when ignoreErrors is false (default)", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({
        score: 0,
        throwError: new Error("LLM call failed"),
      });

      await expect(assertTest(tc, [metric])).rejects.toThrow("LLM call failed");
    });

    it("records errored metric as failure when ignoreErrors is true", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({
        score: 0,
        throwError: new Error("LLM call failed"),
      });

      const err = await assertTest(tc, [metric], {
        errorConfig: { ignoreErrors: true },
      }).catch((e) => e);

      expect(err).toBeInstanceOf(AssertTestError);
      expect(err.failedMetrics[0].error).toBe("LLM call failed");
    });

    it("skips metric (no failure) when MissingTestCaseParamsError and skipOnMissingParams is true", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({
        score: 0,
        throwError: new MissingTestCaseParamsError("Missing retrieval_context"),
      });

      await expect(
        assertTest(tc, [metric], {
          errorConfig: { skipOnMissingParams: true },
        }),
      ).resolves.toBeUndefined();
    });

    it("re-throws MissingTestCaseParamsError when skipOnMissingParams is false (default)", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({
        score: 0,
        throwError: new MissingTestCaseParamsError("Missing retrieval_context"),
      });

      await expect(assertTest(tc, [metric])).rejects.toThrow(
        MissingTestCaseParamsError,
      );
    });
  });

  // -------------------------------------------------------------------------
  // Input validation
  // -------------------------------------------------------------------------

  describe("input validation", () => {
    it("throws when testCase is null", async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await expect(assertTest(null as any, [])).rejects.toThrow(
        "testCase",
      );
    });

    it("throws when metrics array is empty", async () => {
      const tc = makeLLMTestCase();
      await expect(assertTest(tc, [])).rejects.toThrow("metrics");
    });
  });

  // -------------------------------------------------------------------------
  // ConversationalTestCase support
  // -------------------------------------------------------------------------

  describe("ConversationalTestCase", () => {
    it("accepts a ConversationalTestCase with a ConversationalMetric", async () => {
      const tc = new ConversationalTestCase({
        turns: [
          new Turn({ role: "user", content: "Hello" }),
          new Turn({ role: "assistant", content: "Hi" }),
        ],
      });
      const metric = makeConversationalMetric({ score: 0.9, threshold: 0.5 });

      await expect(assertTest(tc, [metric])).resolves.toBeUndefined();
    });
  });

  // -------------------------------------------------------------------------
  // TestRunManager integration
  // -------------------------------------------------------------------------

  describe("TestRunManager integration", () => {
    it("records a passing result in testRunManager", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.9, threshold: 0.5 });

      await assertTest(tc, [metric]);
      const results = testRunManager.getResults();

      expect(results).toHaveLength(1);
      expect(results[0].success).toBe(true);
    });

    it("records a failing result in testRunManager even when an error is thrown", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.2, threshold: 0.5 });

      await assertTest(tc, [metric]).catch(() => {});
      const results = testRunManager.getResults();

      expect(results).toHaveLength(1);
      expect(results[0].success).toBe(false);
    });

    it("records results across multiple calls", async () => {
      const tc1 = makeLLMTestCase("case-1");
      const tc2 = makeLLMTestCase("case-2");
      const passing = makeMetric({ score: 0.9, threshold: 0.5 });
      const failing = makeMetric({ score: 0.1, threshold: 0.5 });

      await assertTest(tc1, [passing]);
      await assertTest(tc2, [failing]).catch(() => {});

      const results = testRunManager.getResults();
      expect(results).toHaveLength(2);
      expect(results[0].success).toBe(true);
      expect(results[1].success).toBe(false);
    });

    it("does NOT record a result when metric throws and ignoreErrors is false", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({
        score: 0,
        throwError: new Error("fatal"),
      });

      await assertTest(tc, [metric]).catch(() => {});
      // runMetric re-throws before buildTestResult is called in this path,
      // so the record should not appear.
      // Actually the test verifies what happens: the error propagates out of
      // Promise.all before record() is called.
      const results = testRunManager.getResults();
      expect(results).toHaveLength(0);
    });
  });

  // -------------------------------------------------------------------------
  // showIndicator option
  // -------------------------------------------------------------------------

  describe("showIndicator option", () => {
    it("restores metric.showIndicator to its original value after the call", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.9 });
      metric.showIndicator = true; // explicitly set

      await assertTest(tc, [metric], { showIndicator: false });
      expect(metric.showIndicator).toBe(true); // restored
    });

    it("restores showIndicator even when a metric throws", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({
        score: 0,
        throwError: new Error("boom"),
      });
      metric.showIndicator = true;

      await assertTest(tc, [metric]).catch(() => {});
      expect(metric.showIndicator).toBe(true);
    });
  });
});

// ===========================================================================
// TestRunManager
// ===========================================================================

describe("TestRunManager", () => {
  // Use a fresh instance for most tests to avoid coupling with the singleton.
  let manager: TestRunManager;

  beforeEach(() => {
    manager = new TestRunManager();
  });

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------

  describe("initialization", () => {
    it("starts with no results", () => {
      expect(manager.getResults()).toHaveLength(0);
    });

    it("getSummary returns zeros for a fresh instance", () => {
      const summary = manager.getSummary();
      expect(summary.total).toBe(0);
      expect(summary.passed).toBe(0);
      expect(summary.failed).toBe(0);
      expect(summary.evaluationCost).toBeUndefined();
      expect(summary.results).toHaveLength(0);
    });
  });

  // -------------------------------------------------------------------------
  // Recording results
  // -------------------------------------------------------------------------

  describe("record()", () => {
    it("records a passing result", () => {
      manager.record({
        name: "test_1",
        success: true,
        metricsData: [],
        conversational: false,
        index: 0,
      });

      expect(manager.getResults()).toHaveLength(1);
      expect(manager.getResults()[0].success).toBe(true);
    });

    it("records a failing result", () => {
      manager.record({
        name: "test_1",
        success: false,
        metricsData: [
          {
            name: "M",
            threshold: 0.5,
            success: false,
            score: 0.2,
            strictMode: false,
            skipped: false,
          },
        ],
        conversational: false,
        index: 0,
      });

      expect(manager.getResults()[0].success).toBe(false);
    });

    it("records multiple results sequentially", () => {
      for (let i = 0; i < 5; i++) {
        manager.record({
          name: `test_${i}`,
          success: i % 2 === 0,
          metricsData: [],
          conversational: false,
          index: i,
        });
      }

      expect(manager.getResults()).toHaveLength(5);
    });

    it("getResults returns a snapshot, not the internal array", () => {
      manager.record({
        name: "t",
        success: true,
        metricsData: [],
        conversational: false,
        index: 0,
      });

      const snapshot1 = manager.getResults();
      manager.record({
        name: "t2",
        success: false,
        metricsData: [],
        conversational: false,
        index: 1,
      });
      const snapshot2 = manager.getResults();

      expect(snapshot1).toHaveLength(1);
      expect(snapshot2).toHaveLength(2);
    });
  });

  // -------------------------------------------------------------------------
  // Aggregation
  // -------------------------------------------------------------------------

  describe("getSummary()", () => {
    it("counts passes and fails correctly", () => {
      manager.record({ name: "a", success: true, metricsData: [], conversational: false, index: 0 });
      manager.record({ name: "b", success: false, metricsData: [], conversational: false, index: 1 });
      manager.record({ name: "c", success: true, metricsData: [], conversational: false, index: 2 });

      const { total, passed, failed } = manager.getSummary();
      expect(total).toBe(3);
      expect(passed).toBe(2);
      expect(failed).toBe(1);
    });

    it("sums evaluationCost across metrics", () => {
      manager.record({
        name: "a",
        success: true,
        metricsData: [
          { name: "M1", threshold: 0.5, success: true, strictMode: false, skipped: false, evaluationCost: 0.001 },
          { name: "M2", threshold: 0.5, success: true, strictMode: false, skipped: false, evaluationCost: 0.002 },
        ],
        conversational: false,
        index: 0,
      });
      manager.record({
        name: "b",
        success: true,
        metricsData: [
          { name: "M1", threshold: 0.5, success: true, strictMode: false, skipped: false, evaluationCost: 0.003 },
        ],
        conversational: false,
        index: 1,
      });

      const { evaluationCost } = manager.getSummary();
      expect(evaluationCost).toBeCloseTo(0.006);
    });

    it("returns undefined evaluationCost when no metric reported cost", () => {
      manager.record({
        name: "a",
        success: true,
        metricsData: [
          { name: "M", threshold: 0.5, success: true, strictMode: false, skipped: false },
        ],
        conversational: false,
        index: 0,
      });

      expect(manager.getSummary().evaluationCost).toBeUndefined();
    });

    it("returns independent snapshot objects on each call", () => {
      manager.record({ name: "a", success: true, metricsData: [], conversational: false, index: 0 });

      const s1 = manager.getSummary();
      manager.record({ name: "b", success: false, metricsData: [], conversational: false, index: 1 });
      const s2 = manager.getSummary();

      expect(s1.total).toBe(1);
      expect(s2.total).toBe(2);
    });
  });

  // -------------------------------------------------------------------------
  // reset()
  // -------------------------------------------------------------------------

  describe("reset()", () => {
    it("clears all recorded results", () => {
      manager.record({ name: "t", success: true, metricsData: [], conversational: false, index: 0 });
      manager.reset();

      expect(manager.getResults()).toHaveLength(0);
    });

    it("getSummary returns zeros after reset", () => {
      manager.record({ name: "t", success: true, metricsData: [], conversational: false, index: 0 });
      manager.reset();

      const { total, passed, failed } = manager.getSummary();
      expect(total).toBe(0);
      expect(passed).toBe(0);
      expect(failed).toBe(0);
    });

    it("allows recording after reset", () => {
      manager.record({ name: "old", success: true, metricsData: [], conversational: false, index: 0 });
      manager.reset();
      manager.record({ name: "new", success: false, metricsData: [], conversational: false, index: 0 });

      const results = manager.getResults();
      expect(results).toHaveLength(1);
      expect(results[0].name).toBe("new");
    });
  });

  // -------------------------------------------------------------------------
  // Multiple sequential runs
  // -------------------------------------------------------------------------

  describe("multiple sequential runs", () => {
    it("accumulates results across multiple simulate assertTest calls", () => {
      // Simulate run 1
      for (let i = 0; i < 3; i++) {
        manager.record({ name: `run1_${i}`, success: true, metricsData: [], conversational: false, index: i });
      }
      expect(manager.getSummary().total).toBe(3);

      // Simulate reset + run 2
      manager.reset();
      for (let i = 0; i < 2; i++) {
        manager.record({ name: `run2_${i}`, success: false, metricsData: [], conversational: false, index: i });
      }

      const summary = manager.getSummary();
      expect(summary.total).toBe(2);
      expect(summary.passed).toBe(0);
      expect(summary.failed).toBe(2);
    });
  });

  // -------------------------------------------------------------------------
  // Singleton (testRunManager)
  // -------------------------------------------------------------------------

  describe("singleton testRunManager", () => {
    it("is a TestRunManager instance", () => {
      expect(testRunManager).toBeInstanceOf(TestRunManager);
    });

    it("is reset before each test (via beforeEach in this file)", () => {
      // The beforeEach at the top-level calls testRunManager.reset(),
      // so this test should always see an empty state.
      expect(testRunManager.getResults()).toHaveLength(0);
    });

    it("records results when assertTest is called", async () => {
      const tc = makeLLMTestCase();
      const metric = makeMetric({ score: 0.8 });

      await assertTest(tc, [metric]);

      expect(testRunManager.getResults()).toHaveLength(1);
    });
  });

  // -------------------------------------------------------------------------
  // Edge cases
  // -------------------------------------------------------------------------

  describe("edge cases", () => {
    it("handles result with null metricsData gracefully in getSummary", () => {
      manager.record({
        name: "t",
        success: true,
        metricsData: null as unknown as [],
        conversational: false,
        index: 0,
      });

      expect(() => manager.getSummary()).not.toThrow();
    });

    it("handles a large number of results efficiently", () => {
      for (let i = 0; i < 1000; i++) {
        manager.record({
          name: `tc_${i}`,
          success: i % 3 !== 0,
          metricsData: [],
          conversational: false,
          index: i,
        });
      }

      const summary = manager.getSummary();
      expect(summary.total).toBe(1000);
      expect(summary.passed + summary.failed).toBe(1000);
    });
  });
});
