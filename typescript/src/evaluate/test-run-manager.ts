/**
 * TestRunManager — in-memory accumulator for assertTest() results.
 *
 * Mirrors the responsibility of Python's `global_test_run_manager` but
 * scoped only to what the TypeScript SDK needs today:
 *
 *   - Record one {@link TestResult} per `assertTest()` call.
 *   - Expose aggregation via {@link getSummary}.
 *   - Support a clean-slate {@link reset} for multi-run scenarios.
 *
 * This implementation is framework-agnostic (no Jest / Vitest dependency).
 * It is intentionally kept simple; persistence, caching, and CI reporters
 * are out of scope for this PR and can be added on top without breaking
 * the public API.
 *
 * Concurrency: Node.js is single-threaded; `record()` is synchronous and
 * therefore safe to call from multiple concurrently-resolved Promises
 * (the event loop serialises the microtask callbacks).
 */

import { TestResult, MetricData } from "./types";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Aggregate statistics produced by {@link TestRunManager.getSummary}. */
export interface TestRunSummary {
  /** Total number of `assertTest()` calls recorded. */
  total: number;
  /** How many test cases passed (all metrics succeeded). */
  passed: number;
  /** How many test cases failed (at least one metric failed). */
  failed: number;
  /**
   * Cumulative evaluation cost across all recorded test cases.
   * `undefined` when no metric reported a cost.
   */
  evaluationCost: number | undefined;
  /** Read-only snapshot of every recorded result. */
  results: readonly TestResult[];
}

// ---------------------------------------------------------------------------
// TestRunManager
// ---------------------------------------------------------------------------

/**
 * Singleton in-memory store for `assertTest()` results.
 *
 * Use the exported {@link testRunManager} instance rather than constructing
 * one directly.
 *
 * @example
 * ```ts
 * import { assertTest, testRunManager } from "deepeval";
 *
 * await assertTest(testCase, [metric]);
 * const summary = testRunManager.getSummary();
 * console.log(`${summary.passed}/${summary.total} passed`);
 * testRunManager.reset();
 * ```
 */
export class TestRunManager {
  private _results: TestResult[] = [];

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /**
   * Clear all recorded results and reset internal state.
   *
   * Mirrors Python's `TestRunManager.reset()`. Call this between unrelated
   * test suites when you reuse the same process.
   */
  reset(): void {
    this._results = [];
  }

  // ---------------------------------------------------------------------------
  // Mutation
  // ---------------------------------------------------------------------------

  /**
   * Append a completed {@link TestResult} to the in-memory store.
   *
   * Called automatically by {@link assertTest} — you rarely need to call
   * this directly.
   */
  record(result: TestResult): void {
    this._results.push(result);
  }

  // ---------------------------------------------------------------------------
  // Queries
  // ---------------------------------------------------------------------------

  /**
   * Return a read-only snapshot of every result recorded so far.
   *
   * The array is a shallow copy; mutating its elements does **not** affect
   * the manager's internal state (but modifying nested object properties
   * would, by reference — treat the contents as read-only).
   */
  getResults(): readonly TestResult[] {
    return this._results.slice();
  }

  /**
   * Compute aggregate statistics over all recorded results.
   *
   * Returns a plain object snapshot — calling this multiple times returns
   * independent objects reflecting the state at each call site.
   */
  getSummary(): TestRunSummary {
    const results = this._results.slice();
    let passed = 0;
    let failed = 0;
    let totalCost: number | undefined = undefined;

    for (const result of results) {
      if (result.success) {
        passed++;
      } else {
        failed++;
      }

      // Accumulate evaluation cost, respecting the undefined-means-not-tracked
      // convention used by MetricData.evaluationCost.
      const caseCost = _sumMetricCosts(result.metricsData ?? []);
      if (caseCost !== undefined) {
        totalCost = (totalCost ?? 0) + caseCost;
      }
    }

    return {
      total: results.length,
      passed,
      failed,
      evaluationCost: totalCost,
      results,
    };
  }
}

// ---------------------------------------------------------------------------
// Module-level singleton (mirrors Python's `global_test_run_manager`)
// ---------------------------------------------------------------------------

/**
 * The shared `TestRunManager` instance used by `assertTest()`.
 *
 * Exported for consumers that need to inspect or reset the run state
 * without importing the class itself.
 */
export const testRunManager = new TestRunManager();

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Sum `evaluationCost` fields across a list of {@link MetricData}.
 * Returns `undefined` when no metric reported a cost.
 */
function _sumMetricCosts(metricsData: MetricData[]): number | undefined {
  let total: number | undefined = undefined;
  for (const m of metricsData) {
    if (m.evaluationCost != null) {
      total = (total ?? 0) + m.evaluationCost;
    }
  }
  return total;
}
