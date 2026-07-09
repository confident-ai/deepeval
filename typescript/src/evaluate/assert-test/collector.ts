import { EvaluatedCase } from "../types";
import { getIsRunningDeepEval } from "../../utils";
import { persistCase } from "./test-run-utils";

/**
 * Accumulates the `EvaluatedCase`s produced by `assertTest` / the matchers
 * within a single worker process, so they can be turned into one posted
 * TestRun.
 *
 * Why per-worker: Vitest/Jest run each test file in an isolated worker with its
 * own module registry, so this singleton only ever sees one file's results.
 * Aggregation across files happens in the runner's reporter (parent process),
 * which drains each worker — that wiring is the Vitest adapter's job. This
 * collector is the framework-agnostic accumulation point they share.
 */
class ResultCollector {
  private cases: EvaluatedCase[] = [];

  record(evaluatedCase: EvaluatedCase): void {
    if (getIsRunningDeepEval()) {
      this.cases.push(evaluatedCase);
      persistCase(evaluatedCase);
    }
  }

  add(evaluatedCase: EvaluatedCase): void {
    this.cases.push(evaluatedCase);
  }

  getCases(): EvaluatedCase[] {
    return [...this.cases];
  }

  get size(): number {
    return this.cases.length;
  }

  reset(): void {
    this.cases = [];
  }
}

export const globalResultCollector = new ResultCollector();
