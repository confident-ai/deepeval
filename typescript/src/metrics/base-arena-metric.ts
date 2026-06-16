import { BaseMetricCore } from "./base-metrics";
import { ArenaTestCase } from "../test-case";

/**
 * Base for arena metrics, which COMPARE contestants rather than score one
 * output. `measure` returns the winning contestant's name (a string), not a
 * numeric score. Reuses {@link BaseMetricCore}'s spinner/cost machinery.
 */
export abstract class BaseArenaMetric extends BaseMetricCore {
  /** The winning contestant's (real) name, set by `measure`. */
  winner?: string;

  abstract measure(
    testCase: ArenaTestCase,
    ...args: unknown[]
  ): string | Promise<string>;

  get name(): string {
    return "Base Arena Metric";
  }
}
