import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { checkSingleTurnParams, constructVerboseLogs } from "@/metrics/utils";

export interface PatternMatchMetricOptions {
  pattern: string;
  ignoreCase?: boolean;
  threshold?: number | null;
  flaky?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Pattern Match — does `actualOutput` (trimmed) FULLY match `pattern`?
 * Deterministic, no model. Score is 1 or 0. (Full match = the regex must span
 * the entire string, like Python's `re.fullmatch`.)
 */
export class PatternMatchMetric extends BaseMetric {
  pattern: string;
  private readonly regex: RegExp;

  constructor(options: PatternMatchMetricOptions) {
    super(resolveThreshold(options.threshold, 1), {
      verboseMode: options.verboseMode,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.pattern = options.pattern.trim();
    try {
      // Anchor to emulate Python's re.fullmatch (whole-string match).
      this.regex = new RegExp(
        `^(?:${this.pattern})$`,
        options.ignoreCase ? "i" : "",
      );
    } catch (e) {
      throw new Error(`Invalid regex pattern: ${options.pattern} — ${e}`);
    }
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);

      const actual = testCase.actualOutput.trim();
      const fullMatch = this.regex.test(actual);

      this.score = fullMatch ? 1 : 0;
      this.reason = fullMatch
        ? "The actual output fully matches the pattern."
        : "The actual output does not match the pattern.";
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Pattern: ${this.pattern}`,
        `Actual: ${actual}`,
        `Score: ${this.score.toFixed(2)}`,
        `Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  get name(): string {
    return "Pattern Match";
  }
}
