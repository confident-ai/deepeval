import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams } from "../../test-case";
import { checkSingleTurnParams, constructVerboseLogs } from "../utils";

export interface ExactMatchMetricOptions {
  threshold?: number;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Exact Match — is `actualOutput` (trimmed) identical to `expectedOutput`
 * (trimmed)? Deterministic, no model. Score is 1 or 0.
 */
export class ExactMatchMetric extends BaseMetric {
  precision?: number;
  recall?: number;
  f1?: number;

  constructor(options: ExactMatchMetricOptions = {}) {
    super(options.threshold ?? 1, {
      verboseMode: options.verboseMode,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.EXPECTED_OUTPUT,
    ];
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);

      const expected = (testCase.expectedOutput ?? "").trim();
      const actual = testCase.actualOutput.trim();

      if (expected === actual) {
        this.score = this.precision = this.recall = this.f1 = 1;
        this.reason = "The actual and expected outputs are exact matches.";
      } else {
        this.score = this.precision = this.recall = this.f1 = 0;
        this.reason = "The actual and expected outputs are different.";
      }
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Score: ${this.score.toFixed(2)}`,
        `Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Exact Match";
  }
}
