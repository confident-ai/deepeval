import { DeepEvalBaseLLM } from "../models";
import { LLMTestCaseParams } from "../test-case";

export abstract class BaseMetric {
  protected requiredParams: Array<LLMTestCaseParams>;

  threshold: number;
  score?: number;
  scoreBreakdown?: Record<string, any>;
  reason?: string;
  success?: boolean;
  evaluationModel?: string;
  strictMode: boolean = false;
  asyncMode: boolean = true;
  verboseMode: boolean = true;
  includeReason: boolean = false;
  error?: string;
  evaluationCost?: number;
  verboseLogs?: string;
  skipped: boolean = false;
  model?: DeepEvalBaseLLM;
  usingNativeModel?: boolean = undefined;

  constructor(
    threshold: number,
    options?: {
      strictMode?: boolean;
      asyncMode?: boolean;
      verboseMode?: boolean;
      includeReason?: boolean;
    },
  ) {
    this.threshold = threshold;
    this.requiredParams = [];
    if (options) {
      this.strictMode = options.strictMode ?? this.strictMode;
      this.asyncMode = options.asyncMode ?? this.asyncMode;
      this.verboseMode = options.verboseMode ?? this.verboseMode;
      this.includeReason = options.includeReason ?? this.includeReason;
    }
  }

  abstract measure(testCase: any, ...args: any[]): number | Promise<number>;

  abstract isSuccessful(): boolean;

  get className(): string {
    return "Base Metric";
  }
}
