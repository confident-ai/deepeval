import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "@/metrics/utils";
import { evaluateGEvalPrompt } from "@/metrics/g-eval/utils";
import {
  type JudgeEvalMessage,
  type JudgeEvalVariable,
  renderMessages,
  resolveVariables,
  validateMessages,
  validateScoreRange,
  validateVariables,
} from "@/metrics/judge-eval/utils";

const TEMPLATE_CLASS = "JudgeEval";

export interface JudgeEvalOptions {
  name: string;
  messages: JudgeEvalMessage[];
  variables?: Record<string, JudgeEvalVariable>;
  scoreRange?: [number, number];
  model?: DeepEvalBaseLLM | string;
  threshold?: number | null;
  topLogprobs?: number;
  flaky?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  includeJudgeEvalSuffix?: boolean;
}

export class JudgeEval extends BaseMetric {
  readonly metricName: string;
  readonly messages: JudgeEvalMessage[];
  readonly variables: Record<string, JudgeEvalVariable>;
  readonly scoreRange: [number, number];
  private readonly scoreRangeSpan: number;
  private readonly includeJudgeEvalSuffix: boolean;
  private readonly topLogprobs: number;

  constructor(options: JudgeEvalOptions) {
    const messages = validateMessages(options.messages);
    const variables = validateVariables(messages, options.variables);
    const scoreRange = validateScoreRange(options.scoreRange ?? [0, 10]);
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
    });
    this.templateClass = TEMPLATE_CLASS;

    this.metricName = options.name;
    this.messages = messages;
    this.variables = variables;
    this.scoreRange = scoreRange;
    this.scoreRangeSpan = scoreRange[1] - scoreRange[0];
    this.requiredParams = [];
    this.includeJudgeEvalSuffix = options.includeJudgeEvalSuffix ?? true;
    this.topLogprobs = options.topLogprobs ?? 20;

    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      const prompt = this.resultsPrompt(testCase);
      const [judgeScore, reason] = await evaluateGEvalPrompt(this, prompt, {
        topLogprobs: this.topLogprobs,
        strictMode: this.strictMode,
      });

      const normalized =
        (judgeScore - this.scoreRange[0]) / this.scoreRangeSpan;
      this.score = this.strictMode
        ? normalized >= 1
          ? 1
          : 0
        : Math.min(Math.max(normalized, 0), 1);
      this.success = this.isSuccessful();
      this.reason = reason;

      this.verboseLogs = constructVerboseLogs(this, [
        `Prompt:\n${prompt}`,
        `Score Range: ${this.scoreRange[0]} to ${this.scoreRange[1]}`,
        `Score: ${this.score}`,
        `Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private resultsPrompt(testCase: LLMTestCase): string {
    const values = resolveVariables(testCase, this.variables);
    return this.getPrompt("generate_evaluation_results", {
      prompt: renderMessages(this.messages, values),
      score_range: this.scoreRange,
      strict_mode: this.strictMode,
    });
  }

  get name(): string {
    return this.includeJudgeEvalSuffix
      ? `${this.metricName} [JudgeEval]`
      : this.metricName;
  }
}
