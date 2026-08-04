import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  prettifyList,
} from "@/metrics/utils";
import { StepsSchema } from "@/metrics/g-eval/schema";
import {
  type Rubric,
  constructGEvalParamsString,
  constructTestCaseString,
  evaluateGEvalPrompt,
  numberEvaluationSteps,
  formatRubrics,
  getScoreRange,
  validateAndSortRubrics,
  validateCriteriaAndEvaluationSteps,
} from "@/metrics/g-eval/utils";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "GEval";

export type GEvalTemplateOverride = MetricTemplateOverride<"GEval">;

export interface GEvalMetricOptions {
  name: string;
  evaluationParams: SingleTurnParams[];
  criteria?: string;
  evaluationSteps?: string[];
  rubric?: Rubric[];
  model?: DeepEvalBaseLLM | string;
  threshold?: number | null;
  /** Score-token alternatives to weigh, on models that report log probabilities. */
  topLogprobs?: number;
  flaky?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  includeGEvalSuffix?: boolean;
  evaluationTemplate?: GEvalTemplateOverride;
}

export class GEval extends BaseMetric {
  evaluationParams: SingleTurnParams[];
  criteria?: string;
  evaluationSteps?: string[];
  rubric?: Rubric[];
  readonly metricName: string;
  private readonly scoreRange: [number, number];
  private readonly scoreRangeSpan: number;
  private readonly includeGEvalSuffix: boolean;
  private readonly topLogprobs: number;

  constructor(options: GEvalMetricOptions) {
    if (!options.evaluationParams || options.evaluationParams.length === 0) {
      throw new Error("evaluationParams cannot be an empty list.");
    }
    if (options.criteria != null || options.evaluationSteps != null) {
      validateCriteriaAndEvaluationSteps(
        options.criteria,
        options.evaluationSteps,
      );
    }
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.multimodalAware = true;
    this.templateClass = TEMPLATE_CLASS;

    this.metricName = options.name;
    this.evaluationParams = options.evaluationParams;
    this.requiredParams = options.evaluationParams;
    this.criteria = options.criteria;
    this.rubric = validateAndSortRubrics(options.rubric);
    this.scoreRange = getScoreRange(this.rubric);
    this.scoreRangeSpan = this.scoreRange[1] - this.scoreRange[0];
    this.evaluationSteps =
      options.evaluationSteps && options.evaluationSteps.length > 0
        ? options.evaluationSteps
        : undefined;
    this.includeGEvalSuffix = options.includeGEvalSuffix ?? true;
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

      this.evaluationSteps = await this.generateEvaluationSteps();
      const [gScore, reason] = await this.evaluate(testCase);

      this.score = this.strictMode
        ? Math.trunc(gScore)
        : (gScore - this.scoreRange[0]) / this.scoreRangeSpan;
      this.success = this.isSuccessful();
      this.reason = reason;

      this.verboseLogs = constructVerboseLogs(this, [
        `Criteria:\n${this.criteria}`,
        `Evaluation Steps:\n${prettifyList(this.evaluationSteps)}`,
        `Rubric:\n${formatRubrics(this.rubric)}`,
        `Score: ${this.score}`,
        `Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateEvaluationSteps(): Promise<string[]> {
    if (this.evaluationSteps) return this.evaluationSteps;
    const prompt = this.getPrompt("generate_evaluation_steps", {
      criteria: this.criteria,
      parameters: constructGEvalParamsString(this.evaluationParams),
    });
    const { steps } = await generateWithSchema(this, prompt, StepsSchema);
    return steps;
  }

  private async evaluate(testCase: LLMTestCase): Promise<[number, string]> {
    const testCaseContent = constructTestCaseString(
      this.evaluationParams,
      testCase,
    );
    const parameters = constructGEvalParamsString(this.evaluationParams);
    const numberedSteps = numberEvaluationSteps(this.evaluationSteps ?? []);

    const prompt = this.strictMode
      ? this.getPrompt("generate_strict_evaluation_results", {
          evaluation_steps: numberedSteps,
          test_case_content: testCaseContent,
          parameters,
          _additional_context: null,
        })
      : this.getPrompt("generate_evaluation_results", {
          evaluation_steps: numberedSteps,
          test_case_content: testCaseContent,
          parameters,
          rubric: this.rubric ? formatRubrics(this.rubric) : null,
          score_range: this.scoreRange,
          _additional_context: null,
        });

    return evaluateGEvalPrompt(this, prompt, {
      topLogprobs: this.topLogprobs,
      strictMode: this.strictMode,
    });
  }

  get name(): string {
    return this.includeGEvalSuffix
      ? `${this.metricName} [GEval]`
      : this.metricName;
  }
}
