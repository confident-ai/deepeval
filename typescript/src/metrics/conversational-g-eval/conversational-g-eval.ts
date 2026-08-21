import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
} from "@/metrics/utils";
import {
  checkConversationalTestCaseParams,
  convertTurnToDict,
} from "@/metrics/conversational-utils";
import { StepsSchema } from "@/metrics/g-eval/schema";
import {
  type Rubric,
  evaluateGEvalPrompt,
  numberEvaluationSteps,
  formatRubrics,
  validateAndSortRubrics,
  validateCriteriaAndEvaluationSteps,
} from "@/metrics/g-eval/utils";
import {
  constructConversationalGEvalTurnParamsString,
  constructNonTurnsTestCaseString,
} from "@/metrics/conversational-g-eval/utils";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "ConversationalGEval";

export type ConversationalGEvalTemplateOverride =
  MetricTemplateOverride<"ConversationalGEval">;

export interface ConversationalGEvalMetricOptions {
  name: string;
  /** Which turn/conversation fields the judgement considers (CONTENT + ROLE always added). */
  evaluationParams: MultiTurnParams[];
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
  evaluationTemplate?: ConversationalGEvalTemplateOverride;
}

/**
 * Conversational GEval — a flexible LLM judge over a whole conversation: generate
 * eval steps from `criteria` (or use supplied `evaluationSteps`), then score the
 * conversation 0–10 (normalized to 0–1) with a reason. Like single-turn GEval but
 * over turns + conversation-level fields.
 */
export class ConversationalGEval extends BaseConversationalMetric {
  evaluationParams: MultiTurnParams[];
  criteria?: string;
  evaluationSteps?: string[];
  rubric?: Rubric[];
  readonly metricName: string;
  private readonly includeGEvalSuffix: boolean;
  private readonly topLogprobs: number;

  constructor(options: ConversationalGEvalMetricOptions) {
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
    this.templateClass = TEMPLATE_CLASS;

    // CONTENT + ROLE are always required (mirrors Python).
    const params = [...options.evaluationParams];
    if (!params.includes(MultiTurnParams.CONTENT))
      params.push(MultiTurnParams.CONTENT);
    if (!params.includes(MultiTurnParams.ROLE))
      params.push(MultiTurnParams.ROLE);
    this.evaluationParams = params;
    this.requiredParams = params;

    this.metricName = options.name;
    this.criteria = options.criteria;
    this.rubric = validateAndSortRubrics(options.rubric);
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

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      this.evaluationSteps = await this.generateEvaluationSteps();
      const [gScore, reason] = await this.evaluate(testCase);

      this.score = this.applyStrictMode(gScore / 10);
      this.reason = reason;
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Criteria:\n${this.criteria}`,
        `Evaluation Steps:\n${prettifyList(this.evaluationSteps)}`,
        `Rubric:\n${formatRubrics(this.rubric)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
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
      parameters: constructConversationalGEvalTurnParamsString(
        this.evaluationParams,
      ),
    });
    const { steps } = await generateWithSchema(this, prompt, StepsSchema);
    return steps;
  }

  private async evaluate(
    testCase: ConversationalTestCase,
  ): Promise<[number, string]> {
    const prompt = this.getPrompt("generate_evaluation_results", {
      evaluation_steps: numberEvaluationSteps(this.evaluationSteps ?? []),
      test_case_content: constructNonTurnsTestCaseString(
        this.evaluationParams,
        testCase,
      ),
      turns: testCase.turns.map((t) =>
        convertTurnToDict(t, this.evaluationParams),
      ),
      parameters: constructConversationalGEvalTurnParamsString(
        this.evaluationParams,
      ),
      rubric: this.rubric ? formatRubrics(this.rubric) : null,
    });
    return evaluateGEvalPrompt(this, prompt, {
      topLogprobs: this.topLogprobs,
      strictMode: this.strictMode,
    });
  }

  get name(): string {
    return this.includeGEvalSuffix
      ? `${this.metricName} [Conversational GEval]`
      : this.metricName;
  }
}
