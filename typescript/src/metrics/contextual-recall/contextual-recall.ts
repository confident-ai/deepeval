import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  prettifyList,
  resolveRetrievalContext,
} from "@/metrics/utils";
import {
  VerdictsSchema,
  ContextualRecallScoreReasonSchema,
  type ContextualRecallVerdict,
} from "@/metrics/contextual-recall/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import {
  contextualRecallVerdictVars,
  contextualRecallReasonContentType,
} from "@/metrics/retrieval-context-display";

const TEMPLATE_CLASS = "ContextualRecallMetric";

export type ContextualRecallTemplateOverride =
  MetricTemplateOverride<"ContextualRecallMetric">;

export interface ContextualRecallMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: ContextualRecallTemplateOverride;
}

/**
 * Contextual Recall — can each sentence of `expectedOutput` be attributed to the
 * `retrievalContext`? Score = attributable sentences / total.
 */
export class ContextualRecallMetric extends BaseMetric {
  verdicts: ContextualRecallVerdict[] = [];

  constructor(options: ContextualRecallMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.multimodalAware = true;
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.RETRIEVAL_CONTEXT,
      SingleTurnParams.EXPECTED_OUTPUT,
    ];
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

      this.verdicts = await this.generateVerdicts(
        testCase.expectedOutput ?? "",
        resolveRetrievalContext(testCase.retrievalContext ?? []),
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.expectedOutput ?? "");
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdicts(
    expectedOutput: string,
    retrievalContext: string[],
  ): Promise<ContextualRecallVerdict[]> {
    const prompt = this.getPrompt("generate_verdicts", {
      expected_output: expectedOutput,
      ...contextualRecallVerdictVars(retrievalContext, this.multimodal),
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(
    expectedOutput: string,
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const supportiveReasons: (string | null | undefined)[] = [];
    const unsupportiveReasons: (string | null | undefined)[] = [];
    for (const v of this.verdicts) {
      if (v.verdict.toLowerCase() === "yes") supportiveReasons.push(v.reason);
      else unsupportiveReasons.push(v.reason);
    }
    const prompt = this.getPrompt("generate_reason", {
      expected_output: expectedOutput,
      supportive_reasons: supportiveReasons,
      unsupportive_reasons: unsupportiveReasons,
      score: (this.score ?? 0).toFixed(2),
      content_type: contextualRecallReasonContentType(this.multimodal),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRecallScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 0;
    const justified = this.verdicts.filter(
      (v) => v.verdict.toLowerCase() === "yes",
    ).length;
    const score = justified / total;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Contextual Recall";
  }
}
