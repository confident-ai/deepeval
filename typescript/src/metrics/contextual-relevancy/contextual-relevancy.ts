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
  ContextualRelevancyVerdictsSchema,
  ContextualRelevancyScoreReasonSchema,
  type ContextualRelevancyVerdicts,
} from "@/metrics/contextual-relevancy/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { contextualRelevancyVerdictVars } from "@/metrics/retrieval-context-display";

const TEMPLATE_CLASS = "ContextualRelevancyMetric";

export type ContextualRelevancyTemplateOverride =
  MetricTemplateOverride<"ContextualRelevancyMetric">;

export interface ContextualRelevancyMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: ContextualRelevancyTemplateOverride;
}

/**
 * Contextual Relevancy — what fraction of statements across `retrievalContext`
 * are relevant to the `input`? Judge each node's statements, then
 * score = relevant statements / total statements.
 */
export class ContextualRelevancyMetric extends BaseMetric {
  verdictsList: ContextualRelevancyVerdicts[] = [];

  constructor(options: ContextualRelevancyMetricOptions = {}) {
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

      const retrievalContext = resolveRetrievalContext(
        testCase.retrievalContext ?? [],
      );
      this.verdictsList = await Promise.all(
        retrievalContext.map((context) =>
          this.generateVerdicts(testCase.input, context),
        ),
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.input);
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Verdicts:\n${prettifyList(this.verdictsList)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdicts(
    input: string,
    context: string,
  ): Promise<ContextualRelevancyVerdicts> {
    const prompt = this.getPrompt("generate_verdicts", {
      input,
      context,
      ...contextualRelevancyVerdictVars(this.multimodal),
    });
    return generateWithSchema(this, prompt, ContextualRelevancyVerdictsSchema);
  }

  private async generateReason(input: string): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const irrelevantStatements: (string | null | undefined)[] = [];
    const relevantStatements: string[] = [];
    for (const { verdicts } of this.verdictsList) {
      for (const v of verdicts) {
        if (v.verdict.toLowerCase() === "no")
          irrelevantStatements.push(v.reason);
        else relevantStatements.push(v.statement);
      }
    }
    const prompt = this.getPrompt("generate_reason", {
      input,
      irrelevant_statements: irrelevantStatements,
      relevant_statements: relevantStatements,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRelevancyScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    let totalVerdicts = 0;
    let relevant = 0;
    for (const { verdicts } of this.verdictsList) {
      for (const v of verdicts) {
        totalVerdicts++;
        if (v.verdict.toLowerCase() === "yes") relevant++;
      }
    }
    if (totalVerdicts === 0) return 0;
    const score = relevant / totalVerdicts;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Contextual Relevancy";
  }
}
