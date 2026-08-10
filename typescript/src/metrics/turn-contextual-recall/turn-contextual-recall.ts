import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams, Turn } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
  resolveRetrievalContext,
} from "@/metrics/utils";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
  getTurnsInSlidingWindow,
} from "@/metrics/conversational-utils";
import {
  VerdictsSchema,
  ContextualRecallScoreReasonSchema,
  type ContextualRecallVerdict,
  type InteractionContextualRecallScore,
} from "@/metrics/turn-contextual-recall/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import {
  contextualRecallVerdictVars,
  contextualRecallReasonContentType,
} from "@/metrics/retrieval-context-display";

const TEMPLATE_CLASS = "TurnContextualRecallMetric";

export type TurnContextualRecallTemplateOverride =
  MetricTemplateOverride<"TurnContextualRecallMetric">;

export interface TurnContextualRecallMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  windowSize?: number;
  evaluationTemplate?: TurnContextualRecallTemplateOverride;
}

/**
 * Turn Contextual Recall — per sliding window, can each part of the
 * `expectedOutcome` be attributed to the window's retrieval context?
 * Attributable / total per window, averaged. **Higher is better**.
 */
export class TurnContextualRecallMetric extends BaseConversationalMetric {
  scores: InteractionContextualRecallScore[] = [];
  private readonly windowSize: number;

  constructor(options: TurnContextualRecallMetricOptions = {}) {
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
      MultiTurnParams.ROLE,
      MultiTurnParams.CONTENT,
      MultiTurnParams.RETRIEVAL_CONTEXT,
      MultiTurnParams.EXPECTED_OUTCOME,
    ];
    this.windowSize = options.windowSize ?? 10;
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

      const expectedOutcome = testCase.expectedOutcome ?? "";
      const turnsWindows: Turn[][] = getTurnsInSlidingWindow(
        getUnitInteractions(testCase.turns),
        this.windowSize,
      ).map((window) => window.flat());

      this.scores = await Promise.all(
        turnsWindows.map((w) => this.getInteractionScore(w, expectedOutcome)),
      );
      this.score = this.calculateScore();
      this.success = this.isSuccessful();
      this.reason = await this.generateFinalReason();

      this.verboseLogs = constructVerboseLogs(this, [
        `Windows (size=${this.windowSize}): ${turnsWindows.length}`,
        `Interaction Scores:\n${prettifyList(this.scores)}`,
        `Final Score: ${this.score}\nFinal Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async getInteractionScore(
    window: Turn[],
    expectedOutcome: string,
  ): Promise<InteractionContextualRecallScore> {
    const retrievalContext: string[] = [];
    for (const turn of window) {
      if (turn.role !== "user" && turn.retrievalContext != null)
        retrievalContext.push(
          ...resolveRetrievalContext(turn.retrievalContext),
        );
    }

    const verdicts = await this.generateVerdicts(
      expectedOutcome,
      retrievalContext,
    );
    if (verdicts.length === 0) {
      return {
        score: 1,
        reason:
          "There were no retrieval contexts in the given turns to evaluate the contextual recall.",
        verdicts,
      };
    }
    const score = this.calculateInteractionScore(verdicts);
    const reason = await this.getInteractionReason(
      expectedOutcome,
      score,
      verdicts,
    );
    return {
      score: this.applyStrictMode(score),
      reason,
      verdicts,
    };
  }

  private async generateVerdicts(
    expectedOutcome: string,
    retrievalContext: string[],
  ): Promise<ContextualRecallVerdict[]> {
    if (retrievalContext.length === 0) return [];
    const prompt = this.getPrompt("generate_verdicts", {
      expected_outcome: expectedOutcome,
      ...contextualRecallVerdictVars(retrievalContext, this.multimodal),
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async getInteractionReason(
    expectedOutcome: string,
    score: number,
    verdicts: ContextualRecallVerdict[],
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const supportive: (string | null | undefined)[] = [];
    const unsupportive: (string | null | undefined)[] = [];
    for (const v of verdicts) {
      if (v.verdict.toLowerCase() === "yes") supportive.push(v.reason);
      else unsupportive.push(v.reason);
    }
    const prompt = this.getPrompt("generate_reason", {
      expected_outcome: expectedOutcome,
      supportive_reasons: supportive,
      unsupportive_reasons: unsupportive,
      score: score.toFixed(2),
      content_type: contextualRecallReasonContentType(this.multimodal),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRecallScoreReasonSchema,
    );
    return reason;
  }

  private calculateInteractionScore(
    verdicts: ContextualRecallVerdict[],
  ): number {
    const justified = verdicts.filter(
      (v) => v.verdict.toLowerCase() === "yes",
    ).length;
    return justified / verdicts.length;
  }

  private calculateScore(): number {
    if (this.scores.length === 0) return 1;
    return this.scores.reduce((s, x) => s + x.score, 0) / this.scores.length;
  }

  private async generateFinalReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    if (this.scores.length === 0) {
      return "There were no interactions with retrieval context to evaluate, hence the score is 1";
    }
    const prompt = this.getPrompt("generate_final_reason", {
      final_score: this.score,
      success: this.success,
      reasons: this.scores.map((s) => s.reason),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRecallScoreReasonSchema,
    );
    return reason;
  }

  get name(): string {
    return "Turn Contextual Recall";
  }
}
