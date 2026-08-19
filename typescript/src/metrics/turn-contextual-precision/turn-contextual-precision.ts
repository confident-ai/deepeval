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
  ContextualPrecisionScoreReasonSchema,
  type ContextualPrecisionVerdict,
  type InteractionContextualPrecisionScore,
} from "@/metrics/turn-contextual-precision/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { idRetrievalContext } from "@/metrics/retrieval-context-display";

const TEMPLATE_CLASS = "TurnContextualPrecisionMetric";

export type TurnContextualPrecisionTemplateOverride =
  MetricTemplateOverride<"TurnContextualPrecisionMetric">;

export interface TurnContextualPrecisionMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  windowSize?: number;
  evaluationTemplate?: TurnContextualPrecisionTemplateOverride;
}

/**
 * Turn Contextual Precision — per sliding window, are relevant retrieval nodes
 * (w.r.t. the conversation's `expectedOutcome`) ranked above irrelevant ones?
 * Rank-weighted precision per window, averaged. **Higher is better**.
 */
export class TurnContextualPrecisionMetric extends BaseConversationalMetric {
  scores: InteractionContextualPrecisionScore[] = [];
  private readonly windowSize: number;

  constructor(options: TurnContextualPrecisionMetricOptions = {}) {
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
  ): Promise<InteractionContextualPrecisionScore> {
    let userContent = "";
    const retrievalContext: string[] = [];
    for (const turn of window) {
      if (turn.role === "user") userContent += `\n${turn.content} `;
      else if (turn.retrievalContext != null)
        retrievalContext.push(
          ...resolveRetrievalContext(turn.retrievalContext),
        );
    }

    const verdicts = await this.generateVerdicts(
      userContent,
      expectedOutcome,
      retrievalContext,
    );
    if (verdicts.length === 0) {
      return {
        score: 1,
        reason:
          "There were no retrieval contexts in the given turns to evaluate the contextual precision.",
        verdicts,
      };
    }
    const score = this.calculateInteractionScore(verdicts);
    const reason = await this.getInteractionReason(
      userContent,
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
    input: string,
    expectedOutcome: string,
    retrievalContext: string[],
  ): Promise<ContextualPrecisionVerdict[]> {
    if (retrievalContext.length === 0) return [];
    const n = retrievalContext.length;
    const prompt = this.getPrompt("generate_verdicts", {
      input,
      expected_outcome: expectedOutcome,
      document_count_str: ` (${n} document${n > 1 ? "s" : ""})`,
      context_to_display: this.multimodal
        ? idRetrievalContext(retrievalContext)
        : retrievalContext,
      multimodal_note: this.multimodal
        ? " (which can be text or an image)"
        : "",
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async getInteractionReason(
    input: string,
    score: number,
    verdicts: ContextualPrecisionVerdict[],
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const prompt = this.getPrompt("generate_reason", {
      input,
      verdicts: verdicts.map((v) => ({
        verdict: v.verdict,
        reason: v.reason,
      })),
      score: score.toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualPrecisionScoreReasonSchema,
    );
    return reason;
  }

  /** Rank-weighted precision (same as single-turn ContextualPrecision). */
  private calculateInteractionScore(
    verdicts: ContextualPrecisionVerdict[],
  ): number {
    const nodeVerdicts = verdicts.map((v) =>
      v.verdict.trim().toLowerCase() === "yes" ? 1 : 0,
    );
    let sumWeighted = 0;
    let relevant = 0;
    for (let i = 0; i < nodeVerdicts.length; i++) {
      if (nodeVerdicts[i]) {
        relevant++;
        sumWeighted += relevant / (i + 1);
      }
    }
    return relevant === 0 ? 0 : sumWeighted / relevant;
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
      ContextualPrecisionScoreReasonSchema,
    );
    return reason;
  }

  get name(): string {
    return "Turn Contextual Precision";
  }
}
