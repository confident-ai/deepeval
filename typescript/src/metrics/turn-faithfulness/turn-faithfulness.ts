import { BaseConversationalMetric } from "../base-conversational-metric";
import {
  ConversationalTestCase,
  MultiTurnParams,
  Turn,
} from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
  resolveRetrievalContext,
} from "../utils";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
  getTurnsInSlidingWindow,
} from "../conversational-utils";
import {
  TruthsSchema,
  ClaimsSchema,
  VerdictsSchema,
  FaithfulnessScoreReasonSchema,
  type FaithfulnessVerdict,
  type InteractionFaithfulnessScore,
} from "./schema";

const TEMPLATE_CLASS = "TurnFaithfulnessMetric";

function limitDescription(limit?: number): string {
  if (limit == null) return "factual, explicit truths";
  if (limit === 1) return "one factual, explicit truth";
  return `${limit} factual, explicit truths`;
}

export interface TurnFaithfulnessMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  truthsExtractionLimit?: number;
  penalizeAmbiguousClaims?: boolean;
  windowSize?: number;
}

/**
 * Turn Faithfulness — over each sliding window of the conversation, do the
 * assistant's claims align with that window's retrieval context? Per window:
 * truths + claims → verdicts → interaction score; the final score averages the
 * per-window scores. **Higher is better**.
 */
export class TurnFaithfulnessMetric extends BaseConversationalMetric {
  scores: InteractionFaithfulnessScore[] = [];
  private readonly windowSize: number;
  private readonly truthsExtractionLimit?: number;
  private readonly penalizeAmbiguousClaims: boolean;

  constructor(options: TurnFaithfulnessMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      MultiTurnParams.ROLE,
      MultiTurnParams.CONTENT,
      MultiTurnParams.RETRIEVAL_CONTEXT,
    ];
    this.windowSize = options.windowSize ?? 10;
    this.truthsExtractionLimit =
      options.truthsExtractionLimit != null
        ? Math.max(options.truthsExtractionLimit, 0)
        : undefined;
    this.penalizeAmbiguousClaims = options.penalizeAmbiguousClaims ?? false;
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

      const unitInteractions = getUnitInteractions(testCase.turns);
      const turnsWindows: Turn[][] = getTurnsInSlidingWindow(
        unitInteractions,
        this.windowSize,
      ).map((window) => window.flat());

      this.scores = await Promise.all(
        turnsWindows.map((window) => this.getInteractionScore(window)),
      );
      this.score = this.calculateScore();
      this.success = this.score >= this.threshold;
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
  ): Promise<InteractionFaithfulnessScore> {
    let userContent = "";
    let assistantContent = "";
    const retrievalContext: string[] = [];
    for (const turn of window) {
      if (turn.role === "user") {
        userContent += `\n${turn.content} `;
      } else {
        assistantContent += `\n${turn.content}`;
        if (turn.retrievalContext != null) {
          retrievalContext.push(
            ...resolveRetrievalContext(turn.retrievalContext),
          );
        }
      }
    }

    const [truths, claims] = await Promise.all([
      this.generateTruths(retrievalContext),
      this.generateClaims(userContent, assistantContent),
    ]);
    const verdicts = await this.generateVerdicts(claims, truths);
    const { score, reason } = await this.getInteractionScoreAndReason(verdicts);
    return { score, reason, claims, truths, verdicts };
  }

  private async generateTruths(retrievalContext: string[]): Promise<string[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_truths", {
      reference_context: retrievalContext.join("\n\n"),
      limit_description: limitDescription(this.truthsExtractionLimit),
    });
    const { truths } = await generateWithSchema(this, prompt, TruthsSchema);
    return truths;
  }

  private async generateClaims(
    userContent: string,
    assistantContent: string,
  ): Promise<string[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_claims", {
      input: userContent,
      assistant_output: assistantContent,
    });
    const { claims } = await generateWithSchema(this, prompt, ClaimsSchema);
    return claims;
  }

  private async generateVerdicts(
    claims: string[],
    truths: string[],
  ): Promise<FaithfulnessVerdict[]> {
    if (claims.length === 0) return [];
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      claims,
      reference_context: truths.join("\n\n"),
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async getInteractionScoreAndReason(
    verdicts: FaithfulnessVerdict[],
  ): Promise<{ score: number; reason?: string }> {
    if (verdicts.length === 0) {
      return {
        score: 1,
        reason: this.includeReason ? "<no claims to verify>" : undefined,
      };
    }
    let count = 0;
    for (const v of verdicts) {
      const vd = v.verdict.trim().toLowerCase();
      if (vd !== "no") count++;
      if (this.penalizeAmbiguousClaims && vd === "idk") count--;
    }
    const score = count / verdicts.length;
    const reason = await this.getInteractionReason(score, verdicts);
    if (this.strictMode && score < this.threshold) return { score: 0, reason };
    return { score, reason };
  }

  private async getInteractionReason(
    score: number,
    verdicts: FaithfulnessVerdict[],
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const contradictions = verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "no")
      .map((v) => v.reason);
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      contradictions,
      score: score.toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      FaithfulnessScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    if (this.scores.length === 0) return 1;
    const total = this.scores.reduce((sum, s) => sum + s.score, 0);
    return total / this.scores.length;
  }

  private async generateFinalReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    if (this.scores.length === 0) {
      return "There were no retrieval contexts in your turns to evaluate, hence the score is 1";
    }
    const reasons = this.scores.map((s) => s.reason);
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_final_reason", {
      final_score: this.score,
      success: this.success,
      reasons,
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      FaithfulnessScoreReasonSchema,
    );
    return reason;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Turn Faithfulness";
  }
}
