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
  ContextualRelevancyVerdictsSchema,
  ContextualRelevancyScoreReasonSchema,
  type ContextualRelevancyVerdict,
  type InteractionContextualRelevancyScore,
} from "./schema";

const TEMPLATE_CLASS = "TurnContextualRelevancyMetric";

const EXTRACTION_INSTRUCTIONS =
  "You should first extract statements found in the context, which are " +
  "high level information found in the context, before deciding on a " +
  "verdict and optionally a reason for each statement.";
const EMPTY_CONTEXT_INSTRUCTION =
  '\nIf provided context contains no actual content or statements then: ' +
  'give "no" as a "verdict",\nput context into "statement", and ' +
  '"No statements found in provided context." into "reason".';

export interface TurnContextualRelevancyMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  windowSize?: number;
}

/**
 * Turn Contextual Relevancy — per sliding window, what fraction of statements
 * across the window's retrieval context are relevant to the user input?
 * Relevant / total per window, averaged. **Higher is better**.
 */
export class TurnContextualRelevancyMetric extends BaseConversationalMetric {
  scores: InteractionContextualRelevancyScore[] = [];
  private readonly windowSize: number;

  constructor(options: TurnContextualRelevancyMetricOptions = {}) {
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

      const turnsWindows: Turn[][] = getTurnsInSlidingWindow(
        getUnitInteractions(testCase.turns),
        this.windowSize,
      ).map((window) => window.flat());

      this.scores = await Promise.all(
        turnsWindows.map((w) => this.getInteractionScore(w)),
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
  ): Promise<InteractionContextualRelevancyScore> {
    let userContent = "";
    const retrievalContext: string[] = [];
    for (const turn of window) {
      if (turn.role === "user") userContent += `\n${turn.content} `;
      else if (turn.retrievalContext != null)
        retrievalContext.push(...resolveRetrievalContext(turn.retrievalContext));
    }

    const verdicts = await this.generateVerdicts(userContent, retrievalContext);
    if (verdicts.length === 0) {
      return {
        score: 1,
        reason:
          "There were no retrieval contexts in the given turns to evaluate the contextual relevancy.",
        verdicts,
      };
    }
    const score = this.calculateInteractionScore(verdicts);
    const reason = await this.getInteractionReason(userContent, score, verdicts);
    return {
      score: this.strictMode && score < this.threshold ? 0 : score,
      reason,
      verdicts,
    };
  }

  private async generateVerdicts(
    input: string,
    retrievalContext: string[],
  ): Promise<ContextualRelevancyVerdict[]> {
    if (retrievalContext.length === 0) return [];
    const perNode = await Promise.all(
      retrievalContext.map(async (context) => {
        const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
          input,
          context,
          context_type: "context",
          statement_or_image: "statement",
          extraction_instructions: EXTRACTION_INSTRUCTIONS,
          empty_context_instruction: EMPTY_CONTEXT_INSTRUCTION,
        });
        const { verdicts } = await generateWithSchema(
          this,
          prompt,
          ContextualRelevancyVerdictsSchema,
        );
        return verdicts;
      }),
    );
    return perNode.flat();
  }

  private async getInteractionReason(
    input: string,
    score: number,
    verdicts: ContextualRelevancyVerdict[],
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const irrelevant: (string | null | undefined)[] = [];
    const relevant: string[] = [];
    for (const v of verdicts) {
      if (v.verdict.toLowerCase() === "no") irrelevant.push(v.reason);
      else relevant.push(v.statement);
    }
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      input,
      irrelevant_statements: irrelevant,
      relevant_statements: relevant,
      score: score.toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRelevancyScoreReasonSchema,
    );
    return reason;
  }

  private calculateInteractionScore(
    verdicts: ContextualRelevancyVerdict[],
  ): number {
    const relevant = verdicts.filter(
      (v) => v.verdict.toLowerCase() === "yes",
    ).length;
    return relevant / verdicts.length;
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
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_final_reason", {
      final_score: this.score,
      success: this.success,
      reasons: this.scores.map((s) => s.reason),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRelevancyScoreReasonSchema,
    );
    return reason;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Turn Contextual Relevancy";
  }
}
