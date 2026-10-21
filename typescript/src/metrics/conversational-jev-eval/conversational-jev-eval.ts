import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams } from "@/test-case";
import { DeepEvalBaseSystemOneModel } from "@/models/system-one/base-system-one-model";
import { constructVerboseLogs } from "@/metrics/utils";
import { checkConversationalTestCaseParams } from "@/metrics/conversational-utils";
import { formatSystemOneReason } from "@/metrics/system-one/reason";
import type {
  JevQuestion,
  QuestionOutcome,
} from "@/metrics/jev-eval/questions";
import {
  aggregate,
  aggregateStrict,
  buildQuestions,
  constructMultiTurnState,
  formatOutcomesForLogs,
  initializeJevModel,
  markStrict,
  minConfidence,
  outcomesFromAnswers,
  validateQuestions,
} from "@/metrics/jev-eval/utils";

export interface ConversationalJevEvalOptions {
  name: string;
  /** Turn and conversation fields your questions refer to (CONTENT + ROLE always added). */
  evaluationParams: MultiTurnParams[];
  questions: JevQuestion[];
  /** Defaults to a `TypeSafeModel` built from TYPESAFE_* settings. */
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  includeReason?: boolean;
  threshold?: number | null;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  flaky?: boolean;
  includeJevEvalSuffix?: boolean;
}

/** `JevEval` over a whole conversation. Always Jev, never an LLM. */
export class ConversationalJevEval extends BaseConversationalMetric {
  readonly metricName: string;
  evaluationParams: MultiTurnParams[];
  questions: JevQuestion[];
  declare systemOneModel: DeepEvalBaseSystemOneModel;
  private readonly includeJevEvalSuffix: boolean;

  constructor(options: ConversationalJevEvalOptions) {
    if (!options.evaluationParams || options.evaluationParams.length === 0) {
      throw new Error(
        "evaluationParams cannot be empty; list the turn and conversation " +
          "fields your questions refer to.",
      );
    }
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
    });
    // Every question is about the conversation, so the turns' content and
    // roles are always part of the state.
    const params = [...options.evaluationParams];
    if (!params.includes(MultiTurnParams.CONTENT)) {
      params.push(MultiTurnParams.CONTENT);
    }
    if (!params.includes(MultiTurnParams.ROLE)) {
      params.push(MultiTurnParams.ROLE);
    }
    this.metricName = options.name;
    this.evaluationParams = params;
    this.requiredParams = params;
    this.questions = validateQuestions(options.questions);
    this.systemOneModel = initializeJevModel(options.systemOneModel);
    this.evaluationModel = this.systemOneModel.getModelName();
    this.includeJevEvalSuffix = options.includeJevEvalSuffix ?? true;
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      if (testCase instanceof ConversationalTestCase && testCase.multimodal) {
        this.error =
          `${this.name} evaluates text only: Jev has no image input. Pass a ` +
          "text-only ConversationalTestCase.";
        throw new Error(this.error);
      }
      checkConversationalTestCaseParams(testCase, this.requiredParams, this, {
        requireChatbotRole: this.evaluationParams.includes(
          MultiTurnParams.CHATBOT_ROLE,
        ),
      });
      this.evaluationCost = 0;

      const state = constructMultiTurnState(this.evaluationParams, testCase);
      const { answers, cost } = await this.systemOneModel.decide(
        state,
        buildQuestions(this.questions),
      );
      this.accrueCost(cost);
      this.finalize(outcomesFromAnswers(this.questions, answers));
      return this.score as number;
    } finally {
      this.stopProgress();
    }
  }

  private finalize(outcomes: QuestionOutcome[]): void {
    if (this.strictMode) {
      outcomes = markStrict(this.questions, outcomes);
      this.score = aggregateStrict(outcomes);
    } else {
      this.score = aggregate(outcomes);
    }
    this.systemOneOutcomes = outcomes;
    this.scoreBreakdown = outcomes.map((o) => ({ ...o }));
    this.confidence = minConfidence(outcomes);
    this.reason = this.includeReason
      ? formatSystemOneReason(this, outcomes)
      : undefined;
    this.success = this.isSuccessful();
    this.verboseLogs = constructVerboseLogs(this, [
      `Questions:\n${formatOutcomesForLogs(outcomes)}`,
      `Score: ${this.score}\nConfidence: ${this.confidence}`,
      `Reason: ${this.reason}`,
    ]);
  }

  get name(): string {
    return this.includeJevEvalSuffix
      ? `${this.metricName} [Conversational JevEval]`
      : this.metricName;
  }
}
