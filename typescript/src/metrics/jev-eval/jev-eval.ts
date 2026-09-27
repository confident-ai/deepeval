import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseSystemOneModel } from "@/models/system-one/base-system-one-model";
import { checkSingleTurnParams, constructVerboseLogs } from "@/metrics/utils";
import { formatSystemOneReason } from "@/metrics/system-one/reason";
import type {
  JevQuestion,
  QuestionOutcome,
} from "@/metrics/jev-eval/questions";
import {
  aggregate,
  aggregateStrict,
  buildQuestions,
  constructSingleTurnState,
  formatOutcomesForLogs,
  initializeJevModel,
  markStrict,
  minConfidence,
  outcomesFromAnswers,
  validateQuestions,
} from "@/metrics/jev-eval/utils";

export interface JevEvalOptions {
  name: string;
  /** The test case fields your questions refer to. */
  evaluationParams: SingleTurnParams[];
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

/**
 * A score metric whose decision points are answered by TypeSafe AI's System
 * One model (Jev) with calibrated probabilities, never generated text. The
 * questions go to Jev in one request, each answer maps onto [0, 1], and the
 * score is their weighted mean. No LLM is involved at any point, whatever the
 * eval mode: the reason is built from Jev's answers.
 */
export class JevEval extends BaseMetric {
  readonly metricName: string;
  evaluationParams: SingleTurnParams[];
  questions: JevQuestion[];
  declare systemOneModel: DeepEvalBaseSystemOneModel;
  private readonly includeJevEvalSuffix: boolean;

  constructor(options: JevEvalOptions) {
    if (!options.evaluationParams || options.evaluationParams.length === 0) {
      throw new Error(
        "evaluationParams cannot be empty; list the test case fields your " +
          "questions refer to.",
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
    this.metricName = options.name;
    this.evaluationParams = [...options.evaluationParams];
    this.requiredParams = this.evaluationParams;
    this.questions = validateQuestions(options.questions);
    this.systemOneModel = initializeJevModel(options.systemOneModel);
    this.evaluationModel = this.systemOneModel.getModelName();
    this.includeJevEvalSuffix = options.includeJevEvalSuffix ?? true;
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      if (testCase instanceof LLMTestCase && testCase.multimodal) {
        this.error =
          `${this.name} evaluates text only: Jev has no image input. Pass a ` +
          "text-only LLMTestCase.";
        throw new Error(this.error);
      }
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = 0;

      const state = constructSingleTurnState(this.evaluationParams, testCase);
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
      ? `${this.metricName} [JevEval]`
      : this.metricName;
  }
}
