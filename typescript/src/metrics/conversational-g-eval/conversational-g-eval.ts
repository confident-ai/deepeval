import { BaseConversationalMetric } from "../base-conversational-metric";
import { ConversationalTestCase, MultiTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
} from "../utils";
import {
  checkConversationalTestCaseParams,
  convertTurnToDict,
} from "../conversational-utils";
import { StepsSchema, ReasonScoreSchema } from "../g-eval/schema";
import {
  type Rubric,
  numberEvaluationSteps,
  formatRubrics,
  validateAndSortRubrics,
  validateCriteriaAndEvaluationSteps,
} from "../g-eval/utils";
import {
  constructConversationalGEvalTurnParamsString,
  constructNonTurnsTestCaseString,
} from "./utils";

const TEMPLATE_CLASS = "ConversationalGEval";

export interface ConversationalGEvalMetricOptions {
  name: string;
  /** Which turn/conversation fields the judgement considers (CONTENT + ROLE always added). */
  evaluationParams: MultiTurnParams[];
  criteria?: string;
  evaluationSteps?: string[];
  rubric?: Rubric[];
  model?: DeepEvalBaseLLM | string;
  threshold?: number;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  includeGEvalSuffix?: boolean;
}

/**
 * Conversational GEval — a flexible LLM judge over a whole conversation: generate
 * eval steps from `criteria` (or use supplied `evaluationSteps`), then score the
 * conversation 0–10 (normalized to 0–1) with a reason. Like single-turn GEval but
 * over turns + conversation-level fields. (No log-prob weighting — TS has none.)
 */
export class ConversationalGEval extends BaseConversationalMetric {
  evaluationParams: MultiTurnParams[];
  criteria?: string;
  evaluationSteps?: string[];
  rubric?: Rubric[];
  private readonly metricName: string;
  private readonly includeGEvalSuffix: boolean;

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
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      showIndicator: options.showIndicator,
    });

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

      let score = gScore / 10;
      if (this.strictMode && score < this.threshold) score = 0;
      this.score = score;
      this.reason = reason;
      this.success = this.score >= this.threshold;

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
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_evaluation_steps", {
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
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_evaluation_results", {
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
    const { score, reason } = await generateWithSchema(
      this,
      prompt,
      ReasonScoreSchema,
    );
    return [score, reason];
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return this.includeGEvalSuffix
      ? `${this.metricName} [Conversational GEval]`
      : this.metricName;
  }
}
