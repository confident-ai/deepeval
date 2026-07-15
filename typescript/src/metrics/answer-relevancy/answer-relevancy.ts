import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  prettifyList,
} from "../utils";
import {
  StatementsSchema,
  VerdictsSchema,
  AnswerRelevancyScoreReasonSchema,
  type AnswerRelevancyVerdict,
} from "./schema";

// Must match the key in templates.json (and the Python metric class name).
const TEMPLATE_CLASS = "AnswerRelevancyMetric";

export interface AnswerRelevancyMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

export class AnswerRelevancyMetric extends BaseMetric {
  statements: string[] = [];
  verdicts: AnswerRelevancyVerdict[] = [];

  constructor(options: AnswerRelevancyMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
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

      this.statements = await this.generateStatements(testCase.actualOutput);
      this.verdicts = await this.generateVerdicts(testCase.input);
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.input);
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Statements:\n${prettifyList(this.statements)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateStatements(actualOutput: string): Promise<string[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_statements", {
      actual_output: actualOutput,
    });
    const { statements } = await generateWithSchema(
      this,
      prompt,
      StatementsSchema,
    );
    return statements;
  }

  private async generateVerdicts(
    input: string,
  ): Promise<AnswerRelevancyVerdict[]> {
    if (this.statements.length === 0) return [];
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      input,
      statements: this.statements,
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(input: string): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const irrelevantStatements = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "no")
      .map((v) => v.reason);
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      irrelevant_statements: irrelevantStatements,
      input,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      AnswerRelevancyScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 1;
    let relevant = 0;
    for (const v of this.verdicts) {
      if (v.verdict.trim().toLowerCase() !== "no") relevant++;
    }
    const score = relevant / total;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Answer Relevancy";
  }
}
