import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
} from "@/metrics/utils";
import {
  generateQagVerdicts,
  parseQuestions,
  runSystemOneEval,
  type SystemOneEvalSpec,
  type SystemOneVerdictSpec,
} from "@/metrics/system-one";
import {
  StatementsSchema,
  VerdictsSchema,
  AnswerRelevancyScoreReasonSchema,
  type AnswerRelevancyVerdict,
} from "@/metrics/answer-relevancy/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { prepareMeasure } from "@/metrics/prepare-measure";

// Must match the key in templates.json (and the Python metric class name).
const TEMPLATE_CLASS = "AnswerRelevancyMetric";

export type AnswerRelevancyTemplateOverride =
  MetricTemplateOverride<"AnswerRelevancyMetric">;

export interface AnswerRelevancyMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  /** The System One model (Jev) used under `hybrid` / `system_one`. */
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  /** Who decides; defaults to `DEEPEVAL_EVAL_MODE`, then `llm`. */
  evalMode?: EvalModeName;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: AnswerRelevancyTemplateOverride;
}

export class AnswerRelevancyMetric extends BaseMetric {
  statements: string[] = [];
  verdicts: AnswerRelevancyVerdict[] = [];

  constructor(options: AnswerRelevancyMetricOptions = {}) {
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
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      this.statements = await this.generateStatements(testCase.actualOutput);
      this.verdicts = await this.generateVerdicts(testCase.input);
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.input);
      this.success = this.isSuccessful();

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

  systemOneEvalSpec(testCase: LLMTestCase): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: this.requiredParams,
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
  }

  private systemOneVerdictSpec(
    input: string,
  ): SystemOneVerdictSpec<string, AnswerRelevancyVerdict> {
    return {
      instructions: this.getPrompt("_experimental_system_one_verdict"),
      items: this.statements,
      itemKey: "statement",
      state: { input },
      borderline: "idk",
    };
  }

  private async generateStatements(actualOutput: string): Promise<string[]> {
    const prompt = this.getPrompt("generate_statements", {
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
    return generateQagVerdicts(this, {
      systemOne: this.systemOneVerdictSpec(input),
      llm: async () => {
        const prompt = this.getPrompt("generate_verdicts", {
          input,
          statements: this.statements,
        });
        const { verdicts } = await generateWithSchema(
          this,
          prompt,
          VerdictsSchema,
        );
        return verdicts;
      },
    });
  }

  private async generateReason(input: string): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const irrelevantStatements = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "no")
      .map((v) => v.reason);
    const prompt = this.getPrompt("generate_reason", {
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
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Answer Relevancy";
  }
}
