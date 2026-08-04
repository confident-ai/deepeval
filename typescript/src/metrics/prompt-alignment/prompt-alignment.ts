import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  prettifyList,
} from "@/metrics/utils";
import {
  VerdictsSchema,
  PromptAlignmentScoreReasonSchema,
  type PromptAlignmentVerdict,
} from "@/metrics/prompt-alignment/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "PromptAlignmentMetric";

export type PromptAlignmentTemplateOverride =
  MetricTemplateOverride<"PromptAlignmentMetric">;

export interface PromptAlignmentMetricOptions {
  /** The prompt instructions the output must follow. Required, non-empty. */
  promptInstructions: string[];
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: PromptAlignmentTemplateOverride;
}

/**
 * Prompt Alignment — does the `actualOutput` follow the given
 * `promptInstructions`? Judge each instruction, then
 * score = aligned / total. **Higher is better** (`success = score >= threshold`).
 */
export class PromptAlignmentMetric extends BaseMetric {
  promptInstructions: string[];
  verdicts: PromptAlignmentVerdict[] = [];

  constructor(options: PromptAlignmentMetricOptions) {
    if (
      !options.promptInstructions ||
      options.promptInstructions.length === 0
    ) {
      throw new Error("'promptInstructions' must not be empty.");
    }
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.promptInstructions = options.promptInstructions;
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

      this.verdicts = await this.generateVerdicts(
        testCase.input,
        testCase.actualOutput,
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason(
        testCase.input,
        testCase.actualOutput,
      );
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Prompt Instructions:\n${prettifyList(this.promptInstructions)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdicts(
    input: string,
    actualOutput: string,
  ): Promise<PromptAlignmentVerdict[]> {
    const prompt = this.getPrompt("generate_verdicts", {
      prompt_instructions: this.promptInstructions,
      input,
      actual_output: actualOutput,
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(
    input: string,
    actualOutput: string,
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const unalignmentReasons = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "no")
      .map((v) => v.reason);
    const prompt = this.getPrompt("generate_reason", {
      unalignment_reasons: unalignmentReasons,
      input,
      actual_output: actualOutput,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      PromptAlignmentScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 1;
    const alignmentCount = this.verdicts.filter(
      (v) => v.verdict.trim().toLowerCase() !== "no",
    ).length;
    const score = alignmentCount / total;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Prompt Alignment";
  }
}
