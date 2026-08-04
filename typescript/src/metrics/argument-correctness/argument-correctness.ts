import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams, ToolCall } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  prettifyList,
  printToolsCalled,
} from "@/metrics/utils";
import {
  VerdictsSchema,
  ArgumentCorrectnessScoreReasonSchema,
  type ArgumentCorrectnessVerdict,
} from "@/metrics/argument-correctness/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "ArgumentCorrectnessMetric";

export type ArgumentCorrectnessTemplateOverride =
  MetricTemplateOverride<"ArgumentCorrectnessMetric">;

export interface ArgumentCorrectnessMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: ArgumentCorrectnessTemplateOverride;
}

/**
 * Argument Correctness — do the arguments passed in `toolsCalled` correctly and
 * relevantly address the `input`? Judge each tool call, then
 * score = correct / total. **Higher is better** (`success = score >= threshold`).
 */
export class ArgumentCorrectnessMetric extends BaseMetric {
  verdicts: ArgumentCorrectnessVerdict[] = [];

  constructor(options: ArgumentCorrectnessMetricOptions = {}) {
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
      SingleTurnParams.TOOLS_CALLED,
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

      const toolsCalled = testCase.toolsCalled ?? [];
      if (toolsCalled.length === 0) {
        this.verdicts = [];
        this.score = 1;
        this.reason = "No tool calls provided";
      } else {
        this.verdicts = await this.generateVerdicts(
          testCase.input,
          toolsCalled,
        );
        this.score = this.calculateScore();
        this.reason = await this.generateReason(testCase.input);
      }
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
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
    toolsCalled: ToolCall[],
  ): Promise<ArgumentCorrectnessVerdict[]> {
    const prompt = this.getPrompt("generate_verdicts", {
      input,
      stringified_tools_called: printToolsCalled(toolsCalled),
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(input: string): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const incorrectToolCallsReasons = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "no")
      .map((v) => v.reason);
    const prompt = this.getPrompt("generate_reason", {
      incorrect_tool_calls_reasons: incorrectToolCallsReasons,
      input,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ArgumentCorrectnessScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 1;
    const correctCount = this.verdicts.filter(
      (v) => v.verdict.trim().toLowerCase() !== "no",
    ).length;
    const score = correctCount / total;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Argument Correctness";
  }
}
