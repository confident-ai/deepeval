import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams, ToolCall } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  prettifyList,
  printToolsCalled,
} from "../utils";
import {
  VerdictsSchema,
  ArgumentCorrectnessScoreReasonSchema,
  type ArgumentCorrectnessVerdict,
} from "./schema";

const TEMPLATE_CLASS = "ArgumentCorrectnessMetric";

export interface ArgumentCorrectnessMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
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
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
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
        this.verdicts = await this.generateVerdicts(testCase.input, toolsCalled);
        this.score = this.calculateScore();
        this.reason = await this.generateReason(testCase.input);
      }
      this.success = this.score >= this.threshold;

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
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
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
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
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
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Argument Correctness";
  }
}
