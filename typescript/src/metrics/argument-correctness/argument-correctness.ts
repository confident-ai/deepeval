import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams, ToolCall } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
  printToolsCalled,
} from "@/metrics/utils";
import {
  generateQagVerdicts,
  parseQuestions,
  runSystemOneEval,
  type SystemOneEvalSpec,
  type SystemOneVerdictSpec,
} from "@/metrics/system-one";
import {
  VerdictsSchema,
  ArgumentCorrectnessScoreReasonSchema,
  type ArgumentCorrectnessVerdict,
} from "@/metrics/argument-correctness/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { prepareMeasure } from "@/metrics/prepare-measure";

const TEMPLATE_CLASS = "ArgumentCorrectnessMetric";

export type ArgumentCorrectnessTemplateOverride =
  MetricTemplateOverride<"ArgumentCorrectnessMetric">;

export interface ArgumentCorrectnessMetricOptions {
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
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);

      const toolsCalled = testCase.toolsCalled ?? [];
      if (toolsCalled.length === 0) {
        this.verdicts = [];
        this.score = 1;
        this.reason = "No tool calls provided";
      } else if (await runSystemOneEval(this, testCase)) {
        return this.score as number;
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
    return generateQagVerdicts(this, {
      systemOne: this.systemOneVerdictSpec(input, toolsCalled),
      llm: async () => {
        const prompt = this.getPrompt("generate_verdicts", {
          input,
          stringified_tools_called: printToolsCalled(toolsCalled),
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

  private systemOneVerdictSpec(
    input: string,
    toolsCalled: ToolCall[],
  ): SystemOneVerdictSpec<string, ArgumentCorrectnessVerdict> {
    return {
      instructions: this.getPrompt("_experimental_system_one_verdict"),
      items: toolsCalled.map((toolCall) => printToolsCalled([toolCall])),
      itemKey: "tool_call",
      state: { input },
    };
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
