import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "@/metrics/utils";
import {
  TaskSchema,
  EfficiencyVerdictSchema,
} from "@/metrics/step-efficiency/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "StepEfficiencyMetric";

export type StepEfficiencyTemplateOverride =
  MetricTemplateOverride<"StepEfficiencyMetric">;

/** Serialize the trace dict the way the templates expect (pretty JSON). */
function traceJson(d: unknown): string {
  return d != null && typeof d === "object"
    ? JSON.stringify(d, null, 2)
    : String(d ?? {});
}

export interface StepEfficiencyMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: StepEfficiencyTemplateOverride;
}

/**
 * Step Efficiency — did the agent reach the task with an efficient sequence of
 * steps (no redundant/wasteful tool calls)? Reads the serialized trace.
 * **Higher is better**. Trace-based (`requiresTrace`).
 */
export class StepEfficiencyMetric extends BaseMetric {
  constructor(options: StepEfficiencyMetricOptions = {}) {
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
    this.requiresTrace = true;
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
      const json = traceJson(testCase._traceDict);

      const { task } = await generateWithSchema(
        this,
        this.getPrompt("extract_task_from_trace", {
          trace_json: json,
        }),
        TaskSchema,
      );
      const { score, reason } = await generateWithSchema(
        this,
        this.getPrompt("get_execution_efficiency", {
          task,
          trace_json_str: json,
        }),
        EfficiencyVerdictSchema,
      );

      this.score = this.applyStrictMode(score);
      this.reason = reason;
      this.success = this.isSuccessful();
      this.verboseLogs = constructVerboseLogs(this, [
        `Task: ${task}`,
        `Efficiency Score: ${this.score}\nEfficiency Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  get name(): string {
    return "Step Efficiency";
  }
}
