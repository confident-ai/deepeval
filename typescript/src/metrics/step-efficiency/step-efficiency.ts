import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "../utils";
import { TaskSchema, EfficiencyVerdictSchema } from "./schema";

const TEMPLATE_CLASS = "StepEfficiencyMetric";

/** Serialize the trace dict the way the templates expect (pretty JSON). */
function traceJson(d: unknown): string {
  return d != null && typeof d === "object"
    ? JSON.stringify(d, null, 2)
    : String(d ?? {});
}

export interface StepEfficiencyMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Step Efficiency — did the agent reach the task with an efficient sequence of
 * steps (no redundant/wasteful tool calls)? Reads the serialized trace.
 * **Higher is better**. Trace-based (`requiresTrace`).
 */
export class StepEfficiencyMetric extends BaseMetric {
  constructor(options: StepEfficiencyMetricOptions = {}) {
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
        resolveTemplate("metrics", TEMPLATE_CLASS, "extract_task_from_trace", {
          trace_json: json,
        }),
        TaskSchema,
      );
      const { score, reason } = await generateWithSchema(
        this,
        resolveTemplate("metrics", TEMPLATE_CLASS, "get_execution_efficiency", {
          task,
          trace_json_str: json,
        }),
        EfficiencyVerdictSchema,
      );

      this.score = this.strictMode && score < this.threshold ? 0 : score;
      this.reason = reason;
      this.success = this.score >= this.threshold;
      this.verboseLogs = constructVerboseLogs(this, [
        `Task: ${task}`,
        `Efficiency Score: ${this.score}\nEfficiency Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Step Efficiency";
  }
}
