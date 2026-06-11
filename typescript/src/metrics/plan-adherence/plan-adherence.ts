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
import { TaskSchema, AgentPlanSchema, PlanAdherenceScoreSchema } from "./schema";

// `extract_task_from_trace` lives under StepEfficiencyMetric (shared, mirrors Python).
const TASK_TEMPLATE_CLASS = "StepEfficiencyMetric";
const TEMPLATE_CLASS = "PlanAdherenceMetric";

const NO_PLAN_REASON =
  "There were no plans to evaluate within the trace of your agent's execution. " +
  "Please check if the agent's planning or reasoning or thinking is stored in any one of the trace attributes.";

function traceJson(d: unknown): string {
  return d != null && typeof d === "object"
    ? JSON.stringify(d, null, 2)
    : String(d ?? {});
}

export interface PlanAdherenceMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Plan Adherence — did the agent's execution follow the plan it laid out?
 * Extracts the task + plan from the trace, then judges adherence against the
 * execution. **Higher is better**. Trace-based (`requiresTrace`).
 */
export class PlanAdherenceMetric extends BaseMetric {
  constructor(options: PlanAdherenceMetricOptions = {}) {
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
        resolveTemplate("metrics", TASK_TEMPLATE_CLASS, "extract_task_from_trace", {
          trace_json: json,
        }),
        TaskSchema,
      );
      const { plan } = await generateWithSchema(
        this,
        resolveTemplate("metrics", TEMPLATE_CLASS, "extract_plan_from_trace", {
          trace_json_str: json,
        }),
        AgentPlanSchema,
      );

      if (plan.length === 0) {
        this.score = 1;
        this.reason = NO_PLAN_REASON;
      } else {
        const { score, reason } = await generateWithSchema(
          this,
          resolveTemplate("metrics", TEMPLATE_CLASS, "evaluate_adherence", {
            user_task: task,
            agent_plan: plan.join("\n"),
            execution_trace_json: json,
          }),
          PlanAdherenceScoreSchema,
        );
        this.score = this.strictMode && score < this.threshold ? 0 : score;
        this.reason = reason;
      }
      this.success = this.score >= this.threshold;
      this.verboseLogs = constructVerboseLogs(this, [
        `Task: ${task}`,
        `Plan steps: ${plan.length}`,
        `Final Score: ${this.score}\nFinal Reason: ${this.reason}`,
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
    return "Plan Adherence";
  }
}
