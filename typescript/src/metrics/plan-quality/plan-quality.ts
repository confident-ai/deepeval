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
import { TaskSchema, AgentPlanSchema, PlanQualityScoreSchema } from "./schema";

// Shared templates (mirror Python): task extraction → StepEfficiencyMetric,
// plan extraction → PlanAdherenceMetric.
const TASK_TEMPLATE_CLASS = "StepEfficiencyMetric";
const PLAN_TEMPLATE_CLASS = "PlanAdherenceMetric";
const TEMPLATE_CLASS = "PlanQualityMetric";

const NO_PLAN_REASON =
  "There were no plans to evaluate within the trace of your agent's execution. " +
  "Please check if the agent's planning or reasoning or thinking is stored in any one of the trace attributes.";

function traceJson(d: unknown): string {
  return d != null && typeof d === "object"
    ? JSON.stringify(d, null, 2)
    : String(d ?? {});
}

export interface PlanQualityMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Plan Quality — was the plan the agent laid out well-formed and appropriate for
 * the task (independent of how faithfully it was executed)? Extracts task + plan
 * from the trace and judges the plan. **Higher is better**. Trace-based.
 */
export class PlanQualityMetric extends BaseMetric {
  constructor(options: PlanQualityMetricOptions = {}) {
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
        resolveTemplate("metrics", PLAN_TEMPLATE_CLASS, "extract_plan_from_trace", {
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
          resolveTemplate("metrics", TEMPLATE_CLASS, "evaluate_plan_quality", {
            user_task: task,
            agent_plan: plan.join("\n"),
          }),
          PlanQualityScoreSchema,
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
    return "Plan Quality";
  }
}
