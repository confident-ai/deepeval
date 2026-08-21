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
  AgentPlanSchema,
  PlanQualityScoreSchema,
} from "@/metrics/plan-quality/schema";
import { type MetricTemplateOverride } from "@/templates/override";

// Shared templates (mirror Python): task extraction → StepEfficiencyMetric,
// plan extraction → PlanAdherenceMetric.
const TASK_TEMPLATE_CLASS = "StepEfficiencyMetric";
const PLAN_TEMPLATE_CLASS = "PlanAdherenceMetric";
const TEMPLATE_CLASS = "PlanQualityMetric";

export type PlanQualityTemplateOverride =
  MetricTemplateOverride<"PlanQualityMetric">;

const NO_PLAN_REASON =
  "There were no plans to evaluate within the trace of your agent's execution. " +
  "Please check if the agent's planning or reasoning or thinking is stored in any one of the trace attributes.";

function traceJson(d: unknown): string {
  return d != null && typeof d === "object"
    ? JSON.stringify(d, null, 2)
    : String(d ?? {});
}

export interface PlanQualityMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: PlanQualityTemplateOverride;
}

/**
 * Plan Quality — was the plan the agent laid out well-formed and appropriate for
 * the task (independent of how faithfully it was executed)? Extracts task + plan
 * from the trace and judges the plan. **Higher is better**. Trace-based.
 */
export class PlanQualityMetric extends BaseMetric {
  constructor(options: PlanQualityMetricOptions = {}) {
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
        this.getPrompt(
          "extract_task_from_trace",
          {
            trace_json: json,
          },
          { templateClass: TASK_TEMPLATE_CLASS },
        ),
        TaskSchema,
      );
      const { plan } = await generateWithSchema(
        this,
        this.getPrompt(
          "extract_plan_from_trace",
          {
            trace_json_str: json,
          },
          { templateClass: PLAN_TEMPLATE_CLASS },
        ),
        AgentPlanSchema,
      );

      if (plan.length === 0) {
        this.score = 1;
        this.reason = NO_PLAN_REASON;
      } else {
        const { score, reason } = await generateWithSchema(
          this,
          this.getPrompt("evaluate_plan_quality", {
            user_task: task,
            agent_plan: plan.join("\n"),
          }),
          PlanQualityScoreSchema,
        );
        this.score = this.applyStrictMode(score);
        this.reason = reason;
      }
      this.success = this.isSuccessful();
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

  get name(): string {
    return "Plan Quality";
  }
}
