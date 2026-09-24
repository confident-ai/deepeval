import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "@/metrics/utils";
import {
  compactTrace,
  formatDecisionReason,
  parseQuestions,
  runSystemOneEval,
  systemOneScore,
  type SystemOneEvalSpec,
  type SystemOneScoreSpec,
} from "@/metrics/system-one";
import {
  TaskSchema,
  AgentPlanSchema,
  PlanAdherenceScoreSchema,
} from "@/metrics/plan-adherence/schema";
import { type MetricTemplateOverride } from "@/templates/override";

// `extract_task_from_trace` lives under StepEfficiencyMetric (shared, mirrors Python).
const TASK_TEMPLATE_CLASS = "StepEfficiencyMetric";
const TEMPLATE_CLASS = "PlanAdherenceMetric";

export type PlanAdherenceTemplateOverride =
  MetricTemplateOverride<"PlanAdherenceMetric">;

const PLAN_ADHERENCE_LEVELS = [
  "No adherence",
  "Weak adherence",
  "Partial adherence",
  "Strong adherence",
  "Perfect adherence",
];

const NO_PLAN_REASON =
  "There were no plans to evaluate within the trace of your agent's execution. " +
  "Please check if the agent's planning or reasoning or thinking is stored in any one of the trace attributes.";

function traceJson(d: unknown): string {
  return d != null && typeof d === "object"
    ? JSON.stringify(d, null, 2)
    : String(d ?? {});
}

export interface PlanAdherenceMetricOptions {
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
  evaluationTemplate?: PlanAdherenceTemplateOverride;
}

/**
 * Plan Adherence — did the agent's execution follow the plan it laid out?
 * Extracts the task + plan from the trace, then judges adherence against the
 * execution. **Higher is better**. Trace-based (`requiresTrace`).
 */
export class PlanAdherenceMetric extends BaseMetric {
  constructor(options: PlanAdherenceMetricOptions = {}) {
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
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      if (await runSystemOneEval(this, testCase)) return this.score as number;
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
        this.getPrompt("extract_plan_from_trace", {
          trace_json_str: json,
        }),
        AgentPlanSchema,
      );

      if (plan.length === 0) {
        this.score = 1;
        this.reason = NO_PLAN_REASON;
      } else {
        const { score, reason } = await this.getPlanAdherenceScore(
          task,
          plan,
          testCase,
          json,
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

  private async getPlanAdherenceScore(
    task: string,
    plan: string[],
    testCase: LLMTestCase,
    json: string,
  ): Promise<{ score: number; reason: string }> {
    const value = await systemOneScore(
      this,
      this.systemOneScoreSpec(task, plan, testCase),
    );
    if (value !== undefined) {
      return {
        score: value,
        reason: formatDecisionReason(this, "plan adherence", value),
      };
    }
    return generateWithSchema(
      this,
      this.getPrompt("evaluate_adherence", {
        user_task: task,
        agent_plan: plan.join("\n"),
        execution_trace_json: json,
      }),
      PlanAdherenceScoreSchema,
    );
  }

  private systemOneScoreSpec(
    task: string,
    plan: string[],
    testCase: LLMTestCase,
  ): SystemOneScoreSpec | undefined {
    if (testCase.multimodal || testCase._traceDict == null) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_score"),
      levels: PLAN_ADHERENCE_LEVELS,
      state: {
        task,
        plan,
        trace: compactTrace(testCase._traceDict),
      },
    };
  }

  systemOneEvalSpec(testCase: LLMTestCase): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    const hasTrace = testCase._traceDict != null;
    return {
      evaluationParams: hasTrace
        ? []
        : [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.TOOLS_CALLED,
          ],
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions", {
          has_trace: hasTrace,
        }),
      ),
      extraState: {
        trace: hasTrace ? compactTrace(testCase._traceDict) : undefined,
      },
    };
  }

  get name(): string {
    return "Plan Adherence";
  }
}
