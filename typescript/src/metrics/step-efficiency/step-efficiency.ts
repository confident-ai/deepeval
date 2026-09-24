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
  EfficiencyVerdictSchema,
} from "@/metrics/step-efficiency/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "StepEfficiencyMetric";

export type StepEfficiencyTemplateOverride =
  MetricTemplateOverride<"StepEfficiencyMetric">;

const STEP_EFFICIENCY_LEVELS = [
  "Highly inefficient",
  "Low efficiency",
  "Moderate efficiency",
  "Strong efficiency",
  "Perfectly efficient",
];

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
  /** The System One model (Jev) used under `hybrid` / `system_one`. */
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  /** Who decides; defaults to `DEEPEVAL_EVAL_MODE`, then `llm`. */
  evalMode?: EvalModeName;
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
        this.getPrompt("extract_task_from_trace", {
          trace_json: json,
        }),
        TaskSchema,
      );
      const { score, reason } = await this.getScore(task, testCase, json);

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

  private async getScore(
    task: string,
    testCase: LLMTestCase,
    json: string,
  ): Promise<{ score: number; reason: string }> {
    const value = await systemOneScore(
      this,
      this.systemOneScoreSpec(task, testCase),
    );
    if (value !== undefined) {
      return {
        score: value,
        reason: formatDecisionReason(this, "step efficiency", value),
      };
    }
    return generateWithSchema(
      this,
      this.getPrompt("get_execution_efficiency", {
        task,
        trace_json_str: json,
      }),
      EfficiencyVerdictSchema,
    );
  }

  private systemOneScoreSpec(
    task: string,
    testCase: LLMTestCase,
  ): SystemOneScoreSpec | undefined {
    if (testCase.multimodal || testCase._traceDict == null) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_score"),
      levels: STEP_EFFICIENCY_LEVELS,
      state: {
        task,
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
    return "Step Efficiency";
  }
}
