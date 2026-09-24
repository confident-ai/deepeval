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
  printToolsCalled,
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
  TaskAndOutcomeSchema,
  TaskCompletionVerdictSchema,
} from "@/metrics/task-completion/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "TaskCompletionMetric";

export type TaskCompletionTemplateOverride =
  MetricTemplateOverride<"TaskCompletionMetric">;

const TASK_COMPLETION_LEVELS = [
  "Not achieved",
  "Partly achieved",
  "Mostly achieved",
  "Fully achieved",
];

export interface TaskCompletionMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  /** The task to evaluate against; auto-extracted from the trace when omitted. */
  task?: string;
  model?: DeepEvalBaseLLM | string;
  /** The System One model (Jev) used under `hybrid` / `system_one`. */
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  /** Who decides; defaults to `DEEPEVAL_EVAL_MODE`, then `llm`. */
  evalMode?: EvalModeName;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: TaskCompletionTemplateOverride;
}

/**
 * Task Completion — did the agent accomplish the task? Extracts the task +
 * outcome from the trace (or input/output as a fallback) and judges completion.
 * **Higher is better**. Trace-based (`requiresTrace`).
 */
export class TaskCompletionMetric extends BaseMetric {
  private readonly providedTask?: string;

  constructor(options: TaskCompletionMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.requiresTrace = true;
    this.providedTask = options.task;
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      const extracted = await this.extractTaskAndOutcome(testCase);
      const task = this.providedTask ?? extracted.task;
      const outcome = extracted.outcome;

      const { verdict, reason } = await this.generateVerdict(task, outcome);

      this.score = this.applyStrictMode(verdict);
      this.reason = reason ?? undefined;
      this.success = this.isSuccessful();
      this.verboseLogs = constructVerboseLogs(this, [
        `Task: ${task}`,
        `Outcome: ${outcome}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdict(
    task: string,
    outcome: string,
  ): Promise<{ verdict: number; reason?: string | null }> {
    const value = await systemOneScore(
      this,
      this.systemOneScoreSpec(task, outcome),
    );
    if (value !== undefined) {
      return {
        verdict: value,
        reason: this.includeReason
          ? formatDecisionReason(this, "task completion", value)
          : undefined,
      };
    }
    return generateWithSchema(
      this,
      this.getPrompt("generate_verdict", {
        task,
        actual_outcome: outcome,
      }),
      TaskCompletionVerdictSchema,
    );
  }

  private systemOneScoreSpec(
    task: string,
    outcome: string,
  ): SystemOneScoreSpec {
    return {
      instructions: this.getPrompt("_experimental_system_one_score"),
      levels: TASK_COMPLETION_LEVELS,
      state: { task, outcome },
    };
  }

  systemOneEvalSpec(testCase: LLMTestCase): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    const hasTrace = testCase._traceDict != null;
    const task = this.providedTask;
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
          has_task: task !== undefined,
        }),
      ),
      extraState: {
        trace: hasTrace ? compactTrace(testCase._traceDict) : undefined,
        task,
      },
    };
  }

  private async extractTaskAndOutcome(
    testCase: LLMTestCase,
  ): Promise<{ task: string; outcome: string }> {
    const prompt =
      testCase._traceDict != null
        ? this.getPrompt("extract_task_and_outcome_from_trace", {
            trace_json: JSON.stringify(testCase._traceDict),
          })
        : this.getPrompt("extract_goal_and_outcome", {
            input: testCase.input,
            actual_output: testCase.actualOutput,
            tools_called_formatted: printToolsCalled(
              testCase.toolsCalled ?? [],
            ),
          });
    return generateWithSchema(this, prompt, TaskAndOutcomeSchema);
  }

  get name(): string {
    return "Task Completion";
  }
}
