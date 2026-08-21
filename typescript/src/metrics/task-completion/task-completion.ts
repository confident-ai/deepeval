import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  printToolsCalled,
} from "@/metrics/utils";
import {
  TaskAndOutcomeSchema,
  TaskCompletionVerdictSchema,
} from "@/metrics/task-completion/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "TaskCompletionMetric";

export type TaskCompletionTemplateOverride =
  MetricTemplateOverride<"TaskCompletionMetric">;

export interface TaskCompletionMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  /** The task to evaluate against; auto-extracted from the trace when omitted. */
  task?: string;
  model?: DeepEvalBaseLLM | string;
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

      const extracted = await this.extractTaskAndOutcome(testCase);
      const task = this.providedTask ?? extracted.task;
      const outcome = extracted.outcome;

      const { verdict, reason } = await generateWithSchema(
        this,
        this.getPrompt("generate_verdict", {
          task,
          actual_outcome: outcome,
        }),
        TaskCompletionVerdictSchema,
      );

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
