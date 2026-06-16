import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  printToolsCalled,
} from "../utils";
import { TaskAndOutcomeSchema, TaskCompletionVerdictSchema } from "./schema";

const TEMPLATE_CLASS = "TaskCompletionMetric";

export interface TaskCompletionMetricOptions {
  threshold?: number;
  /** The task to evaluate against; auto-extracted from the trace when omitted. */
  task?: string;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
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
        resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdict", {
          task,
          actual_outcome: outcome,
        }),
        TaskCompletionVerdictSchema,
      );

      this.score = this.strictMode && verdict < this.threshold ? 0 : verdict;
      this.reason = reason ?? undefined;
      this.success = this.score >= this.threshold;
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
        ? resolveTemplate(
            "metrics",
            TEMPLATE_CLASS,
            "extract_task_and_outcome_from_trace",
            { trace_json: JSON.stringify(testCase._traceDict) },
          )
        : resolveTemplate("metrics", TEMPLATE_CLASS, "extract_goal_and_outcome", {
            input: testCase.input,
            actual_output: testCase.actualOutput,
            tools_called_formatted: printToolsCalled(testCase.toolsCalled ?? []),
          });
    return generateWithSchema(this, prompt, TaskAndOutcomeSchema);
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Task Completion";
  }
}
