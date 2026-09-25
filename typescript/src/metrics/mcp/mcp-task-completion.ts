import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
} from "@/metrics/utils";
import { getUnitInteractions } from "@/metrics/conversational-utils";
import { getTasks, taskStepsTakenText } from "@/metrics/mcp/utils";
import {
  formatDecisionReason,
  parseQuestions,
  runSystemOneEval,
  systemOneScore,
  type SystemOneEvalSpec,
  type SystemOneScoreSpec,
} from "@/metrics/system-one";
import {
  TaskScoreSchema,
  ReasonSchema,
  type TaskScore,
  type Task,
} from "@/metrics/mcp/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { prepareMeasure } from "@/metrics/prepare-measure";

const TEMPLATE_CLASS = "MCPTaskCompletionMetric";

export type MCPTaskCompletionTemplateOverride =
  MetricTemplateOverride<"MCPTaskCompletionMetric">;

const TASK_COMPLETION_LEVELS = [
  "Not completed",
  "Partly completed",
  "Mostly completed",
  "Fully completed",
];

export interface MCPTaskCompletionMetricOptions {
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
  evaluationTemplate?: MCPTaskCompletionTemplateOverride;
}

/**
 * MCP Task Completion — across the conversation, how well did the agent's MCP
 * interactions complete each user task? Score = mean of per-task completion
 * scores. **Higher is better**. Requires non-empty `mcpServers`.
 */
export class MCPTaskCompletionMetric extends BaseConversationalMetric {
  constructor(options: MCPTaskCompletionMetricOptions = {}) {
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
    this.requiredParams = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT];
    this.requiresMcpServers = true;
    initializeMetricModels(this, options);
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      const tasks = getTasks(getUnitInteractions(testCase.turns));
      const taskScores = await Promise.all(
        tasks.map((task) => this.getTaskScore(task, testCase.multimodal)),
      );

      const mean =
        taskScores.reduce((s, t) => s + t.score, 0) /
        Math.max(taskScores.length, 1);
      this.score = this.applyStrictMode(mean);
      this.success = this.isSuccessful();
      this.reason = await this.generateReason(taskScores);

      this.verboseLogs = constructVerboseLogs(this, [
        `Tasks: ${tasks.length}`,
        `Scores: ${taskScores.map((t) => t.score).join(", ")}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async getTaskScore(
    task: Task,
    multimodal: boolean,
  ): Promise<TaskScore> {
    const value = await systemOneScore(
      this,
      this.systemOneScoreSpec(task, multimodal),
    );
    if (value !== undefined) {
      return {
        score: value,
        reason: formatDecisionReason(this, "task completion", value),
      };
    }
    return generateWithSchema(
      this,
      this.getPrompt("get_task_completion_score", {
        task,
        steps_taken: taskStepsTakenText(task),
      }),
      TaskScoreSchema,
    );
  }

  private systemOneScoreSpec(
    task: Task,
    multimodal: boolean,
  ): SystemOneScoreSpec | undefined {
    if (multimodal) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_score"),
      levels: TASK_COMPLETION_LEVELS,
      state: { task: task.task, steps_taken: task.steps_taken },
    };
  }

  systemOneEvalSpec(
    testCase: ConversationalTestCase,
  ): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
        MultiTurnParams.MCP_TOOLS,
        MultiTurnParams.MCP_RESOURCES,
        MultiTurnParams.MCP_PROMPTS,
        MultiTurnParams.TOOLS_CALLED,
      ],
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
  }

  private async generateReason(
    taskScores: TaskScore[],
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const { reason } = await generateWithSchema(
      this,
      this.getPrompt("generate_final_reason", {
        final_score: this.score,
        success: this.success,
        reasons: taskScores.map((t) => t.reason),
      }),
      ReasonSchema,
    );
    return reason;
  }

  get name(): string {
    return "MCP Task Completion";
  }
}
