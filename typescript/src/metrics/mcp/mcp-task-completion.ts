import { BaseConversationalMetric } from "../base-conversational-metric";
import { ConversationalTestCase, MultiTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import { MissingTestCaseParamsError } from "../../errors";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
} from "../utils";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
} from "../conversational-utils";
import { getTasks, taskStepsTakenText } from "./utils";
import { TaskScoreSchema, ReasonSchema, type TaskScore } from "./schema";

const TEMPLATE_CLASS = "MCPTaskCompletionMetric";

export interface MCPTaskCompletionMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * MCP Task Completion — across the conversation, how well did the agent's MCP
 * interactions complete each user task? Score = mean of per-task completion
 * scores. **Higher is better**. Requires non-empty `mcpServers`.
 */
export class MCPTaskCompletionMetric extends BaseConversationalMetric {
  constructor(options: MCPTaskCompletionMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT];
    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
      if (!testCase.mcpServers || testCase.mcpServers.length === 0) {
        const msg =
          "'mcpServers' in a conversational test case cannot be empty for the 'MCPTaskCompletionMetric' metric.";
        this.error = msg;
        throw new MissingTestCaseParamsError(msg);
      }
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      const tasks = getTasks(getUnitInteractions(testCase.turns));
      const taskScores = await Promise.all(
        tasks.map((task) =>
          generateWithSchema(
            this,
            resolveTemplate("metrics", TEMPLATE_CLASS, "get_task_completion_score", {
              task,
              steps_taken: taskStepsTakenText(task),
            }),
            TaskScoreSchema,
          ),
        ),
      );

      const mean =
        taskScores.reduce((s, t) => s + t.score, 0) /
        Math.max(taskScores.length, 1);
      this.score = this.strictMode && mean < this.threshold ? 0 : mean;
      this.success = this.score >= this.threshold;
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

  private async generateReason(
    taskScores: TaskScore[],
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const { reason } = await generateWithSchema(
      this,
      resolveTemplate("metrics", TEMPLATE_CLASS, "generate_final_reason", {
        final_score: this.score,
        success: this.success,
        reasons: taskScores.map((t) => t.reason),
      }),
      ReasonSchema,
    );
    return reason;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "MCP Task Completion";
  }
}
