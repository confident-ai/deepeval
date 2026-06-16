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
import { getTasks, taskStepsTakenText, availableMcpServersBlock } from "./utils";
import {
  ToolScoreSchema,
  ArgsScoreSchema,
  ReasonSchema,
  type ToolScore,
  type ArgsScore,
} from "./schema";

// Reuses the MCPTaskCompletionMetric template namespace (mirrors Python).
const TEMPLATE_CLASS = "MCPTaskCompletionMetric";

export interface MultiTurnMCPUseMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Multi-Turn MCP Use — across the conversation, did the agent select correct
 * MCP tools and pass correct arguments? Final score = min(mean tool-correctness,
 * mean argument-correctness). **Higher is better**. Requires non-empty `mcpServers`.
 */
export class MultiTurnMCPUseMetric extends BaseConversationalMetric {
  constructor(options: MultiTurnMCPUseMetricOptions = {}) {
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
          "'mcpServers' in a conversational test case cannot be empty for the 'MultiTurnMCPUseMetric' metric.";
        this.error = msg;
        throw new MissingTestCaseParamsError(msg);
      }
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      const tasks = getTasks(getUnitInteractions(testCase.turns));
      const { availableTools, availableResources, availablePrompts } =
        availableMcpServersBlock(testCase.mcpServers);

      const toolScores = await Promise.all(
        tasks.map((task) =>
          generateWithSchema(
            this,
            resolveTemplate("metrics", TEMPLATE_CLASS, "get_tool_correctness_score", {
              task,
              available_tools: availableTools,
              steps_taken: taskStepsTakenText(task),
            }),
            ToolScoreSchema,
          ),
        ),
      );
      const argScores = await Promise.all(
        tasks.map((task) =>
          generateWithSchema(
            this,
            resolveTemplate("metrics", TEMPLATE_CLASS, "get_args_correctness_score", {
              task,
              available_tools: availableTools,
              available_resources: availableResources,
              available_prompts: availablePrompts,
              steps_taken: taskStepsTakenText(task),
            }),
            ArgsScoreSchema,
          ),
        ),
      );

      this.score = this.calculateScore(toolScores, argScores);
      this.success = this.score >= this.threshold;
      this.reason = await this.generateReason(toolScores, argScores);

      this.verboseLogs = constructVerboseLogs(this, [
        `Tasks: ${tasks.length}`,
        `Tool scores: ${toolScores.map((t) => t.score).join(", ")}`,
        `Args scores: ${argScores.map((a) => a.score).join(", ")}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private calculateScore(
    toolScores: ToolScore[],
    argScores: ArgsScore[],
  ): number {
    const mean = (xs: { score: number }[]) =>
      xs.reduce((s, x) => s + x.score, 0) / Math.max(xs.length, 1);
    const score = Math.min(mean(toolScores), mean(argScores));
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  private async generateReason(
    toolScores: ToolScore[],
    argScores: ArgsScore[],
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const reasons = [
      ...toolScores.map((t) => t.reason),
      ...argScores.map((a) => a.reason),
    ];
    const { reason } = await generateWithSchema(
      this,
      resolveTemplate("metrics", TEMPLATE_CLASS, "generate_final_reason", {
        final_score: this.score,
        success: this.success,
        reasons,
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
    return "Multi-Turn MCP Use";
  }
}
