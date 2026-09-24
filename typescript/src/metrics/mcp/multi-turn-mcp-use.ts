import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import { MissingTestCaseParamsError } from "@/errors";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
} from "@/metrics/utils";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
} from "@/metrics/conversational-utils";
import {
  getTasks,
  taskStepsTakenText,
  availableMcpServersBlock,
  mcpServersState,
} from "@/metrics/mcp/utils";
import {
  formatDecisionReason,
  parseQuestions,
  runSystemOneEval,
  systemOneScore,
  type SystemOneEvalSpec,
  type SystemOneScoreSpec,
} from "@/metrics/system-one";
import {
  ToolScoreSchema,
  ArgsScoreSchema,
  ReasonSchema,
  type ToolScore,
  type ArgsScore,
  type Task,
} from "@/metrics/mcp/schema";
// Owns no templates: every prompt is borrowed, so there is no `evaluationTemplate`
// (as in Python, whose constructor also has no `evaluation_template`).
const BORROWED_TEMPLATE_CLASS = "MCPTaskCompletionMetric";

const PRIMITIVE_USAGE_LEVELS = [
  "Wrong primitives",
  "Poor choice",
  "Reasonable choice",
  "Best choice",
];
const ARGUMENT_CORRECTNESS_LEVELS = [
  "Incorrect",
  "Mostly incorrect",
  "Mostly correct",
  "Fully correct",
];

export interface MultiTurnMCPUseMetricOptions {
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
}

/**
 * Multi-Turn MCP Use — across the conversation, did the agent select correct
 * MCP tools and pass correct arguments? Final score = min(mean tool-correctness,
 * mean argument-correctness). **Higher is better**. Requires non-empty `mcpServers`.
 */
export class MultiTurnMCPUseMetric extends BaseConversationalMetric {
  constructor(options: MultiTurnMCPUseMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
    });
    this.multimodalAware = true;
    this.requiredParams = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT];
    initializeMetricModels(this, options);
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
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      const tasks = getTasks(getUnitInteractions(testCase.turns));
      const { availableTools, availableResources, availablePrompts } =
        availableMcpServersBlock(testCase.mcpServers);

      const toolScores = await Promise.all(
        tasks.map(async (task) => {
          const value = await systemOneScore(
            this,
            this.systemOnePrimitivesSpec(task, testCase),
          );
          if (value !== undefined) {
            return {
              score: value,
              reason: formatDecisionReason(this, "primitive usage", value),
            };
          }
          return generateWithSchema(
            this,
            this.getPrompt(
              "get_tool_correctness_score",
              {
                task,
                available_tools: availableTools,
                steps_taken: taskStepsTakenText(task),
              },
              { templateClass: BORROWED_TEMPLATE_CLASS },
            ),
            ToolScoreSchema,
          );
        }),
      );
      const argScores = await Promise.all(
        tasks.map(async (task) => {
          const value = await systemOneScore(
            this,
            this.systemOneArgsSpec(task, testCase),
          );
          if (value !== undefined) {
            return {
              score: value,
              reason: formatDecisionReason(this, "argument correctness", value),
            };
          }
          return generateWithSchema(
            this,
            this.getPrompt(
              "get_args_correctness_score",
              {
                task,
                available_tools: availableTools,
                available_resources: availableResources,
                available_prompts: availablePrompts,
                steps_taken: taskStepsTakenText(task),
              },
              { templateClass: BORROWED_TEMPLATE_CLASS },
            ),
            ArgsScoreSchema,
          );
        }),
      );

      this.score = this.calculateScore(toolScores, argScores);
      this.success = this.isSuccessful();
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
    return this.applyStrictMode(score);
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
      this.getPrompt(
        "generate_final_reason",
        {
          final_score: this.score,
          success: this.success,
          reasons,
        },
        { templateClass: BORROWED_TEMPLATE_CLASS },
      ),
      ReasonSchema,
    );
    return reason;
  }

  private systemOneTaskState(
    task: Task,
    testCase: ConversationalTestCase,
  ): Record<string, unknown> {
    return {
      task: task.task,
      steps_taken: task.steps_taken,
      mcp_servers: mcpServersState(testCase.mcpServers),
    };
  }

  private systemOnePrimitivesSpec(
    task: Task,
    testCase: ConversationalTestCase,
  ): SystemOneScoreSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      instructions: this.getPrompt(
        "_experimental_system_one_mcp_use_primitive_score",
        {},
        { templateClass: BORROWED_TEMPLATE_CLASS },
      ),
      levels: PRIMITIVE_USAGE_LEVELS,
      state: this.systemOneTaskState(task, testCase),
    };
  }

  private systemOneArgsSpec(
    task: Task,
    testCase: ConversationalTestCase,
  ): SystemOneScoreSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      instructions: this.getPrompt(
        "_experimental_system_one_mcp_use_args_score",
        {},
        { templateClass: BORROWED_TEMPLATE_CLASS },
      ),
      levels: ARGUMENT_CORRECTNESS_LEVELS,
      state: this.systemOneTaskState(task, testCase),
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
        this.getPrompt(
          "_experimental_system_one_mcp_use_questions",
          {},
          { templateClass: BORROWED_TEMPLATE_CLASS },
        ),
      ),
      extraState: {
        mcp_servers: mcpServersState(testCase.mcpServers),
      },
    };
  }

  get name(): string {
    return "Multi-Turn MCP Use";
  }
}
