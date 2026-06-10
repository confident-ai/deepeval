import { BaseConversationalMetric } from "../base-conversational-metric";
import {
  ConversationalTestCase,
  MultiTurnParams,
  Turn,
  ToolCall,
} from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  printToolsCalled,
} from "../utils";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
} from "../conversational-utils";
import {
  ToolSelectionScoreSchema,
  ArgumentCorrectnessScoreSchema,
  ReasonSchema,
  type ToolSelectionScore,
  type ArgumentCorrectnessScore,
  type UserInputAndTools,
} from "./schema";

const TEMPLATE_CLASS = "ToolUseMetric";

export interface ToolUseMetricOptions {
  /** The tools the agent had access to. Required. */
  availableTools: ToolCall[];
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Tool Use — across the conversation, did the agent select the right tools and
 * pass correct arguments? Per interaction, score tool selection and argument
 * correctness; final score = min(mean tool-selection, mean argument-correctness).
 * **Higher is better**. Requires `availableTools`.
 */
export class ToolUseMetric extends BaseConversationalMetric {
  private readonly availableTools: ToolCall[];

  constructor(options: ToolUseMetricOptions) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT];
    this.availableTools = options.availableTools;
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
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      const userInputAndTools = this.getUserInputAndTurns(
        getUnitInteractions(testCase.turns),
      );
      const toolSelectionScores = await Promise.all(
        userInputAndTools.map((u) => this.getToolSelectionScore(u)),
      );
      const argumentCorrectnessScores = await Promise.all(
        userInputAndTools
          .filter((u) => u.tools_used)
          .map((u) => this.getArgumentCorrectnessScore(u)),
      );

      this.score = this.calculateScore(
        toolSelectionScores,
        argumentCorrectnessScores,
      );
      this.success = this.score >= this.threshold;
      this.reason = [
        await this.generateFinalReason(toolSelectionScores),
        await this.generateFinalReason(argumentCorrectnessScores),
      ].join("\n");

      this.verboseLogs = constructVerboseLogs(this, [
        `Tool Selection Scores: ${toolSelectionScores.map((s) => s.score).join(", ")}`,
        `Argument Correctness Scores: ${argumentCorrectnessScores.map((s) => s.score).join(", ")}`,
        `Final Score: ${this.score}\nFinal Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private getUserInputAndTurns(
    unitInteractions: Turn[][],
  ): UserInputAndTools[] {
    const availableTools = printToolsCalled(this.availableTools);
    const result: UserInputAndTools[] = [];
    for (const interaction of unitInteractions) {
      if (interaction.length < 2) continue;
      let userMessages = "";
      for (const turn of interaction) {
        if (turn.role === "user") userMessages += `${turn.content} \n`;
        else break;
      }
      let assistantMessages = "";
      const toolsCalled: ToolCall[] = [];
      let toolsUsed = false;
      for (const turn of interaction.slice(1)) {
        if (turn.role === "assistant") {
          assistantMessages += `${turn.content} \n`;
          if (turn.toolsCalled && turn.toolsCalled.length > 0) {
            toolsCalled.push(...turn.toolsCalled);
            toolsUsed = true;
          }
        }
      }
      result.push({
        user_messages: userMessages,
        assistant_messages: assistantMessages,
        tools_called: printToolsCalled(toolsCalled),
        available_tools: availableTools,
        tools_used: toolsUsed,
      });
    }
    return result;
  }

  private async getToolSelectionScore(
    u: UserInputAndTools,
  ): Promise<ToolSelectionScore> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "get_tool_selection_score", {
      user_input: u.user_messages,
      assistant_messages: u.assistant_messages,
      tools_called: u.tools_called,
      available_tools: u.available_tools,
    });
    return generateWithSchema(this, prompt, ToolSelectionScoreSchema);
  }

  private async getArgumentCorrectnessScore(
    u: UserInputAndTools,
  ): Promise<ArgumentCorrectnessScore> {
    const prompt = resolveTemplate("metrics", 
      TEMPLATE_CLASS,
      "get_argument_correctness_score",
      {
        user_input: u.user_messages,
        assistant_messages: u.assistant_messages,
        tools_called: u.tools_called,
        available_tools: u.available_tools,
      },
    );
    return generateWithSchema(this, prompt, ArgumentCorrectnessScoreSchema);
  }

  private calculateScore(
    toolScores: ToolSelectionScore[],
    argScores: ArgumentCorrectnessScore[],
  ): number {
    const mean = (xs: { score: number }[]) =>
      xs.reduce((s, x) => s + x.score, 0) / Math.max(xs.length, 1);
    const score = Math.min(mean(toolScores), mean(argScores));
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  // Both reason types use the same template (mirrors Python; `get_tool_argument_final_reason` is unused).
  private async generateFinalReason(
    scores: { score: number; reason: string }[],
  ): Promise<string> {
    const allScoresAndReasons = scores
      .map((s) => `\nScore: ${s.score} \nReason: ${s.reason} \n`)
      .join("");
    const prompt = resolveTemplate("metrics", 
      TEMPLATE_CLASS,
      "get_tool_selection_final_reason",
      {
        all_scores_and_reasons: allScoresAndReasons,
        final_score: this.score,
        threshold: this.threshold,
      },
    );
    const { reason } = await generateWithSchema(this, prompt, ReasonSchema);
    return reason;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Tool Use";
  }
}
