import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import {
  ConversationalTestCase,
  MultiTurnParams,
  ToolCall,
  ToolCallParams,
  Turn,
} from "@/test-case";
import { MissingTestCaseParamsError } from "@/errors";
import { constructVerboseLogs } from "@/metrics/utils";
import { prepareMeasure } from "@/metrics/prepare-measure";
import { ToolCorrectnessMetric } from "@/metrics/tool-correctness/tool-correctness";

export interface ConversationToolCorrectnessMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  evaluationParams?: ToolCallParams[];
  shouldExactMatch?: boolean;
  shouldConsiderOrdering?: boolean;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

export interface ToolCheckScore {
  check: string;
  score: number;
  reason: string;
}

export class ConversationToolCorrectnessMetric extends BaseConversationalMetric {
  toolsCalled: ToolCall[] = [];
  checkScores: ToolCheckScore[] = [];
  private readonly evaluationParams: ToolCallParams[];
  private readonly shouldExactMatch: boolean;
  private readonly shouldConsiderOrdering: boolean;

  constructor(options: ConversationToolCorrectnessMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
    });
    this.requiredParams = [
      MultiTurnParams.ROLE,
      MultiTurnParams.CONTENT,
      MultiTurnParams.EXPECTED_TOOLS,
    ];
    this.evaluationParams = options.evaluationParams ?? [];
    this.shouldExactMatch = options.shouldExactMatch ?? false;
    this.shouldConsiderOrdering = options.shouldConsiderOrdering ?? false;
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    prepareMeasure(this, testCase);
    const expectedTools = testCase.expectedTools ?? [];
    const pinned = this.pinnedTools(testCase.turns, expectedTools);
    const unpinned = expectedTools.filter((t) => t.turn == null);

    await this.startProgress();
    try {
      const checks: [string, ToolCall[], ToolCall[]][] = [...pinned.entries()]
        .sort(([a], [b]) => a - b)
        .map(([index, tools]) => [
          `Turn ${index}`,
          testCase.turns[index].toolsCalled ?? [],
          tools,
        ]);
      if (unpinned.length > 0 || pinned.size === 0) {
        checks.push([
          pinned.size > 0 ? "Rest of the conversation" : "Conversation",
          assistantToolCalls(testCase.turns, new Set(pinned.keys())),
          unpinned,
        ]);
      }

      this.checkScores = checks.map(([check, called, expected]) => ({
        check,
        ...ToolCorrectnessMetric.scoreToolCalls(called, expected, {
          evaluationParams: this.evaluationParams,
          shouldExactMatch: this.shouldExactMatch,
          shouldConsiderOrdering: this.shouldConsiderOrdering,
        }),
      }));
      this.toolsCalled = assistantToolCalls(testCase.turns);
      const mean =
        this.checkScores.reduce((sum, c) => sum + c.score, 0) /
        this.checkScores.length;
      this.score = this.applyStrictMode(mean);
      this.reason = this.includeReason ? this.buildReason() : undefined;
      this.success = this.isSuccessful();
      this.verboseLogs = constructVerboseLogs(this, [
        ...this.checkScores.map(
          (c) => `${c.check}: score ${c.score.toFixed(2)}. ${c.reason}`,
        ),
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private pinnedTools(
    turns: Turn[],
    expectedTools: ToolCall[],
  ): Map<number, ToolCall[]> {
    const pinned = new Map<number, ToolCall[]>();
    for (const tool of expectedTools) {
      if (tool.turn == null) continue;
      if (tool.turn < 0 || tool.turn >= turns.length) {
        this.error =
          `Expected tool '${tool.name}' is pinned to turn ${tool.turn}, but ` +
          `the conversation only has ${turns.length} turns.`;
        throw new MissingTestCaseParamsError(this.error);
      }
      if (turns[tool.turn].role !== "assistant") {
        this.error =
          `Expected tool '${tool.name}' is pinned to turn ${tool.turn}, ` +
          `which is not an assistant turn.`;
        throw new MissingTestCaseParamsError(this.error);
      }
      pinned.set(tool.turn, [...(pinned.get(tool.turn) ?? []), tool]);
    }
    return pinned;
  }

  private buildReason(): string {
    if (this.checkScores.length === 1) return this.checkScores[0].reason;
    const failing = this.checkScores.filter((c) => c.score < 1);
    const passed = this.checkScores.length - failing.length;
    return [
      `The score is ${(this.score as number).toFixed(2)}: ${passed} of ` +
        `${this.checkScores.length} checks passed.`,
      ...failing.map(
        (c) => `- ${c.check} (${c.score.toFixed(2)}): ${c.reason}`,
      ),
    ].join("\n");
  }

  get name(): string {
    return "Conversation Tool Correctness";
  }
}

function assistantToolCalls(turns: Turn[], skip?: Set<number>): ToolCall[] {
  return turns.flatMap((turn, index) =>
    turn.role === "assistant" && !skip?.has(index)
      ? (turn.toolsCalled ?? [])
      : [],
  );
}
