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

export interface ConversationalToolCorrectnessMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  /** Which `ToolCall` fields to compare (input parameters / output). */
  evaluationParams?: ToolCallParams[];
  shouldExactMatch?: boolean;
  shouldConsiderOrdering?: boolean;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

export interface ToolGroupScore {
  /** "Turn N" for tools pinned to a turn, otherwise the rest of the conversation. */
  group: string;
  score: number;
  reason: string;
}

/**
 * Conversational Tool Correctness — Tool Correctness for a conversation.
 * An expected tool with `turn` set is pinned to that turn: the turn's own
 * `toolsCalled` are scored against the tools pinned to it. Expected tools
 * without `turn` are scored against the tools called on every other
 * assistant turn, in order. The score is the mean over these groups.
 * Deterministic, no model. Mirrors Python's `ConversationalToolCorrectnessMetric`.
 */
export class ConversationalToolCorrectnessMetric extends BaseConversationalMetric {
  toolsCalled: ToolCall[] = [];
  groupScores: ToolGroupScore[] = [];
  private readonly evaluationParams: ToolCallParams[];
  private readonly shouldExactMatch: boolean;
  private readonly shouldConsiderOrdering: boolean;

  constructor(options: ConversationalToolCorrectnessMetricOptions = {}) {
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
      const groups: [string, ToolCall[], ToolCall[]][] = [...pinned.entries()]
        .sort(([a], [b]) => a - b)
        .map(([index, tools]) => [
          `Turn ${index}`,
          testCase.turns[index].toolsCalled ?? [],
          tools,
        ]);
      if (unpinned.length > 0 || pinned.size === 0) {
        groups.push([
          pinned.size > 0 ? "Rest of the conversation" : "Conversation",
          assistantToolCalls(testCase.turns, new Set(pinned.keys())),
          unpinned,
        ]);
      }

      this.groupScores = groups.map(([group, called, expected]) => ({
        group,
        ...ToolCorrectnessMetric.scoreToolCalls(called, expected, {
          evaluationParams: this.evaluationParams,
          shouldExactMatch: this.shouldExactMatch,
          shouldConsiderOrdering: this.shouldConsiderOrdering,
        }),
      }));
      this.toolsCalled = assistantToolCalls(testCase.turns);
      const mean =
        this.groupScores.reduce((sum, g) => sum + g.score, 0) /
        this.groupScores.length;
      this.score = this.applyStrictMode(mean);
      this.reason = this.includeReason ? this.buildReason() : undefined;
      this.success = this.isSuccessful();
      this.verboseLogs = constructVerboseLogs(this, [
        ...this.groupScores.map(
          (g) => `${g.group}: score ${g.score.toFixed(2)}. ${g.reason}`,
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
          `the conversation has ${turns.length} turns (indexes 0 to ${turns.length - 1}).`;
        throw new MissingTestCaseParamsError(this.error);
      }
      if (turns[tool.turn].role !== "assistant") {
        this.error =
          `Expected tool '${tool.name}' is pinned to turn ${tool.turn}, which ` +
          `is a '${turns[tool.turn].role}' turn. 'turn' is the index in ` +
          `'turns' of an assistant turn.`;
        throw new MissingTestCaseParamsError(this.error);
      }
      pinned.set(tool.turn, [...(pinned.get(tool.turn) ?? []), tool]);
    }
    return pinned;
  }

  private buildReason(): string {
    if (this.groupScores.length === 1) return this.groupScores[0].reason;
    const failing = this.groupScores.filter((g) => g.score < 1);
    const summary =
      `The score is ${(this.score as number).toFixed(2)}: ` +
      `${this.groupScores.length - failing.length} of ` +
      `${this.groupScores.length} checks passed.`;
    if (failing.length === 0) return summary;
    return [
      summary,
      ...failing.map(
        (g) => `- ${g.group} (${g.score.toFixed(2)}): ${g.reason}`,
      ),
    ].join("\n");
  }

  get name(): string {
    return "Conversational Tool Correctness";
  }
}

function assistantToolCalls(turns: Turn[], skip?: Set<number>): ToolCall[] {
  return turns.flatMap((turn, index) =>
    turn.role === "assistant" && !skip?.has(index)
      ? (turn.toolsCalled ?? [])
      : [],
  );
}
