import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import {
  ConversationalTestCase,
  MultiTurnParams,
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

export interface TurnToolScore {
  turn: number;
  score: number;
  reason: string;
}

/**
 * Conversational Tool Correctness — Tool Correctness per assistant turn.
 * Every assistant turn that sets `expectedTools` is scored against its own
 * `toolsCalled`; the score is the mean over those turns. Deterministic, no
 * model. Mirrors Python's `ConversationalToolCorrectnessMetric`.
 */
export class ConversationalToolCorrectnessMetric extends BaseConversationalMetric {
  turnScores: TurnToolScore[] = [];
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
    const scored = scoredTurns(testCase.turns);
    if (scored.length === 0) {
      this.error =
        `No assistant turn in the conversational test case sets 'expectedTools', ` +
        `which the '${this.name}' metric needs. Set expectedTools on the ` +
        `assistant turns to check ([] means the turn should call no tools).`;
      throw new MissingTestCaseParamsError(this.error);
    }

    await this.startProgress();
    try {
      this.turnScores = scored.map(({ index, turn }) => ({
        turn: index,
        ...ToolCorrectnessMetric.scoreToolCalls(
          turn.toolsCalled ?? [],
          turn.expectedTools ?? [],
          {
            evaluationParams: this.evaluationParams,
            shouldExactMatch: this.shouldExactMatch,
            shouldConsiderOrdering: this.shouldConsiderOrdering,
          },
        ),
      }));
      const mean =
        this.turnScores.reduce((sum, t) => sum + t.score, 0) /
        this.turnScores.length;
      this.score = this.applyStrictMode(mean);
      this.reason = this.includeReason ? this.buildReason() : undefined;
      this.success = this.isSuccessful();
      this.verboseLogs = constructVerboseLogs(this, [
        ...this.turnScores.map(
          (t) => `Turn ${t.turn}: score ${t.score.toFixed(2)}. ${t.reason}`,
        ),
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private buildReason(): string {
    const total = this.turnScores.length;
    const failing = this.turnScores.filter((t) => t.score < 1);
    const summary =
      `The score is ${(this.score as number).toFixed(2)}: ` +
      `${total - failing.length} of ${total} scored assistant turns called the expected tools.`;
    if (failing.length === 0) return summary;
    return [
      summary,
      ...failing.map(
        (t) => `- Turn ${t.turn} (${t.score.toFixed(2)}): ${t.reason}`,
      ),
    ].join("\n");
  }

  get name(): string {
    return "Conversational Tool Correctness";
  }
}

function scoredTurns(turns: Turn[]): { index: number; turn: Turn }[] {
  return turns
    .map((turn, index) => ({ index, turn }))
    .filter(
      ({ turn }) => turn.role === "assistant" && turn.expectedTools != null,
    );
}
