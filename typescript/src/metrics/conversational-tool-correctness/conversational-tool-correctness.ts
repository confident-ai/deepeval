import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import {
  ConversationalTestCase,
  MultiTurnParams,
  ToolCall,
  ToolCallParams,
  Turn,
} from "@/test-case";
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

/**
 * Conversational Tool Correctness — Tool Correctness over a whole
 * conversation. Every tool the assistant called, across all turns and in
 * order, is compared with the test case's `expectedTools`. Deterministic, no
 * model. Mirrors Python's `ConversationalToolCorrectnessMetric`.
 */
export class ConversationalToolCorrectnessMetric extends BaseConversationalMetric {
  toolsCalled: ToolCall[] = [];
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

    await this.startProgress();
    try {
      this.toolsCalled = assistantToolCalls(testCase.turns);
      const expectedTools = testCase.expectedTools ?? [];
      const { score, reason } = ToolCorrectnessMetric.scoreToolCalls(
        this.toolsCalled,
        expectedTools,
        {
          evaluationParams: this.evaluationParams,
          shouldExactMatch: this.shouldExactMatch,
          shouldConsiderOrdering: this.shouldConsiderOrdering,
        },
      );
      this.score = this.applyStrictMode(score);
      this.reason = this.includeReason ? reason : undefined;
      this.success = this.isSuccessful();
      this.verboseLogs = constructVerboseLogs(this, [
        `Tools called across the conversation: ${JSON.stringify(this.toolsCalled.map((t) => t.name))}`,
        `Expected tools: ${JSON.stringify(expectedTools.map((t) => t.name))}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  get name(): string {
    return "Conversational Tool Correctness";
  }
}

function assistantToolCalls(turns: Turn[]): ToolCall[] {
  return turns
    .filter((turn) => turn.role === "assistant")
    .flatMap((turn) => turn.toolsCalled ?? []);
}
