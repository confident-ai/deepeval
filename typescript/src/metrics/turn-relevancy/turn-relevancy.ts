import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams, Turn } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
} from "@/metrics/utils";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
  getTurnsInSlidingWindow,
  convertTurnToDict,
} from "@/metrics/conversational-utils";
import {
  TurnRelevancyVerdictSchema,
  TurnRelevancyScoreReasonSchema,
  type TurnRelevancyVerdict,
} from "@/metrics/turn-relevancy/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "TurnRelevancyMetric";

export type TurnRelevancyTemplateOverride =
  MetricTemplateOverride<"TurnRelevancyMetric">;

export interface TurnRelevancyMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  /** Sliding-window size over unit-interactions (default 10). */
  windowSize?: number;
  evaluationTemplate?: TurnRelevancyTemplateOverride;
}

/**
 * Turn Relevancy — over a sliding window of the conversation, is each assistant
 * turn relevant to the dialogue so far? One verdict per window; score =
 * relevant / total. **Higher is better** (`success = score >= threshold`).
 */
export class TurnRelevancyMetric extends BaseConversationalMetric {
  verdicts: TurnRelevancyVerdict[] = [];
  private readonly windowSize: number;

  constructor(options: TurnRelevancyMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [MultiTurnParams.CONTENT, MultiTurnParams.ROLE];
    this.windowSize = options.windowSize ?? 10;
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

      const unitInteractions = getUnitInteractions(testCase.turns);
      const turnsWindows: Turn[][] = getTurnsInSlidingWindow(
        unitInteractions,
        this.windowSize,
      ).map((window) => window.flat());

      this.verdicts = await Promise.all(
        turnsWindows.map((window) => this.generateVerdict(window)),
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Turns Sliding Windows (size=${this.windowSize}):\n${prettifyList(turnsWindows)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdict(window: Turn[]): Promise<TurnRelevancyVerdict> {
    const prompt = this.getPrompt("generate_verdicts", {
      sliding_window: window.map((turn) => convertTurnToDict(turn)),
    });
    return generateWithSchema(this, prompt, TurnRelevancyVerdictSchema);
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const irrelevancies = this.verdicts
      .map((verdict, index) => ({ verdict, index }))
      .filter(
        ({ verdict }) =>
          verdict?.verdict != null &&
          verdict.verdict.trim().toLowerCase() === "no",
      )
      .map(({ verdict, index }) => ({
        "message number": `${index + 1}`,
        reason: verdict.reason,
      }));
    const prompt = this.getPrompt("generate_reason", {
      score: this.score,
      irrelevancies,
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      TurnRelevancyScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const valid = this.verdicts.filter((v) => v != null && v.verdict != null);
    const total = valid.length;
    if (total === 0) return 1;
    const relevant = valid.filter(
      (v) => v.verdict.trim().toLowerCase() !== "no",
    ).length;
    const score = relevant / total;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Turn Relevancy";
  }
}
