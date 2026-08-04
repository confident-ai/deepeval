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
  convertTurnToDict,
} from "@/metrics/conversational-utils";
import {
  UserIntentionsSchema,
  ConversationCompletenessVerdictSchema,
  ConversationCompletenessScoreReasonSchema,
  type ConversationCompletenessVerdict,
} from "@/metrics/conversation-completeness/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "ConversationCompletenessMetric";

export type ConversationCompletenessTemplateOverride =
  MetricTemplateOverride<"ConversationCompletenessMetric">;

export interface ConversationCompletenessMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: ConversationCompletenessTemplateOverride;
}

/**
 * Conversation Completeness — extract the user's intentions across the whole
 * conversation, then judge whether each was satisfied. Score = satisfied /
 * total intentions. **Higher is better** (`success = score >= threshold`).
 */
export class ConversationCompletenessMetric extends BaseConversationalMetric {
  userIntentions: string[] = [];
  verdicts: ConversationCompletenessVerdict[] = [];

  constructor(options: ConversationCompletenessMetricOptions = {}) {
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
    this.requiredParams = [MultiTurnParams.CONTENT, MultiTurnParams.ROLE];
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

      this.userIntentions = await this.extractUserIntentions(testCase.turns);
      this.verdicts = await Promise.all(
        this.userIntentions.map((intention) =>
          this.generateVerdict(testCase.turns, intention),
        ),
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `User Intentions:\n${prettifyList(this.userIntentions)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async extractUserIntentions(turns: Turn[]): Promise<string[]> {
    const prompt = this.getPrompt("extract_user_intentions", {
      turns: turns.map((turn) => convertTurnToDict(turn)),
    });
    const { intentions } = await generateWithSchema(
      this,
      prompt,
      UserIntentionsSchema,
    );
    return intentions;
  }

  private async generateVerdict(
    turns: Turn[],
    intention: string,
  ): Promise<ConversationCompletenessVerdict> {
    const prompt = this.getPrompt("generate_verdicts", {
      turns: turns.map((turn) => convertTurnToDict(turn)),
      intention,
    });
    return generateWithSchema(
      this,
      prompt,
      ConversationCompletenessVerdictSchema,
    );
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const incompletenesses = this.verdicts
      .filter(
        (v) => v?.verdict != null && v.verdict.trim().toLowerCase() === "no",
      )
      .map((v) => v.reason);
    const prompt = this.getPrompt("generate_reason", {
      score: this.score,
      incompletenesses,
      intentions: this.userIntentions,
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ConversationCompletenessScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const valid = this.verdicts.filter((v) => v != null && v.verdict != null);
    const total = valid.length;
    if (total === 0) return 1;
    const satisfied = valid.filter(
      (v) => v.verdict.trim().toLowerCase() !== "no",
    ).length;
    const score = satisfied / total;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Conversation Completeness";
  }
}
