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
  OutOfCharacterResponseVerdictsSchema,
  RoleAdherenceScoreReasonSchema,
  type OutOfCharacterResponseVerdict,
} from "@/metrics/role-adherence/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "RoleAdherenceMetric";

export type RoleAdherenceTemplateOverride =
  MetricTemplateOverride<"RoleAdherenceMetric">;

export interface RoleAdherenceMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: RoleAdherenceTemplateOverride;
}

/**
 * Role Adherence — does the assistant stay in its `chatbotRole`? Identify the
 * out-of-character assistant turns; score = in-character / total assistant
 * turns. **Higher is better**. Requires `chatbotRole` on the test case.
 */
export class RoleAdherenceMetric extends BaseConversationalMetric {
  outOfCharacterVerdicts: OutOfCharacterResponseVerdict[] = [];

  constructor(options: RoleAdherenceMetricOptions = {}) {
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
    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkConversationalTestCaseParams(testCase, this.requiredParams, this, {
        requireChatbotRole: true,
      });
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      const role = testCase.chatbotRole ?? "";
      this.outOfCharacterVerdicts = await this.extractOutOfCharacterVerdicts(
        testCase.turns,
        role,
      );
      this.score = this.calculateScore(testCase.turns);
      this.reason = await this.generateReason(role);
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Chatbot Role:\n${role}`,
        `Out-of-Character Turn(s):\n${prettifyList(this.outOfCharacterVerdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async extractOutOfCharacterVerdicts(
    turns: Turn[],
    role: string,
  ): Promise<OutOfCharacterResponseVerdict[]> {
    const prompt = this.getPrompt(
      "extract_out_of_character_response_verdicts",
      { turns: turns.map((turn) => convertTurnToDict(turn)), role },
    );
    const { verdicts } = await generateWithSchema(
      this,
      prompt,
      OutOfCharacterResponseVerdictsSchema,
    );
    for (const v of verdicts) {
      if (v.index >= 0 && v.index < turns.length) {
        v.ai_message = `${turns[v.index].content} (turn #${v.index + 1})`;
      }
    }
    return verdicts;
  }

  private async generateReason(role: string): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const prompt = this.getPrompt("generate_reason", {
      score: this.score,
      role,
      out_of_character_responses: this.outOfCharacterVerdicts.map(
        (v) => v.ai_message,
      ),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      RoleAdherenceScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(turns: Turn[]): number {
    const assistantTurns = turns.filter((t) => t.role === "assistant").length;
    if (assistantTurns === 0) return 1;
    const outOfChar = Math.min(
      this.outOfCharacterVerdicts.length,
      assistantTurns,
    );
    const score = (assistantTurns - outOfChar) / assistantTurns;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Role Adherence";
  }
}
