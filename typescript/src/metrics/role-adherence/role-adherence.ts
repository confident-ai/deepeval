import { BaseConversationalMetric } from "../base-conversational-metric";
import {
  ConversationalTestCase,
  MultiTurnParams,
  Turn,
} from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
} from "../utils";
import {
  checkConversationalTestCaseParams,
  convertTurnToDict,
} from "../conversational-utils";
import {
  OutOfCharacterResponseVerdictsSchema,
  RoleAdherenceScoreReasonSchema,
  type OutOfCharacterResponseVerdict,
} from "./schema";

const TEMPLATE_CLASS = "RoleAdherenceMetric";

export interface RoleAdherenceMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
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
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
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
      this.success = this.score >= this.threshold;

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
    const prompt = resolveTemplate("metrics", 
      TEMPLATE_CLASS,
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
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
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
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Role Adherence";
  }
}
