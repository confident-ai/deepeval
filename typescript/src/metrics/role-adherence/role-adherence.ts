import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams, Turn } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
} from "@/metrics/utils";
import {
  parseQuestions,
  runSystemOneEval,
  systemOneProbability,
  verdictFromProbability,
  type SystemOneBinarySpec,
  type SystemOneEvalSpec,
} from "@/metrics/system-one";
import { convertTurnToDict } from "@/metrics/conversational-utils";
import {
  OutOfCharacterResponseVerdictsSchema,
  RoleAdherenceScoreReasonSchema,
  type OutOfCharacterResponseVerdict,
} from "@/metrics/role-adherence/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { prepareMeasure } from "@/metrics/prepare-measure";

const TEMPLATE_CLASS = "RoleAdherenceMetric";

export type RoleAdherenceTemplateOverride =
  MetricTemplateOverride<"RoleAdherenceMetric">;

export interface RoleAdherenceMetricOptions {
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
    this.requiresChatbotRole = true;
    initializeMetricModels(this, options);
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      const role = testCase.chatbotRole ?? "";
      this.outOfCharacterVerdicts = await this.extractOutOfCharacterVerdicts(
        testCase.turns,
        role,
        testCase.multimodal,
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
    multimodal: boolean,
  ): Promise<OutOfCharacterResponseVerdict[]> {
    const specs = this.systemOneVerdictSpecs(turns, role, multimodal);
    if (specs !== undefined && specs.length > 0) {
      const probabilities = await Promise.all(
        specs.map(([, spec]) => systemOneProbability(this, spec)),
      );
      if (probabilities.every((p) => p !== undefined)) {
        return this.systemOneVerdicts(
          turns,
          specs.map(([index], i) => [index, probabilities[i] as number]),
        );
      }
    }

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

  private systemOneVerdictSpecs(
    turns: Turn[],
    role: string,
    multimodal: boolean,
  ): [number, SystemOneBinarySpec][] | undefined {
    if (multimodal) return undefined;
    const instructions = this.getPrompt("_experimental_system_one_verdict");
    const specs: [number, SystemOneBinarySpec][] = [];
    turns.forEach((turn, index) => {
      if (turn.role !== "assistant") return;
      specs.push([
        index,
        {
          instructions,
          state: {
            chatbot_role: role,
            previous_turns: turns
              .slice(0, index)
              .map((t) => convertTurnToDict(t)),
            ai_message: turn.content,
          },
        },
      ]);
    });
    return specs;
  }

  private systemOneVerdicts(
    turns: Turn[],
    probabilities: [number, number][],
  ): OutOfCharacterResponseVerdict[] {
    return probabilities
      .filter(([, p]) => verdictFromProbability(p) === "no")
      .map(([index, p]) => ({
        index,
        reason: `P(in character)=${p.toFixed(2)}`,
        ai_message: `${turns[index].content} (turn #${index + 1})`,
      }));
  }

  systemOneEvalSpec(
    testCase: ConversationalTestCase,
  ): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: [...this.requiredParams, MultiTurnParams.CHATBOT_ROLE],
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
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
