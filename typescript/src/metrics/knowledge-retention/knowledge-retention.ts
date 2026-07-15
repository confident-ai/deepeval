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
  KnowledgeSchema,
  KnowledgeRetentionVerdictSchema,
  KnowledgeRetentionScoreReasonSchema,
  type Knowledge,
  type KnowledgeRetentionVerdict,
} from "./schema";

const TEMPLATE_CLASS = "KnowledgeRetentionMetric";

export interface KnowledgeRetentionMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Knowledge Retention — does the assistant remember facts the user established
 * earlier (no attrition)? Extract knowledge per user turn, then judge each
 * assistant turn against the accumulated knowledge. Score = retained / total.
 * **Higher is better** (`success = score >= threshold`).
 */
export class KnowledgeRetentionMetric extends BaseConversationalMetric {
  knowledges: (Knowledge | null)[] = [];
  verdicts: KnowledgeRetentionVerdict[] = [];

  constructor(options: KnowledgeRetentionMetricOptions = {}) {
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
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      this.knowledges = await this.generateKnowledges(testCase.turns);
      this.verdicts = await this.generateVerdicts(testCase.turns);
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Knowledges:\n${prettifyList(this.knowledges)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  /** Extract knowledge from each user turn (assistant turns get `null`). */
  private async generateKnowledges(turns: Turn[]): Promise<(Knowledge | null)[]> {
    const knowledges: (Knowledge | null)[] = new Array(turns.length).fill(null);
    const extracted = await Promise.all(
      turns.map(async (turn, i) => {
        if (turn.role === "assistant") return null;
        const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "extract_data", {
          user_message: turn.content,
          previous_turns: turns.slice(0, i).map((t) => convertTurnToDict(t)),
        });
        return generateWithSchema(this, prompt, KnowledgeSchema);
      }),
    );
    extracted.forEach((k, i) => {
      if (k != null) knowledges[i] = k;
    });
    return knowledges;
  }

  /** One verdict per assistant turn that has prior accumulated knowledge. */
  private async generateVerdicts(
    turns: Turn[],
  ): Promise<KnowledgeRetentionVerdict[]> {
    const results = await Promise.all(
      turns.map(async (turn, i) => {
        if (turn.role !== "assistant") return null;
        const accumulatedKnowledge = this.knowledges
          .slice(0, i)
          .filter((k): k is Knowledge => k != null && k.data != null)
          .map((k) => k.data);
        if (accumulatedKnowledge.length === 0) return null;
        const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdict", {
          llm_message: turn.content,
          accumulated_knowledge: accumulatedKnowledge,
        });
        return generateWithSchema(this, prompt, KnowledgeRetentionVerdictSchema);
      }),
    );
    return results.filter((v): v is KnowledgeRetentionVerdict => v != null);
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const attritions = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "yes")
      .map((v) => v.reason);
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      attritions,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      KnowledgeRetentionScoreReasonSchema,
    );
    return reason;
  }

  /** Score = fraction of assistant turns with NO attrition ("no" verdicts). */
  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 1;
    const retained = this.verdicts.filter(
      (v) => v.verdict.trim().toLowerCase() === "no",
    ).length;
    const score = retained / total;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Knowledge Retention";
  }
}
