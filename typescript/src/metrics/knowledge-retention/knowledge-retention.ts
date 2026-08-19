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
  KnowledgeSchema,
  KnowledgeRetentionVerdictSchema,
  KnowledgeRetentionScoreReasonSchema,
  type Knowledge,
  type KnowledgeRetentionVerdict,
} from "@/metrics/knowledge-retention/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "KnowledgeRetentionMetric";

export type KnowledgeRetentionTemplateOverride =
  MetricTemplateOverride<"KnowledgeRetentionMetric">;

export interface KnowledgeRetentionMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: KnowledgeRetentionTemplateOverride;
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
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      this.knowledges = await this.generateKnowledges(testCase.turns);
      this.verdicts = await this.generateVerdicts(testCase.turns);
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.isSuccessful();

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
  private async generateKnowledges(
    turns: Turn[],
  ): Promise<(Knowledge | null)[]> {
    const knowledges: (Knowledge | null)[] = new Array(turns.length).fill(null);
    const extracted = await Promise.all(
      turns.map(async (turn, i) => {
        if (turn.role === "assistant") return null;
        const prompt = this.getPrompt("extract_data", {
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
        const prompt = this.getPrompt("generate_verdict", {
          llm_message: turn.content,
          accumulated_knowledge: accumulatedKnowledge,
        });
        return generateWithSchema(
          this,
          prompt,
          KnowledgeRetentionVerdictSchema,
        );
      }),
    );
    return results.filter((v): v is KnowledgeRetentionVerdict => v != null);
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const attritions = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "yes")
      .map((v) => v.reason);
    const prompt = this.getPrompt("generate_reason", {
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
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Knowledge Retention";
  }
}
