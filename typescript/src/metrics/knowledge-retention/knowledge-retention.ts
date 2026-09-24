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
  /** The System One model (Jev) used under `hybrid` / `system_one`. */
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  /** Who decides; defaults to `DEEPEVAL_EVAL_MODE`, then `llm`. */
  evalMode?: EvalModeName;
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
    initializeMetricModels(this, options);
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      this.knowledges = await this.generateKnowledges(testCase.turns);
      this.verdicts = await this.generateVerdicts(
        testCase.turns,
        testCase.multimodal,
      );
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
    multimodal: boolean,
  ): Promise<KnowledgeRetentionVerdict[]> {
    const results = await Promise.all(
      turns.map(async (turn, i) => {
        if (turn.role !== "assistant") return null;
        const accumulatedKnowledge = this.knowledges
          .slice(0, i)
          .filter((k): k is Knowledge => k != null && k.data != null)
          .map((k) => k.data);
        if (accumulatedKnowledge.length === 0) return null;
        const p = await systemOneProbability(
          this,
          this.systemOneVerdictSpec(
            turn.content,
            accumulatedKnowledge,
            multimodal,
          ),
        );
        if (p !== undefined) {
          return {
            verdict: verdictFromProbability(p),
            reason: `P(yes)=${p.toFixed(2)}`,
          };
        }
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

  private systemOneVerdictSpec(
    llmMessage: string,
    accumulatedKnowledge: unknown[],
    multimodal: boolean,
  ): SystemOneBinarySpec | undefined {
    if (multimodal) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_verdict"),
      state: {
        llm_message: llmMessage,
        accumulated_knowledge: accumulatedKnowledge,
      },
    };
  }

  systemOneEvalSpec(
    testCase: ConversationalTestCase,
  ): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: this.requiredParams,
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
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
