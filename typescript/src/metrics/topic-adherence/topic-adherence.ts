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
  getUnitInteractions,
} from "../conversational-utils";
import {
  QAPairsSchema,
  RelevancyVerdictSchema,
  TopicAdherenceReasonSchema,
  type QAPair,
  type RelevancyVerdict,
} from "./schema";

const TEMPLATE_CLASS = "TopicAdherenceMetric";
type Verdict = RelevancyVerdict["verdict"];

export interface TopicAdherenceMetricOptions {
  /** Topics the assistant is allowed to engage with. Required. */
  relevantTopics: string[];
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Topic Adherence — does the assistant stay on the `relevantTopics`? Extract
 * Q&A pairs per interaction, classify each into a TP/TN/FP/FN confusion-matrix
 * cell, then score = (TP + TN) / total (accuracy). **Higher is better**.
 */
export class TopicAdherenceMetric extends BaseConversationalMetric {
  private readonly relevantTopics: string[];

  constructor(options: TopicAdherenceMetricOptions) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT];
    this.relevantTopics = options.relevantTopics;
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
      const qaPairs = (
        await Promise.all(unitInteractions.map((u) => this.getQaPairs(u)))
      ).flat();

      const verdicts = await Promise.all(
        qaPairs.map((qa) => this.getQaVerdict(qa)),
      );
      const tally: Record<Verdict, string[]> = { TP: [], TN: [], FP: [], FN: [] };
      for (const v of verdicts) tally[v.verdict].push(v.reason);

      this.score = this.calculateScore(tally);
      this.success = this.score >= this.threshold;
      this.reason = await this.generateReason(tally);

      this.verboseLogs = constructVerboseLogs(this, [
        `Q&A pairs: ${qaPairs.length}`,
        `TP: ${tally.TP.length}, TN: ${tally.TN.length}, FP: ${tally.FP.length}, FN: ${tally.FN.length}`,
        `Final Score: ${this.score}\nFinal Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async getQaPairs(unitInteraction: Turn[]): Promise<QAPair[]> {
    let conversation = "Conversation: \n";
    for (const turn of unitInteraction) {
      conversation += `${turn.role} \n`;
      conversation += `${turn.content} \n\n`;
    }
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "get_qa_pairs", {
      conversation,
    });
    const { qa_pairs } = await generateWithSchema(this, prompt, QAPairsSchema);
    return qa_pairs;
  }

  private async getQaVerdict(qaPair: QAPair): Promise<RelevancyVerdict> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "get_qa_pair_verdict", {
      relevant_topics: this.relevantTopics,
      question: qaPair.question,
      response: qaPair.response,
    });
    return generateWithSchema(this, prompt, RelevancyVerdictSchema);
  }

  private calculateScore(tally: Record<Verdict, string[]>): number {
    const trueValues = tally.TP.length + tally.TN.length;
    const total =
      tally.TP.length + tally.TN.length + tally.FP.length + tally.FN.length;
    const score = total <= 0 ? 0 : trueValues / total;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  private async generateReason(
    tally: Record<Verdict, string[]>,
  ): Promise<string | undefined> {
    const total =
      tally.TP.length + tally.TN.length + tally.FP.length + tally.FN.length;
    if (total <= 0) {
      return "There were no question-answer pairs to evaluate. Please enable verbose logs to look at the evaluation steps taken";
    }
    const line = (reasons: string[]) =>
      reasons.length ? prettifyList(reasons) : "(none)";
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      success: this.success,
      score: this.score,
      threshold: this.threshold,
      true_positives_reason_line: line(tally.TP),
      true_negatives_reason_line: line(tally.TN),
      false_positives_reason_line: line(tally.FP),
      false_negatives_reason_line: line(tally.FN),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      TopicAdherenceReasonSchema,
    );
    return reason;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Topic Adherence";
  }
}
