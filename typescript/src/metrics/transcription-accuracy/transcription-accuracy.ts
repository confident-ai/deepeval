import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams, Turn } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
} from "@/metrics/utils";
import {
  TranscriptionAccuracyScoreReasonSchema,
  VerdictsSchema,
  type TranscriptionAccuracyVerdict,
} from "@/metrics/transcription-accuracy/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { prepareMeasure } from "@/metrics/prepare-measure";

const TEMPLATE_CLASS = "TranscriptionAccuracyMetric";

export type TranscriptionAccuracyTemplateOverride =
  MetricTemplateOverride<"TranscriptionAccuracyMetric">;

export interface TranscriptionAccuracyMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: TranscriptionAccuracyTemplateOverride;
}

export interface TranscribedExchange extends Record<string, unknown> {
  spoken: string;
  transcribed: string;
  agent_reply: string;
}

export function getTranscribedExchanges(turns: Turn[]): TranscribedExchange[] {
  const exchanges: TranscribedExchange[] = [];
  let spoken: string[] = [];
  for (const turn of turns) {
    if (turn.role === "user") {
      if (turn.content) spoken.push(turn.content);
      continue;
    }
    if (turn.providerTranscription != null && spoken.length > 0) {
      exchanges.push({
        spoken: spoken.join(" "),
        transcribed: turn.providerTranscription,
        agent_reply: turn.content,
      });
    }
    spoken = [];
  }
  return exchanges;
}

export class TranscriptionAccuracyMetric extends BaseConversationalMetric {
  verdicts: TranscriptionAccuracyVerdict[] = [];
  exchanges: TranscribedExchange[] = [];

  constructor(options: TranscriptionAccuracyMetricOptions = {}) {
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
    this.requiredParams = [
      MultiTurnParams.CONTENT,
      MultiTurnParams.ROLE,
      MultiTurnParams.PROVIDER_TRANSCRIPTION,
    ];
    initializeMetricModels(this, { ...options, systemOne: false });
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);

      this.exchanges = getTranscribedExchanges(testCase.turns);
      this.verdicts = await this.generateVerdicts();
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Exchanges:\n${prettifyList(this.exchanges)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdicts(): Promise<TranscriptionAccuracyVerdict[]> {
    if (this.exchanges.length === 0) return [];
    const prompt = this.getPrompt("generate_verdicts", {
      exchanges: this.exchanges,
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const mistranscriptions = this.verdicts
      .filter(
        (v) => v?.verdict != null && v.verdict.trim().toLowerCase() === "no",
      )
      .map((v) => v.reason);
    const prompt = this.getPrompt("generate_reason", {
      score: this.score,
      mistranscriptions,
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      TranscriptionAccuracyScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const valid = this.verdicts.filter((v) => v != null && v.verdict != null);
    if (valid.length === 0) return 1;
    const faithful = valid.filter(
      (v) => v.verdict.trim().toLowerCase() !== "no",
    ).length;
    return this.applyStrictMode(faithful / valid.length);
  }

  get name(): string {
    return "Transcription Accuracy";
  }
}
