import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  prettifyList,
} from "../utils";
import {
  VerdictsSchema,
  HallucinationScoreReasonSchema,
  type HallucinationVerdict,
} from "./schema";

const TEMPLATE_CLASS = "HallucinationMetric";

export interface HallucinationMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Hallucination — does the `actualOutput` contradict the provided `context`?
 * Judge the output against each context, then score = contradicting / total.
 * **Lower is better** (`success = score <= threshold`).
 */
export class HallucinationMetric extends BaseMetric {
  verdicts: HallucinationVerdict[] = [];

  constructor(options: HallucinationMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 0 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.CONTEXT,
    ];
    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      this.verdicts = await this.generateVerdicts(
        testCase.actualOutput,
        testCase.context ?? [],
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.score <= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdicts(
    actualOutput: string,
    contexts: string[],
  ): Promise<HallucinationVerdict[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      actual_output: actualOutput,
      contexts,
      contexts_count: contexts.length,
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const factualAlignments: (string | null | undefined)[] = [];
    const contradictions: (string | null | undefined)[] = [];
    for (const v of this.verdicts) {
      if (v.verdict.trim().toLowerCase() === "yes")
        factualAlignments.push(v.reason);
      else contradictions.push(v.reason);
    }
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      factual_alignments: factualAlignments,
      contradictions,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      HallucinationScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 0;
    const hallucinationCount = this.verdicts.filter(
      (v) => v.verdict.trim().toLowerCase() === "no",
    ).length;
    const score = hallucinationCount / total;
    return this.strictMode && score > this.threshold ? 1 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 1) <= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Hallucination";
  }
}
