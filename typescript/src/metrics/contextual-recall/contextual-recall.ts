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
  resolveRetrievalContext,
} from "../utils";
import {
  VerdictsSchema,
  ContextualRecallScoreReasonSchema,
  type ContextualRecallVerdict,
} from "./schema";

const TEMPLATE_CLASS = "ContextualRecallMetric";

export interface ContextualRecallMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Contextual Recall — can each sentence of `expectedOutput` be attributed to the
 * `retrievalContext`? Score = attributable sentences / total.
 */
export class ContextualRecallMetric extends BaseMetric {
  verdicts: ContextualRecallVerdict[] = [];

  constructor(options: ContextualRecallMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.RETRIEVAL_CONTEXT,
      SingleTurnParams.EXPECTED_OUTPUT,
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
        testCase.expectedOutput ?? "",
        resolveRetrievalContext(testCase.retrievalContext ?? []),
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.expectedOutput ?? "");
      this.success = this.score >= this.threshold;

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
    expectedOutput: string,
    retrievalContext: string[],
  ): Promise<ContextualRecallVerdict[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      expected_output: expectedOutput,
      content_type: "sentence",
      content_type_plural: "sentences",
      content_or: "sentence",
      context_to_display: retrievalContext,
      node_instruction: "",
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(
    expectedOutput: string,
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const supportiveReasons: (string | null | undefined)[] = [];
    const unsupportiveReasons: (string | null | undefined)[] = [];
    for (const v of this.verdicts) {
      if (v.verdict.toLowerCase() === "yes") supportiveReasons.push(v.reason);
      else unsupportiveReasons.push(v.reason);
    }
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      expected_output: expectedOutput,
      supportive_reasons: supportiveReasons,
      unsupportive_reasons: unsupportiveReasons,
      score: (this.score ?? 0).toFixed(2),
      content_type: "sentence",
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRecallScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 0;
    const justified = this.verdicts.filter(
      (v) => v.verdict.toLowerCase() === "yes",
    ).length;
    const score = justified / total;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Contextual Recall";
  }
}
