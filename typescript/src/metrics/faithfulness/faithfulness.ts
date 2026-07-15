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
  TruthsSchema,
  ClaimsSchema,
  VerdictsSchema,
  FaithfulnessScoreReasonSchema,
  type FaithfulnessVerdict,
} from "./schema";

const TEMPLATE_CLASS = "FaithfulnessMetric";

function truthsLimitPhrase(limit?: number): string {
  if (limit == null) return " FACTUAL, undisputed truths";
  if (limit === 1) return " the single most important FACTUAL, undisputed truth";
  return ` the ${limit} most important FACTUAL, undisputed truths per document`;
}

export interface FaithfulnessMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  truthsExtractionLimit?: number;
  penalizeAmbiguousClaims?: boolean;
}

/**
 * Faithfulness — does the `actualOutput` contradict the `retrievalContext`?
 * Extract truths from the context and claims from the output, judge each claim
 * against the truths, then score = non-contradicting / total.
 */
export class FaithfulnessMetric extends BaseMetric {
  truths: string[] = [];
  claims: string[] = [];
  verdicts: FaithfulnessVerdict[] = [];
  private readonly truthsExtractionLimit?: number;
  private readonly penalizeAmbiguousClaims: boolean;

  constructor(options: FaithfulnessMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.RETRIEVAL_CONTEXT,
    ];
    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
    this.truthsExtractionLimit =
      options.truthsExtractionLimit != null
        ? Math.max(options.truthsExtractionLimit, 0)
        : undefined;
    this.penalizeAmbiguousClaims = options.penalizeAmbiguousClaims ?? false;
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      const retrievalContext = resolveRetrievalContext(
        testCase.retrievalContext ?? [],
      );

      this.truths = await this.generateTruths(retrievalContext);
      this.claims = await this.generateClaims(testCase.actualOutput);
      this.verdicts = await this.generateVerdicts();
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Truths (limit=${this.truthsExtractionLimit}):\n${prettifyList(this.truths)}`,
        `Claims:\n${prettifyList(this.claims)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateTruths(retrievalContext: string[]): Promise<string[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_truths", {
      retrieval_context: retrievalContext.join("\n\n"),
      limit: truthsLimitPhrase(this.truthsExtractionLimit),
      multimodal_instruction: "",
    });
    const { truths } = await generateWithSchema(this, prompt, TruthsSchema);
    return truths;
  }

  private async generateClaims(actualOutput: string): Promise<string[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_claims", {
      actual_output: actualOutput,
      multimodal_instruction: "",
    });
    const { claims } = await generateWithSchema(this, prompt, ClaimsSchema);
    return claims;
  }

  private async generateVerdicts(): Promise<FaithfulnessVerdict[]> {
    if (this.claims.length === 0) return [];
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      claims: this.claims,
      retrieval_context: this.truths.join("\n\n"),
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const contradictions: (string | null | undefined)[] = [];
    for (const v of this.verdicts) {
      const vd = v.verdict.trim().toLowerCase();
      if (vd === "no") contradictions.push(v.reason);
      else if (vd === "idk" && this.penalizeAmbiguousClaims)
        contradictions.push(`(Ambiguous) ${v.reason}`);
    }
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      contradictions,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      FaithfulnessScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 1;
    let count = 0;
    for (const v of this.verdicts) {
      const vd = v.verdict.trim().toLowerCase();
      if (vd !== "no") count++;
      if (this.penalizeAmbiguousClaims && vd === "idk") count--;
    }
    const score = count / total;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Faithfulness";
  }
}
