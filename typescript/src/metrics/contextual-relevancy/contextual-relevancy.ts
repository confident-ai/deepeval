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
  ContextualRelevancyVerdictsSchema,
  ContextualRelevancyScoreReasonSchema,
  type ContextualRelevancyVerdicts,
} from "./schema";

const TEMPLATE_CLASS = "ContextualRelevancyMetric";

// Text-only values of the verdict template's precomputed vars (Python builds
// these in `_contextual_relevancy_verdict_kwargs`).
const EXTRACTION_INSTRUCTIONS =
  "You should first extract statements found in the context, which are " +
  "high level information found in the context, before deciding on a " +
  "verdict and optionally a reason for each statement.";
const EMPTY_CONTEXT_INSTRUCTION =
  '\nIf provided context contains no actual content or statements then: ' +
  'give "no" as a "verdict",\nput context into "statement", and ' +
  '"No statements found in provided context." into "reason".';

export interface ContextualRelevancyMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Contextual Relevancy — what fraction of statements across `retrievalContext`
 * are relevant to the `input`? Judge each node's statements, then
 * score = relevant statements / total statements.
 */
export class ContextualRelevancyMetric extends BaseMetric {
  verdictsList: ContextualRelevancyVerdicts[] = [];

  constructor(options: ContextualRelevancyMetricOptions = {}) {
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

      const retrievalContext = resolveRetrievalContext(
        testCase.retrievalContext ?? [],
      );
      this.verdictsList = await Promise.all(
        retrievalContext.map((context) =>
          this.generateVerdicts(testCase.input, context),
        ),
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.input);
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Verdicts:\n${prettifyList(this.verdictsList)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async generateVerdicts(
    input: string,
    context: string,
  ): Promise<ContextualRelevancyVerdicts> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      input,
      context,
      context_type: "context",
      statement_or_image: "statement",
      extraction_instructions: EXTRACTION_INSTRUCTIONS,
      empty_context_instruction: EMPTY_CONTEXT_INSTRUCTION,
    });
    return generateWithSchema(this, prompt, ContextualRelevancyVerdictsSchema);
  }

  private async generateReason(input: string): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const irrelevantStatements: (string | null | undefined)[] = [];
    const relevantStatements: string[] = [];
    for (const { verdicts } of this.verdictsList) {
      for (const v of verdicts) {
        if (v.verdict.toLowerCase() === "no")
          irrelevantStatements.push(v.reason);
        else relevantStatements.push(v.statement);
      }
    }
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      input,
      irrelevant_statements: irrelevantStatements,
      relevant_statements: relevantStatements,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualRelevancyScoreReasonSchema,
    );
    return reason;
  }

  private calculateScore(): number {
    let totalVerdicts = 0;
    let relevant = 0;
    for (const { verdicts } of this.verdictsList) {
      for (const v of verdicts) {
        totalVerdicts++;
        if (v.verdict.toLowerCase() === "yes") relevant++;
      }
    }
    if (totalVerdicts === 0) return 0;
    const score = relevant / totalVerdicts;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Contextual Relevancy";
  }
}
