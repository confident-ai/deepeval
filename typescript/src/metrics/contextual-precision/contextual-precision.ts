import { BaseMetric } from "../base-metrics";
import {
  LLMTestCase,
  SingleTurnParams,
  RetrievedContextData,
} from "../../test-case";
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
  ContextualPrecisionScoreReasonSchema,
  type ContextualPrecisionVerdict,
} from "./schema";

const TEMPLATE_CLASS = "ContextualPrecisionMetric";

/**
 * Collapse a `retrievalContext` into evaluation documents: `RetrievedContextData`
 * sharing a `source` are merged into one document (in first-seen order), while
 * plain strings pass through standalone. Mirrors Python's
 * `_group_retrieval_contexts`.
 */
function groupRetrievalContexts(
  retrievalContexts: (string | RetrievedContextData)[],
): string[] {
  const grouped: Record<string, string[]> = {};
  type Ordered =
    | { type: "grouped"; key: string }
    | { type: "standalone"; value: string };
  const ordered: Ordered[] = [];

  for (const context of retrievalContexts) {
    if (context instanceof RetrievedContextData) {
      if (!(context.source in grouped)) {
        ordered.push({ type: "grouped", key: context.source });
        grouped[context.source] = [];
      }
      grouped[context.source].push(context.context);
    } else {
      ordered.push({ type: "standalone", value: context });
    }
  }

  return ordered.map((item) =>
    item.type === "grouped"
      ? `Source: ${item.key}\n` + grouped[item.key].join("\n---\n")
      : item.value,
  );
}

export interface ContextualPrecisionMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Contextual Precision — are the relevant nodes in `retrievalContext` ranked
 * above the irrelevant ones (w.r.t. `expectedOutput`)? Judge each node, then
 * score with rank-weighted precision.
 */
export class ContextualPrecisionMetric extends BaseMetric {
  verdicts: ContextualPrecisionVerdict[] = [];

  constructor(options: ContextualPrecisionMetricOptions = {}) {
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

      const groupedContext = groupRetrievalContexts(
        testCase.retrievalContext ?? [],
      );
      this.verdicts = await this.generateVerdicts(
        testCase.input,
        testCase.expectedOutput ?? "",
        groupedContext,
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.input);
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
    input: string,
    expectedOutput: string,
    retrievalContext: string[],
  ): Promise<ContextualPrecisionVerdict[]> {
    const n = retrievalContext.length;
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      input,
      expected_output: expectedOutput,
      document_count_str: ` (${n} document${n > 1 ? "s" : ""})`,
      context_to_display: retrievalContext,
      multimodal_note: "",
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(input: string): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const verdicts = this.verdicts.map((v) => ({
      verdict: v.verdict,
      reason: v.reason,
    }));
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      input,
      verdicts,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      ContextualPrecisionScoreReasonSchema,
    );
    return reason;
  }

  /** Rank-weighted precision: reward relevant nodes ranked higher. */
  private calculateScore(): number {
    const total = this.verdicts.length;
    if (total === 0) return 0;
    const nodeVerdicts = this.verdicts.map((v) =>
      v.verdict.trim().toLowerCase() === "yes" ? 1 : 0,
    );
    let sumWeightedPrecision = 0;
    let relevantCount = 0;
    for (let i = 0; i < nodeVerdicts.length; i++) {
      if (nodeVerdicts[i]) {
        relevantCount++;
        sumWeightedPrecision += relevantCount / (i + 1);
      }
    }
    if (relevantCount === 0) return 0;
    const score = sumWeightedPrecision / relevantCount;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Contextual Precision";
  }
}
