import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
  resolveRetrievalContext,
} from "@/metrics/utils";
import {
  generateQagVerdicts,
  parseQuestions,
  runSystemOneEval,
  splitSentences,
  type SystemOneEvalSpec,
  type SystemOneVerdictSpec,
} from "@/metrics/system-one";
import {
  VerdictsSchema,
  ContextualRecallScoreReasonSchema,
  type ContextualRecallVerdict,
} from "@/metrics/contextual-recall/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import {
  contextualRecallVerdictVars,
  contextualRecallReasonContentType,
} from "@/metrics/retrieval-context-display";
import { prepareMeasure } from "@/metrics/prepare-measure";

const TEMPLATE_CLASS = "ContextualRecallMetric";

export type ContextualRecallTemplateOverride =
  MetricTemplateOverride<"ContextualRecallMetric">;

export interface ContextualRecallMetricOptions {
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
  evaluationTemplate?: ContextualRecallTemplateOverride;
}

/**
 * Contextual Recall — can each sentence of `expectedOutput` be attributed to the
 * `retrievalContext`? Score = attributable sentences / total.
 */
export class ContextualRecallMetric extends BaseMetric {
  verdicts: ContextualRecallVerdict[] = [];

  constructor(options: ContextualRecallMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.multimodalAware = true;
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.RETRIEVAL_CONTEXT,
      SingleTurnParams.EXPECTED_OUTPUT,
    ];
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      this.verdicts = await this.generateVerdicts(
        testCase.expectedOutput ?? "",
        resolveRetrievalContext(testCase.retrievalContext ?? []),
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason(testCase.expectedOutput ?? "");
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  systemOneEvalSpec(testCase: LLMTestCase): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: [
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
      ],
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
  }

  private systemOneVerdictSpec(
    expectedOutput: string,
    retrievalContext: string[],
  ): SystemOneVerdictSpec<string, ContextualRecallVerdict> | undefined {
    if (this.multimodal) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_verdict"),
      items: splitSentences(expectedOutput),
      itemKey: "sentence",
      state: { retrieval_context: retrievalContext },
    };
  }

  private async generateVerdicts(
    expectedOutput: string,
    retrievalContext: string[],
  ): Promise<ContextualRecallVerdict[]> {
    return generateQagVerdicts(this, {
      systemOne: this.systemOneVerdictSpec(expectedOutput, retrievalContext),
      llm: async () => {
        const prompt = this.getPrompt("generate_verdicts", {
          expected_output: expectedOutput,
          ...contextualRecallVerdictVars(retrievalContext, this.multimodal),
        });
        const { verdicts } = await generateWithSchema(
          this,
          prompt,
          VerdictsSchema,
        );
        return verdicts;
      },
    });
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
    const prompt = this.getPrompt("generate_reason", {
      expected_output: expectedOutput,
      supportive_reasons: supportiveReasons,
      unsupportive_reasons: unsupportiveReasons,
      score: (this.score ?? 0).toFixed(2),
      content_type: contextualRecallReasonContentType(this.multimodal),
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
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Contextual Recall";
  }
}
