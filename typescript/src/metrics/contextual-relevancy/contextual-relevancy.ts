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
  ContextualRelevancyVerdictsSchema,
  ContextualRelevancyScoreReasonSchema,
  type ContextualRelevancyVerdict,
  type ContextualRelevancyVerdicts,
} from "@/metrics/contextual-relevancy/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { contextualRelevancyVerdictVars } from "@/metrics/retrieval-context-display";
import { prepareMeasure } from "@/metrics/prepare-measure";

const TEMPLATE_CLASS = "ContextualRelevancyMetric";

export type ContextualRelevancyTemplateOverride =
  MetricTemplateOverride<"ContextualRelevancyMetric">;

function systemOneRelevancyVerdict(
  statement: string,
  verdict: string,
  probability: number,
): ContextualRelevancyVerdict {
  let reason = `P(yes)=${probability.toFixed(2)}`;
  if (verdict === "no") {
    reason = `'${statement}' is not relevant to the input (${reason}).`;
  }
  return { statement, verdict, reason };
}

export interface ContextualRelevancyMetricOptions {
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
  evaluationTemplate?: ContextualRelevancyTemplateOverride;
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
    ];
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);
      if (await runSystemOneEval(this, testCase)) return this.score as number;

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
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Verdicts:\n${prettifyList(this.verdictsList)}`,
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
      evaluationParams: this.requiredParams,
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
  }

  private systemOneVerdictSpec(
    input: string,
    context: string,
  ): SystemOneVerdictSpec<string, ContextualRelevancyVerdict> | undefined {
    if (this.multimodal) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_verdict"),
      items: splitSentences(context),
      itemKey: "statement",
      state: { input },
      buildVerdict: systemOneRelevancyVerdict,
    };
  }

  private async generateVerdicts(
    input: string,
    context: string,
  ): Promise<ContextualRelevancyVerdicts> {
    const verdicts = await generateQagVerdicts(this, {
      systemOne: this.systemOneVerdictSpec(input, context),
      llm: async () => {
        const prompt = this.getPrompt("generate_verdicts", {
          input,
          context,
          ...contextualRelevancyVerdictVars(this.multimodal),
        });
        const { verdicts } = await generateWithSchema(
          this,
          prompt,
          ContextualRelevancyVerdictsSchema,
        );
        return verdicts;
      },
    });
    return { verdicts };
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
    const prompt = this.getPrompt("generate_reason", {
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
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Contextual Relevancy";
  }
}
