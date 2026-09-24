import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  warnScoreDirectionFlipped,
  prettifyList,
} from "@/metrics/utils";
import {
  generateQagVerdicts,
  parseQuestions,
  runSystemOneEval,
  type SystemOneEvalSpec,
  type SystemOneVerdictSpec,
} from "@/metrics/system-one";
import {
  VerdictsSchema,
  HallucinationScoreReasonSchema,
  type HallucinationVerdict,
} from "@/metrics/hallucination/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "HallucinationMetric";

export type HallucinationTemplateOverride =
  MetricTemplateOverride<"HallucinationMetric">;

export interface HallucinationMetricOptions {
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
  evaluationTemplate?: HallucinationTemplateOverride;
}

/**
 * Hallucination — does the `actualOutput` stay faithful to the provided
 * `context`? Judge the output against each context, then score = aligned /
 * total. **Higher is better** (`success = score >= threshold`).
 */
export class HallucinationMetric extends BaseMetric {
  verdicts: HallucinationVerdict[] = [];

  constructor(options: HallucinationMetricOptions = {}) {
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
    warnScoreDirectionFlipped("HallucinationMetric");
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.CONTEXT,
    ];
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      this.verdicts = await this.generateVerdicts(
        testCase.actualOutput,
        testCase.context ?? [],
      );
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
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
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.CONTEXT,
      ],
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
  }

  private systemOneVerdictSpec(
    actualOutput: string,
    contexts: string[],
  ): SystemOneVerdictSpec<string, HallucinationVerdict> {
    return {
      instructions: this.getPrompt("_experimental_system_one_verdict"),
      items: contexts,
      itemKey: "context",
      state: { actual_output: actualOutput },
    };
  }

  private async generateVerdicts(
    actualOutput: string,
    contexts: string[],
  ): Promise<HallucinationVerdict[]> {
    return generateQagVerdicts(this, {
      systemOne: this.systemOneVerdictSpec(actualOutput, contexts),
      llm: async () => {
        const prompt = this.getPrompt("generate_verdicts", {
          actual_output: actualOutput,
          contexts,
          contexts_count: contexts.length,
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

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const factualAlignments: (string | null | undefined)[] = [];
    const contradictions: (string | null | undefined)[] = [];
    for (const v of this.verdicts) {
      if (v.verdict.trim().toLowerCase() === "yes")
        factualAlignments.push(v.reason);
      else contradictions.push(v.reason);
    }
    const prompt = this.getPrompt("generate_reason", {
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
    if (total === 0) return 1;
    const alignedCount = this.verdicts.filter(
      (v) => v.verdict.trim().toLowerCase() !== "no",
    ).length;
    const score = alignedCount / total;
    return this.applyStrictMode(score);
  }

  get name(): string {
    return "Hallucination";
  }
}
