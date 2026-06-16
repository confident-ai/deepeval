import { BaseArenaMetric } from "../base-arena-metric";
import { ArenaTestCase, SingleTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  prettifyList,
  checkArenaTestCaseParams,
} from "../utils";
import { StepsSchema } from "../g-eval/schema";
import {
  constructGEvalParamsString,
  numberEvaluationSteps,
  validateCriteriaAndEvaluationSteps,
} from "../g-eval/utils";
import { formatArenaTestCase } from "./utils";
import { WinnerSchema, RewrittenReasonSchema } from "./schema";

const TEMPLATE_CLASS = "ArenaGEval";

export interface ArenaGEvalMetricOptions {
  name: string;
  /** Which contestant fields the judge considers (single-turn params). */
  evaluationParams: SingleTurnParams[];
  criteria?: string;
  evaluationSteps?: string[];
  model?: DeepEvalBaseLLM | string;
  verboseMode?: boolean;
  showIndicator?: boolean;
  includeGEvalSuffix?: boolean;
}

/**
 * Arena GEval — a GEval-style judge that picks the WINNING contestant for a
 * shared input. Contestant names are masked (shuffled dummies) before judging
 * and unmasked after, to avoid name/position bias. `measure` returns the
 * winning contestant's name.
 */
export class ArenaGEval extends BaseArenaMetric {
  evaluationParams: SingleTurnParams[];
  criteria?: string;
  evaluationSteps?: string[];
  private readonly metricName: string;
  private readonly includeGEvalSuffix: boolean;

  constructor(options: ArenaGEvalMetricOptions) {
    validateCriteriaAndEvaluationSteps(
      options.criteria,
      options.evaluationSteps,
    );
    super(0, {
      verboseMode: options.verboseMode,
      showIndicator: options.showIndicator,
    });
    this.metricName = options.name;
    this.evaluationParams = options.evaluationParams;
    this.criteria = options.criteria;
    this.evaluationSteps =
      options.evaluationSteps && options.evaluationSteps.length > 0
        ? options.evaluationSteps
        : undefined;
    this.includeGEvalSuffix = options.includeGEvalSuffix ?? true;
    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(
    testCase: ArenaTestCase,
    opts: { onStep?: () => void } = {},
  ): Promise<string> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkArenaTestCaseParams(testCase, this.evaluationParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      this.evaluationSteps = await this.generateEvaluationSteps(
        testCase.multimodal,
      );
      opts.onStep?.();
      const [maskedWinner, maskedReason, dummyToReal] = await this.compareArena(
        testCase,
        testCase.multimodal,
      );
      opts.onStep?.();
      this.winner = dummyToReal[maskedWinner] ?? maskedWinner;
      this.reason = await this.generateRewrittenReason(
        maskedReason,
        dummyToReal,
      );
      opts.onStep?.();
      this.success = true;

      this.verboseLogs = constructVerboseLogs(this, [
        `Criteria:\n${this.criteria}`,
        `Evaluation Steps:\n${prettifyList(this.evaluationSteps)}`,
        `Winner: ${this.winner}`,
        `Reason: ${this.reason}`,
      ]);
      return this.winner;
    } finally {
      this.stopProgress();
    }
  }

  private async generateEvaluationSteps(
    multimodal: boolean,
  ): Promise<string[]> {
    if (this.evaluationSteps) return this.evaluationSteps;
    const prompt = resolveTemplate(
      "metrics",
      TEMPLATE_CLASS,
      "generate_evaluation_steps",
      {
        criteria: this.criteria,
        parameters: constructGEvalParamsString(this.evaluationParams),
        multimodal,
      },
    );
    const { steps } = await generateWithSchema(this, prompt, StepsSchema);
    return steps;
  }

  private async compareArena(
    testCase: ArenaTestCase,
    multimodal: boolean,
  ): Promise<[string, string, Record<string, string>]> {
    const [formatted, dummyToReal] = formatArenaTestCase(
      this.evaluationParams,
      testCase,
    );
    const prompt = resolveTemplate(
      "metrics",
      TEMPLATE_CLASS,
      "generate_arena_winner",
      {
        evaluation_steps: numberEvaluationSteps(this.evaluationSteps ?? []),
        test_case_contents: formatted,
        parameters: constructGEvalParamsString(this.evaluationParams),
        multimodal,
      },
    );
    const { winner, reason } = await generateWithSchema(
      this,
      prompt,
      WinnerSchema,
    );
    return [winner, reason, dummyToReal];
  }

  private async generateRewrittenReason(
    reason: string,
    dummyToReal: Record<string, string>,
  ): Promise<string> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "rewrite_reason", {
      reason,
      // Pass as JSON text: Nunjucks renders a bare object as "[object Object]"
      // (the resolver only gives arrays a Python-repr toString). The template
      // renders {{ dummy_to_real_names }} directly and its example is JSON.
      dummy_to_real_names: JSON.stringify(dummyToReal),
    });
    const { rewritten_reason } = await generateWithSchema(
      this,
      prompt,
      RewrittenReasonSchema,
    );
    return rewritten_reason;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && this.success === true;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return this.includeGEvalSuffix
      ? `${this.metricName} [Arena GEval]`
      : this.metricName;
  }
}
