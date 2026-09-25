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
} from "@/metrics/utils";
import {
  generateQagVerdicts,
  parseQuestions,
  runSystemOneEval,
  type SystemOneEvalSpec,
  type SystemOneVerdictSpec,
} from "@/metrics/system-one";
import {
  RoleViolationsSchema,
  VerdictsSchema,
  RoleViolationScoreReasonSchema,
  type RoleViolationVerdict,
} from "@/metrics/role-violation/schema";
import { type MetricTemplateOverride } from "@/templates/override";
import { prepareMeasure } from "@/metrics/prepare-measure";

const TEMPLATE_CLASS = "RoleViolationMetric";

export type RoleViolationTemplateOverride =
  MetricTemplateOverride<"RoleViolationMetric">;

export interface RoleViolationMetricOptions {
  /** The role the assistant must stay in (e.g. "helpful assistant"). Required. */
  role: string;
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
  evaluationTemplate?: RoleViolationTemplateOverride;
}

/**
 * Role Violation — does the `actualOutput` break out of its expected `role`?
 * Detect candidate violations, judge each, then score is binary: 0 if any real
 * violation, else 1. **Higher is better** (`success = score >= threshold`).
 */
export class RoleViolationMetric extends BaseMetric {
  roleViolations: string[] = [];
  verdicts: RoleViolationVerdict[] = [];
  private readonly role: string;

  constructor(options: RoleViolationMetricOptions) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 0 : resolveThreshold(options.threshold, 0.5), {
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
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.role = options.role;
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      prepareMeasure(this, testCase);
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      this.roleViolations = await this.detectRoleViolations(
        testCase.actualOutput,
      );
      this.verdicts = await this.generateVerdicts();
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `Role violations:\n${prettifyList(this.roleViolations)}`,
        `Verdicts:\n${prettifyList(this.verdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async detectRoleViolations(actualOutput: string): Promise<string[]> {
    const prompt = this.getPrompt("detect_role_violations", {
      actual_output: actualOutput,
      expected_role: this.role,
    });
    const { role_violations } = await generateWithSchema(
      this,
      prompt,
      RoleViolationsSchema,
    );
    return role_violations;
  }

  private systemOneVerdictSpec(): SystemOneVerdictSpec<
    string,
    RoleViolationVerdict
  > {
    return {
      instructions: this.getPrompt("_experimental_system_one_verdict"),
      items: this.roleViolations,
      itemKey: "statement",
      state: { role: this.role },
    };
  }

  systemOneEvalSpec(testCase: LLMTestCase): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: this.requiredParams,
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions", {
          role: this.role,
        }),
      ),
    };
  }

  private async generateVerdicts(): Promise<RoleViolationVerdict[]> {
    if (this.roleViolations.length === 0) return [];
    return generateQagVerdicts(this, {
      systemOne: this.systemOneVerdictSpec(),
      llm: async () => {
        const prompt = this.getPrompt("generate_verdicts", {
          role_violations: this.roleViolations,
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
    const violationReasons = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "yes")
      .map((v) => v.reason);
    const prompt = this.getPrompt("generate_reason", {
      role_violations: violationReasons,
      score: (this.score ?? 0).toFixed(2),
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      RoleViolationScoreReasonSchema,
    );
    return reason;
  }

  /** Binary: 0 if any real violation, else 1 (1 when there are no verdicts). */
  private calculateScore(): number {
    if (this.roleViolations.length === 0) return 1;
    // Every extracted candidate must be judged: an incomplete or empty
    // verdict list is a failed audit, not a clean pass.
    if (this.verdicts.length < this.roleViolations.length) return 0;
    for (const v of this.verdicts) {
      if (v.verdict.trim().toLowerCase() === "yes") return 0;
    }
    return 1;
  }

  get name(): string {
    return "Role Violation";
  }
}
