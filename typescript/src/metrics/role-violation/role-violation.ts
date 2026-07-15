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
  RoleViolationsSchema,
  VerdictsSchema,
  RoleViolationScoreReasonSchema,
  type RoleViolationVerdict,
} from "./schema";

const TEMPLATE_CLASS = "RoleViolationMetric";

export interface RoleViolationMetricOptions {
  /** The role the assistant must stay in (e.g. "helpful assistant"). Required. */
  role: string;
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
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
    super(strictMode ? 0 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.role = options.role;
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

      this.roleViolations = await this.detectRoleViolations(
        testCase.actualOutput,
      );
      this.verdicts = await this.generateVerdicts();
      this.score = this.calculateScore();
      this.reason = await this.generateReason();
      this.success = this.score >= this.threshold;

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
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "detect_role_violations", {
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

  private async generateVerdicts(): Promise<RoleViolationVerdict[]> {
    if (this.roleViolations.length === 0) return [];
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_verdicts", {
      role_violations: this.roleViolations,
    });
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const violationReasons = this.verdicts
      .filter((v) => v.verdict.trim().toLowerCase() === "yes")
      .map((v) => v.reason);
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
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
    if (this.verdicts.length === 0) return 1;
    for (const v of this.verdicts) {
      if (v.verdict.trim().toLowerCase() === "yes") return 0;
    }
    return 1;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Role Violation";
  }
}
