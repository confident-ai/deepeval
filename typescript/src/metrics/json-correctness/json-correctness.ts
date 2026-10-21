import type { ZodType } from "zod";
import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { EvalModeName } from "@/config/eval-mode";
import { toJsonSchema } from "@/models/utils";
import {
  initializeMetricModels,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "@/metrics/utils";
import { JsonCorrectnessScoreReasonSchema } from "@/metrics/json-correctness/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "JsonCorrectnessMetric";

export type JsonCorrectnessTemplateOverride =
  MetricTemplateOverride<"JsonCorrectnessMetric">;
const DEFAULT_CORRECT_REASON =
  "The generated Json matches and is syntactically correct to the expected schema.";

export interface JsonCorrectnessMetricOptions {
  /** The schema the output must conform to (zod ⇄ Python's pydantic model). Required. */
  expectedSchema: ZodType;
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  /** Who decides; defaults to `DEEPEVAL_EVAL_MODE`, then `llm`. */
  evalMode?: EvalModeName;
  includeReason?: boolean;
  /** Defaults to true (matches Python): requires a perfectly valid JSON. */
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: JsonCorrectnessTemplateOverride;
}

/**
 * Json Correctness — does the `actualOutput` parse as JSON and conform to
 * `expectedSchema`? Deterministic check (score 1/0); an LLM only explains
 * failures. **Higher is better** (`success = score >= threshold`).
 */
export class JsonCorrectnessMetric extends BaseMetric {
  private readonly expectedSchema: ZodType;
  private validationError?: string;

  constructor(options: JsonCorrectnessMetricOptions) {
    const strictMode = options.strictMode ?? true;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.expectedSchema = options.expectedSchema;
    initializeMetricModels(this, {
      model: options.model,
      evalMode: options.evalMode,
      systemOne: false,
    });
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      const validJson = this.isValidJson(testCase.actualOutput);
      this.score = validJson ? 1 : 0;
      this.reason = await this.generateReason(testCase.actualOutput);
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        `LLM outputed Json:\n${testCase.actualOutput}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private isValidJson(actualOutput: string): boolean {
    this.validationError = undefined;
    try {
      this.expectedSchema.parse(JSON.parse(actualOutput));
      return true;
    } catch (e) {
      this.validationError = e instanceof Error ? e.message : String(e);
      return false;
    }
  }

  private async generateReason(
    actualOutput: string,
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    if (this.score === 1) return DEFAULT_CORRECT_REASON;
    if (!this.model) return this.validationError;

    const prompt = this.getPrompt("generate_reason", {
      actual_output: actualOutput,
      expected_schema: JSON.stringify(
        toJsonSchema(this.expectedSchema),
        null,
        4,
      ),
      is_valid_json: false,
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      JsonCorrectnessScoreReasonSchema,
    );
    return reason;
  }

  get name(): string {
    return "Json Correctness";
  }
}
