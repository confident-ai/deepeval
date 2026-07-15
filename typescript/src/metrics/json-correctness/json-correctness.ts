import type { ZodType } from "zod";
import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { toJsonSchema } from "../../models/utils";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "../utils";
import { JsonCorrectnessScoreReasonSchema } from "./schema";

const TEMPLATE_CLASS = "JsonCorrectnessMetric";
const DEFAULT_CORRECT_REASON =
  "The generated Json matches and is syntactically correct to the expected schema.";

export interface JsonCorrectnessMetricOptions {
  /** The schema the output must conform to (zod ⇄ Python's pydantic model). Required. */
  expectedSchema: ZodType;
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  /** Defaults to true (matches Python): requires a perfectly valid JSON. */
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Json Correctness — does the `actualOutput` parse as JSON and conform to
 * `expectedSchema`? Deterministic check (score 1/0); an LLM only explains
 * failures. **Higher is better** (`success = score >= threshold`).
 */
export class JsonCorrectnessMetric extends BaseMetric {
  private readonly expectedSchema: ZodType;

  constructor(options: JsonCorrectnessMetricOptions) {
    const strictMode = options.strictMode ?? true;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.expectedSchema = options.expectedSchema;
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

      const validJson = this.isValidJson(testCase.actualOutput);
      this.score = validJson ? 1 : 0;
      this.reason = await this.generateReason(testCase.actualOutput);
      this.success = this.score >= this.threshold;

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
    try {
      this.expectedSchema.parse(JSON.parse(actualOutput));
      return true;
    } catch {
      return false;
    }
  }

  private async generateReason(
    actualOutput: string,
  ): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    if (this.score === 1) return DEFAULT_CORRECT_REASON;

    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      actual_output: actualOutput,
      expected_schema: JSON.stringify(toJsonSchema(this.expectedSchema), null, 4),
      is_valid_json: false,
    });
    const { reason } = await generateWithSchema(
      this,
      prompt,
      JsonCorrectnessScoreReasonSchema,
    );
    return reason;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Json Correctness";
  }
}
