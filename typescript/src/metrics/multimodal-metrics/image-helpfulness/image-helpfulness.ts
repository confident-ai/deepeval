import { BaseMetric } from "../../base-metrics";
import {
  LLMTestCase,
  SingleTurnParams,
  MLLMImage,
  convertToMultiModalArray,
} from "../../../test-case";
import { DeepEvalBaseLLM } from "../../../models";
import { resolveTemplate } from "../../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "../../utils";
import { getImageIndices, getImageContext } from "../utils";
import { ReasonScoreSchema } from "../schema";

const TEMPLATE_CLASS = "ImageHelpfulnessMetric";

export interface ImageHelpfulnessMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  /** Clip the text context around each image to this many chars. */
  maxContextSize?: number;
}

/**
 * Image Helpfulness — does each image in `actualOutput` help convey the
 * surrounding text? Scores each image (0–10) against its text context via a
 * vision model, then averages. **Higher is better**.
 */
export class ImageHelpfulnessMetric extends BaseMetric {
  private readonly maxContextSize?: number;

  constructor(options: ImageHelpfulnessMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.maxContextSize = options.maxContextSize;
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

      const actualOutput = convertToMultiModalArray(testCase.actualOutput);
      const imageIndices = getImageIndices(actualOutput);
      if (imageIndices.length === 0) {
        throw new Error(
          `The test case must have at least one image in the \`actualOutput\` to calculate ${this.name} score`,
        );
      }

      const scores: number[] = [];
      const reasons: string[] = [];
      for (const idx of imageIndices) {
        const [above, below] = getImageContext(
          idx,
          actualOutput,
          this.maxContextSize,
        );
        const image = actualOutput[idx] as MLLMImage;
        const instructions = resolveTemplate(
          "metrics",
          TEMPLATE_CLASS,
          "evaluate_image_helpfulness",
          { context_above: above ?? "", context_below: below ?? "" },
        );
        const { reasoning, score } = await generateWithSchema(
          this,
          `${instructions} \nImages: ${image}`,
          ReasonScoreSchema,
        );
        scores.push(score / 10);
        reasons.push(reasoning);
      }

      const avg = scores.reduce((s, x) => s + x, 0) / scores.length;
      this.score = this.strictMode && avg < this.threshold ? 0 : avg;
      this.reason = reasons
        .map((r, i) => `Reason for image ${i}: ${r}`)
        .join("\n");
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Images scored: ${scores.length}`,
        `Per-image scores: ${scores.map((s) => s.toFixed(2)).join(", ")}`,
        `Score (Average): ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Image Helpfulness";
  }
}
