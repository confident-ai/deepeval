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
import { separateImagesFromText } from "../utils";
import { ListReasonScoreSchema } from "../schema";

const TEMPLATE_CLASS = "ImageEditingMetric";

export interface ImageEditingMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Image Editing — does the edited (output) image reflect the editing
 * instruction applied to the input image, and is it high quality? Combines
 * semantic consistency (instruction + input image ↔ output image) and
 * perceptual quality as `sqrt(min(SC) · min(PQ)) / 10`. **Higher is better**.
 */
export class ImageEditingMetric extends BaseMetric {
  constructor(options: ImageEditingMetricOptions = {}) {
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

      const [inputTexts, inputImages] = separateImagesFromText(
        convertToMultiModalArray(testCase.input),
      );
      const [, outputImages] = separateImagesFromText(
        convertToMultiModalArray(testCase.actualOutput),
      );
      if (outputImages.length === 0) {
        throw new Error(
          `The test case must have at least one image in the \`actualOutput\` to calculate ${this.name} score`,
        );
      }
      const outputImage = outputImages[0];

      // Editing also feeds the original (input) image into semantic consistency.
      const scImages = inputImages[0]
        ? [inputImages[0], outputImage]
        : [outputImage];
      const [scScores, scReasoning] = await this.evaluateSemanticConsistency(
        inputTexts.join("\n"),
        scImages,
      );
      const [pqScores, pqReasoning] =
        await this.evaluatePerceptualQuality(outputImage);

      const score =
        Math.sqrt(Math.min(...scScores) * Math.min(...pqScores)) / 10;
      this.score = this.strictMode && score < this.threshold ? 0 : score;
      this.reason =
        `The overall score is ${this.score.toFixed(2)} because the lowest semantic consistency score was ` +
        `${Math.min(...scScores)} and the lowest perceptual quality score was ${Math.min(...pqScores)}.\n` +
        `Reason for Semantic Consistency score: ${scReasoning}\n` +
        `Reason for Perceptual Quality score: ${pqReasoning}`;
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Semantic Consistency Scores:\n${scScores.join(", ")}`,
        `Semantic Consistency Reasoning:\n${scReasoning}`,
        `Perceptual Quality Scores:\n${pqScores.join(", ")}`,
        `Perceptual Quality Reasoning:\n${pqReasoning}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async evaluateSemanticConsistency(
    textPrompt: string,
    images: MLLMImage[],
  ): Promise<[number[], string]> {
    const instructions = resolveTemplate(
      "metrics",
      TEMPLATE_CLASS,
      "generate_semantic_consistency_evaluation_results",
      { text_prompt: textPrompt },
    );
    const { score, reasoning } = await generateWithSchema(
      this,
      `${instructions}\n${images.map((i) => i.toString()).join("\n")}`,
      ListReasonScoreSchema,
    );
    return [score, reasoning];
  }

  private async evaluatePerceptualQuality(
    outputImage: MLLMImage,
  ): Promise<[number[], string]> {
    const instructions = resolveTemplate(
      "metrics",
      TEMPLATE_CLASS,
      "generate_perceptual_quality_evaluation_results",
      {},
    );
    const { score, reasoning } = await generateWithSchema(
      this,
      `${instructions}\n${outputImage}`,
      ListReasonScoreSchema,
    );
    return [score, reasoning];
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Image Editing";
  }
}
