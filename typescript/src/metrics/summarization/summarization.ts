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
import { TruthsSchema, ClaimsSchema } from "../faithfulness/schema";
import {
  VerdictsSchema,
  QuestionsSchema,
  AnswersSchema,
  SummarizationScoreReasonSchema,
  type SummarizationAlignmentVerdict,
  type SummarizationCoverageVerdict,
} from "./schema";

const TEMPLATE_CLASS = "SummarizationMetric";

// Borrowed from faithfulness (Python imports `_faithfulness_truths_limit_phrase`).
function truthsLimitPhrase(limit?: number): string {
  if (limit == null) return " FACTUAL, undisputed truths";
  if (limit === 1) return " the single most important FACTUAL, undisputed truth";
  return ` the ${limit} most important FACTUAL, undisputed truths per document`;
}

/** Python `repr()` of a list of strings, for the manually-appended reason prompt. */
function pyListRepr(items: string[]): string {
  return (
    "[" +
    items
      .map((s) => "'" + s.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "'")
      .join(", ") +
    "]"
  );
}

export interface SummarizationMetricOptions {
  threshold?: number;
  /** Number of assessment questions to generate (when none are supplied). */
  n?: number;
  model?: DeepEvalBaseLLM | string;
  /** Pre-supplied yes/no assessment questions (skips question generation). */
  assessmentQuestions?: string[];
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  truthsExtractionLimit?: number;
}

/**
 * Summarization — is the summary (`actualOutput`) both faithful to and covers
 * the original text (`input`)? Combines an **alignment** score (summary claims
 * vs. original truths) and a **coverage** score (assessment Q&A on both), and
 * takes `score = min(alignment, coverage)`. **Higher is better**.
 */
export class SummarizationMetric extends BaseMetric {
  truths: string[] = [];
  claims: string[] = [];
  coverageVerdicts: SummarizationCoverageVerdict[] = [];
  alignmentVerdicts: SummarizationAlignmentVerdict[] = [];
  assessmentQuestions?: string[];
  private readonly n: number;
  private readonly truthsExtractionLimit?: number;

  constructor(options: SummarizationMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
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
    this.n = options.n ?? 5;
    this.assessmentQuestions =
      options.assessmentQuestions && options.assessmentQuestions.length > 0
        ? options.assessmentQuestions
        : undefined;
    this.truthsExtractionLimit =
      options.truthsExtractionLimit != null
        ? Math.max(options.truthsExtractionLimit, 0)
        : undefined;
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

      [this.truths, this.claims] = await Promise.all([
        this.generateTruths(testCase.input),
        this.generateClaims(testCase.actualOutput),
      ]);
      [this.coverageVerdicts, this.alignmentVerdicts] = await Promise.all([
        this.generateCoverageVerdicts(testCase),
        this.generateAlignmentVerdicts(),
      ]);

      const alignmentScore = this.calculateAlignmentScore();
      const coverageScore = this.calculateCoverageScore();
      this.scoreBreakdown = {
        Alignment: alignmentScore,
        Coverage: coverageScore,
      };
      this.score = Math.min(alignmentScore, coverageScore);
      this.reason = await this.generateReason();
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Truths (limit=${this.truthsExtractionLimit}):\n${prettifyList(this.truths)}`,
        `Claims:\n${prettifyList(this.claims)}`,
        `Assessment Questions:\n${prettifyList(this.assessmentQuestions ?? [])}`,
        `Coverage Verdicts:\n${prettifyList(this.coverageVerdicts)}`,
        `Alignment Verdicts:\n${prettifyList(this.alignmentVerdicts)}`,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  // --- truths/claims (borrow Faithfulness templates) ---

  private async generateTruths(text: string): Promise<string[]> {
    const prompt = resolveTemplate("metrics", "FaithfulnessMetric", "generate_truths", {
      retrieval_context: text,
      limit: truthsLimitPhrase(this.truthsExtractionLimit),
      multimodal_instruction: "",
    });
    const { truths } = await generateWithSchema(this, prompt, TruthsSchema);
    return truths;
  }

  private async generateClaims(text: string): Promise<string[]> {
    const prompt = resolveTemplate("metrics", "FaithfulnessMetric", "generate_claims", {
      actual_output: text,
      multimodal_instruction: "",
    });
    const { claims } = await generateWithSchema(this, prompt, ClaimsSchema);
    return claims;
  }

  // --- alignment ---

  private async generateAlignmentVerdicts(): Promise<
    SummarizationAlignmentVerdict[]
  > {
    if (this.claims.length === 0) return [];
    const prompt = resolveTemplate("metrics", 
      TEMPLATE_CLASS,
      "generate_alignment_verdicts",
      {
        summary_claims: this.claims,
        original_text: this.truths.join("\n\n"),
      },
    );
    const { verdicts } = await generateWithSchema(this, prompt, VerdictsSchema);
    return verdicts;
  }

  // --- coverage ---

  private async generateCoverageVerdicts(
    testCase: LLMTestCase,
  ): Promise<SummarizationCoverageVerdict[]> {
    if (this.assessmentQuestions == null) {
      this.assessmentQuestions = await this.generateAssessmentQuestions(
        testCase.input,
      );
    }
    const [originalAnswers, summaryAnswers] = await Promise.all([
      this.generateAnswers(testCase.input),
      this.generateAnswers(testCase.actualOutput),
    ]);
    if (originalAnswers.length !== summaryAnswers.length) {
      throw new Error("Number of verdicts generated does not equal.");
    }
    const coverage: SummarizationCoverageVerdict[] = [];
    for (let i = 0; i < originalAnswers.length; i++) {
      coverage.push({
        summary_verdict: summaryAnswers[i],
        original_verdict: originalAnswers[i],
        question: this.assessmentQuestions[i],
      });
    }
    return coverage;
  }

  private async generateAssessmentQuestions(text: string): Promise<string[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_questions", {
      text,
      n: this.n,
    });
    const { questions } = await generateWithSchema(
      this,
      prompt,
      QuestionsSchema,
    );
    return questions;
  }

  private async generateAnswers(text: string): Promise<string[]> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_answers", {
      questions: this.assessmentQuestions,
      text,
    });
    const { answers } = await generateWithSchema(this, prompt, AnswersSchema);
    return answers;
  }

  // --- reason ---

  private async generateReason(): Promise<string | undefined> {
    if (!this.includeReason) return undefined;
    const contradictions: (string | null | undefined)[] = [];
    const redundancies: (string | null | undefined)[] = [];
    for (const v of this.alignmentVerdicts) {
      const vd = v.verdict.trim().toLowerCase();
      if (vd === "no") contradictions.push(v.reason);
      else if (vd === "idk") redundancies.push(v.reason);
    }
    const questions: string[] = [];
    for (const v of this.coverageVerdicts) {
      if (
        v.original_verdict.trim().toLowerCase() === "yes" &&
        v.summary_verdict.trim().toLowerCase() === "no"
      ) {
        questions.push(v.question);
      }
    }

    let prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "generate_reason", {
      contradictions,
      redundancies,
      questions,
      score: (this.score ?? 0).toFixed(2),
    });
    if (questions.length > 0) {
      prompt += `Questions the original text can answer but not the summary:\n${pyListRepr(questions)}\n\n`;
    }
    prompt += "JSON:\n";

    const { reason } = await generateWithSchema(
      this,
      prompt,
      SummarizationScoreReasonSchema,
    );
    return reason;
  }

  // --- scoring (alignment penalizes idk; coverage = answerable-by-both / answerable-by-original) ---

  private calculateAlignmentScore(): number {
    const total = this.alignmentVerdicts.length;
    if (total === 0) return 0;
    const faithful = this.alignmentVerdicts.filter(
      (v) => v.verdict.trim().toLowerCase() === "yes",
    ).length;
    const score = faithful / total;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  private calculateCoverageScore(): number {
    if (this.assessmentQuestions == null) return 1;
    let total = 0;
    let coverage = 0;
    for (const v of this.coverageVerdicts) {
      if (v.original_verdict.trim().toLowerCase() === "yes") {
        total++;
        if (v.summary_verdict.trim().toLowerCase() === "yes") coverage++;
      }
    }
    if (total === 0) return 0;
    const score = coverage / total;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Summarization";
  }
}
