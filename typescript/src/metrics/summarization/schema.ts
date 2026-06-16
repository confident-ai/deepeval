import { z } from "zod";

// Mirrors deepeval/metrics/summarization/schema.py.

export const SummarizationAlignmentVerdictSchema = z.object({
  verdict: z.enum(["yes", "no", "idk"]),
  reason: z.string().nullish(),
});

/** Built in code (not from an LLM call) from the per-question answers. */
export interface SummarizationCoverageVerdict {
  summary_verdict: string;
  original_verdict: string;
  question: string;
}

export const VerdictsSchema = z.object({
  verdicts: z.array(SummarizationAlignmentVerdictSchema),
});

export const QuestionsSchema = z.object({ questions: z.array(z.string()) });
export const AnswersSchema = z.object({ answers: z.array(z.string()) });

export const SummarizationScoreReasonSchema = z.object({ reason: z.string() });

export type SummarizationAlignmentVerdict = z.infer<
  typeof SummarizationAlignmentVerdictSchema
>;
