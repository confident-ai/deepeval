import { z } from "zod";

// Mirrors deepeval/metrics/prompt_alignment/schema.py.

export const PromptAlignmentVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(PromptAlignmentVerdictSchema),
});

export const PromptAlignmentScoreReasonSchema = z.object({
  reason: z.string(),
});

export type PromptAlignmentVerdict = z.infer<
  typeof PromptAlignmentVerdictSchema
>;
