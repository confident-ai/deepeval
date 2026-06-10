import { z } from "zod";

// Mirrors deepeval/metrics/contextual_recall/schema.py.

export const ContextualRecallVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(ContextualRecallVerdictSchema),
});

export const ContextualRecallScoreReasonSchema = z.object({
  reason: z.string(),
});

export type ContextualRecallVerdict = z.infer<
  typeof ContextualRecallVerdictSchema
>;
