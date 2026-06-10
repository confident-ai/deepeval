import { z } from "zod";

// Mirrors deepeval/metrics/non_advice/schema.py.

export const AdvicesSchema = z.object({ advices: z.array(z.string()) });

export const NonAdviceVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(NonAdviceVerdictSchema),
});

export const NonAdviceScoreReasonSchema = z.object({ reason: z.string() });

export type NonAdviceVerdict = z.infer<typeof NonAdviceVerdictSchema>;
