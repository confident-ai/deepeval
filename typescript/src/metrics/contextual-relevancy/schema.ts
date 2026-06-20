import { z } from "zod";

// Mirrors deepeval/metrics/contextual_relevancy/schema.py.

export const ContextualRelevancyVerdictSchema = z.object({
  statement: z.string(),
  verdict: z.string(),
  reason: z.string().nullish(),
});

export const ContextualRelevancyVerdictsSchema = z.object({
  verdicts: z.array(ContextualRelevancyVerdictSchema),
});

export const ContextualRelevancyScoreReasonSchema = z.object({
  reason: z.string(),
});

export type ContextualRelevancyVerdict = z.infer<
  typeof ContextualRelevancyVerdictSchema
>;
export type ContextualRelevancyVerdicts = z.infer<
  typeof ContextualRelevancyVerdictsSchema
>;
