import { z } from "zod";

// Mirrors deepeval/metrics/argument_relevancy/schema.py.

export const ArgumentRelevancyVerdictSchema = z.object({
  verdict: z.enum(["yes", "no", "idk"]),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(ArgumentRelevancyVerdictSchema),
});

export const ArgumentRelevancyScoreReasonSchema = z.object({
  reason: z.string(),
});

export type ArgumentRelevancyVerdict = z.infer<
  typeof ArgumentRelevancyVerdictSchema
>;
