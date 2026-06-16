import { z } from "zod";

// Mirrors deepeval/metrics/turn_relevancy/schema.py.

export const TurnRelevancyVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string().nullish(),
});

export const TurnRelevancyScoreReasonSchema = z.object({
  reason: z.string(),
});

export type TurnRelevancyVerdict = z.infer<typeof TurnRelevancyVerdictSchema>;
