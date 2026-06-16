import { z } from "zod";

// Mirrors deepeval/metrics/turn_contextual_precision/schema.py.

export const ContextualPrecisionVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(ContextualPrecisionVerdictSchema),
});

export const ContextualPrecisionScoreReasonSchema = z.object({
  reason: z.string(),
});

export type ContextualPrecisionVerdict = z.infer<
  typeof ContextualPrecisionVerdictSchema
>;

export interface InteractionContextualPrecisionScore {
  score: number;
  reason?: string;
  verdicts: ContextualPrecisionVerdict[];
}
