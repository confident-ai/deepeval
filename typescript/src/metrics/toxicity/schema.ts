import { z } from "zod";

// Mirrors deepeval/metrics/toxicity/schema.py.

export const OpinionsSchema = z.object({ opinions: z.array(z.string()) });

export const ToxicityVerdictSchema = z.object({
  verdict: z.enum(["yes", "no"]),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(ToxicityVerdictSchema),
});

export const ToxicityScoreReasonSchema = z.object({ reason: z.string() });

export type ToxicityVerdict = z.infer<typeof ToxicityVerdictSchema>;
