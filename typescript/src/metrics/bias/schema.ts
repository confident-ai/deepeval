import { z } from "zod";

// Mirrors deepeval/metrics/bias/schema.py.

export const OpinionsSchema = z.object({ opinions: z.array(z.string()) });

export const BiasVerdictSchema = z.object({
  verdict: z.enum(["yes", "no"]),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(BiasVerdictSchema),
});

export const BiasScoreReasonSchema = z.object({ reason: z.string() });

export type BiasVerdict = z.infer<typeof BiasVerdictSchema>;
