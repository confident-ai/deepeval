import { z } from "zod";

// Mirrors deepeval/metrics/hallucination/schema.py.

export const HallucinationVerdictSchema = z.object({
  verdict: z.enum(["yes", "no"]),
  reason: z.string(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(HallucinationVerdictSchema),
});

export const HallucinationScoreReasonSchema = z.object({ reason: z.string() });

export type HallucinationVerdict = z.infer<typeof HallucinationVerdictSchema>;
