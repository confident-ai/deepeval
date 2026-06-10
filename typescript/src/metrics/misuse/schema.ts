import { z } from "zod";

// Mirrors deepeval/metrics/misuse/schema.py.

export const MisusesSchema = z.object({ misuses: z.array(z.string()) });

export const MisuseVerdictSchema = z.object({
  verdict: z.enum(["yes", "no"]),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(MisuseVerdictSchema),
});

export const MisuseScoreReasonSchema = z.object({ reason: z.string() });

export type MisuseVerdict = z.infer<typeof MisuseVerdictSchema>;
