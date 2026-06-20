import { z } from "zod";

// Mirrors deepeval/metrics/argument_correctness/schema.py.

export const ArgumentCorrectnessVerdictSchema = z.object({
  verdict: z.enum(["yes", "no", "idk"]),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(ArgumentCorrectnessVerdictSchema),
});

export const ArgumentCorrectnessScoreReasonSchema = z.object({
  reason: z.string(),
});

export type ArgumentCorrectnessVerdict = z.infer<
  typeof ArgumentCorrectnessVerdictSchema
>;
