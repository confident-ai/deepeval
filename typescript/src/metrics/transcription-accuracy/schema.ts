import { z } from "zod";

export const TranscriptionAccuracyVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(TranscriptionAccuracyVerdictSchema),
});

export const TranscriptionAccuracyScoreReasonSchema = z.object({
  reason: z.string(),
});

export type TranscriptionAccuracyVerdict = z.infer<
  typeof TranscriptionAccuracyVerdictSchema
>;
