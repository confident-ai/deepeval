import { z } from "zod";

// Mirrors deepeval/metrics/role_adherence/schema.py.

export const OutOfCharacterResponseVerdictSchema = z.object({
  index: z.number().int(),
  reason: z.string(),
  ai_message: z.string().nullish(),
});

export const OutOfCharacterResponseVerdictsSchema = z.object({
  verdicts: z.array(OutOfCharacterResponseVerdictSchema),
});

export const RoleAdherenceScoreReasonSchema = z.object({ reason: z.string() });

export type OutOfCharacterResponseVerdict = z.infer<
  typeof OutOfCharacterResponseVerdictSchema
>;
