import { z } from "zod";

export const StatementsSchema = z.object({
  statements: z.array(z.string()),
});

export const AnswerRelevancyVerdictSchema = z.object({
  verdict: z.enum(["yes", "no", "idk"]),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(AnswerRelevancyVerdictSchema),
});

export const AnswerRelevancyScoreReasonSchema = z.object({
  reason: z.string(),
});

export type AnswerRelevancyVerdict = z.infer<typeof AnswerRelevancyVerdictSchema>;
