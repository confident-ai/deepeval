import { z } from "zod";

// Mirrors deepeval/metrics/conversation_completeness/schema.py.

export const UserIntentionsSchema = z.object({
  intentions: z.array(z.string()),
});

export const ConversationCompletenessVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string().nullish(),
});

export const ConversationCompletenessScoreReasonSchema = z.object({
  reason: z.string(),
});

export type ConversationCompletenessVerdict = z.infer<
  typeof ConversationCompletenessVerdictSchema
>;
