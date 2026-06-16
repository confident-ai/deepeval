import { z } from "zod";

// Mirrors deepeval/metrics/topic_adherence/schema.py.

export const QAPairSchema = z.object({
  question: z.string(),
  response: z.string(),
});

export const QAPairsSchema = z.object({ qa_pairs: z.array(QAPairSchema) });

export const RelevancyVerdictSchema = z.object({
  verdict: z.enum(["TP", "TN", "FP", "FN"]),
  reason: z.string(),
});

export const TopicAdherenceReasonSchema = z.object({ reason: z.string() });

export type QAPair = z.infer<typeof QAPairSchema>;
export type RelevancyVerdict = z.infer<typeof RelevancyVerdictSchema>;
