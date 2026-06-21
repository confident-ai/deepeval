import { z } from "zod";

// Mirrors deepeval/metrics/knowledge_retention/schema.py.

export const KnowledgeSchema = z.object({
  data: z
    .record(z.string(), z.union([z.string(), z.array(z.string())]))
    .nullish(),
});

export const KnowledgeRetentionVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string().nullish(),
});

export const KnowledgeRetentionScoreReasonSchema = z.object({
  reason: z.string(),
});

export type Knowledge = z.infer<typeof KnowledgeSchema>;
export type KnowledgeRetentionVerdict = z.infer<
  typeof KnowledgeRetentionVerdictSchema
>;
