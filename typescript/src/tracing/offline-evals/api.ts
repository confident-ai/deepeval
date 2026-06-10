import { z } from "zod";

export const EvaluateThreadRequestBodySchema = z.object({
  metricCollection: z.string(),
  chatbotRole: z.string().optional(),
  overwriteMetrics: z.boolean().optional(),
});

export const EvaluateTraceRequestBodySchema = z.object({
  metricCollection: z.string(),
  overwriteMetrics: z.boolean().optional(),
});

export const EvaluateSpanRequestBodySchema = z.object({
  metricCollection: z.string(),
  overwriteMetrics: z.boolean().optional(),
});

export type EvaluateThreadRequestBody = z.infer<
  typeof EvaluateThreadRequestBodySchema
>;
export type EvaluateTraceRequestBody = z.infer<
  typeof EvaluateTraceRequestBodySchema
>;
export type EvaluateSpanRequestBody = z.infer<
  typeof EvaluateSpanRequestBodySchema
>;
