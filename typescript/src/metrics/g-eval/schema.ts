import { z } from "zod";

// Mirrors deepeval/metrics/g_eval/schema.py.

export const StepsSchema = z.object({ steps: z.array(z.string()) });

export const ReasonScoreSchema = z.object({
  reason: z.string(),
  score: z.number(),
});
