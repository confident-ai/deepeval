import { z } from "zod";

// Mirrors deepeval/metrics/json_correctness/schema.py.

export const JsonCorrectnessScoreReasonSchema = z.object({
  reason: z.string(),
});
