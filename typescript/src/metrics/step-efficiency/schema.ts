import { z } from "zod";

// Mirrors deepeval/metrics/step_efficiency/schema.py.

export const TaskSchema = z.object({ task: z.string() });

export const EfficiencyVerdictSchema = z.object({
  score: z.number(),
  reason: z.string(),
});
