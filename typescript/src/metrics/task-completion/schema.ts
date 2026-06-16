import { z } from "zod";

// Mirrors deepeval/metrics/task_completion/schema.py.

export const TaskAndOutcomeSchema = z.object({
  task: z.string(),
  outcome: z.string(),
});

export const TaskCompletionVerdictSchema = z.object({
  verdict: z.number(),
  reason: z.string().nullish(),
});
