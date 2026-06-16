import { z } from "zod";

// Mirrors deepeval/metrics/goal_accuracy/schema.py.

export const GoalScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});

export const PlanScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});

export type GoalScore = z.infer<typeof GoalScoreSchema>;
export type PlanScore = z.infer<typeof PlanScoreSchema>;

/** Extracted per-interaction goal + the assistant's accumulated steps. */
export interface GoalSteps {
  user_goal: string;
  steps_taken: string[];
}
