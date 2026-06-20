import { z } from "zod";

// Mirrors deepeval/metrics/plan_adherence/schema.py.

export const TaskSchema = z.object({ task: z.string() });
export const AgentPlanSchema = z.object({ plan: z.array(z.string()) });
export const PlanAdherenceScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});
