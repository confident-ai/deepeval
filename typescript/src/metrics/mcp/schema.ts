import { z } from "zod";

// Mirrors deepeval/metrics/mcp/schema.py.

export const TaskScoreSchema = z.object({ score: z.number(), reason: z.string() });
export const ToolScoreSchema = z.object({ score: z.number(), reason: z.string() });
export const ArgsScoreSchema = z.object({ score: z.number(), reason: z.string() });
export const ReasonSchema = z.object({ reason: z.string() });

export type TaskScore = z.infer<typeof TaskScoreSchema>;
export type ToolScore = z.infer<typeof ToolScoreSchema>;
export type ArgsScore = z.infer<typeof ArgsScoreSchema>;

/** A user goal + the agent's steps (built in code, fed to the score prompts). */
export interface Task {
  task: string;
  steps_taken: string[];
}
