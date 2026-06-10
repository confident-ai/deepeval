import { z } from "zod";

// Mirrors deepeval/metrics/tool_correctness/schema.py.

export const ToolSelectionScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});

export type ToolSelectionScore = z.infer<typeof ToolSelectionScoreSchema>;
