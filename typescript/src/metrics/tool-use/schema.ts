import { z } from "zod";

// Mirrors deepeval/metrics/tool_use/schema.py.

export const ToolSelectionScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});

export const ArgumentCorrectnessScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});

export const ReasonSchema = z.object({ reason: z.string() });

export type ToolSelectionScore = z.infer<typeof ToolSelectionScoreSchema>;
export type ArgumentCorrectnessScore = z.infer<
  typeof ArgumentCorrectnessScoreSchema
>;

/** Per-interaction inputs (built in code, fed to the score prompts). */
export interface UserInputAndTools {
  user_messages: string;
  assistant_messages: string;
  tools_called: string;
  available_tools: string;
  tools_used: boolean;
}
