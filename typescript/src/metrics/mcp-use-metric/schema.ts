import { z } from "zod";

// Mirrors deepeval/metrics/mcp_use_metric/schema.py.

export const MCPPrimitivesScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});

export const MCPArgsScoreSchema = z.object({
  score: z.number(),
  reason: z.string(),
});
