import { z } from "zod";

// Mirrors deepeval/metrics/arena_g_eval/schema.py (Steps is reused from g-eval).

export const WinnerSchema = z.object({
  winner: z.string(),
  reason: z.string(),
});

export const RewrittenReasonSchema = z.object({
  rewritten_reason: z.string(),
});
