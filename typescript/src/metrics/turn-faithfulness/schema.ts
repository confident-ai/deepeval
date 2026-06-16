import { z } from "zod";

// Mirrors deepeval/metrics/turn_faithfulness/schema.py.

export const TruthsSchema = z.object({ truths: z.array(z.string()) });
export const ClaimsSchema = z.object({ claims: z.array(z.string()) });

export const FaithfulnessVerdictSchema = z.object({
  verdict: z.enum(["yes", "no", "idk"]),
  reason: z.string().nullish(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(FaithfulnessVerdictSchema),
});

export const FaithfulnessScoreReasonSchema = z.object({ reason: z.string() });

export type FaithfulnessVerdict = z.infer<typeof FaithfulnessVerdictSchema>;

/** Per-window interaction result (built in code, not from one LLM call). */
export interface InteractionFaithfulnessScore {
  score: number;
  reason?: string;
  claims: string[];
  truths: string[];
  verdicts: FaithfulnessVerdict[];
}
