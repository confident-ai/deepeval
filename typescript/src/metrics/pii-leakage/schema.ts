import { z } from "zod";

// Mirrors deepeval/metrics/pii_leakage/schema.py.

export const ExtractedPIISchema = z.object({
  extracted_pii: z.array(z.string()),
});

export const PIILeakageVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(PIILeakageVerdictSchema),
});

export const PIILeakageScoreReasonSchema = z.object({ reason: z.string() });

export type PIILeakageVerdict = z.infer<typeof PIILeakageVerdictSchema>;
