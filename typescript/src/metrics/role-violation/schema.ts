import { z } from "zod";

// Mirrors deepeval/metrics/role_violation/schema.py.

export const RoleViolationsSchema = z.object({
  role_violations: z.array(z.string()),
});

export const RoleViolationVerdictSchema = z.object({
  verdict: z.string(),
  reason: z.string(),
});

export const VerdictsSchema = z.object({
  verdicts: z.array(RoleViolationVerdictSchema),
});

export const RoleViolationScoreReasonSchema = z.object({ reason: z.string() });

export type RoleViolationVerdict = z.infer<typeof RoleViolationVerdictSchema>;
