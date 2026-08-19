import { z } from "zod";

export const TaskNodeOutputSchema = z.object({
  output: z.union([
    z.string(),
    z.array(z.string()),
    z.record(z.string(), z.string()),
  ]),
});

export const BinaryJudgementVerdictSchema = z.object({
  verdict: z.boolean(),
  reason: z.string(),
});

export const MetricScoreReasonSchema = z.object({
  reason: z.string(),
});

/**
 * Constrain the verdict to the options registered on a non-binary judgement
 * node — the TS stand-in for Python's `create_model` + `Literal[tuple(...)]`.
 */
export function nonBinaryVerdictSchema(options: string[]) {
  return z.object({
    verdict: z.enum(options as [string, ...string[]]),
    reason: z.string(),
  });
}

export type TaskNodeOutput = z.infer<typeof TaskNodeOutputSchema>;
export type BinaryJudgementVerdict = z.infer<
  typeof BinaryJudgementVerdictSchema
>;

/** What a judgement node resolves to, regardless of binary vs non-binary. */
export interface JudgementVerdict {
  verdict: string | boolean;
  reason: string;
}
