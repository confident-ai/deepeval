import { z } from "zod";

// Mirrors deepeval/metrics/multimodal_metrics/*/schema.py.

/** Per-image metrics (coherence/helpfulness/reference): a single 0–10 score. */
export const ReasonScoreSchema = z.object({
  reasoning: z.string(),
  score: z.number(),
});

/** Semantic-consistency / perceptual-quality: a list of 0–10 scores. */
export const ListReasonScoreSchema = z.object({
  reasoning: z.string(),
  score: z.array(z.number()),
});
