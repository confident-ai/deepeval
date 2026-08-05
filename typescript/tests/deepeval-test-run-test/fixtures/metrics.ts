import { metrics, testCase } from "deepeval";

const { SingleTurnParams, MultiTurnParams } = testCase;
const MODEL = "gpt-4o-mini";

// --- single-turn ---
export const answerRelevancy = () =>
  new metrics.AnswerRelevancyMetric({ threshold: 0.5, model: MODEL });

export const faithfulness = () =>
  new metrics.FaithfulnessMetric({ threshold: 0.5, model: MODEL });

export const contextualRelevancy = () =>
  new metrics.ContextualRelevancyMetric({ threshold: 0.3, model: MODEL });

export const correctness = () =>
  new metrics.GEval({
    name: "Correctness",
    criteria:
      "Whether the actual output is factually consistent with the expected output.",
    evaluationParams: [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold: 0.5,
    model: MODEL,
  });

// --- multi-turn ---
export const turnRelevancy = () =>
  new metrics.TurnRelevancyMetric({ threshold: 0.5, model: MODEL });

export const professionalism = () =>
  new metrics.ConversationalGEval({
    name: "Professionalism",
    criteria:
      "Whether the assistant stays polite, helpful and professional across the conversation.",
    evaluationParams: [MultiTurnParams.CONTENT],
    threshold: 0.5,
    model: MODEL,
  });

export const roleAdherence = () =>
  new metrics.RoleAdherenceMetric({ threshold: 0.5, model: MODEL });
