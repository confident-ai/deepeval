import type {
  ChoiceAnswer,
  ChoiceQuestion,
  NoulAnswer,
  NoulQuestion,
  ScoreAnswer,
  ScoreQuestion,
  SystemOneAnswers,
  SystemOneQuestion,
} from "@/models/system-one/schema";

export interface SystemOneDecision {
  answers: SystemOneAnswers;
  cost: number | null;
}

/**
 * A model that answers typed questions about a `state` with calibrated
 * probabilities instead of generating text. Metrics use it for decisions
 * only; extraction and reasons stay on the LLM. Separate from
 * `DeepEvalBaseLLM` on purpose: it cannot generate, and an LLM cannot decide.
 */
export abstract class DeepEvalBaseSystemOneModel {
  modelName: string;

  constructor(modelName: string) {
    this.modelName = modelName;
  }

  abstract decide(
    state: unknown,
    questions: Record<string, SystemOneQuestion>,
  ): Promise<SystemOneDecision>;

  abstract getModelName(): string;

  async noul(
    state: unknown,
    questions: Record<string, NoulQuestion>,
  ): Promise<{ answers: Record<string, NoulAnswer>; cost: number | null }> {
    const { answers, cost } = await this.decide(state, questions);
    return { answers: answers.nouls, cost };
  }

  async choice(
    state: unknown,
    questions: Record<string, ChoiceQuestion>,
  ): Promise<{ answers: Record<string, ChoiceAnswer>; cost: number | null }> {
    const { answers, cost } = await this.decide(state, questions);
    return { answers: answers.choices, cost };
  }

  async score(
    state: unknown,
    questions: Record<string, ScoreQuestion>,
  ): Promise<{ answers: Record<string, ScoreAnswer>; cost: number | null }> {
    const { answers, cost } = await this.decide(state, questions);
    return { answers: answers.scores, cost };
  }
}
