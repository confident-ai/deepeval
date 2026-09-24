// The wire-level vocabulary of a System One model: typed questions about a
// `state`, answered with calibrated probabilities instead of generated text.

export interface NoulQuestion {
  type: "noul";
  instructions: unknown;
  true?: unknown;
  false?: unknown;
}

export interface ChoiceQuestion {
  type: "choice";
  instructions: unknown;
  options: Record<string, unknown>;
}

export interface ScoreQuestion {
  type: "score";
  instructions: unknown;
  levels: unknown[];
}

export type SystemOneQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion;

export class NoulAnswer {
  constructor(public probability: number) {}

  /**
   * How decisive the answer is, on the same scale as the confidence the API
   * returns for Choice and Score answers. TypeSafe derives that as
   * `(n * peak - 1) / (n - 1)`; with the two outcomes of a Noul it reduces to
   * `|2p - 1|`: 0 at `p = 0.5`, 1 at either extreme.
   */
  get confidence(): number {
    return Math.abs(2 * this.probability - 1);
  }
}

export class ChoiceAnswer {
  constructor(
    public choice: string,
    public probabilities: Record<string, number>,
    public confidence: number,
  ) {}
}

export class ScoreAnswer {
  constructor(
    public score: number,
    public probabilities: Record<number, number>,
    public confidence: number,
  ) {}

  get normalized(): number {
    const levels = Object.keys(this.probabilities).map(Number);
    const top = levels.length > 0 ? Math.max(...levels) : 0;
    return top > 0 ? this.score / top : 0;
  }
}

export class SystemOneAnswers {
  nouls: Record<string, NoulAnswer> = {};
  choices: Record<string, ChoiceAnswer> = {};
  scores: Record<string, ScoreAnswer> = {};

  constructor(init?: {
    nouls?: Record<string, NoulAnswer>;
    choices?: Record<string, ChoiceAnswer>;
    scores?: Record<string, ScoreAnswer>;
  }) {
    if (init?.nouls) this.nouls = init.nouls;
    if (init?.choices) this.choices = init.choices;
    if (init?.scores) this.scores = init.scores;
  }

  /**
   * The least decisive answer in the request, or `undefined` when there are
   * none. One question near the fence is enough to make an aggregate over the
   * request untrustworthy, so the minimum rather than the mean is what a
   * confidence gate should look at.
   */
  minConfidence(): number | undefined {
    const values = [
      ...Object.values(this.nouls),
      ...Object.values(this.choices),
      ...Object.values(this.scores),
    ].map((a) => a.confidence);
    return values.length > 0 ? Math.min(...values) : undefined;
  }
}
