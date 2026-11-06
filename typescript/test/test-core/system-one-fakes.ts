// Test doubles for System One (Jev) and the evaluation LLM, shared by the
// JevEval and eval-mode suites so no test needs a network, an API key or
// `@typesafe-ai/sdk`.

import type { ZodType } from "zod";
import { DeepEvalBaseLLM, type GenerationResult } from "@/models";
import {
  ChoiceAnswer,
  DeepEvalBaseSystemOneModel,
  NoulAnswer,
  ScoreAnswer,
  SystemOneAnswers,
  type SystemOneDecision,
  type SystemOneQuestion,
} from "@/models/system-one";

export type AnswerFn = (
  questions: Record<string, SystemOneQuestion>,
) => SystemOneAnswers;

/**
 * Returns canned answers and records what it was asked. `answers` is one
 * `SystemOneAnswers` (returned on every call) or a list consumed one per
 * call; `answerFn` answers however many questions a metric happens to ask.
 */
export class FakeSystemOneModel extends DeepEvalBaseSystemOneModel {
  calls: Array<{
    state: any;
    questions: Record<string, SystemOneQuestion>;
  }> = [];
  private answers?: SystemOneAnswers | SystemOneAnswers[];
  private answerFn?: AnswerFn;
  private cost: number | null;

  constructor(
    options: {
      answers?: SystemOneAnswers | SystemOneAnswers[];
      answerFn?: AnswerFn;
      cost?: number | null;
      name?: string;
    } = {},
  ) {
    super(options.name ?? "fake-jev");
    this.answers = options.answers;
    this.answerFn = options.answerFn;
    this.cost = options.cost === undefined ? 0 : options.cost;
  }

  getModelName(): string {
    return this.modelName;
  }

  async decide(
    state: unknown,
    questions: Record<string, SystemOneQuestion>,
  ): Promise<SystemOneDecision> {
    this.calls.push({ state, questions });
    return { answers: this.next(questions), cost: this.cost };
  }

  private next(questions: Record<string, SystemOneQuestion>): SystemOneAnswers {
    if (this.answerFn) return this.answerFn(questions);
    if (Array.isArray(this.answers)) {
      const answer = this.answers.shift();
      if (!answer) throw new Error("FakeSystemOneModel ran out of answers.");
      return answer;
    }
    return this.answers ?? new SystemOneAnswers();
  }
}

/**
 * Throws `error` on every call: a Jev outage, an auth failure, a context
 * overflow (pass a `SystemOneContextLimitError`)...
 */
export class ExplodingSystemOneModel extends FakeSystemOneModel {
  constructor(
    private readonly error: unknown,
    name = "exploding-jev",
  ) {
    super({ name });
  }

  async decide(
    state: unknown,
    questions: Record<string, SystemOneQuestion>,
  ): Promise<SystemOneDecision> {
    this.calls.push({ state, questions });
    throw this.error;
  }
}

/** An `answerFn` answering every Noul question with the same P(yes). */
export function noulAnswers(probability: number): AnswerFn {
  return (questions) => {
    const nouls: Record<string, NoulAnswer> = {};
    for (const key of Object.keys(questions)) {
      nouls[key] = new NoulAnswer(probability);
    }
    return new SystemOneAnswers({ nouls });
  };
}

/**
 * An `answerFn` answering any mix of questions: every Noul with
 * `probability`, every Score at its top level, every Choice with its first
 * option.
 */
export function answerEverything(
  probability = 0.9,
  confidence = 0.8,
): AnswerFn {
  return (questions) => {
    const out = new SystemOneAnswers();
    for (const [key, question] of Object.entries(questions)) {
      if (question.type === "noul") {
        out.nouls[key] = new NoulAnswer(probability);
      } else if (question.type === "score") {
        const top = question.levels.length - 1;
        const probabilities: Record<number, number> = {};
        for (let i = 0; i <= top; i++) probabilities[i] = i === top ? 1 : 0;
        out.scores[key] = new ScoreAnswer(top, probabilities, confidence);
      } else {
        const first = Object.keys(question.options)[0];
        out.choices[key] = new ChoiceAnswer(first, { [first]: 1 }, confidence);
      }
    }
    return out;
  };
}

export type Reply = unknown | ((prompt: string) => unknown);

/**
 * The evaluation LLM for the legacy chain. `reply` answers every prompt;
 * a function reply picks the answer from the prompt. Records every prompt.
 */
export class ScriptedLLM extends DeepEvalBaseLLM {
  prompts: string[] = [];

  constructor(
    private readonly reply: Reply,
    private readonly name = "scripted-llm",
  ) {
    super(name);
  }

  async generate<T = string>(
    prompt: string,
    _schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    this.prompts.push(prompt);
    const output =
      typeof this.reply === "function"
        ? (this.reply as (p: string) => unknown)(prompt)
        : this.reply;
    return { output: output as T, cost: 0 };
  }

  getModelName(): string {
    return this.name;
  }
}

/** Fails if any LLM call is made. */
export class ExplodingLLM extends DeepEvalBaseLLM {
  constructor() {
    super("exploding-llm");
  }

  async generate<T = string>(): Promise<GenerationResult<T>> {
    throw new Error("LLM must not be called");
  }

  getModelName(): string {
    return "exploding-llm";
  }
}
