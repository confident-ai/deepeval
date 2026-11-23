// The three decision points a JevEval metric is built from.
//
// Each question is answered by Jev with calibrated probabilities, never text,
// and each has a fixed mapping onto a value in [0, 1] (see utils.ts):
//
// - `Noul`: one proposition about the test case, v = P(true).
// - `Score`: one holistic judgement over ordered descriptive levels,
//   v = expected level index / (levels - 1).
// - `Choice`: one selection from an unordered closed set where every option
//   carries a credit in [0, 1], or `null` when that option means the question
//   does not apply to this test case.

import { z } from "zod";
import type {
  ChoiceQuestion,
  NoulQuestion,
  ScoreQuestion,
} from "@/models/system-one/schema";

/** Jev accepts at most this many ordered levels in one Score question. */
export const MAX_SCORE_LEVELS = 10;
export const MIN_SCORE_LEVELS = 2;

const nonEmpty = (label: string) =>
  z.string().refine((value) => value.trim().length > 0, {
    message: `${label} cannot be empty.`,
  });

const weight = z
  .number()
  .gt(0, { message: "weight must be greater than 0." })
  .default(1);

const NoulSchema = z.object({
  statement: nonEmpty("Noul statement"),
  weight,
  true: z.unknown().optional(),
  false: z.unknown().optional(),
});

const ScoreSchema = z.object({
  question: nonEmpty("Score question"),
  weight,
  levels: z
    .array(z.string())
    .superRefine((levels, ctx) => {
      if (
        levels.length < MIN_SCORE_LEVELS ||
        levels.length > MAX_SCORE_LEVELS
      ) {
        ctx.addIssue({
          code: "custom",
          message:
            `Score needs between ${MIN_SCORE_LEVELS} and ${MAX_SCORE_LEVELS} ` +
            `levels; got ${levels.length}.`,
        });
      }
    })
    .refine((levels) => levels.every((level) => level.trim().length > 0), {
      message: "Score levels cannot be empty.",
    })
    .refine((levels) => new Set(levels).size === levels.length, {
      message: "Score levels must be unique.",
    }),
});

const ChoiceSchema = z.object({
  question: nonEmpty("Choice question"),
  weight,
  options: z
    .record(z.string(), z.number().nullable())
    .superRefine((options, ctx) => {
      const entries = Object.entries(options);
      if (entries.length < 2) {
        ctx.addIssue({
          code: "custom",
          message: "Choice needs at least 2 options.",
        });
        return;
      }
      if (entries.some(([name]) => name.trim().length === 0)) {
        ctx.addIssue({
          code: "custom",
          message: "Choice option names cannot be empty.",
        });
      }
      for (const [name, credit] of entries) {
        if (credit !== null && (credit < 0 || credit > 1)) {
          ctx.addIssue({
            code: "custom",
            message:
              `Choice option '${name}' has credit ${credit}; credits must be ` +
              "between 0 and 1, or null for not applicable.",
          });
        }
      }
      if (entries.every(([, credit]) => credit === null)) {
        ctx.addIssue({
          code: "custom",
          message:
            "Choice needs at least one option with a credit; every option " +
            "is null (not applicable).",
        });
      }
    }),
});

function parse<T>(schema: z.ZodType<T>, input: unknown): T {
  const result = schema.safeParse(input);
  if (!result.success) {
    throw new Error(result.error.issues.map((i) => i.message).join(" "));
  }
  return result.data;
}

/** A proposition Jev judges true or false about the state. */
export class Noul {
  readonly type = "noul" as const;
  readonly statement: string;
  readonly weight: number;
  readonly true?: unknown;
  readonly false?: unknown;

  constructor(options: {
    statement: string;
    weight?: number;
    true?: unknown;
    false?: unknown;
  }) {
    const data = parse(NoulSchema, options);
    this.statement = data.statement;
    this.weight = data.weight;
    this.true = data.true ?? undefined;
    this.false = data.false ?? undefined;
  }

  get text(): string {
    return this.statement;
  }

  toSystemOne(): NoulQuestion {
    return {
      type: "noul",
      instructions: this.statement,
      true: this.true,
      false: this.false,
    };
  }
}

/** A holistic judgement over ordered descriptive levels, worst first. */
export class Score {
  readonly type = "score" as const;
  readonly question: string;
  readonly weight: number;
  readonly levels: string[];

  constructor(options: {
    question: string;
    levels: string[];
    weight?: number;
  }) {
    const data = parse(ScoreSchema, options);
    this.question = data.question;
    this.weight = data.weight;
    this.levels = [...data.levels];
  }

  get text(): string {
    return this.question;
  }

  toSystemOne(): ScoreQuestion {
    return { type: "score", instructions: this.question, levels: this.levels };
  }
}

/**
 * A selection from a closed set. `options` maps each option name to the
 * credit it earns in [0, 1]; `null` marks an option that means the question
 * does not apply to this test case.
 */
export class Choice {
  readonly type = "choice" as const;
  readonly question: string;
  readonly weight: number;
  readonly options: Record<string, number | null>;

  constructor(options: {
    question: string;
    options: Record<string, number | null>;
    weight?: number;
  }) {
    const data = parse(ChoiceSchema, options);
    this.question = data.question;
    this.weight = data.weight;
    this.options = { ...data.options };
  }

  get text(): string {
    return this.question;
  }

  get applicableOptions(): Record<string, number> {
    const applicable: Record<string, number> = {};
    for (const [name, credit] of Object.entries(this.options)) {
      if (credit !== null) applicable[name] = credit;
    }
    return applicable;
  }

  get notApplicableOptions(): string[] {
    return Object.entries(this.options)
      .filter(([, credit]) => credit === null)
      .map(([name]) => name);
  }

  toSystemOne(): ChoiceQuestion {
    // Credits stay on the DeepEval side; Jev only sees the option names.
    const options: Record<string, null> = {};
    for (const name of Object.keys(this.options)) options[name] = null;
    return { type: "choice", instructions: this.question, options };
  }
}

export type JevQuestion = Noul | Score | Choice;

/** One entry of `metric.scoreBreakdown`. */
export interface QuestionOutcome {
  question: string;
  type: "noul" | "score" | "choice";
  weight: number;
  value: number | null;
  applicable: boolean;
  probabilities: Record<string, number>;
  confidence: number | null;
  /** Only set under strict mode: did this question meet the strict bar? */
  passed: boolean | null;
}
