// Shared core for `JevEval` and `ConversationalJevEval`, and for the
// whole-chain System One path built-in metrics take under `system_one` eval
// mode (`@/metrics/system-one/runner`).
//
// The metric is one Jev `decide()` call followed by one equation:
//
//     score = sum(w_i * v_i) / sum(w_i)   over applicable questions
//
// where v_i is the value each primitive's answer maps onto in [0, 1].
// Everything here is pure: no LLM, no network.

import { DeepEvalError } from "@/errors";
import { DeepEvalBaseSystemOneModel } from "@/models/system-one/base-system-one-model";
import { TypeSafeModel } from "@/models/system-one/typesafe-model";
import type {
  SystemOneAnswers,
  SystemOneQuestion,
} from "@/models/system-one/schema";
import {
  ConversationalTestCase,
  LLMTestCase,
  MultiTurnParams,
  SingleTurnParams,
} from "@/test-case";
import {
  Choice,
  Noul,
  Score,
  type JevQuestion,
  type QuestionOutcome,
} from "@/metrics/jev-eval/questions";

/**
 * A Choice whose "not applicable" options collect at least this much mass is
 * dropped from the metric for that test case.
 */
export const NOT_APPLICABLE_THRESHOLD = 0.5;

/**
 * Score when nothing applied: there was nothing applicable to fail. Matches
 * the empty-score convention of the QAG helpers.
 */
export const EMPTY_SCORE = 1.0;

///////////////////////////////////////////////
// Model
///////////////////////////////////////////////

/** JevEval needs Jev in every mode; there is no LLM to fall back to. */
export function initializeJevModel(
  model?: DeepEvalBaseSystemOneModel | string,
): DeepEvalBaseSystemOneModel {
  if (model instanceof DeepEvalBaseSystemOneModel) return model;
  if (model !== undefined && typeof model !== "string") {
    throw new TypeError(
      "Unsupported type for systemOneModel. Expected undefined, a string, " +
        "or a DeepEvalBaseSystemOneModel.",
    );
  }
  try {
    return new TypeSafeModel({ model });
  } catch (e) {
    throw new DeepEvalError(
      `JevEval needs Jev to score, but it is not usable: ${(e as Error).message} ` +
        "Pass a configured `TypeSafeModel` as `systemOneModel`, or install " +
        "`@typesafe-ai/sdk` and set TYPESAFE_API_KEY.",
    );
  }
}

///////////////////////////////////////////////
// Questions
///////////////////////////////////////////////

export function validateQuestions(
  questions: readonly unknown[] | undefined,
): JevQuestion[] {
  if (!questions || questions.length === 0) {
    throw new Error(
      "JevEval needs at least one question (Noul, Score or Choice).",
    );
  }
  for (const question of questions) {
    if (
      !(
        question instanceof Noul ||
        question instanceof Score ||
        question instanceof Choice
      )
    ) {
      throw new TypeError(
        `Questions must be Noul, Score or Choice; got ${
          (question as object)?.constructor?.name ?? typeof question
        }.`,
      );
    }
  }
  return [...(questions as JevQuestion[])];
}

export function questionKey(index: number): string {
  return `q_${index}`;
}

export function buildQuestions(
  questions: readonly JevQuestion[],
): Record<string, SystemOneQuestion> {
  const built: Record<string, SystemOneQuestion> = {};
  questions.forEach((question, i) => {
    built[questionKey(i)] = question.toSystemOne();
  });
  return built;
}

///////////////////////////////////////////////
// Answers -> values
///////////////////////////////////////////////

export function valueFromAnswer(
  question: JevQuestion,
  answers: SystemOneAnswers,
  key: string,
): QuestionOutcome {
  if (question instanceof Noul) {
    const answer = answers.nouls[key];
    if (!answer)
      throw new DeepEvalError(`Jev returned no Noul answer for ${key}.`);
    const p = Number(answer.probability);
    return {
      question: question.text,
      type: "noul",
      weight: question.weight,
      value: p,
      applicable: true,
      probabilities: { true: p, false: 1 - p },
      confidence: answer.confidence,
      passed: null,
    };
  }

  if (question instanceof Score) {
    const answer = answers.scores[key];
    if (!answer) {
      throw new DeepEvalError(`Jev returned no Score answer for ${key}.`);
    }
    const top = question.levels.length - 1;
    const probabilities: Record<string, number> = {};
    for (const [level, p] of Object.entries(answer.probabilities)) {
      const index = Number(level);
      if (index >= 0 && index <= top) {
        probabilities[question.levels[index]] = Number(p);
      }
    }
    return {
      question: question.text,
      type: "score",
      weight: question.weight,
      value: Math.min(Math.max(Number(answer.score) / top, 0), 1),
      applicable: true,
      probabilities,
      confidence: answer.confidence,
      passed: null,
    };
  }

  const answer = answers.choices[key];
  if (!answer) {
    throw new DeepEvalError(`Jev returned no Choice answer for ${key}.`);
  }
  const probabilities: Record<string, number> = {};
  for (const name of Object.keys(question.options)) {
    probabilities[name] = Number(answer.probabilities[name] ?? 0);
  }
  const naMass = question.notApplicableOptions.reduce(
    (sum, name) => sum + probabilities[name],
    0,
  );
  let applicable = naMass < NOT_APPLICABLE_THRESHOLD;
  let value: number | null = null;
  if (applicable) {
    const credits = question.applicableOptions;
    const mass = Object.keys(credits).reduce(
      (sum, name) => sum + probabilities[name],
      0,
    );
    if (mass > 0) {
      value =
        Object.entries(credits).reduce(
          (sum, [name, credit]) => sum + probabilities[name] * credit,
          0,
        ) / mass;
    } else {
      applicable = false;
    }
  }
  return {
    question: question.text,
    type: "choice",
    weight: question.weight,
    value,
    applicable,
    probabilities,
    confidence: answer.confidence,
    passed: null,
  };
}

export function outcomesFromAnswers(
  questions: readonly JevQuestion[],
  answers: SystemOneAnswers,
): QuestionOutcome[] {
  return questions.map((question, i) =>
    valueFromAnswer(question, answers, questionKey(i)),
  );
}

export function aggregate(outcomes: readonly QuestionOutcome[]): number {
  const applicable = outcomes.filter((o) => o.applicable && o.value !== null);
  if (applicable.length === 0) return EMPTY_SCORE;
  const totalWeight = applicable.reduce((sum, o) => sum + o.weight, 0);
  return (
    applicable.reduce((sum, o) => sum + o.weight * (o.value as number), 0) /
    totalWeight
  );
}

///////////////////////////////////////////////
// Strict mode
///////////////////////////////////////////////
//
// Strict mode keeps deepeval's contract: the score is 1 for perfection and 0
// otherwise. "Perfection" is every applicable question answered in its best
// possible way, decided from Jev's probabilities without an LLM.

export const STRICT_NOUL_THRESHOLD = 0.5;

function argmax(probabilities: Record<string, number>): string | undefined {
  let best: string | undefined;
  let bestP = -Infinity;
  for (const [name, p] of Object.entries(probabilities)) {
    if (p > bestP) {
      best = name;
      bestP = p;
    }
  }
  return best;
}

export function passesStrictly(
  question: JevQuestion,
  outcome: QuestionOutcome,
): boolean {
  if (question instanceof Noul) {
    return (outcome.probabilities.true ?? 0) >= STRICT_NOUL_THRESHOLD;
  }
  if (question instanceof Score) {
    return argmax(outcome.probabilities) === question.levels.at(-1);
  }
  const credits = question.applicableOptions;
  const scoped: Record<string, number> = {};
  for (const name of Object.keys(credits)) {
    scoped[name] = outcome.probabilities[name] ?? 0;
  }
  const chosen = argmax(scoped);
  return chosen !== undefined && credits[chosen] >= 1;
}

/** Record, per applicable question, whether it met the strict bar. */
export function markStrict(
  questions: readonly JevQuestion[],
  outcomes: readonly QuestionOutcome[],
): QuestionOutcome[] {
  return outcomes.map((outcome, i) => ({
    ...outcome,
    passed: outcome.applicable ? passesStrictly(questions[i], outcome) : null,
  }));
}

export function aggregateStrict(outcomes: readonly QuestionOutcome[]): number {
  const applicable = outcomes.filter((o) => o.applicable);
  if (applicable.length === 0) return EMPTY_SCORE;
  return applicable.every((o) => o.passed) ? 1 : 0;
}

/**
 * The least decisive answer across the questions. Choice and Score carry the
 * API's confidence; a Noul's is derived (`|2p - 1|`), so every outcome
 * contributes.
 */
export function minConfidence(
  outcomes: readonly QuestionOutcome[],
): number | null {
  const values = outcomes
    .map((o) => o.confidence)
    .filter((c): c is number => c !== null && c !== undefined);
  return values.length > 0 ? Math.min(...values) : null;
}

///////////////////////////////////////////////
// Test case -> state
///////////////////////////////////////////////
//
// State keys are the Python parameter names (`actual_output`, not
// `actualOutput`): the questions shipped in the template bundle name them, so
// both SDKs send Jev the same state.

export function snakeCase(key: string): string {
  return key.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

function isPopulated(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string" || Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object" && isPlainObject(value)) {
    return Object.keys(value as object).length > 0;
  }
  return true;
}

function isPlainObject(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * A value as Jev reads it. A class instance (a `ToolCall`, a
 * `RetrievedContextData`) becomes a structured object with snake_case keys
 * and no empty fields, the TS analogue of Python's
 * `model_dump(mode="json", exclude_none=True)`; user-supplied plain objects
 * keep their keys.
 */
export function jsonable(value: unknown): unknown {
  if (
    value === null ||
    value === undefined ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (Array.isArray(value)) return value.map(jsonable);
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "object") {
    const plain = isPlainObject(value);
    const out: Record<string, unknown> = {};
    for (const [key, v] of Object.entries(value as Record<string, unknown>)) {
      if (!plain && (v === undefined || v === null)) continue;
      if (typeof v === "function") continue;
      out[plain ? key : snakeCase(key)] = jsonable(v);
    }
    return out;
  }
  return String(value);
}

export function constructSingleTurnState(
  evaluationParams: readonly SingleTurnParams[],
  testCase: LLMTestCase,
): Record<string, any> {
  const fields: Record<string, unknown> = {};
  for (const param of evaluationParams) {
    const value = (testCase as unknown as Record<string, unknown>)[param];
    if (isPopulated(value)) fields[snakeCase(param)] = jsonable(value);
  }
  return { test_case: fields };
}

const TURN_LEVEL_PARAMS: readonly MultiTurnParams[] = [
  MultiTurnParams.ROLE,
  MultiTurnParams.CONTENT,
  MultiTurnParams.RETRIEVAL_CONTEXT,
  MultiTurnParams.TOOLS_CALLED,
  MultiTurnParams.MCP_TOOLS,
  MultiTurnParams.MCP_RESOURCES,
  MultiTurnParams.MCP_PROMPTS,
];

export function constructMultiTurnState(
  evaluationParams: readonly MultiTurnParams[],
  testCase: ConversationalTestCase,
): Record<string, any> {
  const turnParams = evaluationParams.filter((p) =>
    TURN_LEVEL_PARAMS.includes(p),
  );
  const turns = testCase.turns.map((turn) => {
    const entry: Record<string, unknown> = {};
    for (const param of turnParams) {
      const value = (turn as unknown as Record<string, unknown>)[param];
      if (value !== undefined && value !== null) {
        entry[snakeCase(param)] = jsonable(value);
      }
    }
    return entry;
  });
  const fields: Record<string, unknown> = {};
  for (const param of evaluationParams) {
    if (TURN_LEVEL_PARAMS.includes(param)) continue;
    const value = (testCase as unknown as Record<string, unknown>)[param];
    if (isPopulated(value)) fields[snakeCase(param)] = jsonable(value);
  }
  const state: Record<string, unknown> = { turns };
  if (Object.keys(fields).length > 0) state.test_case = fields;
  return state;
}

///////////////////////////////////////////////
// Verbalised outcomes (for the deterministic reason)
///////////////////////////////////////////////

const RUNNER_UP_MARGIN = 0.15;

function verbaliseNoul(p: number): string {
  if (p >= 0.85) return "clearly holds";
  if (p >= 0.65) return "likely holds";
  if (p >= 0.35) return "unclear";
  if (p >= 0.15) return "likely fails";
  return "clearly fails";
}

function verbaliseRanked(probabilities: Record<string, number>): string {
  // Stable sort, so ties keep insertion order as Python's `sorted` does.
  const ranked = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  if (ranked.length === 0) return "no answer";
  const [top, topP] = ranked[0];
  if (ranked.length > 1) {
    const [runner, runnerP] = ranked[1];
    if (topP - runnerP <= RUNNER_UP_MARGIN) {
      return `"${top}", leaning "${runner}"`;
    }
  }
  return `"${top}"`;
}

export function verbaliseOutcome(outcome: QuestionOutcome): string {
  if (outcome.type === "noul") {
    return verbaliseNoul(outcome.probabilities.true ?? 0);
  }
  if (!outcome.applicable) return "not applicable";
  return verbaliseRanked(outcome.probabilities);
}

/** Python's `f"{x:g}"` for the weights that appear in reasons and logs. */
export function formatG(value: number): string {
  return String(Number(value.toPrecision(6)));
}

function pythonFloat(value: number): string {
  return Number.isInteger(value) ? `${value}.0` : String(value);
}

export function formatOutcomesForLogs(
  outcomes: readonly QuestionOutcome[],
): string {
  return outcomes
    .map((outcome, i) => {
      const value = outcome.value === null ? "n/a" : outcome.value.toFixed(3);
      const probabilities = Object.entries(outcome.probabilities)
        .map(([k, v]) => `'${k}': ${pythonFloat(Math.round(v * 1000) / 1000)}`)
        .join(", ");
      let line =
        `${i + 1}. [${outcome.type}, weight=${formatG(outcome.weight)}] ` +
        `${outcome.question}\n   value=${value} ` +
        `applicable=${outcome.applicable ? "True" : "False"} ` +
        `probabilities={${probabilities}}`;
      if (outcome.confidence !== null) {
        line += ` confidence=${outcome.confidence.toFixed(3)}`;
      }
      if (outcome.passed !== null) {
        line += ` strict_pass=${outcome.passed ? "True" : "False"}`;
      }
      return line;
    })
    .join("\n");
}
