// Port of `deepeval/telemetry/api.py`. Evaluations capture on the way out, once
// the totals and outcome are known, so a hard crash mid-run loses the event.
// Python's context managers become callbacks here, since that is what
// `AsyncLocalStorage` can bound; each takes a sync or async body.

import * as crypto from "crypto";

import { Integration } from "@/tracing/integrations";
import {
  capture,
  registerIntegration,
  telemetryOptOut,
} from "@/telemetry/client";
import {
  pushAmbientRun,
  withRun,
  type RunAccumulator,
} from "@/telemetry/context";
import { Entrypoint, Event, Feature } from "@/telemetry/events";
import {
  getFeatureStatus,
  getLastFeature,
  setLastFeature,
} from "@/telemetry/identity";
import {
  UNKNOWN_CLI_COMMAND,
  LoginMethod,
  LoginOutcome,
  LoginPromptSurface,
  Outcome,
  type EventProperties,
} from "@/telemetry/properties";

const ENTRYPOINT_FEATURE: Partial<Record<Entrypoint, Feature>> = {
  [Entrypoint.EVALUATE]: Feature.EVALUATION,
  [Entrypoint.EVALS_ITERATOR]: Feature.COMPONENT_EVALUATION,
  [Entrypoint.VITEST]: Feature.EVALUATION,
  [Entrypoint.COMPARE]: Feature.EVALUATION,
  [Entrypoint.STANDALONE]: Feature.EVALUATION,
};

function isPromise(value: unknown): value is Promise<unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { then?: unknown }).then === "function"
  );
}

/** Run `body`, then `onSettled`, sync or async, thrown or not. */
function withSettle<T>(body: () => T, onSettled: (error?: unknown) => void): T {
  let result: T;
  try {
    result = body();
  } catch (error) {
    onSettled(error);
    throw error;
  }
  if (!isPromise(result)) {
    onSettled();
    return result;
  }
  return result.then(
    (value) => {
      onSettled();
      return value;
    },
    (error: unknown) => {
      onSettled(error);
      throw error;
    },
  ) as T;
}

/** Class name only. Messages routinely carry prompts, keys, org names. */
function errorTypeOf(error: unknown): string {
  if (error instanceof Error) return error.constructor.name;
  return typeof error;
}

export interface EvaluationRunOptions {
  /**
   * Generated per scope unless supplied, which is how Vitest's worker processes
   * report one session as a single run.
   */
  runId?: string;
  /** Suppress the event when the scope saw no deepeval work. */
  skipIfEmpty?: boolean;
}

export function captureEvaluationRun<T>(
  entrypoint: Entrypoint,
  body: (accumulator: RunAccumulator) => T,
  { runId, skipIfEmpty = false }: EvaluationRunOptions = {},
): T {
  if (telemetryOptOut()) {
    return withRun(entrypoint, "", body);
  }

  const feature = ENTRYPOINT_FEATURE[entrypoint] ?? Feature.EVALUATION;
  const featureStatus = getFeatureStatus(feature);

  return withRun(entrypoint, runId ?? crypto.randomUUID(), (accumulator) =>
    withSettle(
      () => body(accumulator),
      (error) => {
        emitEvaluation({
          accumulator,
          feature,
          featureStatus,
          skipIfEmpty,
          error,
        });
      },
    ),
  );
}

/**
 * `captureEvaluationRun` for a caller whose run spans two hooks rather than one
 * callback, as a Vitest reporter's does. `finish` emits.
 */
export function beginEvaluationRun(
  entrypoint: Entrypoint,
  { runId, skipIfEmpty = false }: EvaluationRunOptions = {},
): { accumulator: RunAccumulator; finish: (error?: unknown) => void } {
  const feature = ENTRYPOINT_FEATURE[entrypoint] ?? Feature.EVALUATION;
  const featureStatus = telemetryOptOut()
    ? undefined
    : getFeatureStatus(feature);
  const { accumulator, pop } = pushAmbientRun(
    entrypoint,
    runId ?? crypto.randomUUID(),
  );

  let finished = false;
  return {
    accumulator,
    finish: (error?: unknown) => {
      if (finished) return;
      finished = true;
      pop();
      if (telemetryOptOut()) return;
      emitEvaluation({
        accumulator,
        feature,
        featureStatus,
        skipIfEmpty,
        error,
      });
    },
  };
}

function emitEvaluation({
  accumulator,
  feature,
  featureStatus,
  skipIfEmpty,
  error,
}: {
  accumulator: RunAccumulator;
  feature: Feature;
  featureStatus: EventProperties["featureStatus"];
  skipIfEmpty: boolean;
  error: unknown;
}): void {
  try {
    if (skipIfEmpty && !accumulator.hasActivity()) return;
    setLastFeature(feature);
    capture(Event.EVALUATION, {
      ...accumulator.snapshot(),
      // `Outcome.INTERRUPTED` is unreachable: SIGINT does not unwind.
      outcome: error === undefined ? Outcome.COMPLETED : Outcome.ERRORED,
      errorType: error === undefined ? undefined : errorTypeOf(error),
      featureName: feature,
      featureStatus,
    });
  } catch {
    // Telemetry must never break a user's evaluation.
  }
}

export interface SynthesizerRunOptions {
  method: string;
  maxGenerations?: number;
  numEvolutions: number;
  evolutions?: Record<string, unknown>;
}

export function captureSynthesizerRun({
  method,
  maxGenerations,
  numEvolutions,
  evolutions,
}: SynthesizerRunOptions): void {
  if (telemetryOptOut()) return;
  const featureStatus = getFeatureStatus(Feature.SYNTHESIZER);
  setLastFeature(Feature.SYNTHESIZER);
  capture(Event.SYNTHESIZER, {
    featureName: Feature.SYNTHESIZER,
    featureStatus,
    synthMethod: method,
    synthMaxGenerations: maxGenerations,
    synthNumEvolutions: numEvolutions,
    synthEvolutions: Object.keys(evolutions ?? {}).sort(),
  });
}

export function captureConversationSimulatorRun(
  numConversations: number,
): void {
  if (telemetryOptOut()) return;
  const featureStatus = getFeatureStatus(Feature.CONVERSATION_SIMULATOR);
  setLastFeature(Feature.CONVERSATION_SIMULATOR);
  capture(Event.CONVERSATION_SIMULATOR, {
    featureName: Feature.CONVERSATION_SIMULATOR,
    featureStatus,
    numConversations,
  });
}

export function captureBenchmarkRun(
  benchmark: string,
  numTasks?: number,
): void {
  if (telemetryOptOut()) return;
  const featureStatus = getFeatureStatus(Feature.BENCHMARK);
  setLastFeature(Feature.BENCHMARK);
  capture(Event.BENCHMARK, {
    featureName: Feature.BENCHMARK,
    featureStatus,
    benchmarkName: benchmark,
    benchmarkNumTasks: Number.isInteger(numTasks) ? numTasks : undefined,
  });
}

/** Fires once per process, not per handler construction. */
export function recordTracingIntegration(integration: Integration): void {
  const firstTime = registerIntegration(integration);
  if (telemetryOptOut() || !firstTime) return;

  const featureStatus = getFeatureStatus(Feature.TRACING_INTEGRATION);
  setLastFeature(Feature.TRACING_INTEGRATION);
  capture(Event.INTEGRATION_INSTALLED, {
    featureName: Feature.TRACING_INTEGRATION,
    featureStatus,
    integration,
  });
}

/**
 * Command name only, never arguments or flag values. `knownCommands` is the
 * dispatch table of the program that routed the call, so there is no second
 * list to keep in sync.
 */
export function captureCliCommand(
  command: string | undefined,
  knownCommands: Iterable<string>,
): void {
  if (telemetryOptOut()) return;
  const known = new Set(knownCommands);
  const name = command && known.has(command) ? command : UNKNOWN_CLI_COMMAND;
  capture(Event.CLI_COMMAND, { cliCommand: name });
}

export function captureLoginPromptShown(surface: LoginPromptSurface): void {
  if (telemetryOptOut()) return;
  capture(Event.LOGIN_PROMPT_SHOWN, { promptSurface: surface });
}

const surfacesSeen = new Set<LoginPromptSurface>();

/** For surfaces that repaint continuously, where each repaint is not exposure. */
export function captureLoginPromptShownOnce(surface: LoginPromptSurface): void {
  if (surfacesSeen.has(surface)) return;
  surfacesSeen.add(surface);
  captureLoginPromptShown(surface);
}

/** A login that happened outside `deepeval login`, e.g. inline in `view`. */
export function recordLoginCompleted(surface: LoginPromptSurface): void {
  if (telemetryOptOut()) return;
  capture(Event.LOGIN, {
    loginOutcome: LoginOutcome.COMPLETED,
    promptSurface: surface,
    lastFeature: getLastFeature(),
  });
}

export class LoginSpan {
  outcome = LoginOutcome.ABANDONED;
  method = LoginMethod.UNKNOWN;

  setOutcome(outcome: LoginOutcome): void {
    this.outcome = outcome;
  }

  setMethod(method: LoginMethod): void {
    this.method = method;
  }
}

/** Captures on exit, so an abandoned login is distinguishable from a completed one. */
export function captureLoginEvent<T>(body: (span: LoginSpan) => T): T {
  const span = new LoginSpan();
  if (telemetryOptOut()) return body(span);

  const lastFeature = getLastFeature();
  return withSettle(
    () => body(span),
    (error) => {
      if (error !== undefined && span.outcome === LoginOutcome.ABANDONED) {
        span.outcome = LoginOutcome.FAILED;
      }
      try {
        capture(Event.LOGIN, {
          loginOutcome: span.outcome,
          loginMethod: span.method,
          lastFeature,
        });
      } catch {
        // Never let a telemetry failure mask a login failure.
      }
    },
  );
}
