// Port of `deepeval/telemetry/context.py`. Per-test-case and per-metric events
// are folded into one event per run. A bounded run (`evaluate()`, Vitest,
// `compare`) opens a `RunAccumulator`; bare `metric.measure()` calls have no
// enclosing run, so they land in a process-level accumulator with its own
// flush policy.

import { AsyncLocalStorage } from "async_hooks";
import * as crypto from "crypto";

import { getLogger } from "@/logger";
import { capture, flush as flushBackend } from "@/telemetry/client";
import { Entrypoint, Event } from "@/telemetry/events";
import { describeJudge } from "@/telemetry/judge";
import {
  FlushReason,
  Outcome,
  TurnKind,
  type EventProperties,
} from "@/telemetry/properties";

const logger = getLogger("telemetry");

// A long-running service can measure metrics for weeks without exiting, so the
// standalone path cannot rely on process exit alone.
export const STANDALONE_FLUSH_THRESHOLD = 50;
export const STANDALONE_FLUSH_INTERVAL_MS = 30 * 60 * 1000;

/**
 * Keys on the `turns` field rather than importing the four classes, which would
 * be circular, and which `instanceof` would not survive subclassing anyway.
 */
export function turnKindOf(item: unknown): TurnKind | undefined {
  if (item === null || item === undefined) return undefined;
  return typeof item === "object" && "turns" in item
    ? TurnKind.MULTI_TURN
    : TurnKind.SINGLE_TURN;
}

function resolveTurnKind(kinds: Set<TurnKind>): TurnKind | undefined {
  if (kinds.size === 0) return undefined;
  if (kinds.size === 1) return [...kinds][0];
  return TurnKind.MIXED;
}

function tracingState(): { enabled: boolean; traceCount: number } {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { traceManager } = require("../tracing/tracing") as {
      traceManager: { tracingEnabled?: boolean; traces?: unknown[] };
    };
    return {
      enabled: Boolean(traceManager.tracingEnabled),
      traceCount: traceManager.traces?.length ?? 0,
    };
  } catch {
    return { enabled: false, traceCount: 0 };
  }
}

// No lock, unlike Python: concurrent metrics interleave on one thread.
export class RunAccumulator {
  testCases = 0;
  goldens = 0;
  metricRuns = 0;
  private readonly metrics = new Map<string, number>();
  private readonly turnKinds = new Set<TurnKind>();
  asyncMode?: boolean;
  inComponent = false;
  provider?: string;
  model?: string;

  constructor(
    readonly entrypoint: Entrypoint,
    readonly runId: string,
    private readonly tracesAtEntry: number,
  ) {}

  recordTestCase(count = 1, kind?: TurnKind): void {
    this.testCases += count;
    if (kind !== undefined) this.turnKinds.add(kind);
  }

  recordGolden(count = 1, kind?: TurnKind): void {
    this.goldens += count;
    if (kind !== undefined) this.turnKinds.add(kind);
  }

  recordMetric(
    metricName: string,
    asyncMode: boolean,
    inComponent: boolean,
    model?: unknown,
  ): void {
    const { provider, model: modelName } = describeJudge(model);
    this.metricRuns += 1;
    this.metrics.set(metricName, (this.metrics.get(metricName) ?? 0) + 1);
    if (asyncMode) this.asyncMode = true;
    else if (this.asyncMode === undefined) this.asyncMode = false;
    if (inComponent) this.inComponent = true;
    if (provider !== undefined && this.provider === undefined) {
      this.provider = provider;
      this.model = modelName;
    }
  }

  /** Lets a scope broader than an evaluation, like a Vitest session, stay quiet. */
  hasActivity(): boolean {
    return this.testCases > 0 || this.goldens > 0 || this.metricRuns > 0;
  }

  snapshot(): EventProperties {
    const metricNames = [...this.metrics.keys()].sort();
    const { enabled, traceCount } = tracingState();
    const tracedHere = Math.max(traceCount - this.tracesAtEntry, 0);
    return {
      entrypoint: this.entrypoint,
      runId: this.runId || undefined,
      testCaseCount: this.testCases,
      goldenCount: this.goldens,
      turnKind: resolveTurnKind(this.turnKinds),
      metricRuns: this.metricRuns,
      metrics: metricNames,
      metricsCount: metricNames.length,
      asyncMode: this.asyncMode,
      inComponent: this.inComponent,
      provider: this.provider,
      model: this.model,
      tracingEnabled: enabled,
      traced: tracedHere > 0,
      traceCount: tracedHere,
    };
  }
}

// A stack so metrics attribute to the innermost run: an `evaluate()` call
// inside a Vitest session belongs to `evaluate`.
const runStack = new AsyncLocalStorage<RunAccumulator[]>();

// AsyncLocalStorage only propagates into the callback it wraps, so a run opened
// by a hook that returns before the work happens lands here instead.
let ambientStack: RunAccumulator[] = [];

function stack(): RunAccumulator[] {
  return runStack.getStore() ?? ambientStack;
}

export function currentRun(): RunAccumulator | undefined {
  const active = stack();
  return active.length > 0 ? active[active.length - 1] : undefined;
}

/** Python's context manager equivalent; a callback is what ALS can bound. */
export function withRun<T>(
  entrypoint: Entrypoint,
  runId: string,
  body: (accumulator: RunAccumulator) => T,
): T {
  const { traceCount } = tracingState();
  const accumulator = new RunAccumulator(entrypoint, runId, traceCount);
  return runStack.run([...stack(), accumulator], () => body(accumulator));
}

/**
 * Open a run with no lexical scope, for a caller that opens and closes it from
 * two separate hooks. The returned function closes it.
 */
export function pushAmbientRun(
  entrypoint: Entrypoint,
  runId: string,
): { accumulator: RunAccumulator; pop: () => void } {
  const { traceCount } = tracingState();
  const accumulator = new RunAccumulator(entrypoint, runId, traceCount);
  ambientStack = [...ambientStack, accumulator];
  return {
    accumulator,
    pop: () => {
      const index = ambientStack.lastIndexOf(accumulator);
      if (index !== -1) ambientStack.splice(index, 1);
    },
  };
}

// Whether the running metric belongs to a span or trace scope. The shared hook
// is `BaseMetricCore.startProgress`, which metrics call with no arguments, so
// the component evaluator publishes the fact instead of passing it down.
const componentScope = new AsyncLocalStorage<boolean>();

export function withComponentScope<T>(inComponent: boolean, body: () => T): T {
  return componentScope.run(inComponent, body);
}

export function inComponentScope(): boolean {
  return componentScope.getStore() ?? false;
}

export function recordTestCase(testCase?: unknown, count = 1): void {
  currentRun()?.recordTestCase(count, turnKindOf(testCase));
}

export function recordGolden(golden?: unknown, count = 1): void {
  currentRun()?.recordGolden(count, turnKindOf(golden));
}

export interface RecordMetricOptions {
  asyncMode: boolean;
  inComponent: boolean;
  model?: unknown;
  /** False for a nested measurement already counted by its parent metric. */
  track?: boolean;
}

export function recordMetric(
  metricName: string,
  { asyncMode, inComponent, model, track = true }: RecordMetricOptions,
): void {
  if (!track) return;
  const run = currentRun();
  if (run !== undefined) {
    run.recordMetric(metricName, asyncMode, inComponent, model);
  } else {
    standalone.record(metricName, asyncMode, inComponent, model);
  }
}

/**
 * Metrics measured outside any `evaluate()` call. Flushes on a threshold, an
 * interval, and at exit -- the exit hook alone loses SIGKILLed containers.
 */
export class StandaloneAccumulator {
  private metricRuns = 0;
  private metrics = new Map<string, number>();
  private asyncMode?: boolean;
  private inComponent = false;
  private provider?: string;
  private model?: string;
  // Set per batch, so `traced` covers this event's window, not the process.
  private tracesAtStart?: number;

  private timer: NodeJS.Timeout | null = null;
  private exitHookRegistered = false;

  reset(): void {
    this.metricRuns = 0;
    this.metrics = new Map();
    this.asyncMode = undefined;
    this.inComponent = false;
    this.provider = undefined;
    this.model = undefined;
    this.tracesAtStart = undefined;
  }

  private ensureScheduled(): void {
    if (!this.exitHookRegistered) {
      process.once("exit", () => this.flushAtExit());
      this.exitHookRegistered = true;
    }
    if (this.timer === null) {
      this.timer = setTimeout(() => {
        this.timer = null;
        this.flush(FlushReason.INTERVAL);
      }, STANDALONE_FLUSH_INTERVAL_MS);
      // Never hold the event loop open for the sake of a telemetry batch.
      this.timer.unref();
    }
  }

  record(
    metricName: string,
    asyncMode: boolean,
    inComponent: boolean,
    model?: unknown,
  ): void {
    const { provider, model: modelName } = describeJudge(model);
    const { traceCount } = tracingState();
    this.tracesAtStart ??= traceCount;
    this.metricRuns += 1;
    this.metrics.set(metricName, (this.metrics.get(metricName) ?? 0) + 1);
    if (asyncMode) this.asyncMode = true;
    else if (this.asyncMode === undefined) this.asyncMode = false;
    if (inComponent) this.inComponent = true;
    if (provider !== undefined && this.provider === undefined) {
      this.provider = provider;
      this.model = modelName;
    }
    this.ensureScheduled();
    if (this.metricRuns >= STANDALONE_FLUSH_THRESHOLD) {
      this.flush(FlushReason.THRESHOLD);
    }
  }

  private drain(): EventProperties | undefined {
    if (this.metricRuns === 0) return undefined;
    const { enabled, traceCount } = tracingState();
    const metricNames = [...this.metrics.keys()].sort();
    const tracedHere = Math.max(traceCount - (this.tracesAtStart ?? 0), 0);
    const properties: EventProperties = {
      entrypoint: Entrypoint.STANDALONE,
      runId: crypto.randomUUID(),
      // Explicit zeros so every Evaluation event has the same shape.
      testCaseCount: 0,
      goldenCount: 0,
      metricRuns: this.metricRuns,
      metrics: metricNames,
      metricsCount: metricNames.length,
      asyncMode: this.asyncMode,
      inComponent: this.inComponent,
      provider: this.provider,
      model: this.model,
      tracingEnabled: enabled,
      traced: tracedHere > 0,
      traceCount: tracedHere,
    };
    this.reset();
    return properties;
  }

  flush(reason: FlushReason): void {
    const properties = this.drain();
    if (properties === undefined) return;
    capture(Event.EVALUATION, {
      ...properties,
      flushReason: reason,
      outcome: Outcome.COMPLETED,
    });
  }

  private flushAtExit(): void {
    try {
      this.flush(FlushReason.PROCESS_EXIT);
      flushBackend();
    } catch (error) {
      logger.debug("Failed to flush standalone metrics at exit", error);
    }
  }
}

const standalone = new StandaloneAccumulator();

export function flushStandaloneMetrics(
  reason: FlushReason = FlushReason.MANUAL,
): void {
  standalone.flush(reason);
}

/**
 * Drop any ambient run scope and buffered standalone metrics. A test process is
 * itself inside the Vitest session's run scope, which would swallow records.
 */
export function resetForTesting(): void {
  ambientStack = [];
  standalone.reset();
}
