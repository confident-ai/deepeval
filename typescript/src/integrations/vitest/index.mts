import { expect, afterAll, afterEach, beforeAll, beforeEach } from "vitest";
import type { BaseMetric, BaseConversationalMetric } from "@/metrics/index.js";
import { getIsRunningDeepEval } from "@/utils.js";
import {
  beginTraceCapture,
  endTraceCapture,
} from "@/evaluate/test-run/trace-scope.js";
import { toPass } from "@/integrations/vitest/matchers.mjs";
import type { ToPassOptions } from "@/evaluate/test-run/index.js";
import {
  Entrypoint,
  TELEMETRY_RUN_ID_ENV_VAR,
  beginEvaluationRun,
  flush,
} from "@/telemetry/index.js";

type AnyMetric = BaseMetric | BaseConversationalMetric;

expect.extend({ toPass });

// One run scope per test file: this file is re-evaluated for each, and a worker
// in the threads pool never sees `process.on("exit")`. Every file's event
// carries the run id minted in global setup, so a session reads as one run.
let session: { finish: (error?: unknown) => void } | null = null;

beforeAll(() => {
  if (!getIsRunningDeepEval()) return;
  session = beginEvaluationRun(Entrypoint.VITEST, {
    runId: process.env[TELEMETRY_RUN_ID_ENV_VAR],
    // A test file with no deepeval assertions in it should stay silent.
    skipIfEmpty: true,
  });
});

afterAll(() => {
  session?.finish();
  session = null;
  // The worker is about to be torn down, so a buffered event would never leave.
  flush();
});

beforeEach(() => {
  if (getIsRunningDeepEval()) beginTraceCapture();
});
afterEach(() => {
  if (getIsRunningDeepEval()) endTraceCapture();
});

declare module "vitest" {
  interface Assertion<T = any> {
    toPass(metrics?: AnyMetric[], options?: ToPassOptions): Promise<T>;
  }
  interface AsymmetricMatchersContaining {
    toPass(metrics?: AnyMetric[], options?: ToPassOptions): Promise<void>;
  }
}

export { toPass } from "@/integrations/vitest/matchers.mjs";
