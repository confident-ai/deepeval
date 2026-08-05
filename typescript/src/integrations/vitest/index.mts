import { expect, beforeEach, afterEach } from "vitest";
import type { BaseMetric, BaseConversationalMetric } from "../../metrics/index.js";
import { getIsRunningDeepEval } from "../../utils.js";
import {
  beginTraceCapture,
  endTraceCapture,
} from "../../evaluate/test-run/trace-scope.js";
import { toPass } from "./matchers.mjs";

type AnyMetric = BaseMetric | BaseConversationalMetric;

expect.extend({ toPass });

beforeEach(() => {
  if (getIsRunningDeepEval()) beginTraceCapture();
});
afterEach(() => {
  if (getIsRunningDeepEval()) endTraceCapture();
});

declare module "vitest" {
  interface Assertion<T = any> {
    toPass(metrics?: AnyMetric[]): Promise<T>;
  }
  interface AsymmetricMatchersContaining {
    toPass(metrics?: AnyMetric[]): Promise<void>;
  }
}

export { toPass } from "./matchers.mjs";
