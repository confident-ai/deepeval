import { expect, beforeEach, afterEach } from "vitest";
import type { BaseMetric, BaseConversationalMetric } from "../../metrics/index.js";
import { getIsRunningDeepEval } from "../../utils.js";
import {
  beginTraceCapture,
  endTraceCapture,
} from "../../evaluate/assert-test/trace-scope.js";
import { toPassMetric, toPassAll } from "./matchers.mjs";

type AnyMetric = BaseMetric | BaseConversationalMetric;

expect.extend({ toPassMetric, toPassAll });

beforeEach(() => {
  if (getIsRunningDeepEval()) beginTraceCapture();
});
afterEach(() => {
  if (getIsRunningDeepEval()) endTraceCapture();
});

declare module "vitest" {
  interface Assertion<T = any> {
    toPassMetric(metric: AnyMetric): Promise<T>;
    toPassAll(metrics: AnyMetric[]): Promise<T>;
  }
  interface AsymmetricMatchersContaining {
    toPassMetric(metric: AnyMetric): Promise<void>;
    toPassAll(metrics: AnyMetric[]): Promise<void>;
  }
}

export { toPassMetric, toPassAll } from "./matchers.mjs";
export { assertTest } from "../../evaluate/assert-test/index.js";
