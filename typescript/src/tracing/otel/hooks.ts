/**
 * Late-bound links from deepeval's tracing API to confident-trace.
 *
 * `setup.ts` fills them in once confident-trace runs, so the tracing core
 * never loads OpenTelemetry for an app that doesn't trace a framework.
 */

import type { BaseSpan, Trace } from "@/tracing/tracing";

export type ExportedScope = "span" | "trace";

export const otelHooks: {
  /** The span deepeval collects for the active OTel span. */
  collectedSpan?: () => BaseSpan | undefined;
  /** The trace of the active collected span. */
  collectedTrace?: () => Trace | undefined;
  /** Send an update with no deepeval target to the exported OTel span. */
  updateExported?: (scope: ExportedScope, values: Record<string, any>) => void;
  /** Wait for confident-trace to export its spans. */
  flush?: (timeoutMillis: number) => Promise<boolean>;
} = {};
