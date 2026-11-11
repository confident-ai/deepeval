/**
 * Late-bound links from deepeval's tracing API to confident-trace.
 *
 * `setup.ts` fills them in once confident-trace runs, since it needs the
 * tracing core loaded first.
 */

import type { BaseSpan, Trace } from "@/tracing/tracing";

export type ExportedScope = "span" | "trace";

export const otelHooks: {
  /** The span deepeval collects for the active OTel span. */
  collectedSpan?: () => BaseSpan | undefined;
  /** The trace of the active collected span. */
  collectedTrace?: () => Trace | undefined;
  /** Pass an update to the active OTel span, if confident-trace exports it. */
  updateExported?: (scope: ExportedScope, values: Record<string, any>) => void;
  /** Keep the active span, or its whole trace, out of the export. */
  drop?: (wholeTrace: boolean) => void;
  /** Wait for confident-trace to export its spans. */
  flush?: (timeoutMillis: number) => Promise<boolean>;
} = {};
