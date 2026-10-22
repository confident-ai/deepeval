// The flush entry point for `CONFIDENT_TRACE_FLUSH` (`set-debug --trace-flush`).

import { envBool } from "@/env-flags";
import { CONFIDENT_TRACE_FLUSH } from "@/constants";
import { otelHooks } from "@/tracing/otel/hooks";
import { traceManager } from "@/tracing/tracing";

export function traceFlushEnabled(): boolean {
  return envBool(CONFIDENT_TRACE_FLUSH) ?? false;
}

/** Send every pending trace, and the spans confident-trace batches. */
export async function flushTraces(timeoutMillis = 30000): Promise<void> {
  await traceManager.flush();
  await otelHooks.flush?.(timeoutMillis);
}
