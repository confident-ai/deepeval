// The flush entry point for `CONFIDENT_TRACE_FLUSH` (`set-debug --trace-flush`).

import { envBool } from "@/env-flags";
import { CONFIDENT_TRACE_FLUSH } from "@/constants";
import { traceManager } from "@/tracing/tracing";

export function traceFlushEnabled(): boolean {
  return envBool(CONFIDENT_TRACE_FLUSH) ?? false;
}

/** Wait for confident-trace to export every ended span. */
export async function flushTraces(timeoutMillis = 30000): Promise<void> {
  await traceManager.flush(timeoutMillis);
}
