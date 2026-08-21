// The flush entry point for `CONFIDENT_TRACE_FLUSH` (`set-debug --trace-flush`).

import { envBool } from "@/env-flags";
import { CONFIDENT_TRACE_FLUSH } from "@/constants";
import { traceManager } from "@/tracing/tracing";

/** Integrations own their own OTel providers, so each needs its own flush. */
const INTEGRATION_MODULES = [
  "../integrations/ai-sdk/index",
  "../integrations/openinference/index",
];

export function traceFlushEnabled(): boolean {
  return envBool(CONFIDENT_TRACE_FLUSH) ?? false;
}

export async function flushTraces(): Promise<void> {
  await traceManager.flush();

  for (const specifier of INTEGRATION_MODULES) {
    try {
      const integration = (await import(specifier)) as {
        forceFlush?: () => Promise<void>;
      };
      await integration.forceFlush?.();
    } catch {}
  }
}
