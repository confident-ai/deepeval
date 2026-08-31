import { patchOpenRouter } from "@/openrouter/patch";
import { recordTracingIntegration } from "@/telemetry";
import { Integration } from "@/tracing/integrations";

let telemetryRecorded = false;

/**
 * Instrument an official `@openrouter/sdk` client so every `chat.send(...)` and
 * `responses.send(...)` call produces an LLM span.
 *
 * If you reach OpenRouter through the OpenAI SDK instead, use
 * `instrumentOpenAI` — it recognizes OpenRouter's base URL on its own.
 */
function instrumentOpenRouter(client: any) {
  // Only the telemetry ping is once-per-process; `patchOpenRouter` owns the
  // patch guard, so instrumenting again after an `unpatchOpenRouter` works.
  if (!telemetryRecorded) {
    recordTracingIntegration(Integration.OPEN_ROUTER);
    telemetryRecorded = true;
  }

  patchOpenRouter(client);
}

export { instrumentOpenRouter };
export {
  OPENROUTER_METADATA_KEY,
  OPENROUTER_PROVIDER,
  type OpenRouterMetadata,
} from "@/openrouter/utils";
