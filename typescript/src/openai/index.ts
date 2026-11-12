/**
 * deepeval's OpenAI integration, built on confident-trace.
 *
 * confident-trace traces each Chat Completions and Responses call of a client
 * it instruments. Stage span fields with `nextLlmSpan(...)` or
 * `setTracingContext(...)`.
 */

import {
  confidentTraceModule,
  ensureConfidentTrace,
} from "@/tracing/otel/setup";

type OpenAIAdapter = typeof import("confident-trace/openai");

const instrumented = new WeakMap<object, () => void>();

/** Trace `client`'s calls; returns a function that stops tracing them. */
export function instrumentOpenAI(client: object): () => void {
  const existing = instrumented.get(client);
  if (existing) return existing;
  ensureConfidentTrace();
  const restore = confidentTraceModule<OpenAIAdapter>(
    "openai",
  ).instrumentOpenAI(
    client as Parameters<OpenAIAdapter["instrumentOpenAI"]>[0],
  );
  const stop = () => {
    instrumented.delete(client);
    restore();
  };
  instrumented.set(client, stop);
  return stop;
}
