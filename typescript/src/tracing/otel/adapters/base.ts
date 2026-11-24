/** The contract every framework adapter implements. */

import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import type { SpanType, Trace } from "@/tracing/tracing";

export interface Adapter {
  /** Whether `span` is this framework's. */
  detect(span: ReadableSpan): boolean;
  /** The span's type, or undefined to leave it to the GenAI conventions. */
  spanType(span: ReadableSpan, atStart: boolean): SpanType | undefined;
  /** deepeval span fields by name; only unset fields take them. */
  spanFields(span: ReadableSpan, type: SpanType): Record<string, unknown>;
  /** `Trace` fields by name; only unset fields take them. */
  traceFields(span: ReadableSpan): Partial<Trace>;
}
