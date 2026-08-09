// Opt-in tracing of DeepEval's own metric and model methods, the counterpart of
// Python's `deepeval/tracing/internal.py`.

import { observe, type SpanType } from "@/tracing/tracing";
import { isTraceInternalEnabled } from "@/tracing/utils";

const instrumented = new WeakSet<object>();

/**
 * Wrap the named methods of `instance`'s own class in internal spans. Called
 * from a base-class constructor, this reaches the concrete subclass prototype.
 *
 * Only safe for methods that already return a promise, since an observed method
 * always does.
 */
export function observeMethods(
  instance: object,
  options: { spanType?: SpanType | string; methods: string[] },
): void {
  if (!isTraceInternalEnabled()) return;

  const prototype = Object.getPrototypeOf(instance);
  if (!prototype || prototype === Object.prototype) return;
  if (instrumented.has(prototype)) return;
  instrumented.add(prototype);

  const className = prototype.constructor?.name ?? "anonymous";

  for (const method of options.methods) {
    const descriptor = Object.getOwnPropertyDescriptor(prototype, method);
    if (!descriptor || typeof descriptor.value !== "function") continue;

    const original = descriptor.value as (...args: any[]) => any;
    Object.defineProperty(prototype, method, {
      ...descriptor,
      value: function (this: unknown, ...args: any[]) {
        return observe({
          type: options.spanType,
          name: `${className}.${method}`,
          _internal: true,
          _dropIfRoot: true,
          fn: (...inner: any[]) => original.apply(this, inner),
        })(...args);
      },
    });
  }
}
