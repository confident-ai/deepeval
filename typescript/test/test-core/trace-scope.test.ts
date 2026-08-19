import {
  beginTraceCapture,
  endTraceCapture,
  isCapturingTraces,
  getCapturedTraces,
  getLatestCapturedTrace,
} from "@/evaluate/test-run/trace-scope";
import { traceManager } from "@/tracing";

describe("trace-scope capture", () => {
  afterEach(() => {
    endTraceCapture();
  });

  it("is not capturing by default", () => {
    endTraceCapture();
    expect(isCapturingTraces()).toBe(false);
  });

  const registeredSinks = (): number =>
    (traceManager as unknown as { traceCaptureSinks: Set<unknown> })
      .traceCaptureSinks.size;

  it("registers a capture sink on begin and clears it on end", () => {
    beginTraceCapture();
    expect(isCapturingTraces()).toBe(true);
    // Sink is registered on the trace manager (suppresses posting).
    expect(registeredSinks()).toBe(1);

    endTraceCapture();
    expect(isCapturingTraces()).toBe(false);
    expect(registeredSinks()).toBe(0);
  });

  it("leaves an unrelated subscriber's sink alone", () => {
    const seen: string[] = [];
    const unsubscribe = traceManager.addTraceCaptureSink((t) =>
      seen.push(t.uuid),
    );
    try {
      beginTraceCapture();
      const trace = traceManager.startNewTrace();
      traceManager.endTrace(trace.uuid);

      // Both subscribers receive the trace...
      expect(seen).toEqual([trace.uuid]);
      expect(getCapturedTraces()).toHaveLength(1);

      // ...and ending the eval scope must not unregister the other one.
      endTraceCapture();
      const second = traceManager.startNewTrace();
      traceManager.endTrace(second.uuid);
      expect(seen).toEqual([trace.uuid, second.uuid]);
    } finally {
      unsubscribe();
    }
  });

  it("captures a completed trace via the sink", () => {
    beginTraceCapture();
    const trace = traceManager.startNewTrace();
    expect(getCapturedTraces()).toHaveLength(0);

    traceManager.endTrace(trace.uuid); // routed to the sink, not posted
    expect(getCapturedTraces()).toHaveLength(1);
    expect(getLatestCapturedTrace()?.uuid).toBe(trace.uuid);
  });

  it("resets captured traces between tests", () => {
    beginTraceCapture();
    traceManager.endTrace(traceManager.startNewTrace().uuid);
    expect(getCapturedTraces()).toHaveLength(1);

    endTraceCapture();
    beginTraceCapture();
    expect(getCapturedTraces()).toHaveLength(0);
  });
});
