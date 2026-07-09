import {
  beginTraceCapture,
  endTraceCapture,
  isCapturingTraces,
  getCapturedTraces,
  getLatestCapturedTrace,
} from "../../src/evaluate/assert-test/trace-scope";
import { traceManager } from "../../src/tracing";

describe("trace-scope capture", () => {
  afterEach(() => {
    endTraceCapture();
  });

  it("is not capturing by default", () => {
    endTraceCapture();
    expect(isCapturingTraces()).toBe(false);
  });

  it("registers a capture sink on begin and clears it on end", () => {
    beginTraceCapture();
    expect(isCapturingTraces()).toBe(true);
    // Sink is registered on the trace manager (suppresses posting).
    expect((traceManager as unknown as { traceCaptureSink?: unknown }).traceCaptureSink).toBeDefined();

    endTraceCapture();
    expect(isCapturingTraces()).toBe(false);
    expect((traceManager as unknown as { traceCaptureSink?: unknown }).traceCaptureSink).toBeUndefined();
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
