/**
 * Local (offline) component-level evals for the Mastra integration.
 *
 * Drives the exporter with synthetic tracing events instead of a live agent, so
 * these tests need no OPENAI_API_KEY and make no network calls. Mastra invokes
 * exporters synchronously from the frame that creates the span (its event bus
 * calls handlers inline), which is what the hand-rolled emit below reproduces.
 */
import {
  TracingEventType,
  SpanType as MastraSpanType,
} from "@mastra/core/observability";
import type { AnyExportedSpan, TracingEvent } from "@mastra/core/observability";

import { DeepEvalExporter } from "../../../src/integrations/mastra";
import { BaseMetric } from "../../../src/metrics/base-metrics";
import { BaseSpan, SpanType, traceManager } from "../../../src/tracing/tracing";
import {
  nextLlmSpan,
  nextToolSpan,
  updateCurrentSpan,
} from "../../../src/tracing";
import { Environment } from "../../../src/tracing/utils";

class StubMetric extends BaseMetric {
  constructor(private label: string) {
    super(0.5, { showIndicator: false });
    this.requiredParams = [];
  }
  get name(): string {
    return this.label;
  }
  async measure(): Promise<number> {
    this.score = 1;
    this.success = true;
    return 1;
  }
  isSuccessful(): boolean {
    return true;
  }
}

let spanCounter = 0;

function exportedSpan(
  overrides: Partial<AnyExportedSpan> & { type: MastraSpanType },
): AnyExportedSpan {
  spanCounter += 1;
  return {
    id: `span-${spanCounter}`,
    traceId: "mastra-trace-1",
    name: `span-${spanCounter}`,
    startTime: new Date(),
    attributes: {},
    isRootSpan: false,
    isEvent: false,
    ...overrides,
  } as AnyExportedSpan;
}

function event(
  type: TracingEventType,
  exportedSpan: AnyExportedSpan,
): TracingEvent {
  return { type, exportedSpan } as TracingEvent;
}

/** Find a span anywhere in the captured trace tree by name. */
function findSpan(spans: BaseSpan[], name: string): BaseSpan | undefined {
  for (const span of spans) {
    if (span.name === name) return span;
    const hit = findSpan(span.children ?? [], name);
    if (hit) return hit;
  }
  return undefined;
}

describe("Mastra local component evals", () => {
  let exporter: DeepEvalExporter;

  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
      confidentApiKey: "test-key-not-used-offline",
    });
    spanCounter = 0;
    exporter = new DeepEvalExporter({ name: "offline-test" });
  });

  it("applies metrics staged with nextToolSpan to the tool span", async () => {
    const metric = new StubMetric("tool-metric");
    const tool = exportedSpan({
      type: MastraSpanType.TOOL_CALL,
      name: "get_weather",
      entityName: "get_weather",
      input: { city: "Tokyo" },
    });

    await nextToolSpan(
      { metrics: [metric], expectedOutput: "Sunny" },
      async () => {
        await exporter.exportTracingEvent(
          event(TracingEventType.SPAN_STARTED, tool),
        );
      },
    );

    const span = traceManager.getSpanByUuid(tool.id);
    expect(span?.metrics).toEqual([metric]);
    expect(span?.expectedOutput).toBe("Sunny");
  });

  it("keeps typed staging isolated: an LLM payload does not land on a tool span", async () => {
    const llmMetric = new StubMetric("llm-metric");
    const tool = exportedSpan({
      type: MastraSpanType.TOOL_CALL,
      name: "get_weather",
    });

    await nextLlmSpan({ metrics: [llmMetric] }, async () => {
      await exporter.exportTracingEvent(
        event(TracingEventType.SPAN_STARTED, tool),
      );
    });

    expect(traceManager.getSpanByUuid(tool.id)?.metrics).toBeUndefined();
  });

  it("staged fields win over the exporter's static metric collection", async () => {
    const configured = new DeepEvalExporter({
      name: "offline-test",
      llmMetricCollection: "static-collection",
    });
    const llm = exportedSpan({
      type: MastraSpanType.MODEL_GENERATION,
      name: "llm",
    });

    await nextLlmSpan({ metricCollection: "staged-collection" }, async () => {
      await configured.exportTracingEvent(
        event(TracingEventType.SPAN_STARTED, llm),
      );
    });

    expect(traceManager.getSpanByUuid(llm.id)?.metricCollection).toBe(
      "staged-collection",
    );
  });

  it("makes updateCurrentSpan reach the span while its body runs", async () => {
    const metric = new StubMetric("from-tool-body");
    const tool = exportedSpan({
      type: MastraSpanType.TOOL_CALL,
      name: "get_weather",
    });

    // Mastra emits SPAN_STARTED synchronously, then runs `execute`.
    await exporter.exportTracingEvent(
      event(TracingEventType.SPAN_STARTED, tool),
    );

    // ...this stands in for the body of the tool's `execute`.
    updateCurrentSpan({
      metrics: [metric],
      expectedOutput: "Sunny in Tokyo",
    });

    const span = traceManager.getSpanByUuid(tool.id);
    expect(span?.metrics).toEqual([metric]);
    expect(span?.expectedOutput).toBe("Sunny in Tokyo");
  });

  it("carries staged metrics through to the completed trace", async () => {
    const metric = new StubMetric("tool-metric");
    const captured: Array<{ rootSpans: BaseSpan[] }> = [];
    const unsubscribe = traceManager.addTraceCaptureSink((t) =>
      captured.push(t),
    );

    try {
      const agent = exportedSpan({
        type: MastraSpanType.AGENT_RUN,
        name: "weather_agent",
        isRootSpan: true,
        input: "What's the weather in Tokyo?",
      });
      const tool = exportedSpan({
        type: MastraSpanType.TOOL_CALL,
        name: "get_weather",
        entityName: "get_weather",
        parentSpanId: agent.id,
        input: { city: "Tokyo" },
      });

      await exporter.exportTracingEvent(
        event(TracingEventType.SPAN_STARTED, agent),
      );
      await nextToolSpan({ metrics: [metric] }, async () => {
        await exporter.exportTracingEvent(
          event(TracingEventType.SPAN_STARTED, tool),
        );
      });
      await exporter.exportTracingEvent(
        event(TracingEventType.SPAN_ENDED, {
          ...tool,
          output: "Sunny in Tokyo",
          endTime: new Date(),
        } as AnyExportedSpan),
      );
      await exporter.exportTracingEvent(
        event(TracingEventType.SPAN_ENDED, {
          ...agent,
          output: "It is sunny in Tokyo",
          endTime: new Date(),
        } as AnyExportedSpan),
      );

      expect(captured).toHaveLength(1);
      const toolSpan = findSpan(captured[0].rootSpans, "get_weather");
      expect(toolSpan?.type).toBe(SpanType.TOOL);
      expect(toolSpan?.metrics).toEqual([metric]);
      expect(toolSpan?.output).toBe("Sunny in Tokyo");
    } finally {
      unsubscribe();
    }
  });
});
