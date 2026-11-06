/**
 * deepeval on confident-trace: converting OTel spans, and collecting the
 * spans of the traces deepeval consumes.
 */

import { context, trace as otelTrace, type Tracer } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { ToolCall } from "@/test-case";
import { nextLlmSpan } from "@/tracing/pending-context";
import {
  type AgentSpan,
  type BaseSpan,
  type LlmSpan,
  observe,
  SpanType,
  type Trace,
  traceManager,
  updateCurrentSpan,
} from "@/tracing/tracing";
import { collectFinishedTrace } from "@/tracing/otel/bridge";
import { convertSpan } from "@/tracing/otel/converter";
import {
  confidentTrace,
  ensureConfidentTrace,
  flushConfidentTrace,
} from "@/tracing/otel/setup";

const exported = new InMemorySpanExporter();
ensureConfidentTrace({ exporter: exported });
const { withSpan } = confidentTrace();

/** Ended spans, recorded on a provider of their own. */
function recordSpans(run: (tracer: Tracer) => void): ReadableSpan[] {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  run(provider.getTracer("test"));
  return exporter.getFinishedSpans();
}

function capture(): { traces: Trace[]; stop: () => void } {
  const traces: Trace[] = [];
  const unsubscribe = traceManager.addTraceCaptureSink((trace) =>
    traces.push(trace),
  );
  const release = traceManager.beginEvaluation();
  return {
    traces,
    stop: () => {
      release();
      unsubscribe();
    },
  };
}

function walk(spans: BaseSpan[]): BaseSpan[] {
  return spans.flatMap((span) => [span, ...walk(span.children)]);
}

const MESSAGES = {
  "gen_ai.operation.name": "chat",
  "gen_ai.request.model": "gpt-4o-mini",
  "gen_ai.usage.input_tokens": 11,
  "gen_ai.input.messages": JSON.stringify([
    { role: "user", parts: [{ type: "text", content: "Weather?" }] },
  ]),
  "gen_ai.output.messages": JSON.stringify([
    {
      role: "assistant",
      parts: [
        { type: "text", content: "" },
        { type: "tool_call", name: "lookup", arguments: '{"city":"Paris"}' },
      ],
    },
  ]),
};

beforeEach(() => traceManager.clearTraces());

describe("convertSpan", () => {
  it("reads a model call's messages, tokens and tool calls", () => {
    const [span] = recordSpans((tracer) =>
      tracer.startSpan("chat gpt-4o-mini", { attributes: MESSAGES }).end(),
    );

    const llm = convertSpan(span).span as LlmSpan;

    expect(llm.type).toBe(SpanType.LLM);
    expect([llm.model, llm.inputTokenCount]).toEqual(["gpt-4o-mini", 11]);
    expect(llm.input).toEqual([{ role: "user", content: "Weather?" }]);
    expect(llm.output).toEqual([
      {
        role: "assistant",
        content: {
          type: "tool_call",
          name: "lookup",
          arguments: '{"city":"Paris"}',
        },
      },
    ]);
    expect(llm.toolsCalled).toEqual([
      new ToolCall({ name: "lookup", inputParameters: { city: "Paris" } }),
    ]);
  });

  it("unwraps a tool's keyword arguments and its reply", () => {
    const [span] = recordSpans((tracer) =>
      tracer
        .startSpan("execute_tool lookup", {
          attributes: {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "lookup",
            "confident.span.input": JSON.stringify({
              args: [],
              kwargs: { city: "Paris" },
            }),
            "confident.span.output": JSON.stringify("sunny"),
          },
        })
        .end(),
    );

    const tool = convertSpan(span).span;

    expect([tool.type, tool.name]).toEqual([SpanType.TOOL, "lookup"]);
    expect([tool.input, tool.output]).toEqual([{ city: "Paris" }, "sunny"]);
  });

  it("treats content confident-trace couldn't serialize as missing", () => {
    const [span] = recordSpans((tracer) =>
      tracer
        .startSpan("agent", {
          attributes: {
            "gen_ai.operation.name": "invoke_agent",
            "confident.span.output": JSON.stringify({
              response: "[unsupported]",
            }),
          },
        })
        .end(),
    );

    expect(convertSpan(span).span.output).toBeUndefined();
  });

  it("answers a conversation with its final assistant turn", () => {
    const text = (role: string, content: string) => ({
      role,
      parts: [{ type: "text", content }],
    });
    const [span] = recordSpans((tracer) =>
      tracer
        .startSpan("graph", {
          attributes: {
            "confident.trace.output": JSON.stringify([
              text("user", "Weather?"),
              text("assistant", "Let me check."),
              text("tool", "sunny"),
              text("assistant", "Sunny."),
            ]),
          },
        })
        .end(),
    );

    expect(convertSpan(span).traceFields.output).toBe("Sunny.");
  });

  it("reads OpenInference spans", () => {
    const [span] = recordSpans((tracer) =>
      tracer
        .startSpan("ChatCompletion", {
          attributes: {
            "openinference.span.kind": "LLM",
            "llm.model_name": "gpt-4o-mini",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "Weather?",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Sunny.",
            "session.id": "session",
          },
        })
        .end(),
    );

    const converted = convertSpan(span);
    const llm = converted.span as LlmSpan;

    expect([llm.type, llm.model, llm.output]).toEqual([
      SpanType.LLM,
      "gpt-4o-mini",
      "Sunny.",
    ]);
    expect(converted.traceFields.threadId).toBe("session");
  });
});

describe("DeepEvalSpanBridge", () => {
  it("collects a trace during an evaluation", async () => {
    exported.reset();
    const metric = { name: "recorded" } as never;
    const { traces, stop } = capture();

    await nextLlmSpan({ metrics: [metric] }, () =>
      withSpan({ name: "agent", type: "agent" }, () =>
        withSpan({ name: "chat", type: "llm" }, (span) => {
          span.setAttributes(MESSAGES);
          updateCurrentSpan({ metadata: { step: "first" } });
        }),
      ),
    );
    stop();

    expect(traces).toHaveLength(1);
    const [collected] = traces;
    const [agent] = collected.rootSpans as AgentSpan[];
    expect([agent.type, agent.name]).toEqual([SpanType.AGENT, "agent"]);
    const [llm] = agent.children as LlmSpan[];
    expect(llm.type).toBe(SpanType.LLM);
    expect(llm.metrics).toEqual([metric]);
    expect(llm.metadata).toEqual({ step: "first" });
    // The agent records no content of its own; it takes its model call's.
    expect(agent.input).toBe("Weather?");
    expect(collected.input).toBe("Weather?");
    await flushConfidentTrace(5000);
    expect(exported.getFinishedSpans()).toHaveLength(0);
  });

  it("nests spans under an enclosing observe span", async () => {
    const traces: Trace[] = [];
    const unsubscribe = traceManager.addTraceCaptureSink((trace) =>
      traces.push(trace),
    );

    await observe({
      type: "agent",
      name: "endpoint",
      fn: async () =>
        withSpan({ name: "lookup", type: "tool" }, () =>
          updateCurrentSpan({ output: "sunny" }),
        ),
    })();
    unsubscribe();

    const [collected] = traces;
    const [endpoint] = collected.rootSpans;
    expect(endpoint.name).toBe("endpoint");
    const [tool] = endpoint.children;
    expect([tool.type, tool.name, tool.output]).toEqual([
      SpanType.TOOL,
      "lookup",
      "sunny",
    ]);
  });

  it("lets a later model call in an observe trace set its output", async () => {
    const traces: Trace[] = [];
    const unsubscribe = traceManager.addTraceCaptureSink((trace) =>
      traces.push(trace),
    );

    // Each call is a trace of its own to OTel, but one deepeval trace.
    const call = (answer: string) =>
      withSpan({ name: "chat", type: "llm" }, (span) =>
        span.setAttributes({
          ...MESSAGES,
          "gen_ai.output.messages": JSON.stringify([
            { role: "assistant", parts: [{ type: "text", content: answer }] },
          ]),
        }),
      );
    await observe({
      name: "endpoint",
      fn: async () => {
        call("Let me check.");
        call("Sunny.");
      },
    })();
    unsubscribe();

    expect(traces[0].output).toBe("Sunny.");
  });

  it("leaves other spans to confident-trace's export", async () => {
    exported.reset();

    withSpan({ name: "standalone", type: "tool" }, () => {
      expect(otelTrace.getActiveSpan()).toBeDefined();
    });
    await flushConfidentTrace(5000);

    expect(exported.getFinishedSpans().map((span) => span.name)).toEqual([
      "standalone",
    ]);
  });

  it("builds a trace from spans that only report once they end", () => {
    const { traces, stop } = capture();
    const spans = recordSpans((tracer) => {
      const root = tracer.startSpan("agent run", {
        attributes: { "confident.span.type": "agent" },
      });
      tracer
        .startSpan(
          "lookup",
          {
            attributes: {
              "confident.span.type": "tool",
              "confident.span.output": JSON.stringify("sunny"),
            },
          },
          otelTrace.setSpan(context.active(), root),
        )
        .end();
      root.end();
    });

    collectFinishedTrace(spans, { name: "finished" });
    stop();

    const [collected] = traces;
    expect(collected.name).toBe("finished");
    const [agent] = collected.rootSpans;
    expect(walk(collected.rootSpans).map((span) => span.name)).toEqual([
      "agent run",
      "lookup",
    ]);
    expect(agent.toolsCalled?.map((call) => call.name)).toEqual(["lookup"]);
  });
});
