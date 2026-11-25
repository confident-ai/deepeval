// ArgumentCorrectness on a traced span with no tool calls of its own judges
// the tool calls of the nearest LLM span: the tool's parent when that is an LLM
// span, else the latest LLM sibling that started before it.

import { MissingTestCaseParamsError } from "@/errors";
import { EvaluationDataset, Golden } from "@/dataset";
import { ArgumentCorrectnessMetric } from "@/metrics";
import { LLMTestCase, ToolCall } from "@/test-case";
import { observe } from "@/tracing";
import { BaseSpan, SpanType, TraceSpanStatus } from "@/tracing/tracing";
import { ScriptedLLM } from "./system-one-fakes";

const TOOL_NAMES = [
  "check_application_status",
  "search_agency_kb",
  "lookup_visa_quota",
  "file_report",
  "declared_tool",
];

function judge(): ScriptedLLM {
  return new ScriptedLLM((prompt: string) =>
    prompt.includes("Tool Calls:")
      ? { verdicts: [{ verdict: "yes" }] }
      : { reason: "Arguments fit the input." },
  );
}

function span(
  type: SpanType,
  name: string,
  start: number,
  fields: { input?: unknown; toolsCalled?: ToolCall[] } = {},
  children: BaseSpan[] = [],
): BaseSpan {
  const result = new BaseSpan({
    uuid: name,
    status: TraceSpanStatus.SUCCESS,
    traceUuid: "trace",
    startTime: new Date(start),
    type,
    name,
    ...fields,
  });
  result.children = children;
  for (const child of children) child.parentUuid = result.uuid;
  return result;
}

function traced(
  target: BaseSpan | undefined,
  rootSpans: BaseSpan[],
  toolsCalled?: ToolCall[],
): LLMTestCase {
  const testCase = new LLMTestCase({
    input: target ? JSON.stringify(target.input) : "trace input",
    actualOutput: "done",
    toolsCalled,
  });
  testCase._traceRootSpans = rootSpans;
  testCase._traceSpan = target;
  return testCase;
}

async function judgedPrompt(testCase: LLMTestCase): Promise<string> {
  const llm = judge();
  await new ArgumentCorrectnessMetric({
    model: llm,
    showIndicator: false,
  }).measure(testCase);
  const verdictPrompts = llm.prompts.filter((p) => p.includes("Tool Calls:"));
  expect(verdictPrompts).toHaveLength(1);
  return verdictPrompts[0];
}

function judgedTools(prompt: string): string[] {
  return TOOL_NAMES.filter((name) => prompt.includes(`"name": "${name}"`));
}

/** An agent whose tools run inside a nested agent's LLM span. */
function businessSgTrace() {
  const llm = span(
    SpanType.LLM,
    "generate_response",
    3,
    { input: "llm prompt" },
    [
      span(SpanType.TOOL, "check_application_status", 4, {
        input: { application_id: "EP-2026-118302" },
      }),
      span(SpanType.TOOL, "search_agency_kb", 5, { input: { query: "EP" } }),
    ],
  );
  const orchestrator = span(
    SpanType.AGENT,
    "orchestrate_business_sg_agent",
    0,
    { input: "What's happening with our EP application?" },
    [
      span(
        SpanType.AGENT,
        "business_sg_agent",
        1,
        { input: "business agent input" },
        [span(SpanType.RETRIEVER, "retrieve_business_context", 2), llm],
      ),
      span(SpanType.AGENT, "business_sg_response_validator", 6),
    ],
  );
  return { orchestrator, llm };
}

/** An agent loop: each tool runs beside the LLM span that chose it. */
function agentLoopTrace() {
  const plan = span(SpanType.LLM, "plan", 1, { input: "plan input" });
  const lookup = span(SpanType.TOOL, "lookup_visa_quota", 2, {
    input: { pass: "EP" },
  });
  const answer = span(SpanType.LLM, "answer", 3, { input: "answer input" });
  const report = span(SpanType.TOOL, "file_report", 4, {
    input: { case: "EP-1" },
  });
  // Children in end order, not start order.
  const agent = span(SpanType.AGENT, "agent", 0, { input: "agent input" }, [
    report,
    answer,
    lookup,
    plan,
  ]);
  return { agent, plan, answer, report };
}

describe("ArgumentCorrectness tool calls from the trace", () => {
  it("judges an agent span on the tool calls of its nested LLM spans", async () => {
    const { orchestrator } = businessSgTrace();

    const prompt = await judgedPrompt(traced(orchestrator, [orchestrator]));

    expect(judgedTools(prompt)).toEqual([
      "check_application_status",
      "search_agency_kb",
    ]);
    expect(prompt).toContain("What's happening with our EP application?");
    expect(prompt).toContain('"application_id": "EP-2026-118302"');
  });

  it("judges an LLM span on its sibling tool calls", async () => {
    const { agent, plan, answer } = agentLoopTrace();

    expect(judgedTools(await judgedPrompt(traced(plan, [agent])))).toEqual([
      "lookup_visa_quota",
    ]);
    expect(judgedTools(await judgedPrompt(traced(answer, [agent])))).toEqual([
      "file_report",
    ]);
  });

  it("judges an LLM span on its child tool calls", async () => {
    const { orchestrator, llm } = businessSgTrace();

    expect(
      judgedTools(await judgedPrompt(traced(llm, [orchestrator]))),
    ).toEqual(["check_application_status", "search_agency_kb"]);
  });

  it("judges a tool span against the LLM span before it", async () => {
    const { agent, report } = agentLoopTrace();

    const prompt = await judgedPrompt(traced(report, [agent]));

    expect(judgedTools(prompt)).toEqual(["file_report"]);
    expect(prompt).toContain("answer input");
    expect(prompt).not.toContain("plan input");
  });

  it("judges a tool span against its LLM parent", async () => {
    const { orchestrator, llm } = businessSgTrace();

    const prompt = await judgedPrompt(traced(llm.children[1], [orchestrator]));

    expect(judgedTools(prompt)).toEqual(["search_agency_kb"]);
    expect(prompt).toContain("llm prompt");
  });

  it("judges the trace on every LLM-owned tool call", async () => {
    const { agent } = agentLoopTrace();

    const prompt = await judgedPrompt(traced(undefined, [agent]));

    expect(judgedTools(prompt)).toEqual(["lookup_visa_quota", "file_report"]);
    expect(prompt).toContain("trace input");
  });

  it("does not duplicate tool calls an LLM span lists itself", async () => {
    const { orchestrator, llm } = businessSgTrace();
    llm.toolsCalled = [new ToolCall({ name: "check_application_status" })];

    const prompt = await judgedPrompt(traced(orchestrator, [orchestrator]));

    expect(prompt.split('"name": "check_application_status"')).toHaveLength(2);
    expect(judgedTools(prompt)).toEqual(["check_application_status"]);
  });

  it("keeps the test case's own tool calls", async () => {
    const { agent, plan } = agentLoopTrace();
    const testCase = traced(
      plan,
      [agent],
      [new ToolCall({ name: "declared_tool" })],
    );

    expect(judgedTools(await judgedPrompt(testCase))).toEqual([
      "declared_tool",
    ]);
  });

  it("does not change the shared test case", async () => {
    const { agent, report } = agentLoopTrace();
    const testCase = traced(report, [agent]);

    await judgedPrompt(testCase);

    expect(testCase.toolsCalled).toBeUndefined();
    expect(testCase.input).toBe(JSON.stringify(report.input));
  });

  it("still reports missing tool calls when no LLM span chose them", async () => {
    const agent = span(SpanType.AGENT, "agent", 0, { input: "agent input" }, [
      span(SpanType.TOOL, "lookup_visa_quota", 1, { input: { pass: "EP" } }),
      span(SpanType.LLM, "summarize", 2, { input: "x" }),
    ]);
    const metric = new ArgumentCorrectnessMetric({
      model: judge(),
      showIndicator: false,
    });

    await expect(metric.measure(traced(agent, [agent]))).rejects.toThrow(
      MissingTestCaseParamsError,
    );
  });

  it("still reports missing tool calls without a trace", async () => {
    const metric = new ArgumentCorrectnessMetric({
      model: judge(),
      showIndicator: false,
    });

    await expect(
      metric.measure(
        new LLMTestCase({
          input: "What's the minimum salary?",
          actualOutput: "",
        }),
      ),
    ).rejects.toThrow(MissingTestCaseParamsError);
  });

  it("gives an agent span metric the span tree during evaluation", async () => {
    const llm = judge();
    const metric = new ArgumentCorrectnessMetric({
      model: llm,
      showIndicator: false,
    });
    const checkApplicationStatus = observe({
      type: "tool",
      name: "check_application_status",
      fn: async (applicationId: string) => `pending: ${applicationId}`,
    });
    const searchAgencyKb = observe({
      type: "tool",
      name: "search_agency_kb",
      fn: async (query: string) => `guidance: ${query}`,
    });
    const generateResponse = observe({
      type: "llm",
      name: "generate_response",
      fn: async (prompt: string) => {
        await checkApplicationStatus("EP-2026-118302");
        await searchAgencyKb("EP processing time");
        return `Your EP application is pending. (${prompt})`;
      },
    });
    const businessSgAgent = observe({
      type: "agent",
      name: "business_sg_agent",
      fn: async (message: string) => generateResponse(message),
    });
    const orchestrate = observe({
      type: "agent",
      name: "orchestrate_business_sg_agent",
      metrics: [metric],
      fn: async (message: string) => businessSgAgent(message),
    });
    const dataset = new EvaluationDataset({
      goldens: [
        new Golden({ input: "What's happening with our EP application?" }),
      ],
    });

    for await (const golden of dataset.evalsIterator({
      displayConfig: { showIndicator: false, printResults: false },
    })) {
      await orchestrate((golden as Golden).input);
    }

    expect(metric.error).toBeUndefined();
    expect(metric.score).toBe(1);
    const verdictPrompts = llm.prompts.filter((p) => p.includes("Tool Calls:"));
    expect(verdictPrompts).toHaveLength(1);
    expect(judgedTools(verdictPrompts[0])).toEqual([
      "check_application_status",
      "search_agency_kb",
    ]);
    expect(verdictPrompts[0]).toContain(
      "What's happening with our EP application?",
    );
  });
});
