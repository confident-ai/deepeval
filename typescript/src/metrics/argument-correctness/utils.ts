/**
 * The tool calls a traced test case gives ArgumentCorrectness when its span
 * has none of its own.
 *
 * Each tool call belongs to the LLM span that chose its arguments: the tool's
 * parent when that is an LLM span, else the latest LLM sibling that started
 * before it. Tool calls with no such LLM span are left out.
 */

import { LLMTestCase, ToolCall } from "@/test-case";
import { asTestCaseString, asToolCalls } from "@/test-case/utils";
import { SpanType, type BaseSpan, type ToolSpan } from "@/tracing/tracing";
import { prepareToolCallInputParameters } from "@/tracing/utils";

interface SpanPosition {
  parent?: BaseSpan;
  siblings: readonly BaseSpan[];
}

/**
 * `testCase` with the tool calls (and, for a tool span, the input) its span
 * tree gives it, or `testCase` itself when it already lists tool calls, has no
 * span tree, or the tree yields none.
 */
export function withToolCallsFromTrace(testCase: LLMTestCase): LLMTestCase {
  const rootSpans = testCase._traceRootSpans;
  if (testCase.toolsCalled?.length || rootSpans === undefined) {
    return testCase;
  }

  const span = testCase._traceSpan;
  let input = testCase.input;
  let toolCalls: ToolCall[];
  if (span === undefined) {
    toolCalls = toolCallsWithin(rootSpans, undefined);
  } else {
    const position = parentAndSiblings(span, rootSpans, undefined);
    if (position === undefined) return testCase;
    const { parent, siblings } = position;
    if (span.type === SpanType.TOOL) {
      const llmSpan = owningLlmSpan(span, parent, siblings);
      if (llmSpan?.input == null) return testCase;
      input = asTestCaseString(llmSpan.input);
      toolCalls = [asToolCall(span)];
    } else if (span.type === SpanType.LLM) {
      toolCalls = ownedToolCalls(span, parent, siblings);
    } else {
      toolCalls = toolCallsWithin(span.children ?? [], span);
    }
  }

  if (toolCalls.length === 0) return testCase;
  const copy = Object.create(Object.getPrototypeOf(testCase)) as LLMTestCase;
  return Object.assign(copy, testCase, { input, toolsCalled: toolCalls });
}

function owningLlmSpan(
  toolSpan: BaseSpan,
  parent: BaseSpan | undefined,
  siblings: readonly BaseSpan[],
): BaseSpan | undefined {
  if (parent?.type === SpanType.LLM) return parent;
  let owner: BaseSpan | undefined;
  for (const sibling of siblings) {
    if (
      sibling.type === SpanType.LLM &&
      sibling.startTime.getTime() <= toolSpan.startTime.getTime() &&
      (owner === undefined ||
        sibling.startTime.getTime() > owner.startTime.getTime())
    ) {
      owner = sibling;
    }
  }
  return owner;
}

function ownedToolCalls(
  llmSpan: BaseSpan,
  parent: BaseSpan | undefined,
  siblings: readonly BaseSpan[],
): ToolCall[] {
  // Tool calls the LLM span lists itself (from its output messages, or set by
  // the user) describe the same calls as its tool spans, so they win.
  if (llmSpan.toolsCalled?.length) {
    return asToolCalls(llmSpan.toolsCalled) ?? [];
  }
  const toolSpans = [
    ...(llmSpan.children ?? []).filter((child) => child.type === SpanType.TOOL),
    ...siblings.filter(
      (sibling) =>
        sibling.type === SpanType.TOOL &&
        owningLlmSpan(sibling, parent, siblings) === llmSpan,
    ),
  ];
  return byStartTime(toolSpans).map(asToolCall);
}

/** Tool calls of every LLM span in `spans` and below, nested agents included. */
function toolCallsWithin(
  spans: readonly BaseSpan[],
  parent: BaseSpan | undefined,
): ToolCall[] {
  const toolCalls: ToolCall[] = [];
  for (const span of byStartTime(spans)) {
    if (span.type === SpanType.LLM) {
      toolCalls.push(...ownedToolCalls(span, parent, spans));
    }
    toolCalls.push(...toolCallsWithin(span.children ?? [], span));
  }
  return toolCalls;
}

/** `span`'s parent and siblings (itself included) within `spans`. */
function parentAndSiblings(
  span: BaseSpan,
  spans: readonly BaseSpan[],
  parent: BaseSpan | undefined,
): SpanPosition | undefined {
  for (const candidate of spans) {
    if (candidate.uuid === span.uuid) return { parent, siblings: spans };
    const position = parentAndSiblings(
      span,
      candidate.children ?? [],
      candidate,
    );
    if (position !== undefined) return position;
  }
  return undefined;
}

function asToolCall(toolSpan: BaseSpan): ToolCall {
  return new ToolCall({
    name: toolSpan.name ?? "unknown",
    description: (toolSpan as ToolSpan).description,
    inputParameters: prepareToolCallInputParameters(toolSpan.input),
    output: toolSpan.output,
  });
}

function byStartTime(spans: readonly BaseSpan[]): BaseSpan[] {
  return [...spans].sort(
    (a, b) => a.startTime.getTime() - b.startTime.getTime(),
  );
}
