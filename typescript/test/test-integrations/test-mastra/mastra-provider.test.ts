import { SpanType as MastraSpanType } from "@mastra/core/observability";
import type { AnyExportedSpan } from "@mastra/core/observability";

import {
  buildDeepEvalSpan,
  updateDeepEvalSpan,
} from "../../../src/integrations/mastra/converter";
import { LlmSpan } from "../../../src/tracing/tracing";
import { Integration, Provider } from "../../../src/tracing/integrations";

const TRACE_UUID = "trace-uuid-1";

// Minimal ExportedSpan; the converter only reads a handful of fields.
function makeSpan(
  partial: Partial<AnyExportedSpan> & { type: MastraSpanType },
): AnyExportedSpan {
  return {
    id: "span-1",
    name: "test-span",
    startTime: new Date(0),
    attributes: {},
    metadata: {},
    ...partial,
  } as unknown as AnyExportedSpan;
}

describe("Mastra converter: integration + provider", () => {
  test("tags every span type with the Mastra integration", () => {
    const llm = buildDeepEvalSpan(
      makeSpan({
        type: MastraSpanType.MODEL_GENERATION,
        attributes: { model: "gpt-4o-mini" } as any,
      }),
      TRACE_UUID,
    );
    expect(llm.integration).toBe(Integration.MASTRA);

    const tool = buildDeepEvalSpan(
      makeSpan({ id: "t1", type: MastraSpanType.TOOL_CALL, name: "search" }),
      TRACE_UUID,
    );
    expect(tool.integration).toBe(Integration.MASTRA);

    const agent = buildDeepEvalSpan(
      makeSpan({ id: "a1", type: MastraSpanType.AGENT_RUN }),
      TRACE_UUID,
    );
    expect(agent.integration).toBe(Integration.MASTRA);
  });

  test("uses Mastra's native provider attribute on LLM spans (normalized)", () => {
    const llm = buildDeepEvalSpan(
      makeSpan({
        type: MastraSpanType.MODEL_GENERATION,
        // model says OpenAI, but the native provider attribute must win
        attributes: { model: "gpt-4o-mini", provider: "anthropic" } as any,
      }),
      TRACE_UUID,
    ) as LlmSpan;

    expect(llm.provider).toBe(Provider.ANTHROPIC);
  });

  test("infers the provider from the model when Mastra gives none", () => {
    const llm = buildDeepEvalSpan(
      makeSpan({
        type: MastraSpanType.MODEL_GENERATION,
        attributes: { model: "claude-3-5-sonnet-latest" } as any,
      }),
      TRACE_UUID,
    ) as LlmSpan;

    expect(llm.provider).toBe(Provider.ANTHROPIC);
  });

  test("leaves provider unset when neither attribute nor model resolves", () => {
    const llm = buildDeepEvalSpan(
      makeSpan({
        type: MastraSpanType.MODEL_GENERATION,
        attributes: {} as any,
      }),
      TRACE_UUID,
    ) as LlmSpan;

    expect(llm.provider).toBeUndefined();
  });

  test("updateDeepEvalSpan refreshes model and provider together", () => {
    const llm = buildDeepEvalSpan(
      makeSpan({
        type: MastraSpanType.MODEL_GENERATION,
        attributes: { model: "gpt-4o-mini" } as any,
      }),
      TRACE_UUID,
    ) as LlmSpan;
    expect(llm.provider).toBe(Provider.OPEN_AI);

    updateDeepEvalSpan(
      llm,
      makeSpan({
        type: MastraSpanType.MODEL_GENERATION,
        attributes: { responseModel: "claude-3-5-sonnet-latest" } as any,
      }),
    );

    expect(llm.model).toBe("claude-3-5-sonnet-latest");
    expect(llm.provider).toBe(Provider.ANTHROPIC);
  });
});
