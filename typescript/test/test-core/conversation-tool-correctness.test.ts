import { describe, expect, it } from "@jest/globals";
import { MissingTestCaseParamsError } from "@/errors";
import { ConversationToolCorrectnessMetric } from "@/metrics";
import {
  ConversationalTestCase,
  ToolCall,
  ToolCallParams,
  Turn,
} from "@/test-case";
import { ConversationalGolden } from "@/dataset/golden";
import { convertConvoGoldensToConvoTestCases } from "@/dataset/utils";

const track = (pin: string) =>
  new ToolCall({ name: "track_parcel", inputParameters: { pin } });
const transfer = () =>
  new ToolCall({ name: "transfer_to_human", inputParameters: {} });

function conversation(
  toolsPerAssistantTurn: (ToolCall[] | undefined)[],
  expectedTools?: ToolCall[],
): ConversationalTestCase {
  const turns: Turn[] = [];
  for (const toolsCalled of toolsPerAssistantTurn) {
    turns.push(new Turn({ role: "user", content: "hi" }));
    turns.push(new Turn({ role: "assistant", content: "ok", toolsCalled }));
  }
  return new ConversationalTestCase({ turns, expectedTools });
}

const metric = (options = {}) =>
  new ConversationToolCorrectnessMetric({ showIndicator: false, ...options });

describe("ConversationToolCorrectnessMetric", () => {
  it("collects tools across all assistant turns", async () => {
    const m = metric();
    const score = await m.measure(
      conversation(
        [[track("1")], undefined, [transfer()]],
        [track("1"), transfer()],
      ),
    );
    expect(score).toBe(1);
    expect(m.toolsCalled.map((t) => t.name)).toEqual([
      "track_parcel",
      "transfer_to_human",
    ]);
    expect(m.success).toBe(true);
  });

  it("lowers the score when an expected tool is never called", async () => {
    const m = metric();
    expect(
      await m.measure(
        conversation([[track("1")], []], [track("1"), transfer()]),
      ),
    ).toBe(0.5);
    expect(m.reason).toContain("transfer_to_human");
  });

  it("treats [] as 'no tools expected'", async () => {
    expect(await metric().measure(conversation([[], []], []))).toBe(1);
    expect(await metric().measure(conversation([[transfer()]], []))).toBe(0);
  });

  it("checks ordering across the conversation", async () => {
    const expected = [track("1"), transfer()];
    const m = metric({ shouldConsiderOrdering: true });
    expect(
      await m.measure(conversation([[track("1")], [transfer()]], expected)),
    ).toBe(1);
    expect(
      await m.measure(conversation([[transfer()], [track("1")]], expected)),
    ).toBeLessThan(1);
  });

  it("compares input parameters when asked", async () => {
    const tc = conversation([[track("999")]], [track("123")]);
    expect(await metric().measure(tc)).toBe(1);
    expect(
      await metric({
        evaluationParams: [ToolCallParams.INPUT_PARAMETERS],
      }).measure(tc),
    ).toBe(0);
  });

  it("errors without expectedTools", async () => {
    await expect(
      metric().measure(conversation([[track("1")]])),
    ).rejects.toBeInstanceOf(MissingTestCaseParamsError);
  });

  it("carries expectedTools from a golden to its test case", () => {
    const golden = new ConversationalGolden({
      scenario: "Parcel is late; user wants a human.",
      turns: [new Turn({ role: "user", content: "Get me a person." })],
      expectedTools: [transfer()],
    });
    const [tc] = convertConvoGoldensToConvoTestCases([golden]);
    expect(tc.expectedTools?.map((t) => t.name)).toEqual(["transfer_to_human"]);
  });
  it("checks a pinned tool on its own turn", async () => {
    // Turns: 0 user, 1 assistant, 2 user, 3 assistant.
    const pin = new ToolCall({ name: "transfer_to_human", turn: 3 });
    expect(
      await metric().measure(conversation([[track("1")], [transfer()]], [pin])),
    ).toBe(1);
    expect(
      await metric().measure(conversation([[transfer()], []], [pin])),
    ).toBe(0);
  });

  it("scores pinned and unpinned tools as separate checks", async () => {
    const m = metric();
    const score = await m.measure(
      conversation(
        [[track("1")], []],
        [track("1"), new ToolCall({ name: "transfer_to_human", turn: 3 })],
      ),
    );
    expect(score).toBe(0.5);
    expect(m.groupScores.map((g) => g.group)).toEqual([
      "Turn 3",
      "Rest of the conversation",
    ]);
    expect(m.reason).toContain("Turn 3 (0.00)");
  });

  it("rejects a turn that isn't an assistant turn", async () => {
    await expect(
      metric().measure(
        conversation(
          [[transfer()]],
          [new ToolCall({ name: "transfer_to_human", turn: 0 })],
        ),
      ),
    ).rejects.toThrow("'user' turn");
  });
});
