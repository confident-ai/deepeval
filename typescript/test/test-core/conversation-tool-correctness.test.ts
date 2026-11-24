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

const search = (destination: string) =>
  new ToolCall({ name: "search_flights", inputParameters: { destination } });
const book = () => new ToolCall({ name: "book_flight", inputParameters: {} });

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
      conversation([[search("1")], undefined, [book()]], [search("1"), book()]),
    );
    expect(score).toBe(1);
    expect(m.toolsCalled.map((t) => t.name)).toEqual([
      "search_flights",
      "book_flight",
    ]);
    expect(m.success).toBe(true);
  });

  it("lowers the score when an expected tool is never called", async () => {
    const m = metric();
    expect(
      await m.measure(conversation([[search("1")], []], [search("1"), book()])),
    ).toBe(0.5);
    expect(m.reason).toContain("book_flight");
  });

  it("treats [] as 'no tools expected'", async () => {
    expect(await metric().measure(conversation([[], []], []))).toBe(1);
    expect(await metric().measure(conversation([[book()]], []))).toBe(0);
  });

  it("checks ordering across the conversation", async () => {
    const expected = [search("1"), book()];
    const m = metric({ shouldConsiderOrdering: true });
    expect(
      await m.measure(conversation([[search("1")], [book()]], expected)),
    ).toBe(1);
    expect(
      await m.measure(conversation([[book()], [search("1")]], expected)),
    ).toBeLessThan(1);
  });

  it("compares input parameters when asked", async () => {
    const tc = conversation([[search("999")]], [search("123")]);
    expect(await metric().measure(tc)).toBe(1);
    expect(
      await metric({
        evaluationParams: [ToolCallParams.INPUT_PARAMETERS],
      }).measure(tc),
    ).toBe(0);
  });

  it("errors without expectedTools", async () => {
    await expect(
      metric().measure(conversation([[search("1")]])),
    ).rejects.toBeInstanceOf(MissingTestCaseParamsError);
  });

  it("carries expectedTools from a golden to its test case", () => {
    const golden = new ConversationalGolden({
      scenario: "User wants to book a flight.",
      turns: [
        new Turn({ role: "user", content: "Book me a flight to Tokyo." }),
      ],
      expectedTools: [book()],
    });
    const [tc] = convertConvoGoldensToConvoTestCases([golden]);
    expect(tc.expectedTools?.map((t) => t.name)).toEqual(["book_flight"]);
  });
  it("checks a pinned tool on its own turn", async () => {
    const pinned = new ToolCall({ name: "book_flight", turn: 3 });
    expect(
      await metric().measure(conversation([[search("1")], [book()]], [pinned])),
    ).toBe(1);
    expect(await metric().measure(conversation([[book()], []], [pinned]))).toBe(
      0,
    );
  });

  it("scores pinned and unpinned tools as separate checks", async () => {
    const m = metric();
    const score = await m.measure(
      conversation(
        [[search("1")], []],
        [search("1"), new ToolCall({ name: "book_flight", turn: 3 })],
      ),
    );
    expect(score).toBe(0.5);
    expect(m.checkScores.map((c) => c.check)).toEqual([
      "Turn 3",
      "Rest of the conversation",
    ]);
    expect(m.reason).toContain("Turn 3 (0.00)");
  });

  it("rejects a turn that isn't an assistant turn", async () => {
    await expect(
      metric().measure(
        conversation(
          [[book()]],
          [new ToolCall({ name: "book_flight", turn: 0 })],
        ),
      ),
    ).rejects.toThrow("not an assistant turn");
  });
});
