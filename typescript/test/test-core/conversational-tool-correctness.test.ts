import { describe, expect, it } from "@jest/globals";
import { MissingTestCaseParamsError } from "@/errors";
import { ConversationalToolCorrectnessMetric } from "@/metrics";
import {
  ConversationalTestCase,
  ToolCall,
  ToolCallParams,
  Turn,
} from "@/test-case";

const track = (pin: string) =>
  new ToolCall({ name: "track_parcel", inputParameters: { pin } });
const transfer = () =>
  new ToolCall({ name: "transfer_to_human", inputParameters: {} });

function conversation(
  ...assistantTurns: [ToolCall[] | undefined, ToolCall[] | undefined][]
): ConversationalTestCase {
  const turns: Turn[] = [];
  for (const [toolsCalled, expectedTools] of assistantTurns) {
    turns.push(new Turn({ role: "user", content: "hi" }));
    turns.push(
      new Turn({
        role: "assistant",
        content: "ok",
        toolsCalled,
        expectedTools,
      }),
    );
  }
  return new ConversationalTestCase({ turns });
}

describe("ConversationalToolCorrectnessMetric", () => {
  it("averages only turns with expectedTools", async () => {
    const metric = new ConversationalToolCorrectnessMetric({
      showIndicator: false,
    });
    const score = await metric.measure(
      conversation(
        [[track("1")], [track("1")]],
        [[track("1")], [transfer()]],
        [[transfer()], undefined],
      ),
    );
    expect(score).toBe(0.5);
    expect(metric.turnScores.map((t) => t.turn)).toEqual([1, 3]);
    expect(metric.success).toBe(true);
    expect(metric.reason).toContain("1 of 2");
  });

  it("treats [] as 'no tools expected'", async () => {
    const metric = new ConversationalToolCorrectnessMetric({
      showIndicator: false,
    });
    expect(await metric.measure(conversation([[], []]))).toBe(1);
    expect(await metric.measure(conversation([[transfer()], []]))).toBe(0);
  });

  it("compares input parameters when asked", async () => {
    const tc = conversation([[track("999")], [track("123")]]);
    expect(
      await new ConversationalToolCorrectnessMetric({
        showIndicator: false,
      }).measure(tc),
    ).toBe(1);
    expect(
      await new ConversationalToolCorrectnessMetric({
        showIndicator: false,
        evaluationParams: [ToolCallParams.INPUT_PARAMETERS],
      }).measure(tc),
    ).toBe(0);
  });

  it("errors without any expectedTools", async () => {
    const metric = new ConversationalToolCorrectnessMetric({
      showIndicator: false,
    });
    await expect(
      metric.measure(conversation([[track("1")], undefined])),
    ).rejects.toBeInstanceOf(MissingTestCaseParamsError);
  });
});
