// GEval without `evaluationParams` judges the trace (`_traceDict`) instead of
// individual test case fields.

import { MissingTestCaseParamsError } from "@/errors";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import { GEval } from "@/metrics";
import { ScriptedLLM } from "./system-one-fakes";

const TRACE = {
  name: "support_agent",
  type: "agent",
  input: { input: "Where is my order #1234?" },
  output: "Your order is in transit.",
  children: [
    {
      name: "order_lookup",
      type: "tool",
      input: { order_id: "1234" },
      output: { status: "in_transit" },
      children: [],
    },
  ],
};

function judge(): ScriptedLLM {
  return new ScriptedLLM((prompt: string) =>
    prompt.includes('"steps"') && !prompt.includes("Evaluation Steps:")
      ? { steps: ["Check the tool call.", "Check the output."] }
      : { score: 8, reason: "order_lookup was used correctly." },
  );
}

function tracedTestCase(): LLMTestCase {
  const testCase = new LLMTestCase({
    input: "Where is my order #1234?",
    actualOutput: "Your order is in transit.",
  });
  testCase._traceDict = TRACE;
  return testCase;
}

describe("GEval trajectory mode", () => {
  it("requires a trace only when evaluationParams is omitted", () => {
    expect(
      new GEval({ name: "t", criteria: "c", model: judge() }).requiresTrace,
    ).toBe(true);
    expect(
      new GEval({
        name: "t",
        criteria: "c",
        evaluationParams: [SingleTurnParams.INPUT],
        model: judge(),
      }).requiresTrace,
    ).toBe(false);
  });

  it.each([false, true])(
    "renders the trace prompts (strictMode=%s)",
    async (strictMode) => {
      const llm = judge();
      const metric = new GEval({
        name: "Trajectory",
        criteria: "Did the agent use the right tool and report its result?",
        model: llm,
        strictMode,
        showIndicator: false,
      });

      await metric.measure(tracedTestCase());

      const [stepsPrompt, resultsPrompt] = llm.prompts;
      expect(stepsPrompt).toContain("execution trace");
      expect(resultsPrompt).toContain('"name": "order_lookup"');
      expect(resultsPrompt).not.toContain("Test Case:");
      expect(metric.reason).toBe("order_lookup was used correctly.");
    },
  );

  it("throws when the test case has no trace", async () => {
    const metric = new GEval({
      name: "Trajectory",
      criteria: "c",
      model: judge(),
      showIndicator: false,
    });
    await expect(
      metric.measure(new LLMTestCase({ input: "hi", actualOutput: "hello" })),
    ).rejects.toThrow(MissingTestCaseParamsError);
  });
});
