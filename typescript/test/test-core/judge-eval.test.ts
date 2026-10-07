import { LLMTestCase, ToolCall } from "@/test-case";
import {
  JudgeEval,
  JudgeEvalField,
  JudgeEvalRole,
  type JudgeEvalMessage,
  type JudgeEvalVariable,
} from "@/metrics";
import {
  renderMessages,
  resolvePath,
  resolveVariables,
} from "@/metrics/judge-eval/utils";
import { ScriptedLLM } from "./system-one-fakes";

const INPUT = {
  messages: [
    { role: "system", content: "You are a payroll assistant." },
    { role: "user", content: "Where do I add a super fund?" },
  ],
};
const OUTPUT = {
  messages: [{ role: "assistant", content: "Go to Employees > Super." }],
};

function judge(score = 8): ScriptedLLM {
  return new ScriptedLLM({ score, reason: "Grounded answer." });
}

function messages(): JudgeEvalMessage[] {
  return [
    { role: JudgeEvalRole.SYSTEM, content: "You grade payroll answers." },
    {
      role: JudgeEvalRole.USER,
      content: "QUESTION: {{question}}\nANSWER: {{ answer }}",
    },
  ];
}

function variables(): Record<string, JudgeEvalVariable> {
  return {
    question: {
      field: JudgeEvalField.INPUT,
      path: ["messages", -1, "content"],
    },
    answer: { field: JudgeEvalField.OUTPUT, path: ["messages", 0, "content"] },
  };
}

function testCase(): LLMTestCase {
  return new LLMTestCase({
    input: JSON.stringify(INPUT),
    actualOutput: JSON.stringify(OUTPUT),
    additionalMetadata: { tenant: { region: "au" } },
    toolsCalled: [
      new ToolCall({ name: "search_docs", inputParameters: { q: "super" } }),
    ],
  });
}

describe("JudgeEval paths", () => {
  it("supports keys, indexes, last and all items", () => {
    expect(resolvePath(INPUT, ["messages", 0, "role"])).toBe("system");
    expect(resolvePath(INPUT, ["messages", -1, "content"])).toBe(
      "Where do I add a super fund?",
    );
    expect(resolvePath(INPUT, ["messages", "*", "role"])).toEqual([
      "system",
      "user",
    ]);
    expect(resolvePath(INPUT, [])).toBe(INPUT);
  });

  it("returns undefined on a miss", () => {
    expect(resolvePath(INPUT, ["messages", 5, "content"])).toBeUndefined();
    expect(resolvePath(INPUT, ["missing"])).toBeUndefined();
    expect(resolvePath("plain text", ["messages"])).toBeUndefined();
  });

  it("parses nested JSON strings", () => {
    const value = { payload: JSON.stringify({ answer: "42" }) };
    expect(resolvePath(value, ["payload", "answer"])).toBe("42");
  });

  it("reads every field", () => {
    const values = resolveVariables(testCase(), {
      question: {
        field: JudgeEvalField.INPUT,
        path: ["messages", 1, "content"],
      },
      region: { field: JudgeEvalField.METADATA, path: ["tenant", "region"] },
      tools: { field: JudgeEvalField.TOOLS_CALLED, path: ["*", "name"] },
      wholeOutput: { field: JudgeEvalField.OUTPUT },
      missing: { field: JudgeEvalField.OUTPUT, path: ["messages", 9] },
    });

    expect(values.question).toBe("Where do I add a super fund?");
    expect(values.region).toBe("au");
    expect(values.tools).toBe('["search_docs"]');
    expect(JSON.parse(values.wholeOutput)).toEqual(OUTPUT);
    expect(values.missing).toBe("");
  });

  it("keeps plain text input as is", () => {
    const values = resolveVariables(
      new LLMTestCase({ input: "hello", actualOutput: "hi" }),
      { question: { field: JudgeEvalField.INPUT } },
    );
    expect(values.question).toBe("hello");
  });

  it("labels roles and fills variables", () => {
    expect(renderMessages(messages(), { question: "Q?", answer: "A." })).toBe(
      "System:\nYou grade payroll answers.\n\nUser:\nQUESTION: Q?\nANSWER: A.",
    );
  });
});

describe("JudgeEval validation", () => {
  it("rejects unmapped variables", () => {
    expect(
      () =>
        new JudgeEval({
          name: "Decline Quality",
          messages: messages(),
          variables: { question: variables().question },
          model: judge(),
        }),
    ).toThrow(/answer/);
  });

  it("requires the system message to come first", () => {
    expect(
      () =>
        new JudgeEval({
          name: "Decline Quality",
          messages: [...messages()].reverse(),
          variables: variables(),
          model: judge(),
        }),
    ).toThrow(/system/);
  });

  it.each([
    [5, 5],
    [10, 1],
  ])("rejects score range [%s, %s]", (minimum, maximum) => {
    expect(
      () =>
        new JudgeEval({
          name: "Decline Quality",
          messages: messages(),
          variables: variables(),
          scoreRange: [minimum, maximum],
          model: judge(),
        }),
    ).toThrow(/scoreRange/);
  });
});

describe("JudgeEval measure", () => {
  it("scales the score to the unit range", async () => {
    const llm = judge(4);
    const metric = new JudgeEval({
      name: "Decline Quality",
      messages: messages(),
      variables: variables(),
      scoreRange: [1, 5],
      model: llm,
      showIndicator: false,
    });

    const score = await metric.measure(testCase());

    expect(score).toBeCloseTo(0.75);
    expect(metric.success).toBe(true);
    expect(metric.reason).toBe("Grounded answer.");
    expect(llm.prompts[0]).toContain("QUESTION: Where do I add a super fund?");
    expect(llm.prompts[0]).toContain("ANSWER: Go to Employees > Super.");
    expect(llm.prompts[0]).toContain("between 1 and 5");
  });

  it("only passes strict mode on the max score", async () => {
    const metric = new JudgeEval({
      name: "Decline Quality",
      messages: messages(),
      variables: variables(),
      model: judge(9),
      strictMode: true,
      showIndicator: false,
    });

    expect(await metric.measure(testCase())).toBe(0);
    expect(metric.success).toBe(false);
  });

  it("adds the JudgeEval suffix to the name", () => {
    const metric = new JudgeEval({
      name: "Decline Quality",
      messages: messages(),
      variables: variables(),
      model: judge(),
    });
    expect(metric.name).toBe("Decline Quality [JudgeEval]");
  });
});
