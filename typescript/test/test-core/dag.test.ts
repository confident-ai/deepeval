import type { ZodType } from "zod";

import { DeepEvalBaseLLM, type GenerationResult } from "@/models";
import {
  BinaryJudgementNode,
  ConversationalBinaryJudgementNode,
  ConversationalDAGMetric,
  ConversationalTaskNode,
  DAGMetric,
  DeepAcyclicGraph,
  GEval,
  NonBinaryJudgementNode,
  TaskNode,
  VerdictNode,
  dagFromJson,
  dagToJson,
  isValidTurnWindow,
} from "@/metrics";
import {
  ConversationalTestCase,
  LLMTestCase,
  MultiTurnParams,
  SingleTurnParams,
  Turn,
} from "@/test-case";

/**
 * Answers prompts from canned replies picked by a substring match, so a test
 * can steer the traversal without hitting a provider.
 */
class StubModel extends DeepEvalBaseLLM {
  readonly prompts: string[] = [];

  constructor(private readonly replies: Array<[string, unknown]>) {
    super("stub");
  }

  async generate<T = string>(
    prompt: string,
    _schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    this.prompts.push(prompt);
    const match = this.replies.find(([needle]) => prompt.includes(needle));
    if (!match) {
      throw new Error(
        `StubModel has no reply for prompt: ${prompt.slice(0, 80)}`,
      );
    }
    return { output: match[1] as T, cost: 0 };
  }

  getModelName(): string {
    return "stub";
  }
}

const testCase = new LLMTestCase({
  input: "What is 2 + 2?",
  actualOutput: "4",
});

function binaryGate(criteria: string) {
  const gate = new BinaryJudgementNode({
    criteria,
    evaluationParams: [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
  });
  return gate;
}

describe("DAGMetric", () => {
  test("scores from the verdict the traversal lands on", async () => {
    const gate = binaryGate("Is the answer correct?");
    gate.addVerdict(true, { score: 10 });
    gate.addVerdict(false, { score: 0 });

    const metric = new DAGMetric({
      name: "Correctness",
      dag: new DeepAcyclicGraph({ rootNodes: [gate] }),
      model: new StubModel([
        ["Is the answer correct?", { verdict: true, reason: "It is 4." }],
      ]),
      includeReason: false,
      showIndicator: false,
    });

    expect(await metric.measure(testCase)).toBe(1);
    expect(metric.isSuccessful()).toBe(true);
    expect(metric.name).toBe("Correctness [DAG]");
  });

  test("prunes the branch whose verdict was not chosen", async () => {
    const gate = binaryGate("Is the answer correct?");
    gate.addVerdict(true, { score: 10 });
    const pruned = new TaskNode({
      instructions: "Summarize the failure.",
      outputLabel: "Failure",
      evaluationParams: [SingleTurnParams.ACTUAL_OUTPUT],
    });
    gate.addVerdict(false, { then: pruned });

    const model = new StubModel([
      ["Is the answer correct?", { verdict: true, reason: "It is 4." }],
    ]);
    const metric = new DAGMetric({
      name: "Correctness",
      dag: new DeepAcyclicGraph({ rootNodes: [gate] }),
      model,
      includeReason: false,
      showIndicator: false,
    });

    expect(await metric.measure(testCase)).toBe(1);
    expect(
      model.prompts.some((p) => p.includes("Summarize the failure.")),
    ).toBe(false);
  });

  test("runs a node shared by two parents exactly once", async () => {
    const shared = binaryGate("Is the answer correct?");
    shared.addVerdict(true, { score: 10 });
    shared.addVerdict(false, { score: 0 });

    const first = new TaskNode({
      instructions: "Extract the claim.",
      outputLabel: "Claim",
      evaluationParams: [SingleTurnParams.ACTUAL_OUTPUT],
    });
    const second = new TaskNode({
      instructions: "Extract the question.",
      outputLabel: "Question",
      evaluationParams: [SingleTurnParams.INPUT],
    });
    first.addNode(shared);
    second.addNode(shared);

    const model = new StubModel([
      ["Extract the claim.", { output: "4" }],
      ["Extract the question.", { output: "2 + 2" }],
      ["Is the answer correct?", { verdict: true, reason: "It is 4." }],
    ]);
    const metric = new DAGMetric({
      name: "Correctness",
      dag: new DeepAcyclicGraph({ rootNodes: [first, second] }),
      model,
      includeReason: false,
      showIndicator: false,
    });

    await metric.measure(testCase);
    const gateCalls = model.prompts.filter((p) =>
      p.includes("Is the answer correct?"),
    );
    expect(gateCalls).toHaveLength(1);
  });

  test("adopts the score and reason of a child metric", async () => {
    const model = new StubModel([
      ["Is the answer correct?", { verdict: true, reason: "It is 4." }],
      [
        "generate 3-4 concise evaluation steps",
        { steps: ["Check the wording."] },
      ],
      ["Evaluation Steps:", { score: 6, reason: "Terse but clear." }],
    ]);
    const gate = binaryGate("Is the answer correct?");
    gate.addVerdict(false, { score: 0 });
    gate.addVerdict(true, {
      then: new GEval({
        name: "Clarity",
        criteria: "Is the answer clearly worded?",
        evaluationParams: [SingleTurnParams.ACTUAL_OUTPUT],
        model,
      }),
    });

    const metric = new DAGMetric({
      name: "Correctness",
      dag: new DeepAcyclicGraph({ rootNodes: [gate] }),
      model,
      showIndicator: false,
    });

    expect(await metric.measure(testCase)).toBeCloseTo(0.6);
    expect(metric.reason).toBe("Terse but clear.");
  });

  test("rejects a cycle", () => {
    const task = new TaskNode({
      instructions: "Extract the claim.",
      outputLabel: "Claim",
      evaluationParams: [SingleTurnParams.ACTUAL_OUTPUT],
    });
    task.addNode(task);
    expect(() => new DeepAcyclicGraph({ rootNodes: [task] })).toThrow(
      /Cycle detected/,
    );
  });

  test("rejects a verdict with both a score and a child", () => {
    expect(
      () =>
        new VerdictNode({
          verdict: true,
          score: 10,
          child: new TaskNode({
            instructions: "Extract the claim.",
            outputLabel: "Claim",
            evaluationParams: [SingleTurnParams.ACTUAL_OUTPUT],
          }),
        }),
    ).toThrow(/either a 'score' or a 'child'/);
  });

  test("rejects duplicate verdicts on a non-binary judgement", () => {
    const node = new NonBinaryJudgementNode({
      criteria: "How complete is the answer?",
      evaluationParams: [SingleTurnParams.ACTUAL_OUTPUT],
    });
    node.addVerdict("partial", { score: 5 });
    node.addVerdict("partial", { score: 4 });
    expect(() => new DeepAcyclicGraph({ rootNodes: [node] })).toThrow(
      /Duplicate verdict/,
    );
  });

  test("rejects mixing single-turn and multi-turn roots", () => {
    const single = binaryGate("Is the answer correct?");
    single.addVerdict(true, { score: 10 });
    single.addVerdict(false, { score: 0 });
    const multi = new ConversationalBinaryJudgementNode({
      criteria: "Was the user helped?",
      evaluationParams: [MultiTurnParams.CONTENT],
    });
    multi.addVerdict(true, { score: 10 });
    multi.addVerdict(false, { score: 0 });

    expect(() => new DeepAcyclicGraph({ rootNodes: [single, multi] })).toThrow(
      /cannot mix multi and single turn nodes/,
    );
  });
});

describe("ConversationalDAGMetric", () => {
  const conversation = new ConversationalTestCase({
    turns: [
      new Turn({ role: "user", content: "My order never arrived." }),
      new Turn({ role: "assistant", content: "I've refunded you in full." }),
    ],
  });

  test("scores a conversation from a windowed judgement", async () => {
    const gate = new ConversationalBinaryJudgementNode({
      criteria: "Was the user helped?",
      evaluationParams: [MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
      turnWindow: [0, 1],
    });
    gate.addVerdict(true, { score: 10 });
    gate.addVerdict(false, { score: 0 });

    const metric = new ConversationalDAGMetric({
      name: "Helpfulness",
      dag: new DeepAcyclicGraph({ rootNodes: [gate] }),
      model: new StubModel([
        ["Was the user helped?", { verdict: true, reason: "Refund issued." }],
      ]),
      includeReason: false,
      showIndicator: false,
    });

    expect(await metric.measure(conversation)).toBe(1);
  });

  test("rejects an out-of-range turn window", () => {
    expect(() => isValidTurnWindow([0, 2], conversation.turns)).toThrow(
      /'turnWindow' passed is invalid/,
    );
    expect(isValidTurnWindow([0, 1], conversation.turns)).toBe(true);
  });
});

describe("DAG serialization", () => {
  test("round trips through JSON", () => {
    const task = new ConversationalTaskNode({
      instructions: "Summarize the conversation.",
      outputLabel: "Summary",
      evaluationParams: [MultiTurnParams.CONTENT],
      turnWindow: [0, 1],
      label: "summary",
    });
    const gate = new ConversationalBinaryJudgementNode({
      criteria: "Was the user helped?",
      evaluationParams: [MultiTurnParams.CONTENT],
    });
    gate.addVerdict(true, { score: 10 });
    gate.addVerdict(false, { score: 0 });
    task.addNode(gate);

    const json = dagToJson(new DeepAcyclicGraph({ rootNodes: [task] }));
    const restored = dagFromJson(json, true);

    expect(restored.multiturn).toBe(true);
    const restoredTask = restored.rootNodes[0] as ConversationalTaskNode;
    expect(restoredTask.outputLabel).toBe("Summary");
    expect(restoredTask.turnWindow).toEqual([0, 1]);
    expect(restoredTask.label).toBe("summary");
    expect(restoredTask.children).toHaveLength(1);

    const restoredGate = restoredTask
      .children[0] as ConversationalBinaryJudgementNode;
    expect(restoredGate.criteria).toBe("Was the user helped?");
    expect(restoredGate.children.map((child) => child.score)).toEqual([10, 0]);
  });
});
