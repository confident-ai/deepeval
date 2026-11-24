// Every metric wired for eval modes, checked under each mode (a port of
// Python's tests/test_metrics/test_eval_mode_metrics.py):
//
// (a) `system_one`: Jev runs the whole metric in one request, no LLM call,
//     and the reason is deterministic.
// (b) `hybrid`: the LLM extracts, Jev takes the decisions, and the LLM
//     decision prompt is never sent.
// (c) `hybrid` with Jev down: falls back to the LLM and records why.
// (d) `llm`: no System One model is built and Jev is never called.

import type { ZodType } from "zod";
import {
  AnswerRelevancyMetric,
  ArgumentCorrectnessMetric,
  BiasMetric,
  ContextualPrecisionMetric,
  ContextualRecallMetric,
  ContextualRelevancyMetric,
  ConversationCompletenessMetric,
  FaithfulnessMetric,
  GoalAccuracyMetric,
  HallucinationMetric,
  JsonCorrectnessMetric,
  KnowledgeRetentionMetric,
  MCPTaskCompletionMetric,
  MCPUseMetric,
  MisuseMetric,
  MultiTurnMCPUseMetric,
  NonAdviceMetric,
  PIILeakageMetric,
  PlanAdherenceMetric,
  PlanQualityMetric,
  PromptAlignmentMetric,
  RoleAdherenceMetric,
  RoleViolationMetric,
  StepEfficiencyMetric,
  SummarizationMetric,
  TaskCompletionMetric,
  ToolCorrectnessMetric,
  ToolUseMetric,
  TopicAdherenceMetric,
  ToxicityMetric,
  TurnContextualPrecisionMetric,
  TurnContextualRecallMetric,
  TurnContextualRelevancyMetric,
  TurnFaithfulnessMetric,
  TurnRelevancyMetric,
} from "@/metrics";
import { DeepEvalBaseLLM, type GenerationResult } from "@/models";
import {
  ConversationalTestCase,
  LLMTestCase,
  MCPServer,
  MCPToolCall,
  ToolCall,
  Turn,
} from "@/test-case";
import {
  ExplodingLLM,
  ExplodingSystemOneModel,
  FakeSystemOneModel,
  answerEverything,
} from "./system-one-fakes";

beforeEach(() => {
  delete process.env.DEEPEVAL_EVAL_MODE;
  delete process.env.DEEPEVAL_MODE;
});

/** Replies with the first route whose key is in the prompt, else `fallback`. */
class RoutingLLM extends DeepEvalBaseLLM {
  prompts: string[] = [];

  constructor(
    private readonly routes: Array<[string, unknown]>,
    private readonly fallback: unknown,
  ) {
    super("routing-llm");
  }

  async generate<T = string>(
    prompt: string,
    _schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    this.prompts.push(prompt);
    const route = this.routes.find(([key]) => prompt.includes(key));
    return { output: (route ? route[1] : this.fallback) as T, cost: 0 };
  }

  wasSent(key: string): boolean {
    return this.prompts.some((p) => p.includes(key));
  }

  getModelName(): string {
    return "routing-llm";
  }
}

const jev = () => new FakeSystemOneModel({ answerFn: answerEverything() });
const jevDown = () =>
  new ExplodingSystemOneModel(
    Object.assign(new Error("down"), { name: "APIConnectionError" }),
  );

// Test cases

const TRACE = {
  name: "agent",
  type: "agent",
  input: { input: "Book a flight to Paris" },
  output: "Booked",
  tokens: 123,
  children: [
    {
      name: "search_flights",
      type: "tool",
      input: { dest: "Paris" },
      output: ["AF1"],
      toolsCalled: [{ name: "search" }],
    },
  ],
};
const TOOLS = [
  new ToolCall({ name: "get_weather", description: "Weather for a city" }),
];

function tracedCase(trace = true): LLMTestCase {
  const tc = new LLMTestCase({
    input: "Book a flight to Paris",
    actualOutput: "Booked AF1",
    toolsCalled: [new ToolCall({ name: "get_weather" })],
    expectedTools: [new ToolCall({ name: "get_weather" })],
  });
  if (trace) tc._traceDict = TRACE;
  return tc;
}

const ragCase = () =>
  new LLMTestCase({
    input: "What did Einstein win?",
    actualOutput: "Einstein won the Nobel Prize. He was born in Ulm.",
    expectedOutput: "Einstein won the Nobel Prize. He was born in Ulm.",
    retrievalContext: [
      "Einstein won the Nobel Prize. There was a cat.",
      "Einstein was born in Ulm.",
    ],
    context: ["Einstein won the Nobel Prize.", "Einstein was born in Ulm."],
  });

const ragConversation = () =>
  new ConversationalTestCase({
    turns: [
      new Turn({ role: "user", content: "What did Einstein win?" }),
      new Turn({
        role: "assistant",
        content: "The Nobel Prize.",
        retrievalContext: ["Einstein won the Nobel Prize. There was a cat."],
      }),
      new Turn({ role: "user", content: "Where was he born?" }),
      new Turn({
        role: "assistant",
        content: "Ulm.",
        retrievalContext: ["Einstein was born in Ulm."],
      }),
    ],
    expectedOutcome: "Einstein won the Nobel Prize. He was born in Ulm.",
  });

const supportConversation = () =>
  new ConversationalTestCase({
    chatbotRole: "A polite airline support agent",
    turns: [
      new Turn({
        role: "user",
        content: "Hi, I'm Ann. I want to book a flight to Paris.",
      }),
      new Turn({
        role: "assistant",
        content: "Sure Ann, when would you like to fly?",
      }),
      new Turn({
        role: "user",
        content: "Next Monday. Also can I get a refund for my last trip?",
      }),
      new Turn({
        role: "assistant",
        content: "Booked for Monday. Your refund is on its way.",
      }),
    ],
  });

const toolConversation = () =>
  new ConversationalTestCase({
    turns: [
      new Turn({ role: "user", content: "Weather in Paris?" }),
      new Turn({
        role: "assistant",
        content: "It's sunny.",
        toolsCalled: [
          new ToolCall({
            name: "get_weather",
            inputParameters: { city: "Paris" },
          }),
        ],
      }),
    ],
  });

function mcpServerAndCall(): [MCPServer, MCPToolCall] {
  const call = new MCPToolCall({
    name: "get_weather",
    args: { city: "Paris" },
    result: {
      content: [{ type: "text", text: "sunny" }],
      structuredContent: { result: "sunny" },
    },
  });
  const server = new MCPServer({
    serverName: "weather",
    availableTools: [
      {
        name: "get_weather",
        description: "Get weather",
        inputSchema: {
          type: "object",
          properties: { city: { type: "string" } },
        },
      } as any,
    ],
  });
  return [server, call];
}

function mcpCase(): LLMTestCase {
  const [server, call] = mcpServerAndCall();
  return new LLMTestCase({
    input: "Weather in Paris?",
    actualOutput: "It's sunny.",
    mcpServers: [server],
    mcpToolsCalled: [call],
  });
}

function mcpConversation(): ConversationalTestCase {
  const [server, call] = mcpServerAndCall();
  return new ConversationalTestCase({
    turns: [
      new Turn({ role: "user", content: "Weather in Paris?" }),
      new Turn({
        role: "assistant",
        content: "Checking",
        mcpToolsCalled: [call],
      }),
      new Turn({ role: "assistant", content: "It's sunny." }),
    ],
    mcpServers: [server],
  });
}

const safetyCase = () =>
  new LLMTestCase({
    input: "Should I buy this stock?",
    actualOutput: "Sure, put your savings in it. My email is a@b.com.",
  });

// Metric cases

const REASON = { reason: "LLM reason." };
const SCORE = { score: 0.4, reason: "LLM reason." };
const PLAN = {
  task: "Book a flight",
  plan: ["search", "book"],
  score: 0.4,
  reason: "r",
};

type Metric = { measure(tc: any): Promise<number> } & Record<string, any>;

interface Case {
  id: string;
  metric: (options: Record<string, unknown>) => Metric;
  testCase: () => any;
  /** Hybrid coverage is ported for the cases Python checks it on. */
  hybrid?: {
    /** Substring of the LLM decision prompt Jev replaces. */
    decisionKey: string;
    /** Score from the LLM chain (the routes answer every decision negatively). */
    llmScore: number;
    routes?: Array<[string, unknown]>;
    fallback?: unknown;
    jevCalls?: number;
    /** Score when Jev answers every question at its most positive. */
    jevScore?: number;
  };
  /** False when the Jev reason is one part of a composite reason. */
  reasonIsJev?: boolean;
  checkState?: (state: any) => void;
}

const noTraceNoise = (state: any) => {
  expect(state.trace.tokens).toBeUndefined();
  expect(state.trace.children[0].tools_called).toEqual([{ name: "search" }]);
};

const CASES: Case[] = [
  // Single-turn RAG and pilot metrics: whole-chain coverage.
  {
    id: "AnswerRelevancy",
    metric: (o) => new AnswerRelevancyMetric(o),
    testCase: ragCase,
  },
  {
    id: "Faithfulness",
    metric: (o) => new FaithfulnessMetric(o),
    testCase: ragCase,
  },
  {
    id: "ContextualPrecision",
    metric: (o) => new ContextualPrecisionMetric(o),
    testCase: ragCase,
  },
  {
    id: "ContextualRecall",
    metric: (o) => new ContextualRecallMetric(o),
    testCase: ragCase,
    hybrid: {
      decisionKey: "For EACH sentence in the given expected output",
      routes: [
        [
          "For EACH sentence in the given expected output",
          { verdicts: [{ verdict: "no", reason: "r" }] },
        ],
      ],
      fallback: REASON,
      llmScore: 0,
    },
  },
  {
    id: "ContextualRelevancy",
    metric: (o) => new ContextualRelevancyMetric(o),
    testCase: ragCase,
    hybrid: {
      decisionKey: "Based on the input and context, please generate",
      routes: [
        [
          "Based on the input and context, please generate",
          { verdicts: [{ statement: "s", verdict: "no" }] },
        ],
      ],
      fallback: REASON,
      llmScore: 0,
      jevCalls: 2,
    },
  },
  {
    id: "Hallucination",
    metric: (o) => new HallucinationMetric(o),
    testCase: ragCase,
  },
  {
    id: "Summarization",
    metric: (o) => new SummarizationMetric(o),
    testCase: ragCase,
  },
  // Safety.
  { id: "Bias", metric: (o) => new BiasMetric(o), testCase: safetyCase },
  {
    id: "Toxicity",
    metric: (o) => new ToxicityMetric(o),
    testCase: safetyCase,
  },
  {
    id: "PIILeakage",
    metric: (o) => new PIILeakageMetric(o),
    testCase: safetyCase,
  },
  {
    id: "Misuse",
    metric: (o) => new MisuseMetric({ domain: "finance", ...o }),
    testCase: safetyCase,
  },
  {
    id: "NonAdvice",
    metric: (o) => new NonAdviceMetric({ adviceTypes: ["financial"], ...o }),
    testCase: safetyCase,
  },
  {
    id: "RoleViolation",
    metric: (o) => new RoleViolationMetric({ role: "helpful assistant", ...o }),
    testCase: safetyCase,
  },
  {
    id: "PromptAlignment",
    metric: (o) =>
      new PromptAlignmentMetric({
        promptInstructions: ["Reply in English."],
        ...o,
      }),
    testCase: safetyCase,
  },
  // Agentic.
  {
    id: "StepEfficiency",
    metric: (o) => new StepEfficiencyMetric(o),
    testCase: tracedCase,
    checkState: noTraceNoise,
    hybrid: {
      decisionKey: "**efficiency auditor**",
      fallback: PLAN,
      llmScore: 0.4,
    },
  },
  {
    id: "PlanAdherence",
    metric: (o) => new PlanAdherenceMetric(o),
    testCase: tracedCase,
    checkState: noTraceNoise,
    hybrid: {
      decisionKey: "**adversarial plan adherence evaluator**",
      fallback: PLAN,
      llmScore: 0.4,
    },
  },
  {
    id: "PlanQuality",
    metric: (o) => new PlanQualityMetric(o),
    testCase: tracedCase,
    checkState: noTraceNoise,
    hybrid: {
      decisionKey: "**plan quality evaluator**",
      fallback: PLAN,
      llmScore: 0.4,
    },
  },
  {
    id: "TaskCompletion",
    metric: (o) => new TaskCompletionMetric(o),
    testCase: tracedCase,
    checkState: noTraceNoise,
    hybrid: {
      decisionKey: "Given the task (desired outcome) and the actual achieved",
      routes: [
        [
          "Given a nested workflow trace",
          { task: "Book a flight", outcome: "Booked AF1" },
        ],
        [
          "Given the task (desired outcome) and the actual achieved",
          { verdict: 0.4, reason: "r" },
        ],
      ],
      fallback: REASON,
      llmScore: 0.4,
    },
  },
  {
    id: "ArgumentCorrectness",
    metric: (o) => new ArgumentCorrectnessMetric(o),
    testCase: tracedCase,
  },
  {
    id: "ToolCorrectness",
    metric: (o) => new ToolCorrectnessMetric({ availableTools: TOOLS, ...o }),
    testCase: tracedCase,
    reasonIsJev: false,
    hybrid: {
      decisionKey: "assessing the **Tool Selection** quality",
      fallback: SCORE,
      llmScore: 0.4,
    },
  },
  {
    id: "MCPUse",
    metric: (o) => new MCPUseMetric(o),
    testCase: mcpCase,
    hybrid: {
      decisionKey: "Evaluate whether the tools (primitives) selected",
      fallback: SCORE,
      llmScore: 0.4,
      jevCalls: 2,
    },
  },
  {
    id: "MultiTurnMCPUse",
    metric: (o) => new MultiTurnMCPUseMetric(o),
    testCase: mcpConversation,
    hybrid: {
      decisionKey: "Evaluate whether the tools, resources, and prompts",
      fallback: SCORE,
      llmScore: 0.4,
      jevCalls: 2,
    },
  },
  {
    id: "MCPTaskCompletion",
    metric: (o) => new MCPTaskCompletionMetric(o),
    testCase: mcpConversation,
    hybrid: {
      decisionKey: "Evaluate whether the user's task has been successfully",
      fallback: SCORE,
      llmScore: 0.4,
    },
  },
  // Multi-turn.
  {
    id: "TurnRelevancy",
    metric: (o) => new TurnRelevancyMetric(o),
    testCase: ragConversation,
    hybrid: {
      decisionKey: "Based on the given list of message exchanges",
      routes: [
        [
          "Based on the given list of message exchanges",
          { verdict: "no", reason: "r" },
        ],
      ],
      fallback: REASON,
      llmScore: 0,
      jevCalls: 2,
    },
  },
  {
    id: "TurnFaithfulness",
    metric: (o) => new TurnFaithfulnessMetric(o),
    testCase: ragConversation,
  },
  {
    id: "TurnContextualPrecision",
    metric: (o) => new TurnContextualPrecisionMetric(o),
    testCase: ragConversation,
  },
  {
    id: "TurnContextualRecall",
    metric: (o) => new TurnContextualRecallMetric(o),
    testCase: ragConversation,
    hybrid: {
      decisionKey: "For EACH sentence in the given assistant output",
      routes: [
        [
          "For EACH sentence in the given assistant output",
          { verdicts: [{ verdict: "no", reason: "r" }] },
        ],
      ],
      fallback: REASON,
      llmScore: 0,
      jevCalls: 2,
    },
  },
  {
    id: "TurnContextualRelevancy",
    metric: (o) => new TurnContextualRelevancyMetric(o),
    testCase: ragConversation,
    hybrid: {
      decisionKey: "Based on the user message and context, please generate",
      routes: [
        [
          "Based on the user message and context, please generate",
          { verdicts: [{ statement: "s", verdict: "no" }] },
        ],
      ],
      fallback: REASON,
      llmScore: 0,
      jevCalls: 3,
    },
  },
  {
    id: "ConversationCompleteness",
    metric: (o) => new ConversationCompletenessMetric(o),
    testCase: supportConversation,
    hybrid: {
      decisionKey: "whether given user intention was satisfied",
      routes: [
        [
          "extract all user intentions",
          { intentions: ["book a flight", "get a refund"] },
        ],
        [
          "whether given user intention was satisfied",
          { verdict: "no", reason: "x" },
        ],
      ],
      fallback: REASON,
      llmScore: 0,
      jevCalls: 2,
    },
  },
  {
    id: "KnowledgeRetention",
    metric: (o) => new KnowledgeRetentionMetric(o),
    testCase: supportConversation,
    hybrid: {
      decisionKey: "**contradicts** or **forgets**",
      routes: [
        ["extract **only the factual information", { data: { Name: "Ann" } }],
        ["**contradicts** or **forgets**", { verdict: "no" }],
      ],
      fallback: REASON,
      llmScore: 1,
      jevCalls: 2,
      // A Noul "yes" here means the assistant forgot.
      jevScore: 0,
    },
  },
  {
    id: "RoleAdherence",
    metric: (o) => new RoleAdherenceMetric(o),
    testCase: supportConversation,
    hybrid: {
      decisionKey: "did not adhere to the specified chatbot role",
      routes: [
        [
          "did not adhere to the specified chatbot role",
          { verdicts: [{ index: 1, reason: "x" }] },
        ],
      ],
      fallback: REASON,
      llmScore: 0.5,
      jevCalls: 2,
    },
  },
  {
    id: "GoalAccuracy",
    metric: (o) => new GoalAccuracyMetric(o),
    testCase: supportConversation,
    hybrid: {
      decisionKey: "**goal accuracy**",
      routes: [
        ["**goal accuracy**", SCORE],
        ["**planning quality**", SCORE],
      ],
      fallback: "final llm reason",
      llmScore: 0.4,
      jevCalls: 4,
    },
  },
  {
    id: "TopicAdherence",
    metric: (o) =>
      new TopicAdherenceMetric({
        relevantTopics: ["travel booking", "refunds"],
        ...o,
      }),
    testCase: supportConversation,
    hybrid: {
      decisionKey: "four possible verdicts",
      routes: [
        [
          "extract question-answer (QA) pairs",
          {
            qaPairs: [
              { question: "q1", response: "r1" },
              { question: "q2", response: "r2" },
            ],
            qa_pairs: [
              { question: "q1", response: "r1" },
              { question: "q2", response: "r2" },
            ],
          },
        ],
        ["four possible verdicts", { verdict: "FN", reason: "x" }],
      ],
      fallback: REASON,
      llmScore: 0,
      jevCalls: 8,
    },
  },
  {
    id: "ToolUse",
    metric: (o) => new ToolUseMetric({ availableTools: TOOLS, ...o }),
    testCase: toolConversation,
    checkState: (state) => {
      expect(state.available_tools[0].name).toBe("get_weather");
      expect(state.turns[1].tools_called).toBeTruthy();
    },
    hybrid: {
      decisionKey: "**Tool Selection Quality**",
      fallback: PLAN,
      llmScore: 0.4,
      jevCalls: 2,
    },
  },
];

const HYBRID_CASES = CASES.filter((c) => c.hybrid);

describe.each(CASES)("$id", (c) => {
  it("system_one: decides in one Jev request with no LLM call", async () => {
    const reasons: string[] = [];
    let model = jev();
    for (let i = 0; i < 2; i++) {
      model = jev();
      const metric = c.metric({
        model: new ExplodingLLM(),
        systemOneModel: model,
        evalMode: "system_one",
        showIndicator: false,
      });
      const score = await metric.measure(c.testCase());
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
      expect(model.calls).toHaveLength(1);
      if (c.reasonIsJev === false) {
        expect(metric.reason).toContain("Decided by fake-jev");
      } else {
        expect(metric.reason?.startsWith("Decided by fake-jev")).toBe(true);
      }
      reasons.push(metric.reason as string);
    }
    expect(reasons[0]).toBe(reasons[1]);
    c.checkState?.(model.calls[0].state);
  });

  it("llm: builds no System One model", () => {
    const metric = c.metric({
      model: new RoutingLLM([], REASON),
      systemOneModel: jev(),
      evalMode: "llm",
      showIndicator: false,
    });
    expect(metric.systemOneModel).toBeUndefined();
  });
});

describe.each(HYBRID_CASES)("$id (hybrid)", (c) => {
  const h = c.hybrid!;
  const llm = () => new RoutingLLM(h.routes ?? [], h.fallback ?? REASON);

  it("hybrid: Jev takes the decisions", async () => {
    const model = jev();
    const routing = llm();
    const metric = c.metric({
      model: routing,
      systemOneModel: model,
      evalMode: "hybrid",
      showIndicator: false,
    });
    const score = await metric.measure(c.testCase());
    expect(model.calls).toHaveLength(h.jevCalls ?? 1);
    expect(routing.wasSent(h.decisionKey)).toBe(false);
    expect(score).toBeCloseTo(h.jevScore ?? 1);
    expect(metric.systemOneFallbackReason).toBeUndefined();
    expect(metric.reason).toBeTruthy();
  });

  it("hybrid: falls back to the LLM when Jev fails", async () => {
    const model = jevDown();
    const routing = llm();
    const metric = c.metric({
      model: routing,
      systemOneModel: model,
      evalMode: "hybrid",
      showIndicator: false,
    });
    const score = await metric.measure(c.testCase());
    expect(model.calls.length).toBeGreaterThan(0);
    expect(routing.wasSent(h.decisionKey)).toBe(true);
    expect(score).toBeCloseTo(h.llmScore);
    expect(metric.systemOneFallbackReason).toContain("down");
  });

  it("llm: never calls Jev", async () => {
    const routing = llm();
    const metric = c.metric({
      model: routing,
      evalMode: "llm",
      showIndicator: false,
    });
    await metric.measure(c.testCase());
    expect(routing.wasSent(h.decisionKey)).toBe(true);
    expect(metric.confidence).toBeUndefined();
  });
});

describe("JsonCorrectness", () => {
  it("needs no model of either kind under system_one", async () => {
    const { z } = await import("zod");
    const metric = new JsonCorrectnessMetric({
      expectedSchema: z.object({ a: z.number() }),
      evalMode: "system_one",
      showIndicator: false,
    });
    expect(metric.model).toBeUndefined();
    expect(metric.systemOneModel).toBeUndefined();
    expect(metric.evaluationModel).toBeUndefined();
    expect(
      await metric.measure(
        new LLMTestCase({ input: "x", actualOutput: '{"a": "no"}' }),
      ),
    ).toBe(0);
    expect(metric.reason).toBeTruthy();
  });
});
