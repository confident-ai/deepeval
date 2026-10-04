// Transcription Accuracy: which turns get paired into an exchange, what
// reaches the prompt, the score arithmetic, and the refusal to run on a
// conversation that carries no transcription at all. A scripted judge stands
// in for the LLM, so none of this needs an API key.

import {
  TranscriptionAccuracyMetric,
  getTranscribedExchanges,
} from "@/metrics/transcription-accuracy";
import { ConversationalTestCase, Turn } from "@/test-case";
import { MissingTestCaseParamsError } from "@/errors";
import { DeepEvalBaseLLM, type GenerationResult } from "@/models";

/** Returns one scripted verdict per exchange and records its prompts. */
class ScriptedJudge extends DeepEvalBaseLLM {
  readonly prompts: string[] = [];

  constructor(private readonly scripted: string[]) {
    super("scripted-judge");
  }

  async generate<T = string>(prompt: string): Promise<GenerationResult<T>> {
    this.prompts.push(prompt);
    const output = prompt.includes("transcription accuracy score is")
      ? { reason: "scripted reason" }
      : {
          verdicts: this.scripted.map((verdict) => ({
            verdict,
            reason: "scripted reason",
          })),
        };
    return { output, cost: 0 } as GenerationResult<T>;
  }

  getModelName(): string {
    return "scripted-judge";
  }
}

const measure = async (
  turns: Turn[],
  scripted: string[],
  options: Record<string, unknown> = {},
) => {
  const model = new ScriptedJudge(scripted);
  const metric = new TranscriptionAccuracyMetric({
    model,
    showIndicator: false,
    ...options,
  });
  await metric.measure(new ConversationalTestCase({ turns }));
  return { metric, model };
};

const exchange = (index: number, providerTranscription?: string) => [
  new Turn({ role: "user", content: `question ${index}` }),
  new Turn({
    role: "assistant",
    content: `answer ${index}`,
    providerTranscription: providerTranscription ?? `question ${index}`,
  }),
];

describe("pairing caller speech with what the agent heard", () => {
  it("pairs a caller turn with the reply that transcribed it", () => {
    expect(
      getTranscribedExchanges([
        new Turn({ role: "user", content: "I can start in two weeks" }),
        new Turn({
          role: "assistant",
          content: "Two weeks works.",
          providerTranscription: "I can start in two months",
        }),
      ]),
    ).toEqual([
      {
        spoken: "I can start in two weeks",
        transcribed: "I can start in two months",
        agent_reply: "Two weeks works.",
      },
    ]);
  });

  it("treats consecutive caller turns as one exchange", () => {
    // A barge-in appends a second user turn before the agent answers, and the
    // transcription covers everything it heard since it last spoke.
    const exchanges = getTranscribedExchanges([
      new Turn({ role: "user", content: "Hold on" }),
      new Turn({ role: "user", content: "I meant two weeks" }),
      new Turn({
        role: "assistant",
        content: "Understood.",
        providerTranscription: "hold on i meant two weeks",
      }),
    ]);

    expect(exchanges).toHaveLength(1);
    expect(exchanges[0].spoken).toBe("Hold on I meant two weeks");
  });

  it("drops the caller turn whose reply reported no transcription", () => {
    const exchanges = getTranscribedExchanges([
      new Turn({ role: "user", content: "first question" }),
      new Turn({ role: "assistant", content: "unmeasured reply" }),
      new Turn({ role: "user", content: "second question" }),
      new Turn({
        role: "assistant",
        content: "measured reply",
        providerTranscription: "second question",
      }),
    ]);

    expect(exchanges.map((e) => e.spoken)).toEqual(["second question"]);
  });

  it("opens no exchange for an agent that speaks first", () => {
    expect(
      getTranscribedExchanges([
        new Turn({
          role: "assistant",
          content: "Hello, how can I help?",
          providerTranscription: "",
        }),
        new Turn({ role: "user", content: "hi" }),
      ]),
    ).toEqual([]);
  });

  it("still judges an empty transcription", () => {
    // Hearing nothing is a transcription failure, not a missing field.
    const exchanges = getTranscribedExchanges([
      new Turn({ role: "user", content: "I can start in two weeks" }),
      new Turn({
        role: "assistant",
        content: "Sorry?",
        providerTranscription: "",
      }),
    ]);

    expect(exchanges).toHaveLength(1);
    expect(exchanges[0].transcribed).toBe("");
  });
});

describe("scoring", () => {
  it("scores the share of faithfully heard turns", async () => {
    const turns = [0, 1, 2, 3].flatMap((index) => exchange(index));
    const { metric } = await measure(turns, ["yes", "no", "yes", "yes"]);

    expect(metric.score).toBe(0.75);
    expect(metric.success).toBe(true);
  });

  it("scores one when every turn was heard faithfully", async () => {
    const { metric } = await measure(exchange(0), ["yes"]);

    expect(metric.score).toBe(1);
  });

  it("clamps anything short of perfect to zero in strict mode", async () => {
    const turns = [0, 1].flatMap((index) => exchange(index));
    const { metric } = await measure(turns, ["yes", "no"], {
      strictMode: true,
    });

    expect(metric.score).toBe(0);
    expect(metric.success).toBe(false);
  });
});

describe("the prompt", () => {
  it("shows the judge what was spoken and what was heard", async () => {
    const { model } = await measure(
      [
        new Turn({ role: "user", content: "Siobhan Kavanagh" }),
        new Turn({
          role: "assistant",
          content: "Hello Siobhan!",
          providerTranscription: "Siobhan Cavanaugh",
        }),
      ],
      ["no"],
      { includeReason: false },
    );

    const prompt = model.prompts[0];
    expect(prompt).toContain("Siobhan Kavanagh");
    expect(prompt).toContain("Siobhan Cavanaugh");
    expect(prompt).toContain("Hello Siobhan!");
  });
});

describe("a conversation without the field", () => {
  it("is refused rather than scored zero", async () => {
    // A text conversation, or a phone call, can never satisfy this metric —
    // saying so beats looking like the agent failed.
    const metric = new TranscriptionAccuracyMetric({
      model: new ScriptedJudge(["yes"]),
      showIndicator: false,
    });

    await expect(
      metric.measure(
        new ConversationalTestCase({
          turns: [
            new Turn({ role: "user", content: "hello" }),
            new Turn({ role: "assistant", content: "hi there" }),
          ],
        }),
      ),
    ).rejects.toThrow(MissingTestCaseParamsError);
  });

  it("runs when a single turn carries one", async () => {
    const { metric } = await measure(
      [
        new Turn({ role: "user", content: "first" }),
        new Turn({ role: "assistant", content: "unmeasured" }),
        ...exchange(1, "question 1"),
      ],
      ["yes"],
    );

    expect(metric.score).toBe(1);
  });
});
