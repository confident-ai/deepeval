// Deterministic repro for the TS-side verdict-count denominator bug.
// Mirrors the Python finding (issue #3346): the score denominator is the
// number of verdicts the judge *returned*, not the number of items that
// were up for judgment. A truncated or empty verdict list therefore
// inflates the score. Zero API keys: a stub model returns fixed payloads.

import { AnswerRelevancyMetric } from "@/metrics/answer-relevancy/answer-relevancy";
import {
  StatementsSchema,
  VerdictsSchema as AnswerRelevancyVerdictsSchema,
  AnswerRelevancyScoreReasonSchema,
} from "@/metrics/answer-relevancy/schema";
import { ArgumentCorrectnessMetric } from "@/metrics/argument-correctness/argument-correctness";
import {
  VerdictsSchema as ArgumentCorrectnessVerdictsSchema,
  ArgumentCorrectnessScoreReasonSchema,
} from "@/metrics/argument-correctness/schema";
import { BiasMetric } from "@/metrics/bias/bias";
import {
  OpinionsSchema as BiasOpinionsSchema,
  VerdictsSchema as BiasVerdictsSchema,
  BiasScoreReasonSchema,
} from "@/metrics/bias/schema";
import { FaithfulnessMetric } from "@/metrics/faithfulness/faithfulness";
import {
  TruthsSchema,
  ClaimsSchema,
  VerdictsSchema as FaithfulnessVerdictsSchema,
  FaithfulnessScoreReasonSchema,
} from "@/metrics/faithfulness/schema";
import { HallucinationMetric } from "@/metrics/hallucination/hallucination";
import {
  VerdictsSchema as HallucinationVerdictsSchema,
  HallucinationScoreReasonSchema,
} from "@/metrics/hallucination/schema";
import { MisuseMetric } from "@/metrics/misuse/misuse";
import {
  MisusesSchema,
  VerdictsSchema as MisuseVerdictsSchema,
  MisuseScoreReasonSchema,
} from "@/metrics/misuse/schema";
import { NonAdviceMetric } from "@/metrics/non-advice/non-advice";
import {
  AdvicesSchema,
  VerdictsSchema as NonAdviceVerdictsSchema,
  NonAdviceScoreReasonSchema,
} from "@/metrics/non-advice/schema";
import { PIILeakageMetric } from "@/metrics/pii-leakage/pii-leakage";
import {
  ExtractedPIISchema,
  VerdictsSchema as PIILeakageVerdictsSchema,
  PIILeakageScoreReasonSchema,
} from "@/metrics/pii-leakage/schema";
import { PromptAlignmentMetric } from "@/metrics/prompt-alignment/prompt-alignment";
import {
  VerdictsSchema as PromptAlignmentVerdictsSchema,
  PromptAlignmentScoreReasonSchema,
} from "@/metrics/prompt-alignment/schema";
import { RoleViolationMetric } from "@/metrics/role-violation/role-violation";
import {
  RoleViolationsSchema,
  VerdictsSchema as RoleViolationVerdictsSchema,
  RoleViolationScoreReasonSchema,
} from "@/metrics/role-violation/schema";
import { SummarizationMetric } from "@/metrics/summarization/summarization";
import {
  VerdictsSchema as SummarizationVerdictsSchema,
  QuestionsSchema,
  AnswersSchema,
  SummarizationScoreReasonSchema,
} from "@/metrics/summarization/schema";
import { ToxicityMetric } from "@/metrics/toxicity/toxicity";
import {
  OpinionsSchema as ToxicityOpinionsSchema,
  VerdictsSchema as ToxicityVerdictsSchema,
  ToxicityScoreReasonSchema,
} from "@/metrics/toxicity/schema";
import { LLMTestCase, ToolCall } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { GenerationResult } from "@/models";
import type { ZodType } from "zod";

const THREE_ITEMS = ["item-1", "item-2", "item-3"];

/**
 * Dispatches on the schema each metric stage passes, so one stub covers the
 * whole measure() flow: extraction → verdicts → reason.
 */
class FixedJudge extends DeepEvalBaseLLM {
  constructor(
    private readonly verdicts: { verdict: string; reason: string }[],
    private readonly extractedItems: string[] = THREE_ITEMS,
  ) {
    super("fixed-judge");
  }

  async generate<T = string>(
    _prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    let output: unknown;
    if (schema === (ExtractedPIISchema as unknown as ZodType<T>)) {
      output = { extracted_pii: this.extractedItems };
    } else if (schema === (StatementsSchema as unknown as ZodType<T>)) {
      output = { statements: this.extractedItems };
    } else if (
      schema === (BiasOpinionsSchema as unknown as ZodType<T>) ||
      schema === (ToxicityOpinionsSchema as unknown as ZodType<T>)
    ) {
      output = { opinions: this.extractedItems };
    } else if (schema === (TruthsSchema as unknown as ZodType<T>)) {
      output = { truths: ["truth-1", "truth-2"] };
    } else if (schema === (ClaimsSchema as unknown as ZodType<T>)) {
      output = { claims: this.extractedItems };
    } else if (schema === (MisusesSchema as unknown as ZodType<T>)) {
      output = { misuses: this.extractedItems };
    } else if (schema === (AdvicesSchema as unknown as ZodType<T>)) {
      output = { advices: this.extractedItems };
    } else if (schema === (RoleViolationsSchema as unknown as ZodType<T>)) {
      output = { role_violations: this.extractedItems };
    } else if (schema === (QuestionsSchema as unknown as ZodType<T>)) {
      output = { questions: ["q1"] };
    } else if (schema === (AnswersSchema as unknown as ZodType<T>)) {
      output = { answers: ["yes"] };
    } else if (
      schema === (PIILeakageVerdictsSchema as unknown as ZodType<T>) ||
      schema === (AnswerRelevancyVerdictsSchema as unknown as ZodType<T>) ||
      schema === (ArgumentCorrectnessVerdictsSchema as unknown as ZodType<T>) ||
      schema === (BiasVerdictsSchema as unknown as ZodType<T>) ||
      schema === (FaithfulnessVerdictsSchema as unknown as ZodType<T>) ||
      schema === (HallucinationVerdictsSchema as unknown as ZodType<T>) ||
      schema === (MisuseVerdictsSchema as unknown as ZodType<T>) ||
      schema === (NonAdviceVerdictsSchema as unknown as ZodType<T>) ||
      schema === (PromptAlignmentVerdictsSchema as unknown as ZodType<T>) ||
      schema === (RoleViolationVerdictsSchema as unknown as ZodType<T>) ||
      schema === (SummarizationVerdictsSchema as unknown as ZodType<T>) ||
      schema === (ToxicityVerdictsSchema as unknown as ZodType<T>)
    ) {
      output = { verdicts: this.verdicts };
    } else if (
      schema === (PIILeakageScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (AnswerRelevancyScoreReasonSchema as unknown as ZodType<T>) ||
      schema ===
        (ArgumentCorrectnessScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (BiasScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (FaithfulnessScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (HallucinationScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (MisuseScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (NonAdviceScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (PromptAlignmentScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (RoleViolationScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (SummarizationScoreReasonSchema as unknown as ZodType<T>) ||
      schema === (ToxicityScoreReasonSchema as unknown as ZodType<T>)
    ) {
      output = { reason: "stub reason" };
    } else {
      throw new Error("unexpected schema in test");
    }
    return { output, cost: 0 } as GenerationResult<T>;
  }

  getModelName(): string {
    return "fixed-judge";
  }
}

function testCase(): LLMTestCase {
  return new LLMTestCase({
    input: "Send me the contact card",
    actualOutput:
      "You can reach Alice at alice@example.com, 555-1234, card 4242-4242-4242-4242.",
    retrievalContext: ["ctx-1", "ctx-2", "ctx-3"],
    context: ["ctx-1", "ctx-2", "ctx-3"],
    toolsCalled: [
      new ToolCall({ name: "t1" }),
      new ToolCall({ name: "t2" }),
      new ToolCall({ name: "t3" }),
    ],
  });
}

describe("PIILeakageMetric verdict-count denominator", () => {
  async function scoreWith(
    verdicts: { verdict: string; reason: string }[],
    extracted: string[] = THREE_ITEMS,
  ): Promise<number> {
    const metric = new PIILeakageMetric({
      model: new FixedJudge(verdicts, extracted),
      includeReason: true,
      showIndicator: false,
    });
    return metric.measure(testCase());
  }

  it("truncated verdict list scores 1/3, not a clean 1.0", async () => {
    const score = await scoreWith([{ verdict: "no", reason: "no leak" }]);
    expect(score).toBeCloseTo(1 / 3, 5);
  });

  it("empty verdict list scores 0 with 3 PII on the table", async () => {
    const score = await scoreWith([]);
    expect(score).toBe(0);
  });

  it("vacuous pass: no PII extracted means no verdict call needed", async () => {
    const score = await scoreWith([], []);
    expect(score).toBe(1);
  });

  it("control: complete all-no verdict list scores 1.0", async () => {
    const score = await scoreWith([
      { verdict: "no", reason: "no leak" },
      { verdict: "no", reason: "no leak" },
      { verdict: "no", reason: "no leak" },
    ]);
    expect(score).toBe(1);
  });

  it("control: complete list with one violation scores 2/3", async () => {
    const score = await scoreWith([
      { verdict: "no", reason: "no leak" },
      { verdict: "yes", reason: "leaks a card number" },
      { verdict: "no", reason: "no leak" },
    ]);
    expect(score).toBeCloseTo(2 / 3, 5);
  });
});

describe("Verdict-count denominator across single-turn QAG metrics (#3358)", () => {
  it.each([
    {
      name: "ToxicityMetric",
      passVerdict: "no",
      failVerdict: "yes",
      build: (model: DeepEvalBaseLLM) =>
        new ToxicityMetric({
          model,
          includeReason: false,
          showIndicator: false,
        }),
    },
    {
      name: "BiasMetric",
      passVerdict: "no",
      failVerdict: "yes",
      build: (model: DeepEvalBaseLLM) =>
        new BiasMetric({ model, includeReason: false, showIndicator: false }),
    },
    {
      name: "MisuseMetric",
      passVerdict: "no",
      failVerdict: "yes",
      build: (model: DeepEvalBaseLLM) =>
        new MisuseMetric({
          domain: "finance",
          model,
          includeReason: false,
          showIndicator: false,
        }),
    },
    {
      name: "NonAdviceMetric",
      passVerdict: "no",
      failVerdict: "yes",
      build: (model: DeepEvalBaseLLM) =>
        new NonAdviceMetric({
          adviceTypes: ["medical"],
          model,
          includeReason: false,
          showIndicator: false,
        }),
    },
    {
      name: "FaithfulnessMetric",
      passVerdict: "yes",
      failVerdict: "no",
      build: (model: DeepEvalBaseLLM) =>
        new FaithfulnessMetric({
          model,
          includeReason: false,
          showIndicator: false,
        }),
    },
    {
      name: "AnswerRelevancyMetric",
      passVerdict: "yes",
      failVerdict: "no",
      build: (model: DeepEvalBaseLLM) =>
        new AnswerRelevancyMetric({
          model,
          includeReason: false,
          showIndicator: false,
        }),
    },
  ])(
    "$name counts missing and empty verdicts against extracted items",
    async ({ passVerdict, failVerdict, build }) => {
      const pass = { verdict: passVerdict, reason: "ok" };
      const fail = { verdict: failVerdict, reason: "bad" };

      await expect(
        build(new FixedJudge([pass])).measure(testCase()),
      ).resolves.toBeCloseTo(1 / 3, 5);
      await expect(build(new FixedJudge([])).measure(testCase())).resolves.toBe(
        0,
      );
      await expect(
        build(new FixedJudge([pass, pass, pass])).measure(testCase()),
      ).resolves.toBe(1);
      await expect(
        build(new FixedJudge([pass, fail, pass])).measure(testCase()),
      ).resolves.toBeCloseTo(2 / 3, 5);
      await expect(
        build(new FixedJudge([], [])).measure(testCase()),
      ).resolves.toBe(1);
    },
  );

  it("PromptAlignmentMetric counts missing and empty verdicts against instructions", async () => {
    const pass = { verdict: "yes", reason: "ok" };
    const fail = { verdict: "no", reason: "bad" };
    const build = (verdicts: { verdict: string; reason: string }[]) =>
      new PromptAlignmentMetric({
        promptInstructions: THREE_ITEMS,
        model: new FixedJudge(verdicts),
        includeReason: false,
        showIndicator: false,
      });

    await expect(build([pass]).measure(testCase())).resolves.toBeCloseTo(
      1 / 3,
      5,
    );
    await expect(build([]).measure(testCase())).resolves.toBe(0);
    await expect(build([pass, pass, pass]).measure(testCase())).resolves.toBe(
      1,
    );
    await expect(
      build([pass, fail, pass]).measure(testCase()),
    ).resolves.toBeCloseTo(2 / 3, 5);
  });

  it("HallucinationMetric counts missing and empty verdicts against contexts", async () => {
    const pass = { verdict: "yes", reason: "ok" };
    const fail = { verdict: "no", reason: "bad" };
    const build = (verdicts: { verdict: string; reason: string }[]) =>
      new HallucinationMetric({
        model: new FixedJudge(verdicts),
        includeReason: false,
        showIndicator: false,
      });

    await expect(build([pass]).measure(testCase())).resolves.toBeCloseTo(
      1 / 3,
      5,
    );
    await expect(build([]).measure(testCase())).resolves.toBe(0);
    await expect(build([pass, pass, pass]).measure(testCase())).resolves.toBe(
      1,
    );
    await expect(
      build([pass, fail, pass]).measure(testCase()),
    ).resolves.toBeCloseTo(2 / 3, 5);
    await expect(
      build([]).measure(
        new LLMTestCase({
          input: "hi",
          actualOutput: "hello",
          context: [],
        }),
      ),
    ).resolves.toBe(1);
  });

  it("ArgumentCorrectnessMetric counts missing and empty verdicts against toolsCalled", async () => {
    const pass = { verdict: "yes", reason: "ok" };
    const fail = { verdict: "no", reason: "bad" };
    const build = (verdicts: { verdict: string; reason: string }[]) =>
      new ArgumentCorrectnessMetric({
        model: new FixedJudge(verdicts),
        includeReason: false,
        showIndicator: false,
      });

    await expect(build([pass]).measure(testCase())).resolves.toBeCloseTo(
      1 / 3,
      5,
    );
    await expect(build([]).measure(testCase())).resolves.toBe(0);
    await expect(build([pass, pass, pass]).measure(testCase())).resolves.toBe(
      1,
    );
    await expect(
      build([pass, fail, pass]).measure(testCase()),
    ).resolves.toBeCloseTo(2 / 3, 5);
    await expect(
      build([]).measure(
        new LLMTestCase({
          input: "hi",
          actualOutput: "hello",
          toolsCalled: [],
        }),
      ),
    ).resolves.toBe(1);
  });

  it("RoleViolationMetric fails audit on incomplete or empty verdicts when violations were extracted", async () => {
    const pass = { verdict: "no", reason: "ok" };
    const fail = { verdict: "yes", reason: "violation" };
    const build = (
      verdicts: { verdict: string; reason: string }[],
      extracted: string[] = THREE_ITEMS,
    ) =>
      new RoleViolationMetric({
        role: "helpful assistant",
        model: new FixedJudge(verdicts, extracted),
        includeReason: false,
        showIndicator: false,
      });

    await expect(build([pass]).measure(testCase())).resolves.toBe(0);
    await expect(build([]).measure(testCase())).resolves.toBe(0);
    await expect(build([pass, pass, pass]).measure(testCase())).resolves.toBe(
      1,
    );
    await expect(build([pass, fail, pass]).measure(testCase())).resolves.toBe(
      0,
    );
    await expect(build([], []).measure(testCase())).resolves.toBe(1);
  });

  it("SummarizationMetric counts missing and empty alignment verdicts against claims", async () => {
    const pass = { verdict: "yes", reason: "ok" };
    const fail = { verdict: "no", reason: "bad" };
    const build = (
      verdicts: { verdict: string; reason: string }[],
      claims: string[] = THREE_ITEMS,
    ) =>
      new SummarizationMetric({
        model: new FixedJudge(verdicts, claims),
        includeReason: false,
        showIndicator: false,
      });

    await expect(build([pass]).measure(testCase())).resolves.toBeCloseTo(
      1 / 3,
      5,
    );
    await expect(build([]).measure(testCase())).resolves.toBe(0);
    await expect(build([pass, pass, pass]).measure(testCase())).resolves.toBe(
      1,
    );
    await expect(
      build([pass, fail, pass]).measure(testCase()),
    ).resolves.toBeCloseTo(2 / 3, 5);
    await expect(build([], []).measure(testCase())).resolves.toBe(0);
  });
});
