// Coverage for `evaluationTemplate`, the prompt-override hook on every metric.

import { AnswerRelevancyMetric } from "@/metrics/answer-relevancy/answer-relevancy";
import { FaithfulnessMetric } from "@/metrics/faithfulness/faithfulness";
import { SummarizationMetric } from "@/metrics/summarization/summarization";
import {
  camelizeVars,
  decamelizeVars,
  findOverride,
  snakeToCamel,
} from "@/templates/override";
import { resolveTemplate } from "@/templates";
import metricsBundle from "@/templates/metrics/templates.json";

/** `getPrompt` is protected; tests drive it the way the metric's own code does. */
function prompts(metric: unknown) {
  return metric as {
    getPrompt(
      method: string,
      vars?: Record<string, unknown>,
      opts?: { templateClass?: string; strict?: boolean },
    ): string;
  };
}

// A model is never called here — only prompt construction is under test — but
// the constructors initialize one, so they need a key present.
beforeAll(() => {
  process.env.OPENAI_API_KEY ??= "test-key";
});

describe("name mapping", () => {
  it("camelizes snake_case template method names", () => {
    expect(snakeToCamel("generate_statements")).toBe("generateStatements");
    expect(snakeToCamel("generate_strict_evaluation_results")).toBe(
      "generateStrictEvaluationResults",
    );
    expect(snakeToCamel("verdicts")).toBe("verdicts");
  });

  it("drops the leading underscore on template-internal vars", () => {
    expect(camelizeVars({ _additional_context: 1 })).toEqual({
      additionalContext: 1,
    });
  });

  it("round-trips vars back to their original Jinja names", () => {
    const original = {
      actual_output: "a",
      _additional_context: "b",
      input: "c",
    };
    const keys = Object.keys(original);
    expect(decamelizeVars(camelizeVars(original), keys)).toEqual(original);
  });

  it("finds an override by its camelCase name", () => {
    const fn = () => "x";
    expect(
      findOverride({ generateStatements: fn }, "generate_statements"),
    ).toBe(fn);
    expect(findOverride({}, "generate_statements")).toBeUndefined();
    expect(findOverride(undefined, "generate_statements")).toBeUndefined();
    // A non-function value must not be mistaken for an override.
    expect(
      findOverride({ generateStatements: "text" }, "generate_statements"),
    ).toBeUndefined();
  });
});

describe("override dispatch", () => {
  it("renders the bundled template when no override is given", () => {
    const metric = new AnswerRelevancyMetric();
    const prompt = prompts(metric).getPrompt("generate_statements", {
      actual_output: "The sky is blue.",
    });
    expect(prompt).toBe(
      resolveTemplate(
        "metrics",
        "AnswerRelevancyMetric",
        "generate_statements",
        {
          actual_output: "The sky is blue.",
        },
      ),
    );
  });

  it("uses the override and hands it camelCase vars", () => {
    const metric = new AnswerRelevancyMetric({
      evaluationTemplate: {
        generateStatements: ({ actualOutput }) => `custom: ${actualOutput}`,
      },
    });
    expect(
      prompts(metric).getPrompt("generate_statements", {
        actual_output: "The sky is blue.",
      }),
    ).toBe("custom: The sky is blue.");
  });

  it("leaves the metric's other prompts on the bundled defaults", () => {
    const metric = new AnswerRelevancyMetric({
      evaluationTemplate: {
        generateStatements: () => "custom",
      },
    });
    const verdicts = prompts(metric).getPrompt("generate_verdicts", {
      input: "why?",
      statements: ["a"],
    });
    expect(verdicts).toBe(
      resolveTemplate("metrics", "AnswerRelevancyMetric", "generate_verdicts", {
        input: "why?",
        statements: ["a"],
      }),
    );
  });

  it("exposes multimodal to the override, defaulted to false", () => {
    const seen: unknown[] = [];
    const metric = new AnswerRelevancyMetric({
      evaluationTemplate: {
        generateStatements: (vars) => {
          seen.push(vars.multimodal);
          return "x";
        },
      },
    });
    prompts(metric).getPrompt("generate_statements", { actual_output: "a" });
    prompts(metric).getPrompt("generate_statements", {
      actual_output: "a",
      multimodal: true,
    });
    expect(seen).toEqual([false, true]);
  });
});

describe("renderDefault", () => {
  it("renders the shipped prompt so an override can extend it", () => {
    const metric = new AnswerRelevancyMetric({
      evaluationTemplate: {
        generateStatements: (vars, renderDefault) =>
          `${renderDefault(vars)}\n\nExtra rule.`,
      },
    });
    const base = resolveTemplate(
      "metrics",
      "AnswerRelevancyMetric",
      "generate_statements",
      { actual_output: "The sky is blue." },
    );
    expect(
      prompts(metric).getPrompt("generate_statements", {
        actual_output: "The sky is blue.",
      }),
    ).toBe(`${base}\n\nExtra rule.`);
  });

  it("defaults to the original vars when called with none", () => {
    const metric = new AnswerRelevancyMetric({
      evaluationTemplate: {
        generateStatements: (_vars, renderDefault) => renderDefault(),
      },
    });
    expect(
      prompts(metric).getPrompt("generate_statements", {
        actual_output: "The sky is blue.",
      }),
    ).toBe(
      resolveTemplate(
        "metrics",
        "AnswerRelevancyMetric",
        "generate_statements",
        {
          actual_output: "The sky is blue.",
        },
      ),
    );
  });

  it("accepts substituted vars, mapping them back to Jinja names", () => {
    const metric = new AnswerRelevancyMetric({
      evaluationTemplate: {
        generateStatements: (vars, renderDefault) =>
          renderDefault({ ...vars, actualOutput: "REPLACED" }),
      },
    });
    expect(
      prompts(metric).getPrompt("generate_statements", {
        actual_output: "The sky is blue.",
      }),
    ).toBe(
      resolveTemplate(
        "metrics",
        "AnswerRelevancyMetric",
        "generate_statements",
        {
          actual_output: "REPLACED",
        },
      ),
    );
  });
});

describe("borrowed templates", () => {
  // Summarization renders some of Faithfulness's prompts. Naming a borrowed
  // prompt is already a type error, since the override surface is built from the
  // metric's OWN bundle keys; the cast here checks the runtime guard behind it.
  it("does not expose borrowed prompts on the override surface", () => {
    const own = Object.keys(metricsBundle.SummarizationMetric);
    expect(own).not.toContain("generate_verdicts");
    expect(Object.keys(metricsBundle.FaithfulnessMetric)).toContain(
      "generate_verdicts",
    );
  });

  it("skips the override when a template is borrowed", () => {
    const metric = new SummarizationMetric({
      evaluationTemplate: {
        generateVerdicts: () => "HIJACKED",
      } as never,
    });
    const borrowed = prompts(metric).getPrompt(
      "generate_verdicts",
      { claims: ["a"], retrieval_context: "ctx" },
      { templateClass: "FaithfulnessMetric" },
    );
    expect(borrowed).not.toBe("HIJACKED");
    expect(borrowed).toBe(
      resolveTemplate("metrics", "FaithfulnessMetric", "generate_verdicts", {
        claims: ["a"],
        retrieval_context: "ctx",
      }),
    );
  });

  it("still applies the override to the metric's own prompts", () => {
    const metric = new FaithfulnessMetric({
      evaluationTemplate: { generateVerdicts: () => "MINE" },
    });
    expect(
      prompts(metric).getPrompt("generate_verdicts", {
        claims: ["a"],
        retrieval_context: "ctx",
      }),
    ).toBe("MINE");
  });
});

describe("wiring", () => {
  it("points every metric at a template class present in the bundle", () => {
    // Guards against a typo'd or unset `templateClass` silently falling back to
    // an empty lookup key.
    const metrics = [
      new AnswerRelevancyMetric(),
      new FaithfulnessMetric(),
      new SummarizationMetric(),
    ];
    for (const metric of metrics) {
      const key = (metric as unknown as { templateClass: string })
        .templateClass;
      expect(Object.keys(metricsBundle)).toContain(key);
    }
  });
});
