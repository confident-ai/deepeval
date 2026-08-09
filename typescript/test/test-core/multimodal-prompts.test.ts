// Which metrics switch their `{% if multimodal %}` branches on, and what changes.

import { AnswerRelevancyMetric } from "@/metrics/answer-relevancy/answer-relevancy";
import { FaithfulnessMetric } from "@/metrics/faithfulness/faithfulness";
import { ContextualPrecisionMetric } from "@/metrics/contextual-precision/contextual-precision";
import { ContextualRecallMetric } from "@/metrics/contextual-recall/contextual-recall";
import { ContextualRelevancyMetric } from "@/metrics/contextual-relevancy/contextual-relevancy";
import { SummarizationMetric } from "@/metrics/summarization/summarization";
import { ToxicityMetric } from "@/metrics/toxicity/toxicity";
import { ExactMatchMetric } from "@/metrics/exact-match/exact-match";
import { idRetrievalContext } from "@/metrics/retrieval-context-display";
import { checkSingleTurnParams } from "@/metrics/utils";
import { LLMTestCase, MLLMImage, SingleTurnParams } from "@/test-case";
import { OpenAIModel } from "@/models";
import { bedrockContent, ollamaMessages } from "@/models/multimodal";
import { resolveTemplate } from "@/templates";

function prompts(metric: unknown) {
  return metric as {
    getPrompt(
      method: string,
      vars?: Record<string, unknown>,
      opts?: { templateClass?: string; strict?: boolean },
    ): string;
  };
}

/** Force the flag without needing a real image, for prompt-level assertions. */
function withMultimodal<T>(metric: T, value: boolean): T {
  (metric as { multimodal: boolean }).multimodal = value;
  return metric;
}

const image = () => new MLLMImage({ url: "https://example.com/cat.png" });

/** A 1x1 PNG, so the byte-carrying builders need no network. */
const PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const inlineImage = () =>
  new MLLMImage({ dataBase64: PNG_BASE64, mimeType: "image/png" });

beforeAll(() => {
  process.env.OPENAI_API_KEY ??= "test-key";
});

describe("the param-check seam", () => {
  it("copies multimodal off the test case before prompts are built", () => {
    const metric = new AnswerRelevancyMetric();
    expect(metric.multimodal).toBe(false);

    checkSingleTurnParams(
      new LLMTestCase({
        input: "describe this",
        actualOutput: `here it is ${image()}`,
      }),
      [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
      metric,
    );
    expect(metric.multimodal).toBe(true);
  });

  it("detects multimodality from an embedded image slug", () => {
    const tc = new LLMTestCase({
      input: "hi",
      actualOutput: `look ${image()}`,
    });
    expect(tc.multimodal).toBe(true);
    expect(
      new LLMTestCase({ input: "hi", actualOutput: "plain" }).multimodal,
    ).toBe(false);
  });
});

describe("the vision-model guard", () => {
  it("refuses a multimodal test case on a text-only model", () => {
    const metric = new AnswerRelevancyMetric({
      model: new OpenAIModel({ model: "gpt-4.1-nano" }),
    });
    expect(() =>
      checkSingleTurnParams(
        new LLMTestCase({ input: "hi", actualOutput: `look ${image()}` }),
        [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        metric,
      ),
    ).toThrow(/does not support multimodal evaluations/);
    expect(metric.error).toMatch(/gpt-4.1-nano/);
  });

  it("names vision-capable models from the same provider", () => {
    const metric = new AnswerRelevancyMetric({
      model: new OpenAIModel({ model: "gpt-4.1-nano" }),
    });
    try {
      checkSingleTurnParams(
        new LLMTestCase({ input: "hi", actualOutput: `look ${image()}` }),
        [SingleTurnParams.INPUT],
        metric,
      );
    } catch {
      // Assert on metric.error, which carries the same message.
    }
    expect(metric.error).toContain("gpt-4o");
  });

  it("lets a vision-capable model through", () => {
    const metric = new AnswerRelevancyMetric({
      model: new OpenAIModel({ model: "gpt-4o" }),
    });
    expect(() =>
      checkSingleTurnParams(
        new LLMTestCase({ input: "hi", actualOutput: `look ${image()}` }),
        [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        metric,
      ),
    ).not.toThrow();
  });

  it("ignores text-only test cases entirely", () => {
    const metric = new AnswerRelevancyMetric({
      model: new OpenAIModel({ model: "gpt-4.1-nano" }),
    });
    expect(() =>
      checkSingleTurnParams(
        new LLMTestCase({ input: "hi", actualOutput: "plain" }),
        [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        metric,
      ),
    ).not.toThrow();
  });

  it("refuses multimodal on a model-less deterministic metric, as Python does", () => {
    const metric = new ExactMatchMetric();
    expect(() =>
      checkSingleTurnParams(
        new LLMTestCase({
          input: "hi",
          actualOutput: `look ${image()}`,
          expectedOutput: "x",
        }),
        [SingleTurnParams.ACTUAL_OUTPUT],
        metric,
      ),
    ).toThrow(/has no evaluation model/);
  });
});

describe("provider content builders", () => {
  it("gives Ollama one message per part, images as base64", async () => {
    const img = inlineImage();
    const messages = await ollamaMessages(`before ${img} after`);
    expect(messages).toEqual([
      { role: "user", content: "before " },
      { role: "user", images: [PNG_BASE64] },
      { role: "user", content: " after" },
    ]);
  });

  it("leaves a text-only Ollama prompt as a single message", async () => {
    expect(await ollamaMessages("just text")).toEqual([
      { role: "user", content: "just text" },
    ]);
  });

  it("inlines Bedrock images as raw bytes with a lowercase format", async () => {
    const img = inlineImage();
    const content = await bedrockContent(`describe ${img}`);
    expect(content[0]).toEqual({ text: "describe " });

    const block = content[1] as {
      image: { format: string; source: { bytes: Buffer } };
    };
    expect(block.image.format).toBe("png");
    expect(block.image.source.bytes).toBeInstanceOf(Buffer);
    expect(block.image.source.bytes.toString("base64")).toBe(PNG_BASE64);
  });

  it("leaves a text-only Bedrock prompt as a single text block", async () => {
    expect(await bedrockContent("just text")).toEqual([{ text: "just text" }]);
  });

  it("rejects an image format the Converse API cannot accept", async () => {
    const pdf = new MLLMImage({
      dataBase64: PNG_BASE64,
      mimeType: "application/pdf",
    });
    await expect(bedrockContent(`read ${pdf}`)).rejects.toThrow(
      /does not accept 'application\/pdf'/,
    );
  });
});

describe("multimodal-aware metrics", () => {
  it("renders a different prompt once the flag is on", () => {
    const off = prompts(
      withMultimodal(new AnswerRelevancyMetric(), false),
    ).getPrompt("generate_statements", { actual_output: "a" });
    const on = prompts(
      withMultimodal(new AnswerRelevancyMetric(), true),
    ).getPrompt("generate_statements", { actual_output: "a" });
    expect(on).not.toBe(off);
    // The image rules come from a shared fragment, so just assert it grew.
    expect(on.length).toBeGreaterThan(off.length);
  });

  it("matches the bundle's multimodal branch exactly", () => {
    const on = prompts(
      withMultimodal(new AnswerRelevancyMetric(), true),
    ).getPrompt("generate_statements", { actual_output: "a" });
    expect(on).toBe(
      resolveTemplate(
        "metrics",
        "AnswerRelevancyMetric",
        "generate_statements",
        {
          actual_output: "a",
          multimodal: true,
        },
      ),
    );
  });

  /** Capture the vars a metric hands to `getPrompt`, with the model stubbed out. */
  async function capturedVars(
    metric: unknown,
    method: string,
    args: unknown[],
    output: unknown,
  ): Promise<Record<string, unknown>> {
    const m = metric as Record<string, any>;
    let seen: Record<string, unknown> = {};
    m.getPrompt = (_method: string, vars: Record<string, unknown>) => {
      seen = vars;
      return "prompt";
    };
    m.model = {
      generate: async () => ({ output, cost: 0 }),
      getModelName: () => "stub",
    };
    await m[method](...args);
    return seen;
  }

  it("switches faithfulness's truths instruction on", async () => {
    const on = await capturedVars(
      withMultimodal(new FaithfulnessMetric(), true),
      "generateTruths",
      [["ctx"]],
      { truths: [] },
    );
    expect(on.multimodal_instruction).toBe(
      " The excerpt may contain both text and images.",
    );

    const off = await capturedVars(
      withMultimodal(new FaithfulnessMetric(), false),
      "generateTruths",
      [["ctx"]],
      { truths: [] },
    );
    expect(off.multimodal_instruction).toBe("");
  });

  it("switches faithfulness's claims instruction on", async () => {
    const on = await capturedVars(
      withMultimodal(new FaithfulnessMetric(), true),
      "generateClaims",
      ["output"],
      { claims: [] },
    );
    expect(on.multimodal_instruction).toContain(
      "extract claims from all provided content",
    );
  });

  it("switches contextual recall's content-type wording on", async () => {
    const on = await capturedVars(
      withMultimodal(new ContextualRecallMetric(), true),
      "generateVerdicts",
      ["expected", ["alpha"]],
      { verdicts: [] },
    );
    expect(on.content_type).toBe("sentence and image");
    expect(on.content_type_plural).toBe("sentences and images");
    expect(on.content_or).toBe("sentence or image");
    expect(on.context_to_display).toEqual(["Node 1: alpha"]);
    expect(on.node_instruction).toContain("A node is either a string or image");

    const off = await capturedVars(
      withMultimodal(new ContextualRecallMetric(), false),
      "generateVerdicts",
      ["expected", ["alpha"]],
      { verdicts: [] },
    );
    expect(off.content_type).toBe("sentence");
    expect(off.context_to_display).toEqual(["alpha"]);
    expect(off.node_instruction).toBe("");
  });

  it("switches contextual relevancy's extraction instructions on", async () => {
    const on = await capturedVars(
      withMultimodal(new ContextualRelevancyMetric(), true),
      "generateVerdicts",
      ["i", "c"],
      { verdicts: [] },
    );
    expect(on.context_type).toBe("context (image or string)");
    expect(on.statement_or_image).toBe("statement or image");
    expect(on.extraction_instructions).toContain("If the context is an image");
    // Python drops the empty-context branch when images are in play.
    expect(on.empty_context_instruction).toBe("");

    const off = await capturedVars(
      withMultimodal(new ContextualRelevancyMetric(), false),
      "generateVerdicts",
      ["i", "c"],
      { verdicts: [] },
    );
    expect(off.context_type).toBe("context");
    expect(off.extraction_instructions).not.toContain(
      "If the context is an image",
    );
    expect(off.empty_context_instruction).toContain("no actual content");
  });

  it("labels retrieval context nodes for contextual precision", async () => {
    const on = await capturedVars(
      withMultimodal(new ContextualPrecisionMetric(), true),
      "generateVerdicts",
      ["i", "e", ["alpha"]],
      { verdicts: [] },
    );
    expect(on.context_to_display).toEqual(["Node 1: alpha"]);
    expect(on.multimodal_note).toBe(" (which can be text or an image)");

    const off = await capturedVars(
      withMultimodal(new ContextualPrecisionMetric(), false),
      "generateVerdicts",
      ["i", "e", ["alpha"]],
      { verdicts: [] },
    );
    expect(off.context_to_display).toEqual(["alpha"]);
    expect(off.multimodal_note).toBe("");
  });
});

describe("metrics that stay text-only", () => {
  // Python never threads `multimodal` for these, so neither do we.
  it("does not switch Summarization's branches on", () => {
    const on = prompts(
      withMultimodal(new SummarizationMetric(), true),
    ).getPrompt("generate_answers", { text: "t", questions: ["q"] });
    const off = prompts(
      withMultimodal(new SummarizationMetric(), false),
    ).getPrompt("generate_answers", { text: "t", questions: ["q"] });
    expect(on).toBe(off);
  });

  it("does not switch Toxicity's branches on", () => {
    const on = prompts(withMultimodal(new ToxicityMetric(), true)).getPrompt(
      "generate_verdicts",
      { opinions: ["o"] },
    );
    expect(on).toBe(
      resolveTemplate("metrics", "ToxicityMetric", "generate_verdicts", {
        opinions: ["o"],
        multimodal: false,
      }),
    );
  });

  it("keeps Summarization's borrowed faithfulness prompts text-only", () => {
    const metric = withMultimodal(new SummarizationMetric(), true);
    const borrowed = prompts(metric).getPrompt(
      "generate_claims",
      { actual_output: "a", multimodal_instruction: "" },
      { templateClass: "FaithfulnessMetric" },
    );
    expect(borrowed).toBe(
      resolveTemplate("metrics", "FaithfulnessMetric", "generate_claims", {
        actual_output: "a",
        multimodal_instruction: "",
        multimodal: false,
      }),
    );
  });
});

describe("idRetrievalContext", () => {
  it("labels text nodes by position", () => {
    expect(idRetrievalContext(["alpha", "beta"])).toEqual([
      "Node 1: alpha",
      "Node 2: beta",
    ]);
  });

  it("gives an image its own label line, keeping the image itself", () => {
    const img = image();
    const out = idRetrievalContext([`${img}`]);
    expect(out[0]).toBe("Node 1:");
    expect(out[1]).toBeInstanceOf(MLLMImage);
  });

  it("renders an image as a bare slug inside a prompt (Python repr parity)", () => {
    const img = image();
    const metric = withMultimodal(new ContextualPrecisionMetric(), true);
    const prompt = prompts(metric).getPrompt("generate_verdicts", {
      input: "i",
      expected_output: "e",
      document_count_str: " (1 document)",
      context_to_display: idRetrievalContext([`${img}`]),
      multimodal_note: " (which can be text or an image)",
    });
    // Bare, not quoted and not a dict of MLLMImage fields.
    expect(prompt).toContain(`[DEEPEVAL:IMAGE:${img.id}]`);
    expect(prompt).not.toContain(`'[DEEPEVAL:IMAGE:${img.id}]'`);
    expect(prompt).not.toContain("dataBase64");
  });
});
