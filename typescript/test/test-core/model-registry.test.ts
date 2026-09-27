import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  DEFAULT_MODEL_DATA,
  GENERATED_MODEL_DATA,
  defaultModelName,
  getModelData,
  OpenAIModel,
  AnthropicModel,
  AzureOpenAIModel,
  AISDKModel,
  GeminiModel,
  OpenRouterModel,
} from "@/models";
import { resolveAiSdkNamespace } from "@/models/providers/ai-sdk-model";

describe("generated model registry", () => {
  it("carries every namespace emitted from the Python registries", () => {
    for (const namespace of [
      "openai",
      "anthropic",
      "gemini",
      "grok",
      "kimi",
      "deepseek",
      "ollama",
      "bedrock",
    ]) {
      expect(
        Object.keys(GENERATED_MODEL_DATA[namespace] ?? {}).length,
      ).toBeGreaterThan(0);
    }
  });

  it("excludes the underscored metadata keys from the namespaces", () => {
    expect(GENERATED_MODEL_DATA._meta).toBeUndefined();
    expect(GENERATED_MODEL_DATA._defaults).toBeUndefined();
  });

  it("resolves prices and capabilities for a known model", () => {
    const data = getModelData("openai", "gpt-4.1");
    expect(data.inputPrice).toBeGreaterThan(0);
    expect(data.outputPrice).toBeGreaterThan(0);
    expect(data.supportsStructuredOutputs).toBe(true);
  });

  it("reports reasoning models as rejecting temperature", () => {
    expect(getModelData("openai", "o3-mini").supportsTemperature).toBe(false);
    expect(getModelData("openai", "gpt-5.4").supportsTemperature).toBe(false);
  });

  it("falls back to defaults for unknown models and namespaces", () => {
    expect(getModelData("openai", "not-a-real-model")).toEqual(
      DEFAULT_MODEL_DATA,
    );
    expect(getModelData(undefined, "gpt-4.1")).toEqual(DEFAULT_MODEL_DATA);
    expect(getModelData("openai", undefined)).toEqual(DEFAULT_MODEL_DATA);
  });

  it("defaults supportsTemperature to true, matching DeepEvalModelData", () => {
    expect(DEFAULT_MODEL_DATA.supportsTemperature).toBe(true);
    expect(getModelData("openai", "gpt-4.1").supportsTemperature).toBe(true);
  });
});

describe("AI SDK namespace routing", () => {
  it("routes AI SDK provider ids onto generated namespaces", () => {
    expect(resolveAiSdkNamespace({ provider: "openai.chat" })).toBe("openai");
    expect(resolveAiSdkNamespace({ provider: "anthropic.messages" })).toBe(
      "anthropic",
    );
    expect(resolveAiSdkNamespace({ provider: "google.generative-ai" })).toBe(
      "gemini",
    );
    expect(resolveAiSdkNamespace({ provider: "amazon-bedrock" })).toBe(
      "bedrock",
    );
  });

  it("leaves unknown or malformed providers unrouted", () => {
    expect(resolveAiSdkNamespace({ provider: "mistral.chat" })).toBeUndefined();
    expect(resolveAiSdkNamespace({})).toBeUndefined();
    expect(resolveAiSdkNamespace(undefined)).toBeUndefined();
  });

  it("gives an AI SDK model the same data as its native provider", () => {
    const viaAiSdk = new AISDKModel({
      model: { provider: "openai.chat", modelId: "gpt-4.1" },
    });
    expect(viaAiSdk.supportsLogProbs()).toBe(
      new OpenAIModel({ model: "gpt-4.1" }).supportsLogProbs(),
    );
    expect(viaAiSdk.supportsMultimodal()).toBe(true);
  });

  it("leaves models from unrouted providers on registry defaults", () => {
    const mistral = new AISDKModel({
      model: { provider: "mistral.chat", modelId: "mistral-large-latest" },
    });
    // No Python data for Mistral, so capabilities fall back to the transport's.
    expect(mistral.supportsStructuredOutputs()).toBe(true);
    expect(mistral.supportsLogProbs()).toBeNull();
  });
});

describe("providers backed by the registry", () => {
  it("reads per-model capabilities rather than a hardcoded default", () => {
    // gpt-4.1-nano is text-only in the registry, even though the OpenAI
    // transport itself supports images.
    expect(
      new OpenAIModel({ model: "gpt-4.1-nano" }).supportsMultimodal(),
    ).toBe(false);
    expect(new OpenAIModel({ model: "gpt-4.1" }).supportsMultimodal()).toBe(
      true,
    );
    expect(new OpenAIModel({ model: "gpt-4.1" }).supportsLogProbs()).toBe(true);
    expect(new OpenAIModel({ model: "gpt-5.4" }).maxLogProbs()).toBe(5);
  });

  it("keeps the transport default for models the registry omits", () => {
    expect(
      new OpenAIModel({ model: "some-unknown-model" }).supportsMultimodal(),
    ).toBe(true);
  });

  it("surfaces temperature support per model", () => {
    expect(new OpenAIModel({ model: "o3-mini" }).supportsTemperature()).toBe(
      false,
    );
    expect(new OpenAIModel({ model: "gpt-4.1" }).supportsTemperature()).toBe(
      true,
    );
  });

  it("prices Azure by the underlying model, not the deployment name", () => {
    const azure = new AzureOpenAIModel({
      endpoint: "https://example.openai.azure.com",
      deployment: "my-arbitrary-deployment",
      model: "gpt-4.1",
    });
    expect(azure.supportsLogProbs()).toBe(true);
    expect(azure.supportsMultimodal()).toBe(true);
  });

  it("resolves anthropic models from the anthropic namespace", () => {
    expect(
      new AnthropicModel({
        model: "claude-3-opus-20240229",
      }).supportsMultimodal(),
    ).toBe(true);
  });
});

describe("provider defaults generated from Python", () => {
  // Every `*_MODEL_NAME` a provider under test consults, so a developer's local
  // env cannot make these assertions pass or fail for the wrong reason.
  const MODEL_NAME_VARS = [
    "OPENAI_MODEL_NAME",
    "ANTHROPIC_MODEL_NAME",
    "GEMINI_MODEL_NAME",
    "VERTEX_AI_MODEL_NAME",
    "OPENROUTER_MODEL_NAME",
  ];
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = Object.fromEntries(MODEL_NAME_VARS.map((v) => [v, process.env[v]]));
    for (const v of MODEL_NAME_VARS) delete process.env[v];
  });

  afterEach(() => {
    for (const [v, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[v];
      else process.env[v] = value;
    }
  });

  it("falls back to the generated default when given no model", () => {
    expect(new OpenAIModel().getModelName()).toBe(defaultModelName("openai"));
    expect(new AnthropicModel().getModelName()).toBe(
      defaultModelName("anthropic"),
    );
    expect(new GeminiModel().getModelName()).toBe(defaultModelName("gemini"));
    expect(new OpenRouterModel().getModelName()).toBe(
      defaultModelName("openrouter"),
    );
  });

  it("prefers an explicit model and the env var over the default", () => {
    expect(new OpenAIModel({ model: "gpt-4o" }).getModelName()).toBe("gpt-4o");
    process.env.OPENAI_MODEL_NAME = "gpt-4.1-mini";
    expect(new OpenAIModel().getModelName()).toBe("gpt-4.1-mini");
  });

  it("prices every default it generates", () => {
    for (const namespace of ["openai", "anthropic", "gemini"] as const) {
      const data = getModelData(namespace, defaultModelName(namespace));
      expect(data.inputPrice).toBeGreaterThan(0);
      expect(data.outputPrice).toBeGreaterThan(0);
    }
  });

  // The drift this guards against is how Python and TypeScript ended up on
  // different judges: a provider quietly declaring its own fallback, invisible
  // to `scripts/compile_model_registry.py`.
  it("has no hardcoded default model literal in a synced provider", () => {
    // A real filesystem path, so it cannot use the "@/" module alias.
    const modelsDir = join(__dirname, "../../src/models");
    // Grok, DeepSeek and Kimi deliberately keep TypeScript-only defaults —
    // Python requires their `*_MODEL_NAME` instead. See DEFAULT_MODELS.
    const unsynced = ["grok-model.ts", "deepseek-model.ts", "kimi-model.ts"];
    const offenders: string[] = [];

    for (const dir of ["providers", "gateways"]) {
      for (const file of readdirSync(join(modelsDir, dir))) {
        if (!file.endsWith(".ts") || unsynced.includes(file)) continue;
        const source = readFileSync(join(modelsDir, dir, file), "utf-8");
        source.split("\n").forEach((line, index) => {
          if (/^\s*(const|let)\s+DEFAULT_\w*MODEL\b\s*=\s*["'`]/.test(line)) {
            offenders.push(`${dir}/${file}:${index + 1}: ${line.trim()}`);
          }
        });
      }
    }

    expect(offenders).toEqual([]);
  });
});

describe("temperature resolution", () => {
  // `resolveTemperature` is protected; reach through a cast rather than
  // widening the public surface just for tests.
  const resolve = (model: object) =>
    (
      model as { resolveTemperature(): number | undefined }
    ).resolveTemperature();

  it("defaults to 0, matching Python", () => {
    expect(resolve(new OpenAIModel({ model: "gpt-4.1" }))).toBe(0);
    expect(
      resolve(new AnthropicModel({ model: "claude-3-opus-20240229" })),
    ).toBe(0);
  });

  it("honors an explicit temperature", () => {
    expect(
      resolve(new OpenAIModel({ model: "gpt-4.1", temperature: 0.7 })),
    ).toBe(0.7);
  });

  it("omits temperature for models that reject it", () => {
    expect(resolve(new OpenAIModel({ model: "o3-mini" }))).toBeUndefined();
    expect(
      resolve(new OpenAIModel({ model: "gpt-5.4", temperature: 0.7 })),
    ).toBeUndefined();
  });

  it("omits temperature when explicitly set to null", () => {
    expect(
      resolve(new OpenAIModel({ model: "gpt-4.1", temperature: null })),
    ).toBeUndefined();
  });

  it("defaults to 0 for models the registry does not know", () => {
    expect(resolve(new OpenAIModel({ model: "some-unknown-model" }))).toBe(0);
  });
});

describe("thinking resolution", () => {
  // `resolveThinking` is protected; reach through a cast rather than widening
  // the public surface just for tests.
  const resolve = (model: object) =>
    (
      model as {
        resolveThinking(): {
          maxTokens: number;
          thinking?: Record<string, unknown>;
        };
      }
    ).resolveThinking();

  let saved: string | undefined;

  beforeEach(() => {
    saved = process.env.DEEPEVAL_MODEL_THINKING;
    delete process.env.DEEPEVAL_MODEL_THINKING;
  });

  afterEach(() => {
    if (saved === undefined) delete process.env.DEEPEVAL_MODEL_THINKING;
    else process.env.DEEPEVAL_MODEL_THINKING = saved;
  });

  it("disables thinking by default", () => {
    expect(resolve(new AnthropicModel({ model: "claude-opus-5" }))).toEqual({
      maxTokens: 4096,
      thinking: { type: "disabled" },
    });
  });

  it("sends a budget that leaves room for the response when enabled", () => {
    process.env.DEEPEVAL_MODEL_THINKING = "1";
    const { maxTokens, thinking } = resolve(
      new AnthropicModel({ model: "claude-opus-5" }),
    );
    expect(thinking?.type).toBe("enabled");
    expect(thinking?.budget_tokens as number).toBeLessThan(maxTokens);
    expect(thinking?.budget_tokens as number).toBeGreaterThanOrEqual(1024);
  });

  it("omits the parameter for models that reject it", () => {
    process.env.DEEPEVAL_MODEL_THINKING = "1";
    // Claude 3 has no thinking parameter; claude-fable-5 always thinks and
    // rejects a disabled block.
    for (const model of ["claude-3-haiku", "claude-fable-5"]) {
      expect(resolve(new AnthropicModel({ model })).thinking).toBeUndefined();
    }
  });

  it("raises the default budget only when thinking is on", () => {
    expect(
      resolve(new AnthropicModel({ model: "claude-opus-5" })).maxTokens,
    ).toBe(4096);
    process.env.DEEPEVAL_MODEL_THINKING = "1";
    expect(
      resolve(new AnthropicModel({ model: "claude-opus-5" })).maxTokens,
    ).toBe(8192);
  });

  it("keeps an explicit maxTokens", () => {
    process.env.DEEPEVAL_MODEL_THINKING = "1";
    expect(
      resolve(new AnthropicModel({ model: "claude-opus-5", maxTokens: 6000 }))
        .maxTokens,
    ).toBe(6000);
  });

  it("throws when maxTokens cannot hold thinking and a response", () => {
    process.env.DEEPEVAL_MODEL_THINKING = "1";
    expect(() =>
      resolve(new AnthropicModel({ model: "claude-opus-5", maxTokens: 512 })),
    ).toThrow(/maxTokens/);
  });
});
