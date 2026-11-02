import { describe, expect, it } from "vitest";

import {
  detectJudge,
  judgeEnvValues,
  judgeMissing,
  judgeProviders,
  judgeVariables,
  providersWithKey,
} from "../src/judge.js";

const provider = (id: string) => {
  const found = judgeProviders.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`Unknown provider: ${id}`);
  return found;
};

describe("judge detection", () => {
  it("treats a bare OpenAI key as ready, matching DeepEval's fallback", () => {
    expect(detectJudge({ OPENAI_API_KEY: "sk-test" })).toEqual({
      provider: provider("openai"),
      missing: [],
    });
  });

  it("finds nothing when the environment has no provider credentials", () => {
    expect(detectJudge({ PATH: "/usr/bin" })).toBeUndefined();
  });

  it("ignores an empty or whitespace-only key", () => {
    expect(detectJudge({ OPENAI_API_KEY: "   " })).toBeUndefined();
  });

  it("follows a USE_ flag over the OpenAI fallback", () => {
    expect(
      detectJudge({
        OPENAI_API_KEY: "sk-test",
        USE_ANTHROPIC_MODEL: "true",
        ANTHROPIC_API_KEY: "sk-ant",
        ANTHROPIC_MODEL_NAME: "claude",
      }),
    ).toEqual({ provider: provider("anthropic"), missing: [] });
  });

  it("accepts the boolean spellings DeepEval parses", () => {
    for (const value of ["true", "1", "yes", "YES"]) {
      expect(detectJudge({ USE_GEMINI_MODEL: value })?.provider.id).toBe(
        "gemini",
      );
    }
    expect(detectJudge({ USE_GEMINI_MODEL: "false" })).toBeUndefined();
  });

  it("reports what a selected provider still needs", () => {
    expect(detectJudge({ USE_ANTHROPIC_MODEL: "true" })).toEqual({
      provider: provider("anthropic"),
      // DeepEval defaults the Anthropic model, so only the key is missing.
      missing: ["ANTHROPIC_API_KEY"],
    });
    expect(detectJudge({ USE_PORTKEY_MODEL: "true" })?.missing).toEqual([
      "PORTKEY_API_KEY",
      "PORTKEY_MODEL_NAME",
      "PORTKEY_BASE_URL",
      "PORTKEY_PROVIDER_NAME",
    ]);
  });

  it("matches Ollama on its marker value, and a local server otherwise", () => {
    expect(
      detectJudge({
        USE_LOCAL_MODEL: "true",
        LOCAL_MODEL_API_KEY: "ollama",
        OLLAMA_MODEL_NAME: "llama3.1",
      }),
    ).toEqual({ provider: provider("ollama"), missing: [] });
    expect(
      detectJudge({ USE_LOCAL_MODEL: "true", LOCAL_MODEL_API_KEY: "other" })
        ?.provider.id,
    ).toBe("local");
  });

  it("treats credentials DeepEval can infer as optional", () => {
    // Bedrock falls back to the ambient AWS credential chain.
    expect(judgeMissing(provider("bedrock"), {})).toEqual([
      "AWS_BEDROCK_MODEL_NAME",
      "AWS_BEDROCK_REGION",
    ]);
    // Gemini on Vertex AI authenticates without an API key.
    expect(judgeMissing(provider("gemini"), {})).toEqual(["GOOGLE_API_KEY"]);
    expect(
      judgeMissing(provider("gemini"), { GOOGLE_GENAI_USE_VERTEXAI: "true" }),
    ).toEqual([]);
    // LiteLLM reuses whichever upstream provider key is present.
    expect(judgeMissing(provider("litellm"), {})).toEqual([
      "LITELLM_MODEL_NAME",
    ]);
  });

  it("covers every provider DeepEval can select as a judge", () => {
    expect(judgeProviders).toHaveLength(13);
    expect(
      judgeProviders
        .map((candidate) => candidate.precedence)
        .sort((a, b) => a - b),
    ).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
  });

  it("resolves competing flags by DeepEval's own precedence", () => {
    expect(
      detectJudge({
        USE_GEMINI_MODEL: "true",
        USE_ANTHROPIC_MODEL: "true",
      })?.provider.id,
    ).toBe("gemini");
  });
});

describe("keys already in the environment", () => {
  it("finds a key whose provider flag DeepEval would ignore", () => {
    expect(
      providersWithKey({ ANTHROPIC_API_KEY: "sk-ant" }).map(
        (provider) => provider.id,
      ),
    ).toEqual(["anthropic"]);
    // Without the flag DeepEval itself would not use it.
    expect(detectJudge({ ANTHROPIC_API_KEY: "sk-ant" })).toBeUndefined();
  });

  it("lists every provider with a usable key, in catalog order", () => {
    expect(
      providersWithKey({
        OPENROUTER_API_KEY: "sk-or",
        OPENAI_API_KEY: "sk-test",
        GOOGLE_API_KEY: "  ",
      }).map((provider) => provider.id),
    ).toEqual(["openai", "openrouter"]);
  });

  it("claims nothing for providers that take no key", () => {
    expect(providersWithKey({ USE_LOCAL_MODEL: "true" })).toEqual([]);
  });

  it("does not read Ollama's marker as a local server key", () => {
    expect(providersWithKey({ LOCAL_MODEL_API_KEY: "ollama" })).toEqual([]);
    expect(
      providersWithKey({ LOCAL_MODEL_API_KEY: "vllm" }).map(
        (candidate) => candidate.id,
      ),
    ).toEqual(["local"]);
  });
});

describe("judge environment values", () => {
  it("writes the key and the provider flag", () => {
    expect(
      judgeEnvValues(provider("anthropic"), {
        secrets: { ANTHROPIC_API_KEY: " sk-ant " },
      }),
    ).toEqual({
      USE_ANTHROPIC_MODEL: "true",
      ANTHROPIC_API_KEY: "sk-ant",
    });
  });

  it("writes both AWS secrets and Bedrock's required settings", () => {
    expect(
      judgeEnvValues(provider("bedrock"), {
        secrets: {
          AWS_ACCESS_KEY_ID: "AKIA",
          AWS_SECRET_ACCESS_KEY: "secret",
        },
        settings: {
          AWS_BEDROCK_MODEL_NAME: "anthropic.claude-3-5-sonnet",
          AWS_BEDROCK_REGION: "us-east-1",
        },
      }),
    ).toEqual({
      USE_AWS_BEDROCK_MODEL: "true",
      AWS_ACCESS_KEY_ID: "AKIA",
      AWS_SECRET_ACCESS_KEY: "secret",
      AWS_BEDROCK_MODEL_NAME: "anthropic.claude-3-5-sonnet",
      AWS_BEDROCK_REGION: "us-east-1",
    });
  });

  it("needs no key for a local Ollama judge", () => {
    expect(
      judgeEnvValues(provider("ollama"), {
        settings: { OLLAMA_MODEL_NAME: "llama3.1" },
      }),
    ).toEqual({
      USE_LOCAL_MODEL: "true",
      LOCAL_MODEL_API_KEY: "ollama",
      OLLAMA_MODEL_NAME: "llama3.1",
    });
  });

  it("leaves OpenAI on DeepEval's defaults", () => {
    expect(
      judgeEnvValues(provider("openai"), {
        secrets: { OPENAI_API_KEY: "sk-test" },
      }),
    ).toEqual({ OPENAI_API_KEY: "sk-test" });
    expect(judgeVariables(provider("openai"))).toEqual(["OPENAI_API_KEY"]);
  });
});
