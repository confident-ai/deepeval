import { describe, expect, it } from "vitest";

import { buildAgentPrompt, getCanonicalPrompt } from "../src/prompt.js";

const judgeProvider = {
  id: "anthropic" as const,
  label: "Anthropic",
  hint: "",
  secrets: [{ envVar: "ANTHROPIC_API_KEY", label: "Anthropic API key" }],
  settings: [{ envVar: "ANTHROPIC_MODEL_NAME", label: "Model" }],
  flags: { USE_ANTHROPIC_MODEL: "true" },
  precedence: 12,
};

/**
 * The prompt reaches the bundle through a markdown import, so a bundler or
 * plugin regression could ship an agent an empty or truncated brief without
 * failing anything else.
 */
describe("canonical prompt", () => {
  it("arrives whole, with the contract src/result.ts validates against", () => {
    const prompt = getCanonicalPrompt();
    expect(prompt.startsWith("# Confident AI evaluation setup")).toBe(true);
    expect(prompt.length).toBeGreaterThan(4000);
    expect(prompt).toContain("CONFIDENT_SETUP_RESULT_FILE");
    for (const literal of [
      "completed",
      "partial",
      "failed",
      "component-level",
      "black-box",
      "test-case",
      "span",
      "trace",
      "thread",
      "deepeval-python",
      "deepeval-typescript",
    ]) {
      expect(prompt).toContain(literal);
    }
  });
});

describe("agent prompt", () => {
  it("names the judge provider and its variables, never a value", () => {
    const prompt = buildAgentPrompt("/tmp/app", "/tmp/result.json", {
      provider: judgeProvider,
    });
    expect(prompt).toContain("Judge model: Anthropic");
    expect(prompt).toContain("ANTHROPIC_API_KEY, ANTHROPIC_MODEL_NAME");
    expect(prompt).toContain("never open or echo those values");
  });

  it("tells the agent to stay judge-free without a judge", () => {
    const prompt = buildAgentPrompt("/tmp/app", "/tmp/result.json");
    expect(prompt).toContain("Judge model: none configured");
    expect(prompt).toContain("do not add LLM-judge metrics");
    expect(prompt).toContain("Cloud: Confident AI");
  });

  /** Nothing asks the user to hold judge metrics back, so nothing says it did. */
  it("never claims a consent decision the wizard no longer collects", () => {
    expect(
      buildAgentPrompt("/tmp/app", "/tmp/result.json", {
        provider: judgeProvider,
      }),
    ).not.toContain("consent");
  });

  it("names the SDK the preflight resolved", () => {
    const typescript = buildAgentPrompt(
      "/tmp/app",
      "/tmp/result.json",
      {},
      true,
      {
        sdk: "node",
        shape: "component-level",
      },
    );
    expect(typescript).toContain("SDK: the DeepEval TypeScript SDK");
    expect(typescript).toContain("Instrument and evaluate in that language");

    const python = buildAgentPrompt("/tmp/app", "/tmp/result.json", {}, true, {
      sdk: "python",
      shape: "component-level",
    });
    expect(python).toContain("SDK: the DeepEval Python SDK");
  });

  it("sends an uninstrumentable project down the black-box path", () => {
    const prompt = buildAgentPrompt("/tmp/app", "/tmp/result.json", {}, true, {
      sdk: "python",
      shape: "black-box",
    });
    expect(prompt).toContain("cannot be instrumented");
    expect(prompt).toContain("report the shape as black-box");
    expect(prompt).not.toContain("Instrument and evaluate in that language");
  });

  it("leaves the language to the agent when the preflight could not settle it", () => {
    expect(buildAgentPrompt("/tmp/app", "/tmp/result.json")).not.toContain(
      "- SDK:",
    );
  });

  it("keeps DeepEval local when Confident AI is declined", () => {
    const prompt = buildAgentPrompt("/tmp/app", "/tmp/result.json", {}, false);
    expect(prompt).toContain("Cloud: local only");
    expect(prompt).not.toContain("A completed result requires testRunId");
  });
});
