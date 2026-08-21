// Regression coverage for settings that were declared but never read back.

import { OpenAIModel, AnthropicModel } from "@/models";
import {
  parseBool,
  isReadOnlyFileSystem,
  normalizeFileSystemMode,
} from "@/config/utils";
import { settingsSchema } from "@/config/schema";
import {
  shouldIgnoreErrors,
  shouldSkipOnMissingParams,
  shouldUseCache,
} from "@/env-flags";
import { getBaseApiUrl } from "@/confident/api";

/** `resolveCost` and `resolveTemperature` are protected. */
function resolved(model: unknown) {
  return model as {
    resolveCost(input: number, output: number): number | null;
    resolveTemperature(): number | undefined;
  };
}

const TOUCHED = [
  "OPENAI_COST_PER_INPUT_TOKEN",
  "OPENAI_COST_PER_OUTPUT_TOKEN",
  "ANTHROPIC_COST_PER_INPUT_TOKEN",
  "TEMPERATURE",
  "DEEPEVAL_FILE_SYSTEM",
];

describe("per-provider cost environment variables", () => {
  const original: Record<string, string | undefined> = {};

  beforeEach(() => {
    for (const key of TOUCHED) {
      original[key] = process.env[key];
      delete process.env[key];
    }
  });

  afterEach(() => {
    for (const key of TOUCHED) {
      if (original[key] === undefined) delete process.env[key];
      else process.env[key] = original[key];
    }
  });

  it("prices a model from the environment when the caller passes no cost", () => {
    process.env.OPENAI_COST_PER_INPUT_TOKEN = "0.5";
    process.env.OPENAI_COST_PER_OUTPUT_TOKEN = "1.5";

    const cost = resolved(new OpenAIModel({ apiKey: "sk-test" })).resolveCost(
      2,
      4,
    );

    expect(cost).toBeCloseTo(0.5 * 2 + 1.5 * 4);
  });

  it("lets an explicit constructor cost win over the environment", () => {
    process.env.OPENAI_COST_PER_INPUT_TOKEN = "0.5";

    const model = new OpenAIModel({
      apiKey: "sk-test",
      costPerInputToken: 0.25,
      costPerOutputToken: 0,
    });

    expect(resolved(model).resolveCost(2, 0)).toBeCloseTo(0.5);
  });

  it("keys the variable off the provider, not the process", () => {
    process.env.ANTHROPIC_COST_PER_INPUT_TOKEN = "9";

    const openai = resolved(
      new OpenAIModel({ apiKey: "sk-test", model: "gpt-4.1" }),
    ).resolveCost(1, 0);
    const anthropic = resolved(
      new AnthropicModel({ apiKey: "sk-test" }),
    ).resolveCost(1, 0);

    expect(anthropic).toBeCloseTo(9);
    expect(openai).not.toBeCloseTo(9);
  });

  it("falls back to the registry when the variable is unparseable", () => {
    process.env.OPENAI_COST_PER_INPUT_TOKEN = "not-a-number";

    const cost = resolved(
      new OpenAIModel({ apiKey: "sk-test", model: "gpt-4.1" }),
    ).resolveCost(1, 1);

    expect(cost).not.toBeNull();
  });

  // Pin a model rather than taking the provider default: the default is
  // generated from Python and may be a reasoning model that drops temperature
  // entirely, which would make this read as a TEMPERATURE regression.
  it("reads TEMPERATURE when the caller passes none", () => {
    process.env.TEMPERATURE = "0.7";

    expect(
      resolved(
        new OpenAIModel({ apiKey: "sk-test", model: "gpt-4.1" }),
      ).resolveTemperature(),
    ).toBeCloseTo(0.7);
    expect(
      resolved(
        new OpenAIModel({
          apiKey: "sk-test",
          model: "gpt-4.1",
          temperature: 0.2,
        }),
      ).resolveTemperature(),
    ).toBeCloseTo(0.2);
  });
});

describe("boolean parsing", () => {
  it("accepts 1 and 0 for every boolean, which is the documented contract", () => {
    expect(parseBool("1")).toBe(true);
    expect(parseBool("0")).toBe(false);
  });

  it("accepts the other tokens Python accepts", () => {
    for (const token of ["true", "t", "yes", "y", "on", "enable", "enabled"]) {
      expect(parseBool(token)).toBe(true);
      expect(parseBool(token.toUpperCase())).toBe(true);
    }
    for (const token of [
      "false",
      "f",
      "no",
      "n",
      "off",
      "disable",
      "disabled",
    ]) {
      expect(parseBool(token)).toBe(false);
    }
  });

  it("ignores surrounding whitespace and quotes", () => {
    expect(parseBool('  "1" ')).toBe(true);
    expect(parseBool("'false'")).toBe(false);
  });

  it("reports unset, empty, and unrecognized values as undefined", () => {
    expect(parseBool(undefined)).toBeUndefined();
    expect(parseBool("  ")).toBeUndefined();
    expect(parseBool("2")).toBeUndefined();
    expect(parseBool("maybe")).toBeUndefined();
  });

  it("parses booleans the same way through the schema", () => {
    const parsed = settingsSchema.parse({
      DEEPEVAL_VERBOSE_MODE: "yes",
      CONFIDENT_TRACE_FLUSH: "0",
    });
    expect(parsed.DEEPEVAL_VERBOSE_MODE).toBe(true);
    expect(parsed.CONFIDENT_TRACE_FLUSH).toBe(false);
  });

  it("rejects an unrecognized boolean so the caller can warn and drop it", () => {
    expect(
      settingsSchema.safeParse({ DEEPEVAL_VERBOSE_MODE: "2" }).success,
    ).toBe(false);
  });

  it("defaults CONFIDENT_OPEN_BROWSER to on, as Python does", () => {
    expect(settingsSchema.parse({}).CONFIDENT_OPEN_BROWSER).toBe(true);
  });
});

describe("flags renamed to match Python", () => {
  const names = [
    "IGNORE_DEEPEVAL_ERRORS",
    "SKIP_DEEPEVAL_MISSING_PARAMS",
    "ENABLE_DEEPEVAL_CACHE",
  ] as const;
  const readers = [
    shouldIgnoreErrors,
    shouldSkipOnMissingParams,
    shouldUseCache,
  ];
  const original: Record<string, string | undefined> = {};

  beforeEach(() => {
    for (const name of names) {
      original[name] = process.env[name];
      delete process.env[name];
    }
  });

  afterEach(() => {
    for (const name of names) {
      if (original[name] === undefined) delete process.env[name];
      else process.env[name] = original[name];
    }
  });

  it.each(names.map((name, i) => [name, readers[i]] as const))(
    "reads %s",
    (name, read) => {
      expect(read()).toBe(false);
      process.env[name] = "1";
      expect(read()).toBe(true);
    },
  );

  it("no longer answers to the old TypeScript-only names", () => {
    process.env.DEEPEVAL_IGNORE_ERRORS = "1";
    expect(shouldIgnoreErrors()).toBe(false);
    delete process.env.DEEPEVAL_IGNORE_ERRORS;
  });
});

describe("Confident AI base URL", () => {
  const names = ["CONFIDENT_BASE_URL", "CONFIDENT_REGION"] as const;
  const original: Record<string, string | undefined> = {};

  beforeEach(() => {
    for (const name of names) {
      original[name] = process.env[name];
      delete process.env[name];
    }
  });

  afterEach(() => {
    for (const name of names) {
      if (original[name] === undefined) delete process.env[name];
      else process.env[name] = original[name];
    }
  });

  it("prefers CONFIDENT_BASE_URL over the region and the key prefix", () => {
    process.env.CONFIDENT_BASE_URL = "https://confident.internal/";
    process.env.CONFIDENT_REGION = "EU";

    expect(getBaseApiUrl("confident_eu_abc")).toBe(
      "https://confident.internal",
    );
  });

  it("falls back to the region, then to the key prefix", () => {
    process.env.CONFIDENT_REGION = "EU";
    expect(getBaseApiUrl("confident_eu_abc")).toContain("eu.");

    delete process.env.CONFIDENT_REGION;
    expect(getBaseApiUrl("confident_eu_abc")).toContain("eu.");
    expect(getBaseApiUrl("confident_abc")).toBe("https://api.confident-ai.com");
  });

  it("is read fresh, so a later settings change is picked up", () => {
    expect(getBaseApiUrl("confident_abc")).toBe("https://api.confident-ai.com");
    process.env.CONFIDENT_REGION = "EU";
    expect(getBaseApiUrl("confident_abc")).toContain("eu.");
  });
});

describe("DEEPEVAL_FILE_SYSTEM", () => {
  const original = process.env.DEEPEVAL_FILE_SYSTEM;
  afterEach(() => {
    if (original === undefined) delete process.env.DEEPEVAL_FILE_SYSTEM;
    else process.env.DEEPEVAL_FILE_SYSTEM = original;
  });

  it("accepts the same aliases Python does", () => {
    for (const value of ["READ_ONLY", "read-only", "readonly", " ro "]) {
      expect(normalizeFileSystemMode(value)).toBe("READ_ONLY");
    }
    expect(normalizeFileSystemMode("WRITABLE")).toBeUndefined();
  });

  it("is off unless set", () => {
    delete process.env.DEEPEVAL_FILE_SYSTEM;
    expect(isReadOnlyFileSystem()).toBe(false);
    process.env.DEEPEVAL_FILE_SYSTEM = "RO";
    expect(isReadOnlyFileSystem()).toBe(true);
  });

  it("rejects a value that is not a read-only alias", () => {
    expect(
      settingsSchema.safeParse({ DEEPEVAL_FILE_SYSTEM: "WRITABLE" }).success,
    ).toBe(false);
    expect(
      settingsSchema.parse({ DEEPEVAL_FILE_SYSTEM: "ro" }).DEEPEVAL_FILE_SYSTEM,
    ).toBe("READ_ONLY");
  });
});
