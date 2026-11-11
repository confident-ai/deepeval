import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type * as Prompts from "@clack/prompts";

import { parseEnvContent } from "../src/env.js";
import { detectJudge, judgeProviders, judgeVariables } from "../src/judge.js";
import type * as Ui from "../src/ui.js";

const prompts = vi.hoisted(() => ({
  select: vi.fn(),
  password: vi.fn(),
  text: vi.fn(),
}));

vi.mock("@clack/prompts", async (importOriginal) => ({
  ...(await importOriginal<typeof Prompts>()),
  ...prompts,
}));

vi.mock("../src/ui.js", async (importOriginal) => ({
  ...(await importOriginal<typeof Ui>()),
  log: { info: vi.fn(), warn: vi.fn(), success: vi.fn(), error: vi.fn() },
  spinner: () => ({ start: vi.fn(), stop: vi.fn(), message: vi.fn() }),
}));

const { configureJudgeModel } = await import("../src/wizard.js");

const JUDGE_KEYS = judgeProviders.flatMap((provider) => [
  ...Object.keys(provider.flags),
  ...judgeVariables(provider),
  ...provider.secrets.flatMap(({ optionalWhen }) => optionalWhen ?? []),
]);

describe("switching an incomplete Ollama judge to a local server", () => {
  let directory: string;

  beforeEach(async () => {
    directory = await mkdtemp(join(tmpdir(), "judge-flow-"));
    for (const key of JUDGE_KEYS) vi.stubEnv(key, undefined);
  });

  afterEach(async () => {
    vi.unstubAllEnvs();
    vi.clearAllMocks();
    await rm(directory, { recursive: true, force: true });
  });

  it("asks for a real key instead of reusing Ollama's marker", async () => {
    await writeFile(
      join(directory, ".env.local"),
      "USE_OLLAMA_MODEL=true\nUSE_LOCAL_MODEL=true\nLOCAL_MODEL_API_KEY=ollama\n",
    );
    const local = judgeProviders.find((provider) => provider.id === "local")!;
    prompts.select.mockResolvedValue(local);
    prompts.password.mockResolvedValue("real-key");
    prompts.text.mockImplementation(async ({ message }: { message: string }) =>
      message === "Server base URL" ? "http://localhost:8000/v1" : "my-model",
    );

    await configureJudgeModel(directory);

    expect(prompts.password).toHaveBeenCalledTimes(1);
    const { validate } = prompts.password.mock.calls[0]![0] as {
      validate: (value: string | undefined) => string | undefined;
    };
    expect(validate("ollama")).toMatch(/reserved/);
    expect(validate("")).toBe("A value is required");
    expect(validate("real-key")).toBeUndefined();

    const env = parseEnvContent(
      await readFile(join(directory, ".env.local"), "utf8"),
    );
    expect(env).toMatchObject({
      USE_LOCAL_MODEL: "true",
      LOCAL_MODEL_API_KEY: "real-key",
      LOCAL_MODEL_NAME: "my-model",
      LOCAL_MODEL_BASE_URL: "http://localhost:8000/v1",
    });
    expect(env.USE_OLLAMA_MODEL).toBeUndefined();
    expect(detectJudge(env)).toEqual({ provider: local, missing: [] });
  });
});
