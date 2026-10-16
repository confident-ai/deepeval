import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ensureEnvLocalIgnored,
  mergeEnvContent,
  parseEnvContent,
  readEnvValues,
  writeEnvValues,
} from "../src/env.js";
import type { CommandRunner } from "../src/git.js";

const directories: string[] = [];

const temporaryDirectory = async (): Promise<string> => {
  const directory = await mkdtemp(join(tmpdir(), "confident-env-test-"));
  directories.push(directory);
  return directory;
};

afterEach(async () => {
  await Promise.all(
    directories
      .splice(0)
      .map((directory) => rm(directory, { recursive: true, force: true })),
  );
});

describe("environment merge", () => {
  it("preserves unrelated content and replaces duplicate keys once", () => {
    expect(
      mergeEnvContent(
        "OTHER=1\nexport CONFIDENT_API_KEY=old\nCONFIDENT_API_KEY=duplicate\n",
        { CONFIDENT_API_KEY: "new value" },
      ),
    ).toBe('OTHER=1\nCONFIDENT_API_KEY="new value"\n');
  });

  it("adds judge-model settings without disturbing the project key", () => {
    expect(
      mergeEnvContent('CONFIDENT_API_KEY="confident"\n', {
        USE_ANTHROPIC_MODEL: "true",
        ANTHROPIC_API_KEY: "judge",
      }),
    ).toBe(
      'CONFIDENT_API_KEY="confident"\n\nUSE_ANTHROPIC_MODEL="true"\nANTHROPIC_API_KEY="judge"\n',
    );
  });

  it("atomically writes a mode-0600 file", async () => {
    const directory = await temporaryDirectory();
    await writeFile(join(directory, ".env.local"), "OTHER=1\n");
    await writeEnvValues(directory, { CONFIDENT_API_KEY: "secret" });

    expect(await readFile(join(directory, ".env.local"), "utf8")).toContain(
      'CONFIDENT_API_KEY="secret"',
    );
    expect((await stat(join(directory, ".env.local"))).mode & 0o777).toBe(
      0o600,
    );
  });
});

describe("environment parse", () => {
  it("reads the quoting and export forms DeepEval accepts", () => {
    expect(
      parseEnvContent(
        [
          "# comment",
          "export OPENAI_API_KEY='single'",
          'ANTHROPIC_API_KEY="double"',
          "USE_ANTHROPIC_MODEL=true # trailing",
          "EMPTY=",
        ].join("\n"),
      ),
    ).toEqual({
      OPENAI_API_KEY: "single",
      ANTHROPIC_API_KEY: "double",
      USE_ANTHROPIC_MODEL: "true",
      EMPTY: "",
    });
  });

  it("round-trips what the writer produced", async () => {
    const directory = await temporaryDirectory();
    await writeEnvValues(directory, { OPENAI_API_KEY: "sk-test value" });
    await expect(readEnvValues(directory)).resolves.toEqual({
      OPENAI_API_KEY: "sk-test value",
    });
  });

  it("treats a missing file as an empty environment", async () => {
    await expect(readEnvValues(await temporaryDirectory())).resolves.toEqual(
      {},
    );
  });
});

describe("gitignore merge", () => {
  const runner: CommandRunner = vi.fn();

  it("appends .env.local with a safe newline", async () => {
    const directory = await temporaryDirectory();
    await writeFile(join(directory, ".gitignore"), "dist/");
    await ensureEnvLocalIgnored(directory, runner, async () => false);
    expect(await readFile(join(directory, ".gitignore"), "utf8")).toBe(
      "dist/\n.env.local\n",
    );
  });

  it("does nothing when Git already ignores the file", async () => {
    const directory = await temporaryDirectory();
    await writeFile(join(directory, ".gitignore"), "*.local\n");
    await expect(
      ensureEnvLocalIgnored(directory, runner, async () => true),
    ).resolves.toBe(false);
    expect(await readFile(join(directory, ".gitignore"), "utf8")).toBe(
      "*.local\n",
    );
  });
});
