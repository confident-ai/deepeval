import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { Command } from "commander";
import { SETTING_NAMES } from "@/config/schema";
import {
  getSettingSource,
  getSettings,
  resetSettings,
} from "@/config/settings";
import { registerProviderCommands } from "@/cli/commands/providers";
import { selectProvider } from "@/models/provider-selection";
import { initializeModel } from "@/metrics/utils";

const SET_OLLAMA = ["set-ollama", "--model", "llama3.2"];
const SET_LOCAL_MODEL = [
  "set-local-model",
  "--model",
  "my-vllm-model",
  "--base-url",
  "http://localhost:8000/v1",
];
const SAVE = ["--save", "dotenv"];
const LEGACY = { USE_LOCAL_MODEL: "true", LOCAL_MODEL_API_KEY: "ollama" };

async function run(...argv: string[]): Promise<void> {
  const program = new Command();
  program.exitOverride();
  registerProviderCommands(program);
  await program.parseAsync(["node", "deepeval", ...argv]);
}

function nextProcess(env: Record<string, string> = {}): void {
  for (const key of SETTING_NAMES) delete process.env[key];
  Object.assign(process.env, env);
  process.env.ENV_DIR_PATH = process.cwd();
  resetSettings({ reloadDotenv: true });
}

function writeKeystore(values: Record<string, string>): void {
  fs.mkdirSync(".deepeval", { recursive: true });
  fs.writeFileSync(path.join(".deepeval", ".deepeval"), JSON.stringify(values));
}

describe("Ollama provider selection", () => {
  const original: Record<string, string | undefined> = {};
  const originalCwd = process.cwd();
  let directory: string;

  beforeEach(() => {
    for (const key of SETTING_NAMES) original[key] = process.env[key];
    directory = fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-ollama-"));
    process.chdir(directory);
    nextProcess();
    jest.spyOn(console, "log").mockImplementation(() => undefined);
  });

  afterEach(() => {
    jest.restoreAllMocks();
    process.chdir(originalCwd);
    fs.rmSync(directory, { recursive: true, force: true });
    for (const key of SETTING_NAMES) {
      if (original[key] === undefined) delete process.env[key];
      else process.env[key] = original[key];
    }
    resetSettings();
  });

  describe("selectProvider", () => {
    it("selects Ollama when USE_OLLAMA_MODEL is true", () => {
      nextProcess({ USE_OLLAMA_MODEL: "true" });
      expect(selectProvider()).toBe("ollama");
    });

    it.each(["false", "unknown"])(
      "lets a present USE_OLLAMA_MODEL=%p override the placeholder key",
      (value) => {
        nextProcess({ USE_OLLAMA_MODEL: value, ...LEGACY });
        expect(selectProvider()).toBe("local-model");
      },
    );

    it.each(["env", "keystore"] as const)(
      "reads an empty USE_OLLAMA_MODEL from the %s as set and off",
      (source) => {
        if (source === "env") {
          nextProcess({ USE_OLLAMA_MODEL: "", ...LEGACY });
        } else {
          writeKeystore({ USE_OLLAMA_MODEL: "", USE_LOCAL_MODEL: "YES" });
          nextProcess({ LOCAL_MODEL_API_KEY: "ollama" });
        }
        expect(getSettingSource("USE_OLLAMA_MODEL")).toBe(source);
        expect(getSettings().USE_OLLAMA_MODEL).toBe(false);
        expect(selectProvider()).toBe("local-model");
      },
    );

    it.each([`"'true'"`, `""true""`])(
      "strips quotes like Python, reading USE_OLLAMA_MODEL=%s as on",
      (value) => {
        nextProcess({ USE_OLLAMA_MODEL: value, USE_LOCAL_MODEL: "true" });
        expect(selectProvider()).toBe("ollama");
      },
    );

    it("reads a quoted value with inner spaces as off, like Python", () => {
      nextProcess({ USE_OLLAMA_MODEL: '" true "', ...LEGACY });
      expect(selectProvider()).toBe("local-model");
    });

    it("falls back to the placeholder key when USE_OLLAMA_MODEL is unset", () => {
      nextProcess(LEGACY);
      expect(selectProvider()).toBe("ollama");
    });
  });

  it("selects Ollama after set-ollama", async () => {
    await run(...SET_OLLAMA);
    nextProcess();

    expect(selectProvider()).toBe("ollama");
  });

  it("lets set-local-model override a legacy Ollama config", async () => {
    nextProcess(LEGACY);
    await run(...SET_LOCAL_MODEL);
    nextProcess(LEGACY);

    expect(selectProvider()).toBe("local-model");
  });

  it("selects and builds the local model when both commands save to .env.local", async () => {
    await run(...SET_OLLAMA, ...SAVE);
    nextProcess();
    expect(selectProvider()).toBe("ollama");

    await run(...SET_LOCAL_MODEL, ...SAVE);
    nextProcess({ LOCAL_MODEL_API_KEY: "real-key" });

    const saved = fs.readFileSync(".env.local", "utf-8");
    expect(saved).not.toContain("USE_OLLAMA_MODEL");
    expect(saved).not.toContain("LOCAL_MODEL_API_KEY");
    expect(selectProvider()).toBe("local-model");
    expect(initializeModel().model.constructor.name).toBe("LocalModel");
  });

  it("keeps a later legacy config working after commands that never selected Ollama", async () => {
    await run(...SET_LOCAL_MODEL);
    nextProcess();
    await run("unset-ollama");
    nextProcess(LEGACY);

    expect(selectProvider()).toBe("ollama");
  });

  it("re-selects Ollama after Python saved USE_OLLAMA_MODEL=NO", async () => {
    writeKeystore({ USE_OLLAMA_MODEL: "NO", USE_LOCAL_MODEL: "YES" });
    nextProcess();
    expect(selectProvider()).toBe("local-model");

    await run(...SET_OLLAMA);
    nextProcess();

    expect(selectProvider()).toBe("ollama");
    expect(initializeModel().model.constructor.name).toBe("OllamaModel");
  });

  it("keeps an active local model and its endpoint on unset-ollama", async () => {
    await run(...SET_LOCAL_MODEL);
    nextProcess();
    await run("unset-ollama");
    nextProcess();

    expect(selectProvider()).toBe("local-model");
    expect(getSettings().LOCAL_MODEL_BASE_URL).toBe("http://localhost:8000/v1");
  });

  it.each(["set-ollama", "legacy config"])(
    "deselects Ollama on unset-ollama, starting from %s",
    async (start) => {
      if (start === "set-ollama") {
        await run(...SET_OLLAMA);
        nextProcess({ LOCAL_MODEL_API_KEY: "ollama" });
      } else {
        nextProcess(LEGACY);
      }
      expect(selectProvider()).toBe("ollama");

      await run("unset-ollama");
      nextProcess({ LOCAL_MODEL_API_KEY: "ollama" });

      expect(selectProvider()).toBeNull();
    },
  );

  it("deselects a saved Ollama on unset-ollama by removing the placeholder key", async () => {
    await run(...SET_OLLAMA, ...SAVE);
    await run("unset-ollama", ...SAVE);
    nextProcess();

    expect(selectProvider()).toBeNull();
    expect(getSettings().LOCAL_MODEL_API_KEY).toBeUndefined();
    expect(fs.readFileSync(".env.local", "utf-8")).not.toContain(
      "USE_OLLAMA_MODEL",
    );
  });
});
