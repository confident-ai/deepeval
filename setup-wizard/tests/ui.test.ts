import { afterEach, describe, expect, it, vi } from "vitest";

/** picocolors reads the environment on import, so color needs a fresh module. */
const importWithColor = async () => {
  vi.resetModules();
  vi.stubEnv("NO_COLOR", undefined);
  vi.stubEnv("FORCE_COLOR", "3");
  return import("../src/ui.js");
};

const importThemeWithColor = async () => {
  await importWithColor();
  return import("../src/theme.js");
};

const capture = (write: () => void): string => {
  const chunks: string[] = [];
  const spy = vi
    .spyOn(process.stdout, "write")
    .mockImplementation((chunk: unknown) => {
      chunks.push(String(chunk));
      return true;
    });
  try {
    write();
  } finally {
    spy.mockRestore();
  }
  return chunks.join("");
};

const BRAND = "38;2;118;14;255";
const ACCENT = "38;2;0;229;255";
const ALERT = "38;2;255;65;1";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("themed clack logs", () => {
  it("marks progress with one accent instead of blue and green", async () => {
    const { log } = await importWithColor();
    for (const write of [log.info, log.step, log.success]) {
      const output = capture(() => write("Signing in"));
      expect(output).toContain(ACCENT);
      expect(output).not.toContain(ALERT);
    }
  });

  it("reserves ember for messages the user has to act on", async () => {
    const { log } = await importWithColor();
    for (const write of [log.warn, log.error]) {
      const output = capture(() => write("Could not open the browser"));
      expect(output).toContain(ALERT);
      expect(output).not.toContain(ACCENT);
    }
  });

  it("saves brand violet for the Confident AI name", async () => {
    const { stepHeading, welcomeMessage } = await importThemeWithColor();
    expect(welcomeMessage()).toContain(BRAND);
    expect(welcomeMessage(false)).not.toContain(BRAND);
    expect(welcomeMessage(false)).toContain(ACCENT);
    expect(stepHeading(1)).toContain(ACCENT);
    expect(stepHeading(1)).not.toContain(BRAND);
  });
});
