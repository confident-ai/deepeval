import { describe, expect, it } from "vitest";

import { restoreTerminalFocus, type FocusOptions } from "../src/focus.js";

const record = (overrides: Partial<FocusOptions> = {}) => {
  const written: string[] = [];
  const commands: Array<{ command: string; args: string[] }> = [];
  const options: FocusOptions = {
    platform: "darwin",
    bundleId: "com.todesktop.230313mzl4w4u92",
    isTty: true,
    write: (text) => written.push(text),
    runner: async (command, args) => {
      commands.push({ command, args });
      return { stdout: "", stderr: "" };
    },
    ...overrides,
  };
  return { written, commands, options };
};

describe("returning focus after browser pairing", () => {
  it("raises the app that owns this terminal on macOS", async () => {
    const { written, commands, options } = record();
    await restoreTerminalFocus(options);
    expect(written).toEqual(["\x07"]);
    expect(commands).toEqual([
      {
        command: "osascript",
        args: [
          "-e",
          'tell application id "com.todesktop.230313mzl4w4u92" to activate',
        ],
      },
    ]);
  });

  it("rings the bell alone where no window manager can be trusted", async () => {
    const { written, commands, options } = record({ platform: "linux" });
    await restoreTerminalFocus(options);
    expect(written).toEqual(["\x07"]);
    expect(commands).toEqual([]);
  });

  it("stays quiet when output is not a terminal", async () => {
    const { written, options } = record({ isTty: false });
    await restoreTerminalFocus(options);
    expect(written).toEqual([]);
  });

  it("refuses a bundle id that could carry AppleScript", async () => {
    const { commands, options } = record({
      bundleId: 'com.foo" to activate\ntell application "Finder',
    });
    await restoreTerminalFocus(options);
    expect(commands).toEqual([]);
  });

  it("never fails setup when the activation is refused", async () => {
    const { options } = record({
      runner: () => Promise.reject(new Error("not authorized")),
    });
    await expect(restoreTerminalFocus(options)).resolves.toBeUndefined();
  });
});
