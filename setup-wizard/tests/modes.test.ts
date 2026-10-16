import { describe, expect, it } from "vitest";

import {
  deferredSetupMessage,
  DEEPEVAL_DOCS_URL,
  fullPermissionWarning,
  listLabels,
  MANUAL_QUICKSTART_URL,
  setupModeOptions,
} from "../src/modes.js";

describe("mode helpers", () => {
  it("defines all three setup modes", () => {
    expect(setupModeOptions().map((option) => option.value)).toEqual([
      "built-in",
      "own-agent",
      "manual",
    ]);
    expect(fullPermissionWarning("Codex")).toContain("full permission");
    expect(deferredSetupMessage("manual")).toContain(MANUAL_QUICKSTART_URL);
    expect(deferredSetupMessage("manual", false)).toContain(DEEPEVAL_DOCS_URL);
    expect(deferredSetupMessage("own-agent")).toBe(
      deferredSetupMessage("own-agent", false),
    );
  });

  it("names detected agents and explains paste-your-own-prompt", () => {
    const [detected, own] = setupModeOptions([{ label: "Claude Code" }]);
    expect(detected?.label).toBe("Use detected agent (Claude Code)");
    expect(detected?.hint).toBe(
      "We run Claude Code for you here and stream its progress",
    );
    expect(own?.label).toBe("Paste a setup prompt into your coding agent");
    expect(
      setupModeOptions([{ label: "Claude Code" }, { label: "Codex" }])[0]
        ?.label,
    ).toBe("Use a detected agent (Claude Code or Codex)");
    expect(
      setupModeOptions([
        { label: "Claude Code" },
        { label: "Codex" },
        { label: "Cursor CLI" },
      ])[0]?.label,
    ).toBe("Use a detected agent (Claude Code, Codex, or Cursor CLI)");
    expect(listLabels(["Claude Code", "Codex", "Cursor CLI"], "and")).toBe(
      "Claude Code, Codex, and Cursor CLI",
    );
    expect(listLabels(["Codex"], "and")).toBe("Codex");
  });
});
