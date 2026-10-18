import { describe, expect, it } from "vitest";

import { helpText, parseArgs, requireInteractiveTty } from "../src/args.js";

describe("parseArgs", () => {
  it("uses production defaults and separates source from project path", () => {
    expect(
      parseArgs(["--from", "in_app_setup", "--project-dir", "app"], "/repo"),
    ).toMatchObject({
      from: "in_app_setup",
      projectDir: "/repo/app",
      appUrl: "https://app.confident-ai.com",
      apiUrl: "https://api.confident-ai.com",
      help: false,
    });
  });

  it("labels the entry point deepeval unless --from names another", () => {
    expect(parseArgs([], "/repo").from).toBe("deepeval");
    // deepeval.com/setup.sh still passes this flag explicitly.
    expect(parseArgs(["--from", "deepeval"], "/repo").from).toBe("deepeval");
    expect(() => parseArgs(["--from", "not valid"], "/repo")).toThrow();
  });

  it("documents the deepeval-setup binary and its entry points", () => {
    expect(helpText).toContain("Usage: deepeval-setup [options]");
    expect(helpText).toContain("deepeval setup");
    expect(helpText).toContain("npx deepeval setup");
  });

  it("parses IDs and strips URL trailing slashes", () => {
    expect(
      parseArgs([
        "--app-url",
        "https://app.example/",
        "--api-url",
        "https://api.example///",
        "--org-id",
        "org",
        "--proj-id",
        "project",
      ]),
    ).toMatchObject({
      appUrl: "https://app.example",
      apiUrl: "https://api.example",
      orgId: "org",
      projId: "project",
    });
  });

  it("rejects unknown and missing arguments", () => {
    expect(() => parseArgs(["--wat"])).toThrow("Unknown argument");
    expect(() => parseArgs(["--from"])).toThrow("requires a value");
    expect(() => parseArgs(["--api-url", "nope"])).toThrow();
  });
});

describe("requireInteractiveTty", () => {
  it("rejects non-interactive execution", () => {
    expect(() => requireInteractiveTty(false, true)).toThrow("interactive TTY");
    expect(() => requireInteractiveTty(true, false)).toThrow("interactive TTY");
    expect(() => requireInteractiveTty(true, true)).not.toThrow();
  });
});
