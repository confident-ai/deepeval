import { describe, expect, it, vi } from "vitest";

import {
  describeGitStatus,
  inspectGit,
  type CommandRunner,
} from "../src/git.js";

describe("Git preflight", () => {
  it("reports non-Git directories", async () => {
    const runner: CommandRunner = vi
      .fn()
      .mockRejectedValue(new Error("no git"));
    await expect(inspectGit("/repo", runner)).resolves.toEqual({
      isRepository: false,
      dirty: false,
      files: [],
      totalChanges: 0,
    });
  });

  it("limits dirty file output to twenty", async () => {
    const changes = Array.from(
      { length: 23 },
      (_, index) => ` M src/file-${index}.ts`,
    ).join("\n");
    const runner: CommandRunner = vi
      .fn()
      .mockResolvedValueOnce({ stdout: "true\n", stderr: "" })
      .mockResolvedValueOnce({ stdout: changes, stderr: "" });
    const status = await inspectGit("/repo", runner);

    expect(status.dirty).toBe(true);
    expect(status.files).toHaveLength(20);
    expect(status.totalChanges).toBe(23);
    expect(describeGitStatus(status)).toContain("…plus 3 more");
    expect(describeGitStatus(status)).toContain("1. src/file-0.ts");
  });

  it("recognizes a clean repository", async () => {
    const runner: CommandRunner = vi
      .fn()
      .mockResolvedValueOnce({ stdout: "true\n", stderr: "" })
      .mockResolvedValueOnce({ stdout: "", stderr: "" });
    await expect(inspectGit("/repo", runner)).resolves.toMatchObject({
      isRepository: true,
      dirty: false,
    });
  });
});
