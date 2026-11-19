import { describe, expect, it, vi } from "vitest";

import {
  buildAgentInvocation,
  checkAgent,
  executeAgent,
  getAuthInvocation,
  supportedAgents,
  type AgentRunner,
} from "../src/agents.js";

describe("agent arguments", () => {
  it("uses plan/read-only before Claude full permission", () => {
    const claude = supportedAgents[0]!;
    expect(buildAgentInvocation(claude, "read-only", "prompt").args).toContain(
      "plan",
    );
    expect(buildAgentInvocation(claude, "full", "prompt").args).toContain(
      "--dangerously-skip-permissions",
    );
  });

  it("uses Cursor's ask mode before its unsandboxed run", () => {
    const cursor = supportedAgents[2]!;
    const readOnly = buildAgentInvocation(cursor, "read-only", "prompt");
    expect(readOnly.command).toBe("cursor-agent");
    expect(readOnly.args).toContain("ask");
    expect(readOnly.args).toContain("enabled");
    expect(readOnly.args).not.toContain("--force");

    const full = buildAgentInvocation(cursor, "full", "prompt");
    expect(full.args).toContain("--force");
    expect(full.args).toContain("--trust");
    expect(full.args).toContain("stream-json");
    // The CLI takes the prompt positionally, so it has to come last.
    expect(full.args.at(-1)).toBe("prompt");
  });

  it("asks each agent for its own authentication status", () => {
    expect(supportedAgents.map((agent) => getAuthInvocation(agent))).toEqual([
      { command: "claude", args: ["auth", "status"] },
      { command: "codex", args: ["login", "status"] },
      { command: "cursor-agent", args: ["status", "--format", "json"] },
    ]);
  });

  it("uses Codex sandbox levels", () => {
    const codex = supportedAgents[1]!;
    expect(buildAgentInvocation(codex, "read-only", "prompt").args).toContain(
      "read-only",
    );
    expect(buildAgentInvocation(codex, "full", "prompt").args).toContain(
      "danger-full-access",
    );
    expect(buildAgentInvocation(codex, "full", "prompt").args).toContain(
      "--json",
    );
  });
});

describe("agent checks and execution", () => {
  it("runs discovery, auth, and a read-only smoke check", async () => {
    const runner: AgentRunner = vi.fn().mockResolvedValue({ stdout: "ok" });
    const result = await checkAgent(supportedAgents[0]!, "/repo", runner);
    expect(result).toMatchObject({
      discovered: true,
      authenticated: true,
      readOnlyReady: true,
    });
    expect(runner).toHaveBeenCalledTimes(3);
  });

  it("marks missing executables without further checks", async () => {
    const runner: AgentRunner = vi.fn().mockRejectedValue(new Error("ENOENT"));
    const result = await checkAgent(supportedAgents[1]!, "/repo", runner);
    expect(result.discovered).toBe(false);
    expect(runner).toHaveBeenCalledOnce();
  });

  it("injects secrets through env, never arguments", async () => {
    const runner: AgentRunner = vi.fn().mockResolvedValue({ stdout: "" });
    await executeAgent(
      supportedAgents[1]!,
      "/repo",
      "safe prompt",
      "super-secret",
      "/tmp/result.json",
      runner,
    );
    const [invocation, options] = vi.mocked(runner).mock.calls[0]!;
    expect(invocation.args.join(" ")).not.toContain("super-secret");
    expect(options.env?.CONFIDENT_API_KEY).toBe("super-secret");
    expect(options.env?.CONFIDENT_SETUP_RESULT_FILE).toBe("/tmp/result.json");
    expect(options.stdio).toBe("pipe");
  });

  it("omits CONFIDENT_API_KEY for local runs", async () => {
    const runner: AgentRunner = vi.fn().mockResolvedValue({ stdout: "" });
    vi.stubEnv("CONFIDENT_API_KEY", "existing");
    await executeAgent(
      supportedAgents[1]!,
      "/repo",
      "safe prompt",
      undefined,
      "/tmp/result.json",
      runner,
    );
    expect(vi.mocked(runner).mock.calls[0]?.[1]?.env?.CONFIDENT_API_KEY).toBe(
      undefined,
    );
    vi.unstubAllEnvs();
  });
});
