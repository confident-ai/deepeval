import { spawn } from "node:child_process";

export type AgentKind = "claude" | "codex" | "cursor";
export type AgentPermission = "read-only" | "full";

export interface AgentDefinition {
  kind: AgentKind;
  label: string;
  command: string;
}

export interface AgentInvocation {
  command: string;
  args: string[];
}

export interface AgentCheck {
  agent: AgentDefinition;
  discovered: boolean;
  authenticated: boolean;
  readOnlyReady: boolean;
  detail?: string;
}

export type AgentRunner = (
  invocation: AgentInvocation,
  options: {
    cwd: string;
    env?: NodeJS.ProcessEnv;
    stdio?: "inherit" | "pipe";
    timeoutMs?: number;
    onStdoutLine?: (line: string) => void;
  },
) => Promise<{ stdout: string }>;

export const supportedAgents: AgentDefinition[] = [
  { kind: "claude", label: "Claude Code", command: "claude" },
  { kind: "codex", label: "Codex", command: "codex" },
  { kind: "cursor", label: "Cursor CLI", command: "cursor-agent" },
];

export const buildAgentInvocation = (
  agent: AgentDefinition,
  permission: AgentPermission,
  prompt: string,
): AgentInvocation => {
  if (agent.kind === "claude") {
    return {
      command: agent.command,
      args:
        permission === "read-only"
          ? [
              "-p",
              prompt,
              "--permission-mode",
              "plan",
              "--no-session-persistence",
              "--tools",
              "",
              "--output-format",
              "json",
            ]
          : [
              "-p",
              prompt,
              "--no-session-persistence",
              "--permission-mode",
              "bypassPermissions",
              "--dangerously-skip-permissions",
              "--output-format",
              "stream-json",
              "--verbose",
            ],
    };
  }
  if (agent.kind === "cursor") {
    // The prompt is positional for this CLI, so it stays last.
    return {
      command: agent.command,
      args:
        permission === "read-only"
          ? [
              "--print",
              "--output-format",
              "json",
              "--mode",
              "ask",
              "--sandbox",
              "enabled",
              prompt,
            ]
          : [
              "--print",
              "--output-format",
              "stream-json",
              "--force",
              "--trust",
              "--sandbox",
              "disabled",
              prompt,
            ],
    };
  }
  return {
    command: agent.command,
    args: [
      "exec",
      "--json",
      "--ephemeral",
      "--sandbox",
      permission === "read-only" ? "read-only" : "danger-full-access",
      ...(permission === "full"
        ? ["--dangerously-bypass-approvals-and-sandbox"]
        : []),
      "--skip-git-repo-check",
      prompt,
    ],
  };
};

const authArguments: Record<AgentKind, string[]> = {
  claude: ["auth", "status"],
  codex: ["login", "status"],
  cursor: ["status", "--format", "json"],
};

export const getAuthInvocation = (agent: AgentDefinition): AgentInvocation => ({
  command: agent.command,
  args: authArguments[agent.kind],
});

export const runAgent: AgentRunner = (
  invocation,
  { cwd, env, stdio = "pipe", timeoutMs = 30_000, onStdoutLine },
) =>
  new Promise((resolve, reject) => {
    const child = spawn(invocation.command, invocation.args, {
      cwd,
      env,
      stdio: stdio === "inherit" ? "inherit" : ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let stdoutLineBuffer = "";
    if (stdio === "pipe") {
      child.stdout?.on("data", (chunk: Buffer) => {
        const text = chunk.toString();
        stdout += text;
        stdoutLineBuffer += text;
        let newline = stdoutLineBuffer.indexOf("\n");
        while (newline >= 0) {
          const line = stdoutLineBuffer.slice(0, newline).trim();
          stdoutLineBuffer = stdoutLineBuffer.slice(newline + 1);
          if (line) onStdoutLine?.(line);
          newline = stdoutLineBuffer.indexOf("\n");
        }
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
      });
    }
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(
        new Error(
          `${invocation.command} timed out after ${Math.round(timeoutMs / 1000)}s.`,
        ),
      );
    }, timeoutMs);
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("exit", (code, signal) => {
      clearTimeout(timer);
      const finalLine = stdoutLineBuffer.trim();
      if (finalLine) onStdoutLine?.(finalLine);
      if (code === 0) {
        resolve({ stdout });
      } else {
        const detail = stderr
          .split(/\r?\n/)
          .map((line) => line.trim())
          .find(Boolean);
        reject(
          new Error(
            `${invocation.command} exited with ${signal ?? code}${
              detail ? `: ${detail.slice(0, 300)}` : ""
            }`,
          ),
        );
      }
    });
  });

export const checkAgent = async (
  agent: AgentDefinition,
  cwd: string,
  runner: AgentRunner = runAgent,
): Promise<AgentCheck> => {
  try {
    await runner(
      { command: agent.command, args: ["--version"] },
      { cwd, timeoutMs: 10_000 },
    );
  } catch {
    return {
      agent,
      discovered: false,
      authenticated: false,
      readOnlyReady: false,
      detail: "Executable not found.",
    };
  }

  try {
    await runner(getAuthInvocation(agent), { cwd, timeoutMs: 15_000 });
  } catch (error) {
    return {
      agent,
      discovered: true,
      authenticated: false,
      readOnlyReady: false,
      detail: error instanceof Error ? error.message : "Authentication failed.",
    };
  }

  try {
    await runner(
      buildAgentInvocation(
        agent,
        "read-only",
        "Read-only smoke check: do not inspect files or environment. Reply READY.",
      ),
      { cwd, timeoutMs: 60_000 },
    );
    return {
      agent,
      discovered: true,
      authenticated: true,
      readOnlyReady: true,
    };
  } catch (error) {
    return {
      agent,
      discovered: true,
      authenticated: true,
      readOnlyReady: false,
      detail: error instanceof Error ? error.message : "Smoke check failed.",
    };
  }
};

export const executeAgent = async (
  agent: AgentDefinition,
  cwd: string,
  prompt: string,
  apiKey: string | undefined,
  resultFile: string,
  runner: AgentRunner = runAgent,
  onProgressLine?: (line: string) => void,
): Promise<void> => {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    CONFIDENT_SETUP_RESULT_FILE: resultFile,
  };
  if (apiKey) env.CONFIDENT_API_KEY = apiKey;
  else delete env.CONFIDENT_API_KEY;
  await runner(buildAgentInvocation(agent, "full", prompt), {
    cwd,
    env,
    stdio: "pipe",
    timeoutMs: 30 * 60_000,
    ...(onProgressLine ? { onStdoutLine: onProgressLine } : {}),
  });
};
