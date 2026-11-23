import { relative } from "node:path";

import type { AgentKind } from "./agents.js";

const asObject = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};

const asString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined;

const clean = (value: string, maxLength = 140): string => {
  const singleLine = value
    // eslint-disable-next-line no-control-regex
    .replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, "")
    // eslint-disable-next-line no-control-regex
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return singleLine.length <= maxLength
    ? singleLine
    : `${singleLine.slice(0, maxLength - 3)}...`;
};

const displayPath = (
  value: unknown,
  projectDirectory: string,
): string | undefined => {
  const path = asString(value);
  if (!path) return undefined;
  const candidate = path.startsWith("/")
    ? relative(projectDirectory, path)
    : path;
  return clean(candidate || ".");
};

const action = (name: "read" | "edit" | "run", value?: string): string =>
  value ? `${name}: ${clean(value)}` : name;

const parseClaudeProgress = (
  value: unknown,
  projectDirectory: string,
): string[] => {
  const event = asObject(value);
  if (event.type === "system") return ["Thinking…"];
  if (event.type !== "assistant") return [];

  const message = asObject(event.message);
  const content = Array.isArray(message.content) ? message.content : [];
  return content.flatMap((entry): string[] => {
    const block = asObject(entry);
    if (block.type === "thinking") return ["Thinking…"];
    if (block.type !== "tool_use") return [];

    const name = asString(block.name)?.toLowerCase() ?? "tool";
    const input = asObject(block.input);
    const path = displayPath(
      input.file_path ?? input.path ?? input.notebook_path,
      projectDirectory,
    );
    if (
      ["read", "glob", "grep", "ls", "webfetch", "websearch"].includes(name)
    ) {
      return [action("read", path ?? asString(input.query))];
    }
    if (["write", "edit", "multiedit", "notebookedit"].includes(name)) {
      return [action("edit", path)];
    }
    if (name === "bash") return [action("run", asString(input.command))];
    return [`working: ${clean(name)}`];
  });
};

const parseCodexProgress = (
  value: unknown,
  projectDirectory: string,
): string[] => {
  const event = asObject(value);
  if (event.type === "thread.started" || event.type === "turn.started") {
    return ["Thinking…"];
  }

  const item = asObject(event.item);
  if (item.type === "command_execution" && item.status !== "completed") {
    return [action("run", asString(item.command))];
  }
  if (item.type === "file_change") {
    return [action("edit", displayPath(item.path, projectDirectory))];
  }
  if (item.type === "agent_reasoning") return ["Thinking…"];
  return [];
};

/** Cursor names its tools by the key wrapping them inside `tool_call`. */
const CURSOR_TOOLS: Record<string, "read" | "edit" | "run"> = {
  readToolCall: "read",
  lsToolCall: "read",
  grepToolCall: "read",
  globToolCall: "read",
  editToolCall: "edit",
  writeToolCall: "edit",
  deleteToolCall: "edit",
  shellToolCall: "run",
};

const parseCursorProgress = (
  value: unknown,
  projectDirectory: string,
): string[] => {
  const event = asObject(value);
  if (event.type === "system") return ["Thinking…"];
  // Every call also arrives as "completed", which would duplicate each line.
  if (event.type !== "tool_call" || event.subtype !== "started") return [];

  const call = asObject(event.tool_call);
  const name = Object.keys(call)[0];
  if (!name) return [];
  const args = asObject(asObject(call[name]).args);
  const kind = CURSOR_TOOLS[name];
  if (kind === "run") return [action("run", asString(args.command))];
  if (kind) {
    return [
      action(
        kind,
        displayPath(args.path, projectDirectory) ??
          asString(args.pattern ?? args.globPattern),
      ),
    ];
  }
  return [`working: ${clean(asString(asObject(call.function).name) ?? name)}`];
};

const parsers: Record<
  AgentKind,
  (value: unknown, projectDirectory: string) => string[]
> = {
  claude: parseClaudeProgress,
  codex: parseCodexProgress,
  cursor: parseCursorProgress,
};

export const parseAgentProgressLine = (
  kind: AgentKind,
  line: string,
  projectDirectory: string,
): string[] => {
  try {
    const value: unknown = JSON.parse(line);
    return parsers[kind](value, projectDirectory);
  } catch {
    return [];
  }
};
