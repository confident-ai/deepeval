import { describe, expect, it } from "vitest";

import { parseAgentProgressLine } from "../src/agent-progress.js";

describe("agent progress rendering", () => {
  it("turns Claude tool events into concise progress lines", () => {
    const line = JSON.stringify({
      type: "assistant",
      message: {
        content: [
          {
            type: "tool_use",
            name: "Edit",
            input: { file_path: "/repo/evals/smoke.py" },
          },
          {
            type: "tool_use",
            name: "Bash",
            input: { command: "python -m pytest evals" },
          },
        ],
      },
    });

    expect(parseAgentProgressLine("claude", line, "/repo")).toEqual([
      "edit: evals/smoke.py",
      "run: python -m pytest evals",
    ]);
  });

  it("turns Codex JSON events into concise progress lines", () => {
    const line = JSON.stringify({
      type: "item.started",
      item: {
        type: "command_execution",
        status: "in_progress",
        command: "npm test",
      },
    });

    expect(parseAgentProgressLine("codex", line, "/repo")).toEqual([
      "run: npm test",
    ]);
  });

  it("turns Cursor tool calls into concise progress lines", () => {
    const started = (toolCall: unknown): string =>
      JSON.stringify({
        type: "tool_call",
        subtype: "started",
        tool_call: toolCall,
      });

    expect(
      parseAgentProgressLine(
        "cursor",
        started({ editToolCall: { args: { path: "/repo/evals/smoke.py" } } }),
        "/repo",
      ),
    ).toEqual(["edit: evals/smoke.py"]);
    expect(
      parseAgentProgressLine(
        "cursor",
        started({ shellToolCall: { args: { command: "npm test" } } }),
        "/repo",
      ),
    ).toEqual(["run: npm test"]);
    expect(
      parseAgentProgressLine(
        "cursor",
        started({ grepToolCall: { args: { pattern: "AnswerRelevancy" } } }),
        "/repo",
      ),
    ).toEqual(["read: AnswerRelevancy"]);
  });

  it("reports a Cursor tool call once, not again on completion", () => {
    const completed = JSON.stringify({
      type: "tool_call",
      subtype: "completed",
      tool_call: { readToolCall: { args: { path: "/repo/app.py" } } },
    });
    expect(parseAgentProgressLine("cursor", completed, "/repo")).toEqual([]);
  });

  it("ignores incidental non-JSON output", () => {
    expect(parseAgentProgressLine("claude", "Starting...", "/repo")).toEqual(
      [],
    );
  });
});
