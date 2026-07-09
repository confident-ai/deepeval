import {
  Turn,
  MCPServer,
  MCPToolCall,
  ToolCall,
  ToolCallType,
} from "../../test-case";
import { Task } from "./schema";

/** Stringify a value the way the prompts expect (plain strings pass through). */
function toText(v: unknown): string {
  return typeof v === "string" ? v : JSON.stringify(v);
}

/** A primitive's repr for the "available primitives" blocks (Python uses repr()). */
export function reprPrimitive(obj: unknown): string {
  return JSON.stringify(obj);
}

export function indentMultilineString(s: string, indentLevel = 4): string {
  const indent = " ".repeat(indentLevel);
  return s
    .split("\n")
    .map((line) => `${indent}${line}`)
    .join("\n");
}

/**
 * MCP tool calls for a turn: the native `mcpToolsCalled` when present, otherwise
 * the `toolsCalled` entries explicitly typed as MCP. Mirrors Turn._mcp_tool_calls.
 */
export function mcpToolCalls(turn: Turn): (MCPToolCall | ToolCall)[] {
  if (turn.mcpToolsCalled != null) return turn.mcpToolsCalled;
  return (turn.toolsCalled ?? []).filter((t) => t.type === ToolCallType.MCP);
}

/** A turn involves MCP if any mcp_*_called list is present, or a toolsCalled entry is MCP-typed (mirrors Turn._mcp_interaction). */
export function mcpInteraction(turn: Turn): boolean {
  return (
    turn.mcpToolsCalled != null ||
    turn.mcpResourcesCalled != null ||
    turn.mcpPromptsCalled != null ||
    (turn.toolsCalled?.some((t) => t.type === ToolCallType.MCP) ?? false)
  );
}

/** MCP CallToolResult shape — the prompts read `.structuredContent.result`. */
function structuredResult(result: unknown): unknown {
  const r = result as { structuredContent?: { result?: unknown } } | null;
  return r?.structuredContent?.result ?? result;
}

export function taskStepsTakenText(task: Task): string {
  return task.steps_taken.join("\n\n");
}

/**
 * Build (availableTools, availableResources, availablePrompts) text blocks from
 * the test case's MCP servers. Mirrors `available_mcp_servers_block`.
 */
export function availableMcpServersBlock(servers: MCPServer[]): {
  availableTools: string;
  availableResources: string;
  availablePrompts: string;
} {
  let availableTools = "";
  let availableResources = "";
  let availablePrompts = "";
  for (const server of servers) {
    const header = `MCP Server ${server.serverName}\n`;
    availableTools += header;
    availableResources += header;
    availablePrompts += header;
    if (server.availableTools && server.availableTools.length > 0) {
      availableTools +=
        "\nAvailable Tools:\n[\n" +
        server.availableTools
          .map((t) => indentMultilineString(reprPrimitive(t), 4))
          .join(",\n") +
        "\n]";
    }
    if (server.availableResources && server.availableResources.length > 0) {
      availableResources +=
        "\nAvailable Resources:\n[\n" +
        server.availableResources
          .map((r) => indentMultilineString(reprPrimitive(r), 4))
          .join(",\n") +
        "\n]";
    }
    if (server.availablePrompts && server.availablePrompts.length > 0) {
      availablePrompts +=
        "\nAvailable Prompts:\n[\n" +
        server.availablePrompts
          .map((p) => indentMultilineString(reprPrimitive(p), 4))
          .join(",\n") +
        "\n]";
    }
  }
  return { availableTools, availableResources, availablePrompts };
}

/**
 * Turn unit interactions into tasks (user goal + the agent's MCP/text steps).
 * Shared verbatim by MCPTaskCompletionMetric and MultiTurnMCPUseMetric.
 */
export function getTasks(unitInteractions: Turn[][]): Task[] {
  const tasks: Task[] = [];
  for (const interaction of unitInteractions) {
    if (interaction.length <= 2) continue;
    let userMessages = "";
    for (const turn of interaction) {
      if (turn.role === "user") userMessages += turn.content + "\n";
      else break;
    }
    const task: Task = { task: userMessages, steps_taken: [] };
    for (const turn of interaction.slice(1)) {
      if (mcpInteraction(turn)) {
        let step = "Tools called by agent: \n";
        for (const tool of mcpToolCalls(turn)) {
          let args: unknown;
          let result: unknown;
          if (tool instanceof MCPToolCall) {
            args = tool.args;
            result = structuredResult(tool.result);
          } else {
            args = tool.inputParameters;
            result = tool.output;
          }
          step +=
            `\n<Tool Called>\n` +
            `\n**This does not appear to user**\n` +
            `Name: ${tool.name}\n` +
            `Args: ${toText(args)}\n` +
            `Result: \n${toText(result)}\n` +
            `</Tool Called>\n`;
        }
        if (turn.mcpResourcesCalled != null) {
          for (const resource of turn.mcpResourcesCalled) {
            step +=
              `\n<Resource Called>\n` +
              `\n**This does not appear to user**\n` +
              `URI: ${resource.uri}\n` +
              `Result: ${toText(resource.result)}\n` +
              `</Resource Called>\n`;
          }
        }
        if (turn.mcpPromptsCalled != null) {
          for (const prompt of turn.mcpPromptsCalled) {
            step +=
              `\n<Prompt Called>\n` +
              `\n**This does not appear to user**\n` +
              `Name: ${prompt.name}\n` +
              `Result: ${toText(prompt.result)}\n` +
              `</Prompt Called>\n`;
          }
        }
        task.steps_taken.push(step);
      } else {
        task.steps_taken.push("Agent's response to user: \n" + turn.content);
      }
    }
    tasks.push(task);
  }
  return tasks;
}
