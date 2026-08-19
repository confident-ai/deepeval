// Mirrors deepeval/test_case/mcp.py. Minimal MCP-spec shapes — no
// @modelcontextprotocol/sdk dependency; users pass plain objects.
//
// These interfaces are deliberately not re-exported from `deepeval/test-case`:
// they exist to type our own fields, not to be a second MCP vocabulary users
// have to learn. Structural typing means a real `@modelcontextprotocol/sdk`
// object satisfies them without an import, and the SDK's own types are
// `z.infer` aliases that erase at runtime, so depending on it would buy
// compile-time shapes and no validation. The validation below uses zod, which
// is already a dependency, to get the runtime guarantee instead.
import { z } from "zod";

/** Subset of the MCP `Tool` shape. */
export interface Tool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  [key: string]: unknown;
}

/** Subset of the MCP `Resource` shape. */
export interface Resource {
  uri: string;
  name?: string;
  description?: string;
  mimeType?: string;
  [key: string]: unknown;
}

/** Subset of the MCP `Prompt` shape. */
export interface Prompt {
  name: string;
  description?: string;
  arguments?: unknown[];
  [key: string]: unknown;
}

export class MCPToolCall {
  name: string;
  args: Record<string, unknown>;
  result: unknown;

  constructor(params: {
    name: string;
    args: Record<string, unknown>;
    result: unknown;
  }) {
    this.name = params.name;
    this.args = params.args;
    this.result = params.result;
  }
}

export class MCPPromptCall {
  name: string;
  result: unknown;

  constructor(params: { name: string; result: unknown }) {
    this.name = params.name;
    this.result = params.result;
  }
}

export class MCPResourceCall {
  uri: string;
  result: unknown;

  constructor(params: { uri: string; result: unknown }) {
    this.uri = params.uri;
    this.result = params.result;
  }
}

export type MCPTransport = "stdio" | "sse" | "streamable-http";

export class MCPServer {
  serverName: string;
  transport?: MCPTransport;
  availableTools?: Tool[];
  availableResources?: Resource[];
  availablePrompts?: Prompt[];

  constructor(params: {
    serverName: string;
    transport?: MCPTransport;
    availableTools?: Tool[];
    availableResources?: Resource[];
    availablePrompts?: Prompt[];
  }) {
    this.serverName = params.serverName;
    this.transport = params.transport;
    this.availableTools = params.availableTools;
    this.availableResources = params.availableResources;
    this.availablePrompts = params.availablePrompts;
  }
}

/**
 * Mirrors `validate_mcp_servers`. Without the MCP SDK to `instanceof` against,
 * this does a light structural check that each list holds objects.
 */
export function validateMcpServers(mcpServers: MCPServer[]): void {
  const isObjList = (x: unknown): boolean =>
    Array.isArray(x) && x.every((i) => typeof i === "object" && i !== null);
  for (const s of mcpServers) {
    if (s.availableTools != null && !isObjList(s.availableTools)) {
      throw new TypeError(
        "'availableTools' must be a list of MCP Tool objects",
      );
    }
    if (s.availableResources != null && !isObjList(s.availableResources)) {
      throw new TypeError(
        "'availableResources' must be a list of MCP Resource objects",
      );
    }
    if (s.availablePrompts != null && !isObjList(s.availablePrompts)) {
      throw new TypeError(
        "'availablePrompts' must be a list of MCP Prompt objects",
      );
    }
  }
}

// Only the fields that tell one MCP result from another, so that both an SDK
// object and a hand-built one pass. A tool result carries content,
// structured content, or both — an object with neither is not one.
const CALL_TOOL_RESULT = z
  .object({
    content: z.array(z.unknown()).optional(),
    structuredContent: z.unknown().optional(),
    isError: z.boolean().optional(),
  })
  .refine((r) => r.content !== undefined || r.structuredContent !== undefined);
const READ_RESOURCE_RESULT = z.object({ contents: z.array(z.unknown()) });
const GET_PROMPT_RESULT = z.object({ messages: z.array(z.unknown()) });

function validateCalls(
  calls: { result?: unknown }[],
  field: string,
  className: string,
  resultType: string,
  result: z.ZodType,
): void {
  const ok =
    Array.isArray(calls) &&
    calls.every(
      (call) =>
        call != null &&
        typeof call === "object" &&
        result.safeParse(call.result).success,
    );
  if (!ok) {
    throw new TypeError(
      `'${field}' must be an array of ${className} whose 'result' is an MCP '${resultType}'`,
    );
  }
}

/**
 * Mirrors the `mcp.types` result checks Python runs when these fields are set,
 * catching the common mistake of passing a tool's text output where the whole
 * result object belongs.
 */
export function validateMcpCalls(calls: {
  mcpToolsCalled?: MCPToolCall[];
  mcpResourcesCalled?: MCPResourceCall[];
  mcpPromptsCalled?: MCPPromptCall[];
}): void {
  if (calls.mcpToolsCalled != null) {
    validateCalls(
      calls.mcpToolsCalled,
      "mcpToolsCalled",
      "MCPToolCall",
      "CallToolResult",
      CALL_TOOL_RESULT,
    );
  }
  if (calls.mcpResourcesCalled != null) {
    validateCalls(
      calls.mcpResourcesCalled,
      "mcpResourcesCalled",
      "MCPResourceCall",
      "ReadResourceResult",
      READ_RESOURCE_RESULT,
    );
  }
  if (calls.mcpPromptsCalled != null) {
    validateCalls(
      calls.mcpPromptsCalled,
      "mcpPromptsCalled",
      "MCPPromptCall",
      "GetPromptResult",
      GET_PROMPT_RESULT,
    );
  }
}
