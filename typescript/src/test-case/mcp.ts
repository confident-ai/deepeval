// Mirrors deepeval/test_case/mcp.py. Minimal MCP-spec shapes — no
// @modelcontextprotocol/sdk dependency; users pass plain objects.

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
      throw new TypeError("'availableTools' must be a list of MCP Tool objects");
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
