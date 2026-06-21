import {
  MCPServer,
  MCPToolCall,
  MCPResourceCall,
  MCPPromptCall,
  validateMcpServers,
} from "./mcp";
import { checkIfMultimodal, extractImageIdsFromList, extractImageIdsFromString, MLLM_IMAGE_REGISTRY, MLLMImage, SLUG_PATTERN } from "./mllm-image";

export enum SingleTurnParams {
  INPUT = "input",
  ACTUAL_OUTPUT = "actualOutput",
  EXPECTED_OUTPUT = "expectedOutput",
  CONTEXT = "context",
  RETRIEVAL_CONTEXT = "retrievalContext",
  TOOLS_CALLED = "toolsCalled",
  EXPECTED_TOOLS = "expectedTools",
  MCP_SERVERS = "mcpServers",
  MCP_TOOLS_CALLED = "mcpToolsCalled",
  MCP_RESOURCES_CALLED = "mcpResourcesCalled",
  MCP_PROMPTS_CALLED = "mcpPromptsCalled",
}

export enum ToolCallParams {
  INPUT_PARAMETERS = "inputParameters",
  OUTPUT = "output",
}

export class ToolCall {
  name: string;
  description?: string;
  reasoning?: string;
  output?: any;
  inputParameters?: Record<string, any>;

  constructor(params: {
    name: string;
    description?: string;
    reasoning?: string;
    output?: any;
    inputParameters?: Record<string, any>;
  }) {
    this.name = params.name;
    this.description = params.description;
    this.reasoning = params.reasoning;
    this.output = params.output;
    this.inputParameters = params.inputParameters;
  }
}

export class RetrievedContextData {
  context: string;
  source: string;

  constructor(params: { context: string; source: string }) {
    this.context = params.context;
    this.source = params.source;
  }

  toString(): string {
    return `${this.source}: ${this.context}`;
  }
}

export function resolveRetrievalContext(
  retrievalContext: (string | RetrievedContextData)[],
): string[];
export function resolveRetrievalContext(
  retrievalContext: (string | RetrievedContextData)[] | undefined,
): string[] | undefined;
export function resolveRetrievalContext(
  retrievalContext: (string | RetrievedContextData)[] | undefined,
): string[] | undefined {
  return retrievalContext?.map((c) =>
    typeof c === "string" ? c : c.toString(),
  );
}

export class LLMTestCase {
  input: string;
  actualOutput: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: (string | RetrievedContextData)[];
  additionalMetadata?: Record<string, any>;
  comments?: string;
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  mcpServers?: MCPServer[];
  mcpToolsCalled?: MCPToolCall[];
  mcpResourcesCalled?: MCPResourceCall[];
  mcpPromptsCalled?: MCPPromptCall[];
  reasoning?: string;
  tokenCost?: number;
  completionTime?: number;
  name?: string;
  multimodal: boolean = false;
  _traceDict?: Record<string, unknown>;
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;

  constructor(params: {
    input: string;
    actualOutput: string;
    expectedOutput?: string;
    context?: string[];
    retrievalContext?: (string | RetrievedContextData)[];
    additionalMetadata?: Record<string, any>;
    comments?: string;
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    mcpServers?: MCPServer[];
    mcpToolsCalled?: MCPToolCall[];
    mcpResourcesCalled?: MCPResourceCall[];
    mcpPromptsCalled?: MCPPromptCall[];
    reasoning?: string;
    tokenCost?: number;
    completionTime?: number;
    name?: string;
    multimodal?: boolean;
    _datasetRank?: number;
    _datasetAlias?: string;
    _datasetId?: string;
  }) {
    this.input = params.input;
    this.actualOutput = params.actualOutput;
    this.expectedOutput = params.expectedOutput;
    this.context = params.context;
    this.retrievalContext = params.retrievalContext;
    this.additionalMetadata = params.additionalMetadata;
    this.comments = params.comments;
    this.toolsCalled = params.toolsCalled;
    this.expectedTools = params.expectedTools;
    this.mcpServers = params.mcpServers;
    this.mcpToolsCalled = params.mcpToolsCalled;
    this.mcpResourcesCalled = params.mcpResourcesCalled;
    this.mcpPromptsCalled = params.mcpPromptsCalled;
    this.reasoning = params.reasoning;
    this.tokenCost = params.tokenCost;
    this.completionTime = params.completionTime;
    this.name = params.name;
    this._datasetRank = params._datasetRank;
    this._datasetAlias = params._datasetAlias;
    this._datasetId = params._datasetId;
    this.multimodal = params.multimodal ?? this.detectMultimodal();
    this.validate();
  }

  public getImagesMapping(): Record<string, MLLMImage> | undefined {
    const ids = new Set<string>();

    extractImageIdsFromString(this.input, ids);
    extractImageIdsFromString(this.actualOutput, ids);
    extractImageIdsFromString(this.expectedOutput, ids);
    extractImageIdsFromList(this.context, ids);
    
    extractImageIdsFromList(
      this.retrievalContext?.map((c) => (typeof c === "string" ? c : c.context)),
      ids,
    );

    if (ids.size === 0) return undefined;

    const mapping: Record<string, MLLMImage> = {};
    ids.forEach((id) => {
      const img = MLLM_IMAGE_REGISTRY.get(id);
      if (img) mapping[id] = img;
    });

    return mapping;
  }

  /** Auto-detect multimodality from image slugs in the string fields (mirrors Python). */
  private detectMultimodal(): boolean {
    const has = (s?: string) => s != null && checkIfMultimodal(s);
    let detect =
      has(this.input) || has(this.actualOutput) || has(this.expectedOutput);
    if (this.retrievalContext != null) {
      detect =
        detect ||
        this.retrievalContext.some((c) =>
          checkIfMultimodal(typeof c === "string" ? c : c.context),
        );
    }
    if (this.context != null) {
      detect = detect || this.context.some((c) => checkIfMultimodal(c));
    }
    return detect;
  }

  private validate(): void {
    if (this.context !== undefined && this.context !== null) {
      if (
        !Array.isArray(this.context) ||
        !this.context.every((item) => typeof item === "string")
      ) {
        throw new TypeError(
          "'context' must be undefined or an array of strings",
        );
      }
    }
    if (this.retrievalContext !== undefined && this.retrievalContext !== null) {
      if (
        !Array.isArray(this.retrievalContext) ||
        !this.retrievalContext.every(
          (item) =>
            typeof item === "string" || item instanceof RetrievedContextData,
        )
      ) {
        throw new TypeError(
          "'retrievalContext' must be undefined or an array of strings or RetrievedContextData",
        );
      }
    }
    if (this.toolsCalled !== undefined && this.toolsCalled !== null) {
      if (
        !Array.isArray(this.toolsCalled) ||
        !this.toolsCalled.every((item) => item instanceof ToolCall)
      ) {
        throw new TypeError(
          "'toolsCalled' must be undefined or an array of `ToolCall`",
        );
      }
    }
    if (this.expectedTools !== undefined && this.expectedTools !== null) {
      if (
        !Array.isArray(this.expectedTools) ||
        !this.expectedTools.every((item) => item instanceof ToolCall)
      ) {
        throw new TypeError(
          "'expectedTools' must be undefined or an array of `ToolCall`",
        );
      }
    }
    if (this.mcpServers != null) {
      validateMcpServers(this.mcpServers);
    }
  }
}
