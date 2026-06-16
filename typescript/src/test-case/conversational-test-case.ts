import { ToolCall, RetrievedContextData } from "./llm-test-case";
import { checkIfMultimodal, extractImageIdsFromList, extractImageIdsFromString, MLLM_IMAGE_REGISTRY, MLLMImage } from "./mllm-image";
import {
  MCPServer,
  MCPToolCall,
  MCPResourceCall,
  MCPPromptCall,
  validateMcpServers,
} from "./mcp";

export enum MultiTurnParams {
  ROLE = "role",
  CONTENT = "content",
  METADATA = "metadata",
  TAGS = "tags",
  SCENARIO = "scenario",
  EXPECTED_OUTCOME = "expectedOutcome",
  CONTEXT = "context",
  USER_DESCRIPTION = "userDescription",
  RETRIEVAL_CONTEXT = "retrievalContext",
  CHATBOT_ROLE = "chatbotRole",
  TOOLS_CALLED = "toolsCalled",
  MCP_TOOLS = "mcpToolsCalled",
  MCP_RESOURCES = "mcpResourcesCalled",
  MCP_PROMPTS = "mcpPromptsCalled",
}

/** @deprecated Use {@link MultiTurnParams} instead. */
export { MultiTurnParams as TurnParams };

export class Turn {
  role: "user" | "assistant";
  content: string;
  userId?: string;
  retrievalContext?: (string | RetrievedContextData)[];
  toolsCalled?: ToolCall[];
  mcpToolsCalled?: MCPToolCall[];
  mcpResourcesCalled?: MCPResourceCall[];
  mcpPromptsCalled?: MCPPromptCall[];
  additionalMetadata?: Record<string, any>;

  constructor(params: {
    role: "user" | "assistant";
    content: string;
    userId?: string;
    retrievalContext?: (string | RetrievedContextData)[];
    toolsCalled?: ToolCall[];
    mcpToolsCalled?: MCPToolCall[];
    mcpResourcesCalled?: MCPResourceCall[];
    mcpPromptsCalled?: MCPPromptCall[];
    additionalMetadata?: Record<string, any>;
  }) {
    this.role = params.role;
    this.content = params.content;
    this.userId = params.userId;
    this.retrievalContext = params.retrievalContext;
    this.toolsCalled = params.toolsCalled;
    this.mcpToolsCalled = params.mcpToolsCalled;
    this.mcpResourcesCalled = params.mcpResourcesCalled;
    this.mcpPromptsCalled = params.mcpPromptsCalled;
    this.additionalMetadata = params.additionalMetadata;
    this.validate();
  }

  private validate(): void {
    if (this.retrievalContext != null) {
      if (
        !Array.isArray(this.retrievalContext) ||
        !this.retrievalContext.every(
          (s) => typeof s === "string" || s instanceof RetrievedContextData,
        )
      ) {
        throw new TypeError(
          "'retrievalContext' must be undefined or an array of strings or RetrievedContextData",
        );
      }
    }
    if (this.toolsCalled != null) {
      if (
        !Array.isArray(this.toolsCalled) ||
        !this.toolsCalled.every((t) => t instanceof ToolCall)
      ) {
        throw new TypeError(
          "'toolsCalled' must be undefined or an array of ToolCall",
        );
      }
    }
  }
}

export class ConversationalTestCase {
  turns: Turn[];
  chatbotRole?: string;
  scenario?: string;
  userDescription?: string;
  expectedOutcome?: string;
  context?: string[];
  mcpServers?: MCPServer[];
  name?: string;
  additionalMetadata?: Record<string, any>;
  comments?: string;
  tags?: string[];
  multimodal: boolean = false;
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;

  constructor(params: {
    turns: Turn[];
    chatbotRole?: string;
    scenario?: string;
    userDescription?: string;
    expectedOutcome?: string;
    context?: string[];
    mcpServers?: MCPServer[];
    name?: string;
    additionalMetadata?: Record<string, any>;
    comments?: string;
    tags?: string[];
    multimodal?: boolean;
    _datasetRank?: number;
    _datasetAlias?: string;
    _datasetId?: string;
  }) {
    this.turns = params.turns;
    this.chatbotRole = params.chatbotRole;
    this.scenario = params.scenario;
    this.userDescription = params.userDescription;
    this.expectedOutcome = params.expectedOutcome;
    this.context = params.context;
    this.mcpServers = params.mcpServers;
    this.name = params.name;
    this.additionalMetadata = params.additionalMetadata;
    this.comments = params.comments;
    this.tags = params.tags;
    this.multimodal = params.multimodal ?? this.detectMultimodal();
    this._datasetRank = params._datasetRank;
    this._datasetAlias = params._datasetAlias;
    this._datasetId = params._datasetId;
    this.validate();
  }

  public getImagesMapping(): Record<string, MLLMImage> | undefined {
    const ids = new Set<string>();

    extractImageIdsFromString(this.scenario, ids);
    extractImageIdsFromString(this.expectedOutcome, ids);
    extractImageIdsFromString(this.userDescription, ids);
    extractImageIdsFromList(this.context, ids);

    if (this.turns) {
      this.turns.forEach((turn) => {
        extractImageIdsFromString(turn.content, ids);
        extractImageIdsFromList(
          turn.retrievalContext?.map((c) => (typeof c === "string" ? c : c.context)),
          ids,
        );
      });
    }

    if (ids.size === 0) return undefined;

    const mapping: Record<string, MLLMImage> = {};
    ids.forEach((id) => {
      const img = MLLM_IMAGE_REGISTRY.get(id);
      if (img) mapping[id] = img;
    });

    return mapping;
  }

  /** Auto-detect multimodality from image slugs in the fields/turns (mirrors Python). */
  private detectMultimodal(): boolean {
    const has = (s?: string) => s != null && checkIfMultimodal(s);
    if (has(this.scenario) || has(this.expectedOutcome) || has(this.userDescription)) {
      return true;
    }
    for (const turn of this.turns ?? []) {
      if (checkIfMultimodal(turn.content)) return true;
      if (
        turn.retrievalContext?.some((c) =>
          checkIfMultimodal(typeof c === "string" ? c : c.context),
        )
      ) {
        return true;
      }
    }
    return false;
  }

  private validate(): void {
    if (!this.turns || this.turns.length === 0) {
      throw new TypeError("'turns' must not be empty");
    }
    if (this.context != null) {
      if (
        !Array.isArray(this.context) ||
        !this.context.every((s) => typeof s === "string")
      ) {
        throw new TypeError(
          "'context' must be undefined or an array of strings",
        );
      }
    }
    if (this.mcpServers != null) {
      validateMcpServers(this.mcpServers);
    }
  }
}
