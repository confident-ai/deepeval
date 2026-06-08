export enum LLMTestCaseParams {
  INPUT = "input",
  ACTUAL_OUTPUT = "actual_output",
  EXPECTED_OUTPUT = "expected_output",
  CONTEXT = "context",
  RETRIEVAL_CONTEXT = "retrieval_context",
  TOOLS_CALLED = "tools_called",
  EXPECTED_TOOLS = "expected_tools",
}

export enum ToolCallParams {
  INPUT_PARAMETERS = "input_parameters",
  OUTPUT = "output",
}

export enum TurnParams {
  ROLE = "role",
  CONTENT = "content",
  SCENARIO = "scenario",
  EXPECTED_OUTCOME = "expected_outcome",
  RETRIEVAL_CONTEXT = "retrieval_context",
  TOOLS_CALLED = "tools_called",
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

export class LLMTestCase {
  input: string;
  actualOutput: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: string[];
  additionalMetadata?: Record<string, any>;
  comments?: string;
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  reasoning?: string;
  tokenCost?: number;
  completionTime?: number;
  name?: string;
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;

  constructor(params: {
    input: string;
    actualOutput: string;
    expectedOutput?: string;
    context?: string[];
    retrievalContext?: string[];
    additionalMetadata?: Record<string, any>;
    comments?: string;
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    reasoning?: string;
    tokenCost?: number;
    completionTime?: number;
    name?: string;
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
    this.reasoning = params.reasoning;
    this.tokenCost = params.tokenCost;
    this.completionTime = params.completionTime;
    this.name = params.name;
    this._datasetRank = params._datasetRank;
    this._datasetAlias = params._datasetAlias;
    this._datasetId = params._datasetId;
    this.validate();
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
        !this.retrievalContext.every((item) => typeof item === "string")
      ) {
        throw new TypeError(
          "'retrievalContext' must be undefined or an array of strings",
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
  }
}

export class Turn {
  role: "user" | "assistant";
  content: string;
  userId?: string;
  retrievalContext?: string[];
  toolsCalled?: ToolCall[];
  additionalMetadata?: Record<string, any>;

  constructor(params: {
    role: "user" | "assistant";
    content: string;
    userId?: string;
    retrievalContext?: string[];
    toolsCalled?: ToolCall[];
    additionalMetadata?: Record<string, any>;
  }) {
    this.role = params.role;
    this.content = params.content;
    this.userId = params.userId;
    this.retrievalContext = params.retrievalContext;
    this.toolsCalled = params.toolsCalled;
    this.additionalMetadata = params.additionalMetadata;
    this.validate();
  }

  private validate(): void {
    if (this.retrievalContext != null) {
      if (
        !Array.isArray(this.retrievalContext) ||
        !this.retrievalContext.every((s) => typeof s === "string")
      ) {
        throw new TypeError(
          "'retrievalContext' must be undefined or an array of strings",
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
  name?: string;
  additionalMetadata?: Record<string, any>;
  comments?: string;
  tags?: string[];
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
    name?: string;
    additionalMetadata?: Record<string, any>;
    comments?: string;
    tags?: string[];
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
    this.name = params.name;
    this.additionalMetadata = params.additionalMetadata;
    this.comments = params.comments;
    this.tags = params.tags;
    this._datasetRank = params._datasetRank;
    this._datasetAlias = params._datasetAlias;
    this._datasetId = params._datasetId;
    this.validate();
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
  }
}
