import { ToolCall, Turn } from "../test-case";

export class Golden {
  input: string;
  actualOutput?: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: string[];
  additionalMetadata?: Record<string, any>;
  comments?: string;
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  sourceFile?: string;
  customColumnKeyValues?: Record<string, string>;
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;

  constructor(params: {
    input: string;
    actualOutput?: string;
    expectedOutput?: string;
    context?: string[];
    retrievalContext?: string[];
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    additionalMetadata?: Record<string, any>;
    sourceFile?: string;
    customColumnKeyValues?: Record<string, string>;
    comments?: string;
    _datasetRank?: number;
    _datasetAlias?: string;
    _datasetId?: string;
  }) {
    this.input = params.input;
    this.actualOutput = params.actualOutput;
    this.expectedOutput = params.expectedOutput;
    this.context = params.context;
    this.retrievalContext = params.retrievalContext;
    this.toolsCalled = params.toolsCalled;
    this.expectedTools = params.expectedTools;
    this.additionalMetadata = params.additionalMetadata;
    this.sourceFile = params.sourceFile;
    this.comments = params.comments;
    this._datasetRank = params._datasetRank;
    this._datasetAlias = params._datasetAlias;
    this._datasetId = params._datasetId;
    this.customColumnKeyValues = params.customColumnKeyValues;
  }
}

export class ConversationalGolden {
  scenario: string;
  expectedOutcome?: string;
  userDescription?: string;
  context?: string[];
  additionalMetadata?: Record<string, any>;
  comments?: string;
  name?: string;
  customColumnKeyValues?: Record<string, string>;
  turns?: Turn[];
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;

  constructor(params: {
    scenario: string;
    expectedOutcome?: string;
    userDescription?: string;
    context?: string[];
    additionalMetadata?: Record<string, any>;
    comments?: string;
    name?: string;
    customColumnKeyValues?: Record<string, string>;
    turns?: Turn[];
    _datasetRank?: number;
    _datasetAlias?: string;
    _datasetId?: string;
  }) {
    this.scenario = params.scenario;
    this.expectedOutcome = params.expectedOutcome;
    this.userDescription = params.userDescription;
    this.context = params.context;
    this.additionalMetadata = params.additionalMetadata;
    this.comments = params.comments;
    this.name = params.name;
    this.customColumnKeyValues = params.customColumnKeyValues;
    this.turns = params.turns;
    this._datasetRank = params._datasetRank;
    this._datasetAlias = params._datasetAlias;
    this._datasetId = params._datasetId;
  }
}
