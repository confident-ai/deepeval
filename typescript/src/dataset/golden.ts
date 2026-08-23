import { RetrievedContextData, ToolCall, Turn } from "@/test-case";
import { Persona, resolvePersona } from "@/dataset/persona";

export class Golden {
  id?: string;
  input: string;
  actualOutput?: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: (string | RetrievedContextData)[];
  additionalMetadata?: Record<string, any>;
  comments?: string;
  name?: string;
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  sourceFile?: string;
  customColumnKeyValues?: Record<string, string>;
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;

  constructor(params: {
    id?: string;
    input: string;
    actualOutput?: string;
    expectedOutput?: string;
    context?: string[];
    retrievalContext?: (string | RetrievedContextData)[];
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    additionalMetadata?: Record<string, any>;
    sourceFile?: string;
    customColumnKeyValues?: Record<string, string>;
    comments?: string;
    name?: string;
    _datasetRank?: number;
    _datasetAlias?: string;
    _datasetId?: string;
  }) {
    this.id = params.id;
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
    this.name = params.name;
    this._datasetRank = params._datasetRank;
    this._datasetAlias = params._datasetAlias;
    this._datasetId = params._datasetId;
    this.customColumnKeyValues = params.customColumnKeyValues;
  }
}

export class ConversationalGolden {
  id?: string;
  scenario: string;
  expectedOutcome?: string;
  persona?: Persona;
  /** @deprecated Use `persona`. Kept in sync with `persona.characteristics`. */
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
    id?: string;
    scenario: string;
    expectedOutcome?: string;
    persona?: Persona;
    /** @deprecated Use `persona`. */
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
    this.id = params.id;
    this.scenario = params.scenario;
    this.expectedOutcome = params.expectedOutcome;
    const resolved = resolvePersona(params.persona, params.userDescription);
    this.persona = resolved.persona;
    this.userDescription = resolved.userDescription;
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
