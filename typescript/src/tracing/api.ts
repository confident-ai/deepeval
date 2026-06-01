import { Prompt } from "../prompt";

export enum SpanApiType {
  BASE = "base",
  AGENT = "agent",
  LLM = "llm",
  RETRIEVER = "retriever",
  TOOL = "tool",
}

export enum TraceSpanApiStatus {
  SUCCESS = "SUCCESS",
  ERROR = "ERRORED",
}

export interface ToolCall {
  name: string;
  description?: string;
  inputParameters?: Record<string, any>;
  output?: any;
  reasoning?: string;
}

export interface LLMTestCase {
  input: string;
  actualOutput: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: string[];
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
}

export interface BaseApiSpan {
  uuid: string;
  name?: string;
  status: TraceSpanApiStatus;
  type: SpanApiType | string;
  traceUuid: string;
  parentUuid?: string;
  startTime: string;
  endTime: string;
  input?: any;
  output?: any;
  error?: string;

  // agents
  availableTools?: string[];
  agentHandoffs?: string[];

  // tools
  description?: string;

  // retriever
  embedder?: string;
  topK?: number;
  chunkSize?: number;

  // llm
  model?: string;
  inputTokenCount?: number;
  outputTokenCount?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  prompt?: Prompt;
  promptCommitHash?: string;
  promptAlias?: string;
  promptLabel?: string;
  promptVersion?: string;

  // evals
  llmTestCase?: LLMTestCase;
  metrics?: string[];

  // metadata
  metadata?: Record<string, any>;
  metricCollection?: string;

  // additional test case params
  expectedOutput?: string;
  retrievalContext?: string[];
  context?: string[];
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
}

export interface TraceApi {
  uuid: string;
  status?: TraceSpanApiStatus;
  baseSpans: BaseApiSpan[];
  agentSpans: BaseApiSpan[];
  llmSpans: BaseApiSpan[];
  retrieverSpans: BaseApiSpan[];
  toolSpans: BaseApiSpan[];
  startTime: string;
  endTime: string;
  environment: string;
  metadata?: Record<string, any>;
  tags?: string[];
  threadId?: string;
  userId?: string;
  testCaseId?: string;
  turnId?: string;
  input?: any;
  output?: any;
  name?: string;

  // additional test case params
  expectedOutput?: string;
  retrievalContext?: string[];
  context?: string[];
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  metricCollection?: string;

  // Don't serialize this
  confidentApiKey?: string;
}
