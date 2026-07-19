export const MastraSpanType = {
  AGENT_RUN: "agent_run",
  SCORER_RUN: "scorer_run",
  SCORER_STEP: "scorer_step",
  GENERIC: "generic",
  MODEL_GENERATION: "model_generation",
  MODEL_STEP: "model_step",
  MODEL_INFERENCE: "model_inference",
  MODEL_CHUNK: "model_chunk",
  MCP_TOOL_CALL: "mcp_tool_call",
  PROCESSOR_RUN: "processor_run",
  TOOL_CALL: "tool_call",
  CLIENT_TOOL_CALL: "client_tool_call",
  PROVIDER_TOOL_CALL: "provider_tool_call",
  WORKFLOW_RUN: "workflow_run",
  WORKFLOW_STEP: "workflow_step",
  WORKFLOW_CONDITIONAL: "workflow_conditional",
  WORKFLOW_CONDITIONAL_EVAL: "workflow_conditional_eval",
  WORKFLOW_PARALLEL: "workflow_parallel",
  WORKFLOW_LOOP: "workflow_loop",
  WORKFLOW_SLEEP: "workflow_sleep",
  WORKFLOW_WAIT_EVENT: "workflow_wait_event",
  MEMORY_OPERATION: "memory_operation",
  WORKSPACE_ACTION: "workspace_action",
  RAG_INGESTION: "rag_ingestion",
  RAG_EMBEDDING: "rag_embedding",
  RAG_VECTOR_OPERATION: "rag_vector_operation",
  RAG_ACTION: "rag_action",
  GRAPH_ACTION: "graph_action",
  MAPPING: "mapping",
} as const;

export const MastraTracingEventType = {
  SPAN_STARTED: "span_started",
  SPAN_UPDATED: "span_updated",
  SPAN_ENDED: "span_ended",
} as const;

export type MastraTracingEventTypeValue =
  (typeof MastraTracingEventType)[keyof typeof MastraTracingEventType];

export interface MastraTokenDetails {
  text?: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
  audio?: number;
  image?: number;
}

export interface MastraUsageStats {
  inputTokens?: number;
  outputTokens?: number;
  inputDetails?: MastraTokenDetails;
  outputDetails?: MastraTokenDetails;
}

export interface MastraCostContext {
  provider?: string;
  model?: string;
  estimatedCost?: number;
  costUnit?: string;
  costMetadata?: Record<string, unknown>;
}

export interface MastraSpanAttributes {
  // model spans
  model?: string;
  provider?: string;
  responseModel?: string;
  responseId?: string;
  finishReason?: string;
  streaming?: boolean;
  completionStartTime?: Date | string;
  usage?: MastraUsageStats;
  parameters?: Record<string, unknown>;
  costContext?: MastraCostContext;
  // agent spans
  conversationId?: string;
  instructions?: string;
  availableTools?: string[];
  // tool spans
  toolType?: string;
  toolDescription?: string;
  toolCallId?: string;
  success?: boolean;
  mcpServer?: string;
  // rag spans
  dimensions?: number;
  inputCount?: number;
  topK?: number;
  [key: string]: unknown;
}

export interface MastraSpanErrorInfo {
  message: string;
  id?: string;
  name?: string;
  stack?: string;
  domain?: string;
  category?: string;
  details?: Record<string, any>;
}

export interface MastraExportedSpan {
  id: string;
  traceId: string;
  name: string;
  type: string;
  entityType?: string;
  entityId?: string;
  entityName?: string;
  startTime: Date | string;
  endTime?: Date | string;
  attributes?: MastraSpanAttributes;
  metadata?: Record<string, any>;
  tags?: string[];
  input?: any;
  output?: any;
  errorInfo?: MastraSpanErrorInfo;
  requestContext?: Record<string, any>;
  isEvent: boolean;
  isRootSpan: boolean;
  parentSpanId?: string;
}

export type MastraTracingEvent = {
  type: MastraTracingEventTypeValue | string;
  exportedSpan: MastraExportedSpan;
};

export interface MastraInitExporterOptions {
  mastra?: unknown;
  config?: {
    name?: string;
    serviceName?: string;
    [key: string]: unknown;
  };
  emitDropEvent?: (event: unknown) => void;
}

export interface MastraObservabilityExporter {
  name: string;
  init?(options: MastraInitExporterOptions): void;
  __setLogger?(logger: unknown): void;
  onTracingEvent?(event: MastraTracingEvent): void | Promise<void>;
  exportTracingEvent(event: MastraTracingEvent): Promise<void>;
  flush(): Promise<void>;
  shutdown(): Promise<void>;
}
