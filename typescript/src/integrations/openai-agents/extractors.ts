import {
  BaseSpan,
  LlmSpan,
  ToolSpan,
  AgentSpan,
  Trace,
} from "../../tracing/tracing";

import {
  SpanData,
  ResponseSpanData,
  GenerationSpanData,
  FunctionSpanData,
  AgentSpanData,
  HandoffSpanData,
  CustomSpanData,
  GuardrailSpanData,
} from "@openai/agents";

function makeJsonSerializable(obj: any): string {
  if (typeof obj === "string") return obj;
  try {
    return JSON.stringify(obj);
  } catch {
    return String(obj);
  }
}

export function updateTracePropertiesFromSpanData(
  trace: Trace,
  spanData: any,
): void {
  if (!trace) return;

  // 1. Give priority to Agent span inputs/outputs for the whole Trace
  if (spanData.type === "agent") {
    if (!trace.input && spanData.input) {
      trace.input = spanData.input;
    }
    if (spanData.output) {
      const rawOutput = spanData.output;
      trace.output =
        typeof rawOutput === "string" ? rawOutput : JSON.stringify(rawOutput);
    }
  }

  // 2. LLM Response
  else if (spanData.type === "response") {
    const data = spanData;
    const response = data._response || data.response;
    const input = data._input || data.input;

    if (!trace.input) {
      trace.input = parseResponseInput(input, response?.instructions);
    }

    if (!trace.output) {
      const rawOutput = parseResponseOutput(response?.output);
      if (rawOutput != null) {
        trace.output =
          typeof rawOutput === "string" ? rawOutput : JSON.stringify(rawOutput);
      }
    }
  }

  // 3. LLM Generation
  else if (spanData.type === "generation") {
    if (!trace.input) {
      trace.input = spanData.input;
    }

    // Add safety check so we don't overwrite Agent output
    if (!trace.output) {
      const rawOutput = spanData.output;
      if (rawOutput != null) {
        trace.output =
          typeof rawOutput === "string" ? rawOutput : JSON.stringify(rawOutput);
      }
    }
  }

  // 4. Function/Tool Execution
  if (spanData.type === "function" || spanData.type === "mcp_tools") {
    if (!trace.toolsCalled) {
      trace.toolsCalled = [];
    }

    let inputParams;
    try {
      inputParams = JSON.parse(spanData.input);
    } catch {
      inputParams = { input: spanData.input };
    }

    trace.toolsCalled.push({
      name: spanData.name || "Function tool",
      inputParameters: inputParams,
      output: spanData.output,
      description: spanData.description,
    });
  }
}

export function updateSpanProperties(span: BaseSpan, spanData: SpanData): void {
  // 1. LLM Spans (Response & Generation)
  if (spanData.type === "response") {
    updateSpanPropertiesFromResponseSpanData(span as LlmSpan, spanData);
  } else if (spanData.type === "generation") {
    updateSpanPropertiesFromGenerationSpanData(span as LlmSpan, spanData);
  }

  // 2. Tool Spans
  else if (spanData.type === "function") {
    updateSpanPropertiesFromFunctionSpanData(span as ToolSpan, spanData);
  } else if (spanData.type === "mcp_tools") {
    updateSpanPropertiesFromMcpListToolsSpanData(span as ToolSpan, spanData);
  }

  // 3. Agent Spans
  else if (spanData.type === "agent") {
    updateSpanPropertiesFromAgentSpanData(span as AgentSpan, spanData);
  }

  // 4. Custom / Handoff / Guardrail Spans
  else if (spanData.type === "handoff") {
    updateSpanPropertiesFromHandoffSpanData(span as AgentSpan, spanData);
  } else if (spanData.type === "custom") {
    updateSpanPropertiesFromCustomSpanData(span, spanData);
  } else if (spanData.type === "guardrail") {
    updateSpanPropertiesFromGuardrailSpanData(span, spanData);
  }
}

// ==========================================================================
// LLM Span Updaters
// ==========================================================================

function updateSpanPropertiesFromResponseSpanData(
  span: LlmSpan,
  spanData: ResponseSpanData,
): void {
  const response = spanData._response;
  if (!response) {
    span.model = "NA";
    return;
  }

  // Extract usage tokens
  const usage = response.usage;
  let cachedInputTokens: number | undefined;
  let outputReasoningTokens: number | undefined;

  if (usage) {
    span.outputTokenCount = usage.output_tokens;
    span.inputTokenCount = usage.input_tokens;

    // Check for detailed usage if available
    if (usage.input_tokens_details) {
      cachedInputTokens = usage.input_tokens_details.cached_tokens;
    }
    if (usage.output_tokens_details) {
      outputReasoningTokens = usage.output_tokens_details.reasoning_tokens;
    }
  }

  // Parse Input and Output
  const input = parseResponseInput(spanData._input, response.instructions);
  const rawOutput = parseResponseOutput(response.output);
  const output =
    rawOutput != null
      ? typeof rawOutput === "string"
        ? rawOutput
        : JSON.stringify(rawOutput)
      : undefined;

  // Update Span Fields
  span.input = input;
  span.output = output;
  span.model = response.model || "NA";
  span.name = "LLM Generation";

  span.metadata = {
    ...(span.metadata || {}),
    cached_input_tokens: cachedInputTokens,
    output_reasoning_tokens: outputReasoningTokens,
  };

  const responseDict = response;
  const invocationParams: Record<string, any> = {};

  const keysToCapture = [
    "max_output_tokens",
    "parallel_tool_calls",
    "reasoning",
    "text",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "truncation",
  ];

  keysToCapture.forEach((key) => {
    if (key in responseDict) {
      invocationParams[key] = responseDict[key];
    }
  });

  span.metadata["invocation_params"] = invocationParams;
}

function updateSpanPropertiesFromGenerationSpanData(
  span: LlmSpan,
  spanData: GenerationSpanData,
): void {
  const usage = spanData.usage;
  if (usage) {
    span.outputTokenCount = usage.output_tokens;
    span.inputTokenCount = usage.input_tokens;
  }

  const input = spanData.input;
  const rawOutput = spanData.output;
  const output =
    rawOutput != null
      ? typeof rawOutput === "string"
        ? rawOutput
        : JSON.stringify(rawOutput)
      : undefined;

  span.model = spanData.model || "NA";
  span.input = input;
  span.output = output;
  span.name = "LLM Generation";

  if (spanData.model_config) {
    span.metadata = {
      ...(span.metadata || {}),
      invocation_params: {
        model_config: makeJsonSerializable(spanData.model_config),
      },
    };
  }
}

// ==========================================================================
// Tool Span Updaters
// ==========================================================================

function updateSpanPropertiesFromFunctionSpanData(
  span: ToolSpan,
  spanData: FunctionSpanData,
): void {
  try {
    span.input = JSON.parse(spanData.input);
  } catch {
    span.input = { input: spanData.input };
  }

  span.output = spanData.output;
  span.name = spanData.name
    ? `Function tool: ${spanData.name}`
    : "Function tool";
  span.description = "Function tool";
}

// ==========================================================================
// Agent Span Updaters
// ==========================================================================

function updateSpanPropertiesFromAgentSpanData(
  span: AgentSpan,
  spanData: AgentSpanData,
): void {
  span.name = spanData.name;
  span.agentHandoffs = spanData.handoffs ? [...spanData.handoffs] : [];
  span.availableTools = spanData.tools ? [...spanData.tools] : [];

  // Populate input/output directly from spanData when available.
  // Cast to any because the SDK type definition omits these runtime fields.
  const rawInput = (spanData as any).input;
  if (rawInput != null) {
    span.input = rawInput;
  }

  const rawOutput = (spanData as any).output;
  if (rawOutput != null) {
    span.output =
      typeof rawOutput === "string" ? rawOutput : JSON.stringify(rawOutput);
  }

  const metadata: Record<string, any> = { ...(span.metadata || {}) };
  if (spanData.output_type) {
    metadata["output_type"] = spanData.output_type;
  }
  span.metadata = metadata;
}

function updateSpanPropertiesFromMcpListToolsSpanData(
  span: ToolSpan,
  spanData: any,
): void {
  span.input = null;
  span.output = spanData.result;
  span.name = spanData.server ? `MCP tool: ${spanData.server}` : "MCP tool";
  span.description = "MCP tool";
}

// ==========================================================================
// Custom / Handoff / Guardrail Updaters
// ==========================================================================

function updateSpanPropertiesFromHandoffSpanData(
  span: AgentSpan,
  spanData: HandoffSpanData,
): void {
  span.name = `Handoff → ${spanData.to_agent}`;
  span.input = null; // Handoffs usually don't have distinct IO in this context
  span.output = null;
  span.metadata = {
    ...(span.metadata || {}),
    from_agent: spanData.from_agent,
    to_agent: spanData.to_agent,
  };
}

function updateSpanPropertiesFromCustomSpanData(
  span: BaseSpan,
  spanData: CustomSpanData,
): void {
  span.name = spanData.name;
  span.metadata = {
    ...(span.metadata || {}),
    data: spanData.data,
  };
}

function updateSpanPropertiesFromGuardrailSpanData(
  span: BaseSpan,
  spanData: GuardrailSpanData,
): void {
  span.name = `Guardrail: ${spanData.name}`;
  span.metadata = {
    ...(span.metadata || {}),
    triggered: spanData.triggered,
    type: spanData.type,
  };
}

// ==========================================================================
// Input Parsing Utilities
// ==========================================================================

export function parseResponseInput(
  input: any,
  instructions: any,
): any[] | null {
  const processedInput = [];

  // Case 1: Both are strings -> Create System + User messages
  if (typeof input === "string" && typeof instructions === "string") {
    return [
      { type: "message", role: "system", content: instructions },
      { type: "message", role: "user", content: input },
    ];
  }
  // Case 2: Both are lists -> Concatenate
  else if (Array.isArray(input) && Array.isArray(instructions)) {
    // Treat instructions as prepended context
    const combined = [...instructions, ...input];
    return processInputList(combined);
  }
  // Case 3: List input, String instructions
  else if (Array.isArray(input) && typeof instructions === "string") {
    processedInput.push({
      type: "message",
      role: "system",
      content: instructions,
    });
    // Process the rest of the input list
    const remaining = processInputList(input);
    if (remaining) processedInput.push(...remaining);
    return processedInput;
  }
  // Case 4: String input, List instructions
  else if (typeof input === "string" && Array.isArray(instructions)) {
    const processedInstructions = processInputList(instructions);
    if (processedInstructions) processedInput.push(...processedInstructions);
    processedInput.push({ type: "message", role: "user", content: input });
    return processedInput;
  }

  // Fallback: simple processing of input if it matches none above exactly
  if (Array.isArray(input)) {
    return processInputList(input);
  }

  return null;
}

function processInputList(itemList: any[]): any[] | null {
  const processed: any[] = [];

  for (const item of itemList) {
    // If it's a raw dict with role/content but no type, standardise it
    if (!item.type && item.role && item.content) {
      processed.push({
        type: "message",
        role: item.role,
        content: item.content,
      });
      continue;
    }

    if (item.type === "message") {
      const parsed = parseMessageParam(item);
      if (parsed) processed.push(parsed);
    } else if (item.type === "function_call") {
      processed.push(parseFunctionToolCallParam(item));
    } else if (item.type === "function_call_output") {
      processed.push(parseFunctionCallOutput(item));
    }
  }

  return processed.length > 0 ? processed : null;
}

function parseMessageParam(message: any): any {
  const role = message.role;
  const content = message.content;

  if (typeof content === "string") {
    return { type: "message", role, content };
  } else if (Array.isArray(content)) {
    return { type: "message", role, content: parseMessageContentList(content) };
  }
  return null;
}

function parseMessageContentList(contentList: any[]): any[] | null {
  const processed: any[] = [];
  for (const item of contentList) {
    if (
      item.type === "input_text" ||
      item.type === "output_text" ||
      item.type === "text"
    ) {
      processed.push({ type: "text", text: item.text });
    } else if (item.type === "refusal") {
      processed.push({ type: "refusal", refusal: item.refusal });
    }
  }
  return processed.length > 0 ? processed : null;
}

function parseFunctionToolCallParam(toolCall: any): any {
  return {
    call_id: toolCall.call_id,
    name: toolCall.name,
    arguments: toolCall.arguments,
  };
}

function parseFunctionCallOutput(output: any): any {
  return {
    role: "tool",
    call_id: output.call_id,
    output: output.output,
  };
}
// ==========================================================================
// Output Parsing Utilities
// ==========================================================================

export function parseResponseOutput(responseItems: any): any {
  if (typeof responseItems === "string") return responseItems;
  if (!responseItems) return null;

  if (!Array.isArray(responseItems)) {
    if (typeof responseItems.content === "string") return responseItems.content;
    if (typeof responseItems.output === "string") return responseItems.output;
    return null;
  }

  const processedOutput: any[] = [];
  for (const item of responseItems) {
    if (typeof item === "string") {
      processedOutput.push(item);
    } else if (
      item.type === "message" ||
      item.role === "assistant" ||
      item.status === "completed"
    ) {
      const message = parseMessageOutput(item);
      if (message) {
        if (typeof message === "string") {
          processedOutput.push(message);
        } else if (Array.isArray(message)) {
          processedOutput.push(...message);
        }
      }
    } else if (item.type === "function_call") {
      processedOutput.push(parseFunctionCall(item));
    }
  }

  if (processedOutput.length === 1) return processedOutput[0];
  return processedOutput.length > 0 ? processedOutput : null;
}

function parseMessageOutput(message: any): string | string[] | null {
  if (!message.content) return null;

  // If the SDK returned a flat string, return it immediately
  if (typeof message.content === "string") {
    return message.content;
  }

  // If it's an array, map through the text blocks
  if (Array.isArray(message.content)) {
    const processedContent: string[] = [];

    for (const item of message.content) {
      if (typeof item === "string") {
        processedContent.push(item);
      } else if (
        (item.type === "text" || item.type === "output_text") &&
        item.text
      ) {
        processedContent.push(item.text);
      } else if (item.type === "refusal" && item.refusal) {
        processedContent.push(item.refusal);
      }
    }

    if (processedContent.length === 1) return processedContent[0];
    return processedContent.length > 0 ? processedContent : null;
  }

  return null;
}

function parseFunctionCall(functionCall: any): any {
  return {
    call_id: functionCall.call_id,
    name: functionCall.name,
    arguments: functionCall.arguments,
  };
}
