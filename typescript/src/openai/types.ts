import { ToolCall } from "../test-case";

export interface InputParameters {
  model?: string;
  input?: string;
  tools?: Array<Record<string, any>>;
  instructions?: string;
  messages?: Array<Record<string, any>>;
  toolDescriptions?: Record<string, string>;
}

export interface OutputParameters {
  output?: any;
  promptTokens?: number;
  completionTokens?: number;
  toolsCalled?: ToolCall[];
}
