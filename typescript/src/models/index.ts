export { DeepEvalBaseLLM, type GenerationResult } from "./base-model";

// Shared base for every OpenAI-Chat-Completions-compatible provider/gateway.
// Exported so advanced users can target any OpenAI-compatible endpoint directly.
export {
  DeepEvalOpenAICompatibleModel,
  type OpenAICompatibleModelOptions,
} from "./openai-compatible-model";

// Providers
export { OpenAIModel, type OpenAIModelOptions } from "./providers/openai-model";
export {
  AzureOpenAIModel,
  type AzureOpenAIModelOptions,
} from "./providers/azure-model";
export {
  AnthropicModel,
  type AnthropicModelOptions,
} from "./providers/anthropic-model";
export { GeminiModel, type GeminiModelOptions } from "./providers/gemini-model";
export {
  AmazonBedrockModel,
  type AmazonBedrockModelOptions,
} from "./providers/bedrock-model";
export {
  DeepSeekModel,
  type DeepSeekModelOptions,
} from "./providers/deepseek-model";
export { GrokModel, type GrokModelOptions } from "./providers/grok-model";
export { KimiModel, type KimiModelOptions } from "./providers/kimi-model";
export { LocalModel, type LocalModelOptions } from "./providers/local-model";
export { OllamaModel, type OllamaModelOptions } from "./providers/ollama-model";
export { AISDKModel, type AISDKModelOptions } from "./providers/ai-sdk-model";

// Gateways
export {
  OpenRouterModel,
  type OpenRouterModelOptions,
} from "./gateways/openrouter-model";
export {
  PortkeyModel,
  type PortkeyModelOptions,
} from "./gateways/portkey-model";
